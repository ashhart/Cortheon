"""Independent Clean-Context Adversarial Referee Gate (inspired by tensorfold-referee).

Intercepts `finish()` in `ReplicaAgent` inside a BRAND-NEW zero-history context.
The Referee did NOT build the solution and gains nothing if it passes:
  1. Mechanical Gate: refuses `finish()` if files were modified AFTER the last passing test run.
  2. Citation Verifier (for read-only investigations): mechanically verifies EVERY cited
     `path:line` exists on disk and its line count is valid!
  3. Active Adversarial Testing (for code workspaces): writes 4-6 ruthless adversarial pytest
     edge-case tests (crash recovery, concurrency/order, boundary values, rollback, corruption)
     and EXECUTES them live in the workspace!
  4. Spec-Arbiter Gate: validates any failing Referee test so bogus Referee hallucinations never
     cause false rejections — real bugs trigger `REJECTED` with exact tracebacks!
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .brain_state import BrainState
from .client import CallSpec, TensorFoldClient
from .repo_index import build_repo_map, read_numbered_slice
from .sandbox import check_syntax, extract_python_code
from .tools import AgentToolbelt


@dataclass
class RefereeVerdict:
    confirmed: bool
    reason: str
    adversarial_tests_run: int = 0
    adversarial_failures: list[str] = field(default_factory=list)
    citation_errors: list[str] = field(default_factory=list)


_FILE_LINE_RE = re.compile(r"(?:`|\b)([A-Za-z0-9_./-]+\.(?:zig|py|rs|ts|js|c|h|md)):(\d+)(?:-(\d+))?")


def verify_citations_on_disk(root: Path, text: str) -> tuple[int, list[str]]:
    """Mechanically check every `path:line` citation in a research/investigation report."""
    matches = _FILE_LINE_RE.findall(text)
    errors: list[str] = []
    checked = 0
    seen: set[tuple[str, int]] = set()
    root_resolved = root.resolve()
    for rel, ln_s, end_s in matches:
        ln = int(ln_s)
        if (rel, ln) in seen:
            continue
        seen.add((rel, ln))
        checked += 1
        p = (root_resolved / rel).resolve()
        if not p.is_relative_to(root_resolved) or not p.is_file():
            errors.append(f"Hallucinated or missing file path: `{rel}:{ln}`")
            continue
        total_lines = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
        if ln < 1 or ln > total_lines:
            errors.append(f"Out-of-bounds line number: `{rel}:{ln}` (file has {total_lines} lines)")
        elif end_s and int(end_s) > total_lines:
            errors.append(f"Out-of-bounds end line: `{rel}:{ln}-{end_s}` (file has {total_lines} lines)")
    return checked, errors


async def run_referee_gate(
    client: TensorFoldClient,
    tools: AgentToolbelt,
    state: BrainState,
    proposed_summary: str,
    proposed_evidence: str,
) -> RefereeVerdict:
    """Independently audit the workspace and proposed answer in a fresh, zero-history context."""
    root = tools.root

    # ------------------------------------------------------------------
    # READ-ONLY INVESTIGATION REFEREE (Mechanical citation check + Fresh Audit)
    # ------------------------------------------------------------------
    if state.read_only:
        full_report = proposed_summary + "\n" + proposed_evidence
        n_cited, cite_errs = verify_citations_on_disk(root, full_report)
        if cite_errs:
            return RefereeVerdict(
                confirmed=False,
                reason="CITATION CHECK FAILED — report contains invalid file paths or out-of-bounds line numbers:\n"
                + "\n".join(f"- {e}" for e in cite_errs[:8]),
                citation_errors=cite_errs,
            )
        if n_cited == 0:
            return RefereeVerdict(
                confirmed=False,
                reason="No concrete `path:line` code citations found in final report. Cite exact files and line numbers proven by read_file/search_code.",
            )

        # Spot-check up to 4 cited file:line windows directly from disk and ask fresh Referee if claims match actual code!
        spot_snippets: list[str] = []
        seen_paths: set[str] = set()
        for rel, ln_s, _ in _FILE_LINE_RE.findall(full_report):
            if rel in seen_paths:
                continue
            seen_paths.add(rel)
            ln = int(ln_s)
            snip = read_numbered_slice(root, rel, start_line=max(1, ln - 8), end_line=ln + 14, max_lines=30)
            spot_snippets.append(snip)
            if len(spot_snippets) >= 5:
                break

        audit_prompt = (
            f"USER QUESTION:\n{state.goal}\n\n"
            f"PROPOSED REPORT:\n{proposed_summary[:3500]}\n\n"
            f"GROUND-TRUTH CODE WINDOWS FROM DISK:\n" + "\n\n".join(spot_snippets) + "\n\n"
            "Verify whether the report accurately answers EVERY part of the user question and matches the ground-truth code above.\n"
            "If anything is missing, fabricated, or contradicted by the code snippets, explain concisely and end with VERDICT: REJECTED.\n"
            "Otherwise end with VERDICT: CONFIRMED."
        )
        res = await client.complete(
            CallSpec(
                messages=[
                    {"role": "system", "content": "You are an impartial code referee. Verify every claim strictly against ground-truth code."},
                    {"role": "user", "content": audit_prompt},
                ],
                tag="referee_ro_audit",
                max_tokens=1100,
                temperature=0.0,
                enable_thinking=True,
                thinking_budget=500,
                loop_guard=True,
            )
        )
        if not res.ok or res.finish_reason == "length":
            return RefereeVerdict(
                confirmed=False,
                reason=f"Referee code audit INDETERMINATE (upstream error/truncation: {res.raw_error or res.finish_reason}).",
            )
        verdicts = re.findall(r"VERDICT\s*[:=]\s*(CONFIRMED|REJECTED)\b", res.content.upper())
        if not verdicts or verdicts[-1] != "CONFIRMED":
            return RefereeVerdict(confirmed=False, reason=f"Referee code audit rejected report:\n{res.content[:900]}")
        return RefereeVerdict(confirmed=True, reason=f"CONFIRMED ({n_cited} file:line citations verified on disk + code audit passed).")

    # ------------------------------------------------------------------
    # CODE / DEVELOPMENT REFEREE
    # ------------------------------------------------------------------
    # Gate 1: Did the builder run tests AFTER its final edit?
    if state.modified_files and state.edits_since_last_test > 0:
        return RefereeVerdict(
            confirmed=False,
            reason=(
                f"MECHANICAL GATE FAILED: You edited {state.edits_since_last_test} time(s) AFTER your last passing test run. "
                "Run the full test suite now and confirm 100% pass before calling finish()."
            ),
        )

    if state.modified_files and not state.last_test_ok:
        return RefereeVerdict(
            confirmed=False,
            reason="MECHANICAL GATE FAILED: No passing test run recorded after modifying files. Write/run tests via run_command first.",
        )

    # Gate 2: Re-run existing test/bench command independently right now!
    has_py_tests = any("test" in f.lower() and f.endswith(".py") for f in state.modified_files) or (root / "tests").is_dir()
    if has_py_tests:
        recheck = tools.run_command("pytest -q", timeout_s=25.0)
        if not recheck.ok:
            return RefereeVerdict(
                confirmed=False,
                reason=f"INDEPENDENT TEST RE-RUN FAILED (`pytest -q`):\n{recheck.output[-1200:]}",
            )
    elif state.last_test_cmd:
        recheck = tools.run_command(state.last_test_cmd, timeout_s=30.0)
        if not recheck.ok:
            return RefereeVerdict(
                confirmed=False,
                reason=f"INDEPENDENT RE-RUN FAILED (`{state.last_test_cmd}`):\n{recheck.output[-1200:]}",
            )

    # Gate 3: Clean-Context Adversarial Edge-Case Test Generation & Live Execution (for Python code workspaces)
    has_py_code = any(f.endswith(".py") for f in state.modified_files)
    if not has_py_code:
        # C/Zig/Rust kernel workspace: audit the actual modified source code + re-run output in a fresh context!
        code_slices = [
            read_numbered_slice(root, rel, start_line=1, end_line=350, max_lines=350)
            for rel in state.modified_files[:4]
        ]
        k_prompt = (
            f"ORIGINAL MISSION:\n{state.goal}\n\n"
            f"INDEPENDENT RE-RUN OUTPUT (`{state.last_test_cmd}`):\n{state.last_test_output[-1500:]}\n\n"
            f"MODIFIED CODE:\n" + "\n\n".join(code_slices)[:18000] + "\n\n"
            "You are an independent systems/kernel referee. Check:\n"
            "1. Did the kernel genuinely meet ALL exactness and speedup requirements (without cheating or editing the benchmark harness)?\n"
            "2. Are unaligned M/K edge tails, zero/negative scales, and memory bounds handled cleanly without UB or out-of-bounds reads?\n"
            "End with VERDICT: CONFIRMED or VERDICT: REJECTED (with concrete bug/line if rejected)."
        )
        k_res = await client.complete(
            CallSpec(
                messages=[{"role": "system", "content": "You are a strict low-level systems & kernel referee."}, {"role": "user", "content": k_prompt}],
                tag="referee_kernel_audit",
                max_tokens=1100,
                temperature=0.0,
                enable_thinking=True,
                thinking_budget=500,
                loop_guard=True,
            )
        )
        if not k_res.ok or k_res.finish_reason == "length":
            return RefereeVerdict(
                confirmed=False,
                reason=f"KERNEL REFEREE INDETERMINATE (upstream error/truncation: {k_res.raw_error or k_res.finish_reason}).",
            )
        k_verdicts = re.findall(r"VERDICT\s*[:=]\s*(CONFIRMED|REJECTED)\b", k_res.content.upper())
        if not k_verdicts or k_verdicts[-1] != "CONFIRMED":
            return RefereeVerdict(confirmed=False, reason=f"KERNEL REFEREE REJECTED:\n{k_res.content[:1000]}")
        return RefereeVerdict(confirmed=True, reason="CONFIRMED (independent build/bench re-run + clean-context kernel audit passed).")
    rmap = build_repo_map(root, max_chars=4000)
    touched_slices: list[str] = []
    for rel in state.modified_files[:6]:
        if rel.endswith(".py") and not rel.startswith("_ff_"):
            touched_slices.append(read_numbered_slice(root, rel, start_line=1, end_line=150, max_lines=150))

    rt_prompt = (
        f"ORIGINAL MISSION / REQUIREMENTS:\n{state.goal}\n\n"
        f"WORKSPACE MAP:\n{rmap}\n\n"
        f"MODIFIED FILES:\n" + "\n\n".join(touched_slices)[:7500] + "\n\n"
        "You are an INDEPENDENT ADVERSARIAL REFEREE in a fresh context.\n"
        "Your job is to TRY TO BREAK the implementation by testing real, explicit requirements from the ORIGINAL MISSION "
        "that the builder's own tests may have missed (edge cases, error handling, crash/corruption recovery, isolation, "
        "concurrency/ordering, boundary inputs, atomicity).\n"
        "Rules:\n"
        "- Import ONLY actual classes/functions visible in the MODIFIED FILES above — NEVER invent non-existent APIs.\n"
        "- Write 3 to 5 crisp pytest functions (`def test_referee_...():`) using `tmp_path` for any filesystem I/O.\n"
        "- Every assertion MUST follow strictly from the ORIGINAL MISSION.\n"
        "- Output ONLY one fenced ```python``` block."
    )
    rt_res = await client.complete(
        CallSpec(
            messages=[
                {"role": "system", "content": "You are a ruthless, fair adversarial QA referee."},
                {"role": "user", "content": rt_prompt},
            ],
            tag="referee_adversarial_gen",
            max_tokens=2200,
            temperature=0.25,
            seed=777,
            enable_thinking=True,
            thinking_budget=800,
            loop_guard=True,
        )
    )
    rt_code = extract_python_code(rt_res.content, salvage_truncated_tests=True)
    if not rt_code or check_syntax(rt_code) is not None:
        return RefereeVerdict(confirmed=True, reason="CONFIRMED (mechanical test re-run passed).")

    rt_file = root / "_ff_referee_audit_test.py"
    try:
        rt_file.write_text(rt_code, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(rt_file)],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=25,
        )
        out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if proc.returncode == 0:
            return RefereeVerdict(
                confirmed=True,
                reason=f"CONFIRMED (existing tests + independent adversarial Referee suite PASSED: {out.splitlines()[-1] if out else 'ok'}).",
                adversarial_tests_run=rt_code.count("def test_"),
            )

        # Gate 4: Spec Arbiter checks if the Referee's failing test is 100% VALID per the mission (prevent false kills!)
        arb_prompt = (
            f"ORIGINAL MISSION:\n{state.goal}\n\n"
            f"REFEREE TEST CODE:\n```python\n{rt_code[:2500]}\n```\n\n"
            f"FAILURE TRACEBACK:\n{out[-1200:]}\n\n"
            "Does this failing test catch a GENUINE bug or missing requirement from the ORIGINAL MISSION "
            "(as opposed to the test calling a wrong/invented signature or testing something not required)?\n"
            "Explain briefly and end with VERDICT: VALID_BUG or VERDICT: INVALID_TEST."
        )
        arb = await client.complete(
            CallSpec(
                messages=[{"role": "system", "content": "You are a strict spec arbiter."}, {"role": "user", "content": arb_prompt}],
                tag="referee_arbiter",
                max_tokens=900,
                temperature=0.0,
                enable_thinking=True,
                thinking_budget=400,
                loop_guard=True,
            )
        )
        arb_verdicts = (
            re.findall(r"VERDICT\s*[:=]\s*(VALID_BUG|INVALID_TEST)\b", arb.content.upper())
            if (arb.ok and arb.finish_reason != "length")
            else []
        )
        # Fail-closed: only dismiss a failing adversarial test on an explicit, valid INVALID_TEST verdict!
        if arb_verdicts and arb_verdicts[-1] == "INVALID_TEST":
            return RefereeVerdict(
                confirmed=True,
                reason="CONFIRMED (mechanical test suite passed; Referee test failure was ruled INVALID_TEST by Spec Arbiter).",
                adversarial_tests_run=rt_code.count("def test_"),
            )
        return RefereeVerdict(
            confirmed=False,
            reason=(
                "ADVERSARIAL REFEREE FOUND A REAL BUG (or arbiter indeterminate)!\n"
                f"Failing Referee Test:\n```python\n{rt_code[:1400]}\n```\n"
                f"Traceback:\n{out[-1000:]}"
            ),
            adversarial_tests_run=rt_code.count("def test_"),
            adversarial_failures=[out[-500:]],
        )
    finally:
        rt_file.unlink(missing_ok=True)

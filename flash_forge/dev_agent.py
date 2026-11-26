"""DevForge: Autonomous Multi-File Software Engineering Agent for Qwen3.8-Flash-Next.

Replicates how a frontier coding agent develops real software — adapted for a fast local MoE:
  Phase 1 — Grounded Recon (Deterministic):
    Build compact AST repo map + run baseline verify_cmd (zero hallucinated paths/errors).
  Phase 2 — Surgical Plan & File Selection (libtfgrammar causal JSON schema):
    Select exact files to read + diagnose hypotheses + optionally run a Python diagnostic probe.
  Phase 3 — Parallel Multi-File Worktree Racing (B=4 over cached prefix):
    4 diverse engineers (minimal_surgical, first_principles, edge_skeptic, alt_design) emit
    multi-file <<<EDIT>>> / <<<FILE>>> patches, each applied and verified in its OWN isolated
    ephemeral git worktree!
  Phase 4 — Clean-Context Diff Red-Teamer + Adversarial Regression Gate:
    Fresh context reviews the winning git diff against the spec and writes adversarial regression
    tests executed inside the worktree. Confirmed bugs are promoted and trigger Wave 2!
  Phase 5 — Graveyard-Tracked Hard Reset Wave 2 (B=4 fresh worktrees):
    Wipes polluted chat history, resets worktrees to HEAD, injects exact tombstones + failing trace,
    and races 4 fresh multi-file repairs!
  Phase 6 — Pairwise guided_choice Diff Tournament + Optional Clean Apply + Episodic Lesson Save.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .client import CallSpec, TensorFoldClient
from .memory import EpisodicMemory
from .repo_index import (
    PatchOp,
    apply_patch_ops,
    build_repo_map,
    git_diff_worktree,
    list_repo_files,
    parse_patch_ops,
    read_numbered_slice,
    rg_search,
)
from .sandbox import ExecResult, check_syntax, extract_python_code
from .skill_loader import SkillPack


RECON_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "files_to_read": {"type": "array", "items": {"type": "string"}},
        "search_queries": {"type": "array", "items": {"type": "string"}},
        "hypotheses": {"type": "array", "items": {"type": "string"}},
        "invariants": {"type": "array", "items": {"type": "string"}},
        "edge_cases": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["files_to_read", "search_queries", "hypotheses", "invariants", "edge_cases"],
    "additionalProperties": False,
}


@dataclass
class WorktreeCandidate:
    name: str
    wave: int
    ops: list[PatchOp] = field(default_factory=list)
    touched_files: list[str] = field(default_factory=list)
    patched_contents: dict[str, str] = field(default_factory=dict)
    diff_text: str = ""
    apply_error: str = ""
    verify_ok: bool = False
    verify_output: str = ""
    redteam_ok: bool = True
    redteam_output: str = ""
    elapsed_s: float = 0.0


@dataclass
class DevForgeResult:
    ok: bool
    winner_name: str
    touched_files: list[str]
    patched_contents: dict[str, str]
    diff_text: str
    wave2_triggered: bool
    redteam_kills: list[str]
    graveyard: list[str]
    wall_s: float
    token_stats: dict[str, int]
    verify_output: str = ""
    lesson_learned: Optional[str] = None


def _run_shell(cmd: str, cwd: Path, timeout_s: float = 30.0) -> ExecResult:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return ExecResult(
            ok=(proc.returncode == 0),
            returncode=proc.returncode,
            stdout=proc.stdout[:12000],
            stderr=proc.stderr[:12000],
            elapsed_s=time.perf_counter() - t0,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return ExecResult(
            ok=False,
            returncode=-1,
            stdout=out[:6000],
            stderr=err[:6000],
            elapsed_s=time.perf_counter() - t0,
            timed_out=True,
        )


def _eval_candidate_in_worktree(
    repo_dir: Path,
    cand: WorktreeCandidate,
    verify_cmd: str,
    extra_test_code: Optional[str] = None,
    timeout_s: float = 30.0,
) -> None:
    """Apply candidate's PatchOps inside a fresh detached git worktree and run verify_cmd (+ optional red-team test)."""
    wt = Path(tempfile.mkdtemp(prefix="ff_dev_wt_"))
    try:
        add = subprocess.run(
            ["git", "-C", str(repo_dir), "worktree", "add", "--detach", str(wt), "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if add.returncode != 0:
            cand.apply_error = f"git worktree add failed: {add.stderr}"
            return

        ok_apply, err_msg, touched = apply_patch_ops(wt, cand.ops)
        cand.touched_files = touched
        if not ok_apply:
            cand.apply_error = err_msg
            return

        # Snapshot final patched file contents + git diff
        cand.patched_contents = {rel: (wt / rel).read_text(encoding="utf-8") for rel in touched if (wt / rel).is_file()}
        cand.diff_text = git_diff_worktree(wt)

        # Run repository verify command
        res = _run_shell(verify_cmd, wt, timeout_s=timeout_s)
        cand.verify_ok = res.ok
        cand.elapsed_s = res.elapsed_s
        combined_out = ((res.stdout or "") + "\n" + (res.stderr or "")).strip()
        cand.verify_output = combined_out[-1600:] if len(combined_out) > 1600 else combined_out

        # Optionally run promoted Red-Team regression test inside the SAME worktree!
        if cand.verify_ok and extra_test_code and extra_test_code.strip():
            rt_file = wt / "_ff_redteam_test.py"
            rt_file.write_text(extra_test_code, encoding="utf-8")
            rt_cmd = f"{sys.executable} -m pytest -q _ff_redteam_test.py"
            rt_res = _run_shell(rt_cmd, wt, timeout_s=min(20.0, timeout_s))
            cand.redteam_ok = rt_res.ok
            rt_out = ((rt_res.stdout or "") + "\n" + (rt_res.stderr or "")).strip()
            cand.redteam_output = rt_out[-1200:] if len(rt_out) > 1200 else rt_out
            rt_file.unlink(missing_ok=True)
    finally:
        subprocess.run(
            ["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(wt)],
            capture_output=True,
            timeout=20,
        )
        shutil.rmtree(wt, ignore_errors=True)


async def run_dev_forge(
    client: TensorFoldClient,
    skills: SkillPack,
    memory: EpisodicMemory,
    repo_dir: str | Path,
    task: str,
    verify_cmd: str,
    *,
    num_streams: int = 4,
    apply_winner: bool = False,
    timeout_s: float = 30.0,
) -> DevForgeResult:
    """Run full multi-file autonomous development loop in isolated git worktrees."""
    t0 = time.perf_counter()
    client.reset_stats()
    await client.probe()

    repo = Path(repo_dir).resolve()
    repo_map = build_repo_map(repo)
    all_files = list_repo_files(repo)

    # ------------------------------------------------------------------
    # PHASE 1: Deterministic Recon (Repo Map + Baseline Verify Status)
    # ------------------------------------------------------------------
    baseline_res = _run_shell(verify_cmd, repo, timeout_s=timeout_s)
    baseline_summary = (
        f"BASELINE `{verify_cmd}` -> exit={baseline_res.returncode}\n"
        + ((baseline_res.stdout + "\n" + baseline_res.stderr).strip()[-1400:])
    )

    lessons_block = memory.format_prompt_block(task, domain="code", limit=3)
    shared_sys = (
        "You are a principal software engineer inside DevForge, an automated multi-file worktree development environment.\n"
        "You NEVER guess file contents—you read exact code slices, identify cross-file invariants, emit exact multi-file "
        "<<<EDIT: path>>> / <<<FILE: path>>> patches, and satisfy both existing tests and adversarial edge-case audits."
    )
    if lessons_block:
        shared_sys += "\n\n" + lessons_block

    # ------------------------------------------------------------------
    # PHASE 2: Structured Recon & Context Slicing (libtfgrammar JSON)
    # ------------------------------------------------------------------
    recon_prompt = (
        f"TASK:\n{task.strip()}\n\n"
        f"{repo_map}\n\n"
        f"{baseline_summary}\n\n"
        "Identify EVERY relevant file to read (up to 8 files), useful search queries, root-cause hypotheses, "
        "cross-file invariants, and edge cases. Return strict JSON."
    )
    recon_res = await client.complete(
        CallSpec(
            messages=[{"role": "system", "content": shared_sys}, {"role": "user", "content": recon_prompt}],
            tag="dev_recon",
            max_tokens=1100,
            temperature=0.1,
            enable_thinking=True,
            thinking_budget=450,
            json_schema=RECON_PLAN_SCHEMA,
            schema_name="recon_plan",
            grammar_budget=True,
        )
    )
    plan = recon_res.json_payload() or {}

    # Assemble surgical file slices (ALWAYS include files named by recon, falling back to small repo files)
    requested_files = [f for f in plan.get("files_to_read", []) if f in all_files]
    if not requested_files:
        requested_files = [f for f in all_files if f.endswith(".py")][:8]
    # If repo is small (<=10 python files), include all non-test source files + primary test file for complete visibility
    if len([f for f in all_files if f.endswith(".py")]) <= 10:
        for f in all_files:
            if f.endswith(".py") and f not in requested_files:
                requested_files.append(f)

    slices = [read_numbered_slice(repo, f, max_lines=220) for f in requested_files[:10]]
    search_blocks: list[str] = []
    for q in plan.get("search_queries", [])[:3]:
        if isinstance(q, str) and len(q.strip()) >= 2:
            search_blocks.append(f"SEARCH `{q}`:\n{rg_search(repo, q, max_hits=12)}")

    patch_format_instructions = (
        "PATCH FORMAT (follow EXACTLY):\n"
        "PREFER compact <<<EDIT: relative/path.py>>> SEARCH/REPLACE blocks for targeted edits so every file fits cleanly:\n"
        "<<<EDIT: relative/path.py>>>\n"
        "<<<SEARCH>>>\n"
        "<exact existing lines from the file>\n"
        "<<<REPLACE>>>\n"
        "<replacement lines>\n"
        "<<<END_EDIT>>>\n\n"
        "Or use <<<FILE: relative/path.py>>> when creating a new file or rewriting most of a short file:\n"
        "<<<FILE: relative/path.py>>>\n"
        "<complete file contents>\n"
        "<<<END_FILE>>>\n"
        "You MUST patch EVERY affected file in one reply (double-check callers, reports, and models!). "
        "Emit ONLY patch blocks (no prose outside blocks)."
    )

    shared_dev_context = (
        f"TASK:\n{task.strip()}\n\n"
        f"{baseline_summary}\n\n"
        f"ENGINEERING PLAN:\n```json\n{json.dumps(plan, indent=2)[:1600]}\n```\n\n"
        + "\n\n".join(slices)
        + ("\n\n" + "\n\n".join(search_blocks) if search_blocks else "")
        + "\n\n"
        + patch_format_instructions
        + "\n"
    )

    # Warm TensorFold prompt cache on exact shared context!
    await client.warm_prefix(
        [{"role": "system", "content": shared_sys}, {"role": "user", "content": shared_dev_context}],
        enable_thinking=True,
    )

    # ------------------------------------------------------------------
    # PHASE 3: B=4 Role-Seeded Parallel Multi-File Worktree Racing
    # ------------------------------------------------------------------
    w1_roles = [
        ("minimal_surgical", "Make the smallest surgical cross-file fix addressing every requirement and invariant.", 0.20, 101, 1000),
        ("first_principles", "Reason from cross-file invariants and dataflow first; ensure every module stays consistent.", 0.50, 202, 1300),
        ("edge_skeptic", "Assume naive fixes miss subtle edge cases (atomicity, validation, ties, immutability, empty states). Guard EVERY one.", 0.40, 303, 1200),
        ("alt_architecture", "Consider an alternative cleaner design or factoring across modules that eliminates the entire bug class.", 0.65, 404, 1200),
    ][:num_streams]

    w1_specs = [
        CallSpec(
            messages=[
                {"role": "system", "content": shared_sys},
                {"role": "user", "content": shared_dev_context + f"\nROLE ({name}): {role_note}"},
            ],
            tag=f"dev_w1_{name}",
            max_tokens=tb + 4800,
            temperature=temp,
            seed=seed,
            enable_thinking=True,
            thinking_budget=tb,
            loop_guard=True,
        )
        for name, role_note, temp, seed, tb in w1_roles
    ]
    w1_res = await client.batch(w1_specs)

    w1_cands: list[WorktreeCandidate] = []
    graveyard: list[str] = []
    for (name, _, _, _, _), res in zip(w1_roles, w1_res):
        ops = parse_patch_ops(res.content)
        cand = WorktreeCandidate(name=name, wave=1, ops=ops)
        _eval_candidate_in_worktree(repo, cand, verify_cmd, timeout_s=timeout_s)
        w1_cands.append(cand)
        if cand.apply_error:
            graveyard.append(f"[{name}] PATCH APPLY ERROR: {cand.apply_error[:240]}")
        elif not cand.verify_ok:
            graveyard.append(f"[{name}] VERIFY FAILED:\n{cand.verify_output[-350:]}")

    passing_w1 = [c for c in w1_cands if c.verify_ok]
    redteam_kills: list[str] = []
    promoted_rt_test = ""

    # ------------------------------------------------------------------
    # PHASE 4: Clean-Context Diff Red-Teamer (if we have a green Wave-1 patch!)
    # ------------------------------------------------------------------
    if passing_w1:
        leader = min(passing_w1, key=lambda c: len(c.diff_text))
        rt_prompt = (
            f"TASK SPECIFICATION:\n{task.strip()}\n\n"
            f"REPO MAP:\n{repo_map}\n\n"
            f"CANDIDATE GIT DIFF (`{leader.name}` — passed existing tests):\n```diff\n{leader.diff_text[:4500]}\n```\n\n"
            "You are an adversarial code reviewer in a FRESH context.\n"
            "Try to BREAK this patch on subtle requirements or edge cases NOT covered by the existing test suite!\n"
            "Write 3 to 5 surgical pytest functions (`def test_rt_...():`) importing directly from the repo modules "
            "that assert 100%-spec-required edge cases (do NOT invent speculative requirements not in the task!).\n"
            "Output ONLY one fenced ```python``` block."
        )
        rt_res = await client.complete(
            CallSpec(
                messages=[{"role": "system", "content": "You are a ruthless adversarial software auditor."}, {"role": "user", "content": rt_prompt}],
                tag="dev_redteam",
                max_tokens=2000,
                temperature=0.3,
                seed=808,
                enable_thinking=True,
                thinking_budget=800,
                loop_guard=True,
            )
        )
        rt_code = extract_python_code(rt_res.content, salvage_truncated_tests=True)
        if rt_code and check_syntax(rt_code) is None:
            # Test ALL passing Wave-1 candidates against the adversarial regression test!
            for c in passing_w1:
                _eval_candidate_in_worktree(repo, c, verify_cmd, extra_test_code=rt_code, timeout_s=timeout_s)
            # If the leader FAILED red-team, validate via Spec Arbiter so bogus red-team tests don't false-kill!
            if not leader.redteam_ok:
                arb_prompt = (
                    f"TASK SPECIFICATION:\n{task.strip()}\n\n"
                    f"ADVERSARIAL TEST:\n```python\n{rt_code[:2200]}\n```\n\n"
                    f"FAILURE ON PATCH:\n{leader.redteam_output[:800]}\n\n"
                    "Does this failing test strictly enforce a requirement explicitly stated or implied by the TASK SPECIFICATION "
                    "(not a hallucinated API or bogus expectation)? Reply VALID or INVALID."
                )
                verdict, _ = await client.decide_choice(arb_prompt, ["VALID", "INVALID"], tag="dev_rt_arbiter")
                if verdict == "VALID":
                    promoted_rt_test = rt_code
                    redteam_kills.append(f"{leader.name}: {leader.redteam_output[:260]}")
                    graveyard.append(f"[{leader.name}] KILLED BY RED-TEAM:\n{leader.redteam_output[-350:]}")
                else:
                    # Reset redteam_ok on false kill
                    for c in passing_w1:
                        c.redteam_ok = True

    green_w1 = [c for c in passing_w1 if c.redteam_ok]

    # ------------------------------------------------------------------
    # PHASE 5: Hybrid Wave 2 (2 Surgical Continuations on Best W1 + 2 Clean HEAD Resets)
    # ------------------------------------------------------------------
    wave2_triggered = not bool(green_w1)
    w2_cands: list[WorktreeCandidate] = []

    if wave2_triggered:
        # Pick best Wave-1 candidate that applied cleanly (to enable surgical multi-turn continuation!)
        applied_w1 = [c for c in w1_cands if not c.apply_error and c.patched_contents]
        best_base: Optional[WorktreeCandidate] = None
        if passing_w1:
            best_base = passing_w1[0]
        elif applied_w1:
            # Prefer candidate that touched the most files cleanly
            best_base = max(applied_w1, key=lambda c: len(c.touched_files))

        w2_reset_context = (
            f"TASK:\n{task.strip()}\n\n"
            f"GRAVEYARD OF WAVE-1 FAILURES (HEAD RESET — DO NOT REPEAT THESE MISTAKES):\n"
            + "\n---\n".join(graveyard[:4])
            + (
                f"\n\nADVERSARIAL REGRESSION TEST YOUR PATCH MUST ALSO PASS:\n```python\n{promoted_rt_test[:2200]}\n```"
                if promoted_rt_test
                else ""
            )
            + "\n\nCURRENT SOURCE FILES (CLEAN HEAD):\n"
            + "\n\n".join(slices)
            + "\n\n"
            + patch_format_instructions
            + "\n"
        )

        # Build surgical continuation context from best_base's already-patched files + exact error!
        w2_cont_context = w2_reset_context
        if best_base is not None:
            base_err = best_base.redteam_output if (best_base.verify_ok and not best_base.redteam_ok) else best_base.verify_output
            # Show current state of every requested file AFTER best_base's Wave-1 edits
            post_slices: list[str] = []
            for rel in requested_files[:10]:
                if rel in best_base.patched_contents:
                    raw_lines = best_base.patched_contents[rel].splitlines()
                    numbered = "\n".join(f"{i:4d} | {ln}" for i, ln in enumerate(raw_lines, 1))
                    post_slices.append(f"=== FILE (PATCHED BY W1): {rel} ===\n{numbered}")
                else:
                    post_slices.append(read_numbered_slice(repo, rel, max_lines=220))
            w2_cont_context = (
                f"TASK:\n{task.strip()}\n\n"
                f"YOUR WAVE-1 PATCH (`{best_base.name}`) IS ALREADY APPLIED. IT FAILED ONLY ON:\n"
                f"```\n{base_err[-1200:]}\n```\n"
                + (
                    f"\nADVERSARIAL REGRESSION TEST IT MUST PASS:\n```python\n{promoted_rt_test[:2000]}\n```\n"
                    if promoted_rt_test
                    else ""
                )
                + "\nCURRENT WORKTREE FILES (WITH WAVE-1 ALREADY APPLIED):\n"
                + "\n\n".join(post_slices)
                + "\n\nEmit ONLY the incremental <<<EDIT: path>>> or <<<FILE: path>>> blocks needed to fix the remaining failure "
                "(do NOT repeat already-correct unchanged files!).\n"
                + patch_format_instructions
                + "\n"
            )

        w2_specs_meta: list[tuple[str, str, Optional[WorktreeCandidate], float, int, int]] = [
            ("w2_continue_surgical", w2_cont_context + "\nROLE: Fix the exact remaining error surgically on top of the applied Wave-1 worktree.", best_base, 0.20, 1101, 1100),
            ("w2_continue_skeptic", w2_cont_context + "\nROLE: Fix the remaining error AND audit every cross-file caller/edge case on top of Wave-1.", best_base, 0.45, 1202, 1200),
            ("w2_reset_rewrite", w2_reset_context + "\nROLE: Start fresh from HEAD and write a clean cross-file fix avoiding all graveyard errors.", None, 0.35, 1303, 1300),
            ("w2_reset_alt", w2_reset_context + "\nROLE: Start fresh from HEAD with a distinct clean design across all affected files.", None, 0.60, 1404, 1300),
        ][:num_streams]

        w2_specs = [
            CallSpec(
                messages=[
                    {"role": "system", "content": shared_sys},
                    {"role": "user", "content": ctx},
                ],
                tag=f"dev_{name}",
                max_tokens=tb + 4800,
                temperature=temp,
                seed=seed,
                enable_thinking=True,
                thinking_budget=tb,
                loop_guard=True,
            )
            for name, ctx, _, temp, seed, tb in w2_specs_meta
        ]
        w2_res = await client.batch(w2_specs)
        for (name, _, base_cand, _, _, _), res in zip(w2_specs_meta, w2_res):
            delta_ops = parse_patch_ops(res.content)
            # If continuing from Wave-1 base_cand, replay base_cand.ops FIRST, then delta_ops!
            combined_ops = (list(base_cand.ops) + delta_ops) if (base_cand is not None and delta_ops) else delta_ops
            cand = WorktreeCandidate(name=name, wave=2, ops=combined_ops)
            _eval_candidate_in_worktree(
                repo, cand, verify_cmd, extra_test_code=promoted_rt_test or None, timeout_s=timeout_s
            )
            w2_cands.append(cand)
            if cand.apply_error:
                graveyard.append(f"[{name}] PATCH APPLY ERROR: {cand.apply_error[:200]}")
            elif not cand.verify_ok:
                graveyard.append(f"[{name}] VERIFY FAILED:\n{cand.verify_output[-300:]}")
            elif not cand.redteam_ok:
                graveyard.append(f"[{name}] RED-TEAM FAILED:\n{cand.redteam_output[-300:]}")

    # ------------------------------------------------------------------
    # PHASE 6: Final Ranking + Order-Swapped Pairwise Diff Tournament
    # ------------------------------------------------------------------
    all_cands = w1_cands + w2_cands
    # Never resurrect a Red-Team-killed candidate as ok=True!
    full_green = [c for c in all_cands if c.verify_ok and c.redteam_ok]
    pool = full_green

    winner: Optional[WorktreeCandidate] = None
    if pool:
        # Deduplicate identical diffs
        uniq: list[WorktreeCandidate] = []
        seen_diffs: set[str] = set()
        for c in sorted(pool, key=lambda x: len(x.diff_text)):
            if c.diff_text not in seen_diffs:
                seen_diffs.add(c.diff_text)
                uniq.append(c)
        winner = uniq[0]
        for challenger in uniq[1:3]:
            p_ab = (
                f"TASK:\n{task.strip()}\n\n"
                f"PATCH A (`{winner.name}`):\n```diff\n{winner.diff_text[:5000]}\n```\n\n"
                f"PATCH B (`{challenger.name}`):\n```diff\n{challenger.diff_text[:5000]}\n```\n\n"
                "Both passed all tests. Which patch is cleaner, more complete on hidden edge cases, and strictly faithful to the task? Reply A or B."
            )
            p_ba = (
                f"TASK:\n{task.strip()}\n\n"
                f"PATCH A (`{challenger.name}`):\n```diff\n{challenger.diff_text[:5000]}\n```\n\n"
                f"PATCH B (`{winner.name}`):\n```diff\n{winner.diff_text[:5000]}\n```\n\n"
                "Both passed all tests. Which patch is cleaner, more complete on hidden edge cases, and strictly faithful to the task? Reply A or B."
            )
            if await client.pairwise_vote(p_ab, p_ba, tag_prefix="dev_judge") == "B":
                winner = challenger

    if winner and apply_winner:
        repo_resolved = repo.resolve()
        for rel, content in winner.patched_contents.items():
            dest = (repo_resolved / rel).resolve()
            if not dest.is_relative_to(repo_resolved):
                raise ValueError(f"Refusing path outside repo: {rel}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

    return DevForgeResult(
        ok=bool(winner and winner.verify_ok and winner.redteam_ok),
        winner_name=winner.name if winner else "none",
        touched_files=winner.touched_files if winner else [],
        patched_contents=winner.patched_contents if winner else {},
        diff_text=winner.diff_text if winner else "",
        wave2_triggered=wave2_triggered,
        redteam_kills=redteam_kills,
        graveyard=graveyard,
        wall_s=time.perf_counter() - t0,
        token_stats=client.stats_dict(),
        verify_output=winner.verify_output if winner else (graveyard[-1] if graveyard else ""),
    )

"""Unified Consensus Verified Code & Adversarial Debugging Engine (v4).

Built around one universal loop (mirroring ReasonForge):
  Stage 1 (B=4 single parallel wave — ZERO upfront Stage-1 wait):
    - 3 diverse implementations (`radically_simple` textbook, `first_principles`, `edge_skeptic`)
      in parallel with 1 high-signal Spec-Property Test Suite (`spec_oracle`)
      (or 4 diverse implementations when `external_test_code` is provided).
    - Mechanical Tier-0 per-test execution + Helper-Aware Strict B=2 Consensus Shield
      (any generated test with 0/N passes is audited WITH module helpers; requires 2/2 VALID).
  Stage 2 (Targeted B=1 Clean-Context Adversarial Red-Teamer):
    - Fresh context probes the surviving leader for subtle spec violations.
    - Every killing assertion is gated by the same Helper-Aware Strict B=2 Shield (2/2 VALID).
  Stage 3 (Conditional Graveyard Repair: Wave 2 B=3 + Pre-W3 Re-Audit + Wave 3 B=2):
    - Fires ONLY if no perfect survivor exists.
    - Includes `w2_radically_simple` (strip over-engineering) + `w2_surgical_fix` + `w2_clean_rewrite`.
    - Before Wave 3 runs, any generated test still failing 0/N across >=4 independent W1+W2
      implementations is re-audited strictly so Wave 3 NEVER overfits a poisoned generated test.
  Final Tie-Break:
    - Order-swapped pairwise tournament over top-scoring distinct survivors.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .client import CallSpec, TensorFoldClient
from .memory import EpisodicMemory
from .sandbox import check_syntax, extract_python_code, run_python
from .skill_loader import SkillPack


@dataclass
class CodeCandidate:
    name: str
    wave: int
    code: str
    raw_content: str
    syntax_error: Optional[str] = None
    fatal_error: Optional[str] = None
    redteam_fatal: Optional[str] = None
    truncated: bool = False
    tests_passed: int = 0
    tests_total: int = 0
    failed_tests: list[str] = field(default_factory=list)
    failure_details: dict[str, str] = field(default_factory=dict)
    redteam_passed: int = 0
    redteam_total: int = 0
    redteam_failures: list[str] = field(default_factory=list)
    diff_outputs: dict[str, str] = field(default_factory=dict)
    diff_majority_hits: int = 0
    diff_counterexample_wins: int = 0
    diff_counterexample_losses: int = 0
    starter_api_score: float = 0.0
    runtime_ms: float = 0.0
    latency_s: float = 0.0

    @property
    def total_passed(self) -> int:
        return self.tests_passed + self.redteam_passed

    @property
    def total_checks(self) -> int:
        return self.tests_total + self.redteam_total

    @property
    def perfect(self) -> bool:
        return (
            self.syntax_error is None
            and self.fatal_error is None
            and self.redteam_fatal is None
            and self.total_checks > 0
            and self.total_passed == self.total_checks
        )


@dataclass
class CodeForgeTrace:
    mode: str = "code"
    contract: dict[str, Any] = field(default_factory=dict)
    oracle_code: str = ""
    oracle_test_names: list[str] = field(default_factory=list)
    dropped_bad_oracle_tests: list[str] = field(default_factory=list)
    wave1_candidates: list[CodeCandidate] = field(default_factory=list)
    wave2_candidates: list[CodeCandidate] = field(default_factory=list)
    redteam_tests_generated: int = 0
    redteam_valid_kills: list[str] = field(default_factory=list)
    wave2_triggered: bool = False
    wave2_reason: str = ""
    graveyard: list[str] = field(default_factory=list)
    winner_name: str = ""
    winner_code: str = ""
    wall_s: float = 0.0
    lessons_used: int = 0
    recovered_by_redteam_or_wave2: bool = False
    verified: bool = False


_FUTURE_IMPORT_RE = re.compile(r"^[ \t]*from\s+__future__\s+import\s+[^\n]+$", re.MULTILINE)


def _strip_future_imports(code: str) -> str:
    """Strip `from __future__ import ...` lines before embedding/concatenating test suites."""
    if not code:
        return ""
    return _FUTURE_IMPORT_RE.sub("", code).strip()


_MODULAR_RUNNER_TEMPLATE = r'''from __future__ import annotations
import inspect, json, time, traceback
import solution
from solution import *

# --- BEGIN TEST SUITE ---
__TEST_CODE__
# --- END TEST SUITE ---

# Anti-shadowing guard: restore every public symbol actually defined in `solution.py`
# so generated tests accidentally pasting a starter/buggy copy NEVER shadow `solution`!
for _k in dir(solution):
    if not _k.startswith("_") and not _k.startswith("test_"):
        _obj = getattr(solution, _k)
        if getattr(_obj, "__module__", None) == "solution":
            globals()[_k] = _obj

_results = []
_t_all = time.perf_counter()
_funcs = [
    (name, obj)
    for name, obj in list(globals().items())
    if name.startswith("test_") and callable(obj) and inspect.isfunction(obj)
]
if not _funcs and "run_tests" in globals() and callable(globals()["run_tests"]):
    _funcs = [("run_tests", globals()["run_tests"])]

for _name, _fn in _funcs:
    _t0 = time.perf_counter()
    try:
        _fn()
        _row = {"name": _name, "ok": True, "err": "", "ms": round((time.perf_counter() - _t0) * 1000, 2)}
    except BaseException as _e:
        import re as _re
        _tb = traceback.format_exc()
        _tb = _re.sub(r'File "[^"]*/(solution\.py|_run\.py)"', r'File "\1"', _tb)
        _clean_lines = [ln for ln in _tb.strip().splitlines() if not _re.fullmatch(r"\s*[~^]+\s*", ln)]
        _headline = f"{type(_e).__name__}: {str(_e)[:200]}".rstrip(": ")
        _compact_tb = "\n".join(_clean_lines[-5:])
        _row = {
            "name": _name,
            "ok": False,
            "err": f"{_headline}\n{_compact_tb}"[:550],
            "ms": round((time.perf_counter() - _t0) * 1000, 2),
        }
    _results.append(_row)
    print("__FF_TEST_ROW__:" + json.dumps(_row), flush=True)

_diff_out = {}
if "diff_probe" in globals() and callable(globals()["diff_probe"]):
    try:
        _raw_dp = globals()["diff_probe"]()
        if isinstance(_raw_dp, dict):
            for _dk, _dv in list(_raw_dp.items())[:24]:
                _diff_out[str(_dk)] = repr(_dv)[:300]
    except BaseException as _de:
        _diff_out["__probe_exc__"] = f"{type(_de).__name__}: {str(_de)[:150]}"

print("__FF_TEST_JSON__:" + json.dumps({
    "results": _results,
    "diff_outputs": _diff_out,
    "total_ms": round((time.perf_counter() - _t_all) * 1000, 2),
}), flush=True)
'''


_STDIN_RUNNER_TEMPLATE = r'''from __future__ import annotations
import inspect, json, subprocess, sys, time, traceback

def run_solution(inp: str, timeout: float = 3.0) -> str:
    p = subprocess.run(
        [sys.executable, "solution.py"],
        input=inp,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if p.returncode != 0:
        raise RuntimeError(f"Exit {p.returncode}: {(p.stderr or p.stdout).strip()[:300]}")
    return p.stdout.strip()

# --- BEGIN TEST SUITE ---
__TEST_CODE__
# --- END TEST SUITE ---

_results = []
_t_all = time.perf_counter()
_funcs = [
    (name, obj)
    for name, obj in list(globals().items())
    if name.startswith("test_") and callable(obj) and inspect.isfunction(obj)
]
if not _funcs and "run_tests" in globals() and callable(globals()["run_tests"]):
    _funcs = [("run_tests", globals()["run_tests"])]

for _name, _fn in _funcs:
    _t0 = time.perf_counter()
    try:
        _fn()
        _row = {"name": _name, "ok": True, "err": "", "ms": round((time.perf_counter() - _t0) * 1000, 2)}
    except BaseException as _e:
        import re as _re
        _tb = traceback.format_exc()
        _tb = _re.sub(r'File "[^"]*/(solution\.py|_run\.py)"', r'File "\1"', _tb)
        _clean_lines = [ln for ln in _tb.strip().splitlines() if not _re.fullmatch(r"\s*[~^]+\s*", ln)]
        _headline = f"{type(_e).__name__}: {str(_e)[:200]}".rstrip(": ")
        _compact_tb = "\n".join(_clean_lines[-5:])
        _row = {
            "name": _name,
            "ok": False,
            "err": f"{_headline}\n{_compact_tb}"[:550],
            "ms": round((time.perf_counter() - _t0) * 1000, 2),
        }
    _results.append(_row)
    print("__FF_TEST_ROW__:" + json.dumps(_row), flush=True)

_diff_out = {}
if "diff_probe" in globals() and callable(globals()["diff_probe"]):
    try:
        _raw_dp = globals()["diff_probe"]()
        if isinstance(_raw_dp, dict):
            for _dk, _dv in list(_raw_dp.items())[:24]:
                _diff_out[str(_dk)] = repr(_dv)[:300]
    except BaseException as _de:
        _diff_out["__probe_exc__"] = f"{type(_de).__name__}: {str(_de)[:150]}"

print("__FF_TEST_JSON__:" + json.dumps({
    "results": _results,
    "diff_outputs": _diff_out,
    "total_ms": round((time.perf_counter() - _t_all) * 1000, 2),
}), flush=True)
'''


_STARTER_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


def _extract_starter_imports(problem: str) -> list[str]:
    """Extract top-level import lines from starter code blocks (or raw starter code) in the prompt."""
    imports: list[str] = []
    seen: set[str] = set()
    blocks: list[str] = []
    for m in _STARTER_BLOCK_RE.finditer(problem):
        blk = m.group(1)
        if "def " in blk or "class " in blk:
            blocks.append(blk)
    if not blocks and ("def " in problem or "class " in problem):
        # Raw unfenced starter snippet (e.g. direct docstring/function prompt)
        head = problem.split("def ", 1)[0]
        if "import " in head:
            blocks.append(head)
    for blk in blocks:
        for ln in blk.splitlines():
            s = ln.strip()
            if (s.startswith("import ") or s.startswith("from ")) and not s.startswith("from __future__"):
                if s not in seen:
                    seen.add(s)
                    imports.append(s)
    return imports


def _extract_starter_symbols(starter_imports: list[str]) -> set[str]:
    """Return local module/symbol names bound by starter import statements (for API faithfulness scoring)."""
    syms: set[str] = set()
    for imp in starter_imports:
        try:
            tree = ast.parse(imp)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    syms.add((alias.asname or alias.name.split(".")[0]).strip())
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != "*":
                        syms.add((alias.asname or alias.name).strip())
    return syms


def _starter_api_coverage(code: str, starter_symbols: set[str]) -> float:
    """Fraction of declared starter-import symbols actually used inside non-import statements of `code`."""
    if not code or not starter_symbols:
        return 0.0
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0
    used: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name):
                used.add(node.id)
    return len(used & starter_symbols) / len(starter_symbols)


_RAISES_SECTION_RE = re.compile(
    r"(?:^|\n)[ \t]*Raises?\s*:?\s*\n((?:[ \t]+[^\n]+\n?)+)",
    re.IGNORECASE,
)
_EXC_NAME_RE = re.compile(r"\b(?:[a-z_]+\.)?([A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning))\b")


def _extract_required_exceptions(problem: str) -> set[str]:
    """Extract exception class names explicitly declared in Raises:/Raise: docstring clauses."""
    found: set[str] = set()
    for m in _RAISES_SECTION_RE.finditer(problem):
        for em in _EXC_NAME_RE.finditer(m.group(1)):
            found.add(em.group(1))
    return found


def _ensure_starter_imports(code: str, starter_imports: list[str]) -> str:
    """Prepend any missing starter imports (after any __future__ import) so module-level bindings stay intact."""
    if not code or not starter_imports:
        return code
    existing = {ln.strip() for ln in code.splitlines() if ln.strip().startswith(("import ", "from "))}
    missing = [imp for imp in starter_imports if imp not in existing]
    if not missing:
        return code
    lines = code.splitlines()
    insert_at = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith("from __future__"):
            insert_at = i + 1
        elif ln.strip() and not ln.strip().startswith("#"):
            break
    out_lines = lines[:insert_at] + missing + lines[insert_at:]
    return "\n".join(out_lines) + ("\n" if code.endswith("\n") else "")


def _extract_prompt_io_samples(problem: str) -> list[tuple[str, str]]:
    """Deterministically extract verbatim Sample Input / Sample Output pairs from competitive prompts."""
    text = problem.replace("\r\n", "\n").replace("\r", "\n")
    pairs: list[tuple[str, str]] = []
    # Match fenced AtCoder/Codeforces style first
    pat = re.compile(
        r"(?:^|\n)#*\s*Sample\s+Input\s*\d*\s*:?\s*\n+```[^\n]*\n(.*?)\n```\s*\n+"
        r"#*\s*Sample\s+Output\s*\d*\s*:?\s*\n+```[^\n]*\n(.*?)\n```",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pat.finditer(text):
        pairs.append((m.group(1), m.group(2)))
    if not pairs:
        # Unfenced style: stop BOTH Input and Output cleanly at the first blank line (\n\s*\n)
        # so sample explanation prose after a blank line is NEVER swallowed into expected output!
        pat2 = re.compile(
            r"(?:^|\n)#*\s*Sample\s+Input\s*\d*\s*:?\s*\n+(.*?)\n+#*\s*Sample\s+Output\s*\d*\s*:?\s*\n+(.*?)(?=\n[ \t]*\n|\n#*\s*Sample\s+Input|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        for m in pat2.finditer(text):
            inp, out = m.group(1).strip(), m.group(2).strip()
            # Guard against swallowing multi-word English explanation sentences
            if inp and out and len(inp) <= 4000 and len(out) <= 4000 and not re.search(r"\b[a-z]{4,}\s+[a-z]{4,}\s+[a-z]{4,}\b", out):
                pairs.append((inp, out))
    return pairs[:6]


def _build_deterministic_sample_tests(samples: list[tuple[str, str]]) -> tuple[str, set[str]]:
    """Build immutable `def test_prompt_sample_k():` checks that can NEVER be mistyped or dropped."""
    if not samples:
        return "", set()
    lines: list[str] = []
    names: set[str] = set()
    for i, (inp, exp) in enumerate(samples, 1):
        name = f"test_prompt_sample_{i}"
        names.add(name)
        exp_norm = "\n".join(ln.rstrip() for ln in exp.strip().splitlines())
        lines.append(
            f"def {name}():\n"
            f"    _exp = {exp_norm!r}\n"
            f"    _got = '\\n'.join(ln.rstrip() for ln in run_solution({inp!r}).strip().splitlines())\n"
            f"    assert _got == _exp, f'sample {i}: got={{_got[:120]!r}} exp={{_exp[:120]!r}}'\n"
        )
    return "\n".join(lines), names


def _is_stdin_problem(problem: str) -> bool:
    """Detect CLI / competitive-programming stdin->stdout problems (no Python function signature requested)."""
    low = problem.lower()
    has_io = ("standard input" in low or "sample input" in low or "sys.stdin" in low or "stdin" in low) and (
        "standard output" in low or "sample output" in low or "stdout" in low or "print the answer" in low
    )
    has_fn_sig = bool(re.search(r"(?:^|\n)\s*def\s+[a-zA-Z_]\w*\s*\(", problem))
    return has_io and not has_fn_sig


def _run_modular_suite_with_diff(
    solution_code: str,
    test_code: str,
    timeout_s: float = 8.0,
    stdin_mode: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, str], float, Optional[str]]:
    """Run every test_* function plus optional `diff_probe()` behavioral fingerprint."""
    cleaned_tests = _strip_future_imports(test_code)
    if not cleaned_tests:
        return [], {}, 0.0, None
    tmpl = _STDIN_RUNNER_TEMPLATE if stdin_mode else _MODULAR_RUNNER_TEMPLATE
    harness = tmpl.replace("__TEST_CODE__", cleaned_tests)
    res = run_python(harness, timeout_s=timeout_s, extra_files={"solution.py": solution_code})
    lines = res.stdout.splitlines()
    for line in reversed(lines):
        if line.startswith("__FF_TEST_JSON__:"):
            try:
                payload = json.loads(line[len("__FF_TEST_JSON__:") :])
                diff_out = payload.get("diff_outputs", {})
                if not isinstance(diff_out, dict):
                    diff_out = {}
                return payload.get("results", []), {str(k): str(v) for k, v in diff_out.items()}, float(payload.get("total_ms", 0.0)), None
            except json.JSONDecodeError:
                pass
    partial_rows: list[dict[str, Any]] = []
    for line in lines:
        if line.startswith("__FF_TEST_ROW__:"):
            try:
                partial_rows.append(json.loads(line[len("__FF_TEST_ROW__:") :]))
            except json.JSONDecodeError:
                pass
    if res.timed_out:
        return partial_rows, {}, timeout_s * 1000.0, f"TIMEOUT ({timeout_s}s)"
    if not res.ok:
        return partial_rows, {}, res.elapsed_s * 1000.0, res.short_error(500)
    return partial_rows, {}, res.elapsed_s * 1000.0, None


def _run_modular_suite(
    solution_code: str,
    test_code: str,
    timeout_s: float = 8.0,
    stdin_mode: bool = False,
) -> tuple[list[dict[str, Any]], float, Optional[str]]:
    """Run every test_* function independently and return structured per-test outcomes."""
    rows, _, total_ms, fatal = _run_modular_suite_with_diff(solution_code, test_code, timeout_s, stdin_mode)
    return rows, total_ms, fatal


_TMPDIR_NORM_RE = re.compile(
    r"(?:/private)?/var/folders/[^'\",\s/]+/[^'\",\s/]+/T/(?:tmp[A-Za-z0-9_]+|ff_exec_[A-Za-z0-9_]+)|"
    r"/tmp/(?:tmp[A-Za-z0-9_]+|ff_exec_[A-Za-z0-9_]+)"
)


def _norm_diff_val(val: str) -> str:
    """Normalize ephemeral temp-dir prefixes so identical relative-to-tempdir behavior matches,
    while relative ('dir/a.zip') vs absolute ('<TMPDIR>/dir/a.zip') STILL cleanly diverges!"""
    return _TMPDIR_NORM_RE.sub("<TMPDIR>", val.strip())


def _extract_diff_probe_source(test_code: str) -> str:
    """Extract `def diff_probe(...)` source from the oracle suite for counterexample adjudication."""
    if not test_code.strip():
        return ""
    try:
        tree = ast.parse(test_code)
        lines = test_code.splitlines()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "diff_probe":
                return "\n".join(lines[node.lineno - 1 : getattr(node, "end_lineno", node.lineno)])
    except SyntaxError:
        pass
    return ""


def _is_total_probe_crasher(c: CodeCandidate) -> bool:
    """True when every diff_probe entry threw a trivial setup/binding error (NameError/ImportError/TypeError)."""
    if not c.diff_outputs:
        return True
    vals = [v for k, v in c.diff_outputs.items() if k != "__probe_exc__"]
    if not vals:
        return True
    return all(
        any(tok in v for tok in ("EXC:NameError", "EXC:ImportError", "EXC:ModuleNotFoundError", "EXC:SyntaxError"))
        for v in vals
    )


def _analyze_behavioral_consensus(
    candidates: list[CodeCandidate],
) -> tuple[int, list[tuple[str, dict[str, tuple[str, str]]]]]:
    """Compute cross-candidate behavioral majority hits and collect concrete divergences.

    Filters out 0-pass/NameError crashers whenever >=2 healthy/passing candidates exist,
    so counterexample adjudication focuses 100% on real behavioral disagreements!
    """
    raw_valid = [c for c in candidates if c.syntax_error is None and c.fatal_error is None and c.diff_outputs]
    for c in candidates:
        c.diff_majority_hits = 0
    if len(raw_valid) < 2:
        return 0, []

    perfect_pool = [c for c in raw_valid if c.perfect and not _is_total_probe_crasher(c)]
    if len(perfect_pool) >= 2:
        valid = perfect_pool
    else:
        healthy_pool = [c for c in raw_valid if not _is_total_probe_crasher(c) and (c.tests_total == 0 or c.tests_passed > 0)]
        valid = healthy_pool if len(healthy_pool) >= 2 else raw_valid

    all_cases: list[str] = []
    seen: set[str] = set()
    for c in valid:
        for k in c.diff_outputs:
            if k != "__probe_exc__" and k not in seen:
                seen.add(k)
                all_cases.append(k)

    divergences: list[tuple[str, dict[str, tuple[str, str]]]] = []
    for case in all_cases:
        obs: dict[str, tuple[str, str]] = {}
        counts: dict[str, int] = {}
        for c in valid:
            if case in c.diff_outputs:
                raw = c.diff_outputs[case]
                norm = _norm_diff_val(raw)
                obs[c.name] = (raw, norm)
                counts[norm] = counts.get(norm, 0) + 1
        if len(obs) < 2:
            continue
        ranked_vals = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top_val, top_cnt = ranked_vals[0]
        is_unique_maj = top_cnt >= 2 and (len(ranked_vals) == 1 or ranked_vals[1][1] < top_cnt)
        if is_unique_maj:
            for c in valid:
                if c.name in obs and obs[c.name][1] == top_val:
                    c.diff_majority_hits += 1
        if len(counts) > 1:
            divergences.append((case, obs))
    return len(all_cases), divergences


async def _adjudicate_divergences(
    client: TensorFoldClient,
    problem: str,
    probe_src: str,
    divergences: list[tuple[str, dict[str, tuple[str, str]]]],
    candidates: list[CodeCandidate],
) -> list[str]:
    """Oracle-Free Counterexample Adjudicator (B=2 strict consensus per divergence).

    When two independent candidates disagree on a concrete probe case `x*` (`out_A != out_B`),
    asks 2 independent clean-context arbiters strictly which output (A, B, or NEITHER) matches
    the specification. Updates `diff_counterexample_wins`/`losses` and returns concrete tombstone
    notes for any `NEITHER` or resolved bug!
    """
    if not divergences:
        return []
    by_name = {c.name: c for c in candidates}
    items: list[tuple[str, str, str, list[str], list[str]]] = []

    for case, obs in divergences[:3]:
        # Group candidate names by normalized output, pick the 2 distinct behaviors (most popular first)
        groups: dict[str, list[str]] = {}
        sample_raw: dict[str, str] = {}
        for cname, (raw, norm) in obs.items():
            groups.setdefault(norm, []).append(cname)
            sample_raw.setdefault(norm, raw)
        ordered = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
        if len(ordered) < 2:
            continue
        norm_a, names_a = ordered[0]
        norm_b, names_b = ordered[1]
        items.append((case, sample_raw[norm_a], sample_raw[norm_b], names_a, names_b))

    if not items:
        return []

    specs: list[CallSpec] = []
    meta: list[int] = []
    for idx, (case, raw_a, raw_b, _, _) in enumerate(items):
        prompt = (
            f"PROBLEM SPECIFICATION:\n{problem.strip()}\n\n"
            f"BEHAVIORAL PROBE (`diff_probe`, case `{case}`):\n```python\n{probe_src[:2400]}\n```\n\n"
            f"CONCRETE DIVERGENCE OBSERVED ON `{case}`:\n"
            f"- OUTPUT A: {raw_a[:350]}\n"
            f"- OUTPUT B: {raw_b[:350]}\n\n"
            "Strictly according to the problem specification (exact return format, relative vs absolute path, "
            "exact exception type, edge-case semantics), which output is CORRECT for this case?\n"
            "End your concise analysis (<=5 sentences) with `VERDICT: A`, `VERDICT: B`, or `VERDICT: NEITHER`."
        )
        for j, (temp, seed) in enumerate(((0.0, 41), (0.15, 83))):
            specs.append(
                CallSpec(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a strict formal spec judge resolving a concrete behavioral divergence. End with VERDICT: A, VERDICT: B, or VERDICT: NEITHER.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    tag=f"diff_arb{j+1}_{case}",
                    max_tokens=1000,
                    temperature=temp,
                    seed=seed,
                    enable_thinking=True,
                    thinking_budget=420,
                    loop_guard=True,
                )
            )
            meta.append(idx)

    results = await client.batch(specs)
    votes: dict[int, list[str]] = {i: [] for i in range(len(items))}
    need_fb: list[tuple[int, str, str, str, str]] = []
    for idx, res in zip(meta, results):
        txt = (res.content or "").upper() if res.ok else ""
        ms = re.findall(r"VERDICT\s*[:=]\s*(A|B|NEITHER)\b", txt)
        if ms:
            votes[idx].append(ms[-1])
        elif res.ok and res.content.strip() and hasattr(client, "decide_choice"):
            case, raw_a, raw_b, _, _ = items[idx]
            need_fb.append((idx, case, raw_a, raw_b, res.content[:450]))
        else:
            votes[idx].append("UNKNOWN")

    if need_fb:
        fb_res = await asyncio.gather(
            *[
                client.decide_choice(
                    prompt=(
                        f"Case `{case}` divergence:\nA: {raw_a[:200]}\nB: {raw_b[:200]}\n\n"
                        f"Analysis:\n{analysis}\n\nWhich output matches the spec? Reply A, B, or NEITHER:"
                    ),
                    choices=["A", "B", "NEITHER"],
                    tag=f"diff_arb_fb_{case}",
                )
                for _, case, raw_a, raw_b, analysis in need_fb
            ]
        )
        for (idx, _, _, _, _), (choice, r) in zip(need_fb, fb_res):
            votes[idx].append(choice if (r.ok and choice in ("A", "B", "NEITHER")) else "UNKNOWN")

    notes: list[str] = []
    for idx, (case, raw_a, raw_b, names_a, names_b) in enumerate(items):
        vlist = votes[idx]
        if len(vlist) == 2 and vlist[0] == vlist[1] and vlist[0] in ("A", "B", "NEITHER"):
            verdict = vlist[0]
            if verdict == "A":
                for n in names_a:
                    by_name[n].diff_counterexample_wins += 1
                for n in names_b:
                    by_name[n].diff_counterexample_losses += 1
                    by_name[n].failure_details[f"diff:{case}"] = f"Diverged on `{case}`: produced {raw_b[:140]} but spec requires {raw_a[:140]}"
                notes.append(f"[diff:{case}] Spec requires {raw_a[:140]} (NOT {raw_b[:140]})")
            elif verdict == "B":
                for n in names_b:
                    by_name[n].diff_counterexample_wins += 1
                for n in names_a:
                    by_name[n].diff_counterexample_losses += 1
                    by_name[n].failure_details[f"diff:{case}"] = f"Diverged on `{case}`: produced {raw_a[:140]} but spec requires {raw_b[:140]}"
                notes.append(f"[diff:{case}] Spec requires {raw_b[:140]} (NOT {raw_a[:140]})")
            else:
                for n in names_a + names_b:
                    by_name[n].diff_counterexample_losses += 1
                    by_name[n].failure_details[f"diff:{case}"] = f"BOTH outputs ({raw_a[:100]} / {raw_b[:100]}) violated spec on `{case}`"
                notes.append(f"[diff:{case}] NEITHER {raw_a[:100]} NOR {raw_b[:100]} satisfies spec")
    return notes


async def _spec_compliance_audit(
    client: TensorFoldClient,
    problem: str,
    code: str,
) -> tuple[bool, str]:
    """Parallel B=2 Adversarial Spec-Clause + API-Mockability Audit (catches shared blind spots!).

    Auditor 1 checks every explicit `Raises:`, empty/degenerate inputs, recursion/nesting, exact return format.
    Auditor 2 checks starter-library & mockability faithfulness (`os.path.exists`, `glob.glob`, `os.makedirs`, etc.).
    Requires 2/2 PASS to allow Unanimous Behavioral Fast-Path!
    """
    p_edge = (
        f"PROBLEM SPECIFICATION:\n{problem.strip()}\n\n"
        f"CANDIDATE IMPLEMENTATION:\n```python\n{code[:6000]}\n```\n\n"
        "You are an adversarial spec auditor hunting for SHARED BLIND SPOTS that unit tests missed.\n"
        "Check EVERY explicit clause line-by-line:\n"
        "1. Every `Raise:` / exception condition (e.g., missing `src_dir` AND missing `dest_dir`, invalid inputs, empty strings like `ext=''`).\n"
        "2. Exact domain semantics (e.g., nested subdirectories if recursive search is implied by `glob`/`Path`, inclusive vs exclusive bounds, ALL pairs vs distinct pairs, exact ordering/types).\n"
        "3. Exact return type, path format (relative vs absolute), and side effects.\n"
        "If ANY explicit requirement is missing, weakened, or subtly wrong, reply `VERDICT: FAIL: <one-sentence concrete violation>`.\n"
        "Only if EVERY single clause is 100% satisfied, reply `VERDICT: PASS`."
    )
    p_api = (
        f"PROBLEM SPECIFICATION:\n{problem.strip()}\n\n"
        f"CANDIDATE IMPLEMENTATION:\n```python\n{code[:6000]}\n```\n\n"
        "You are a Python API & mockability auditor verifying faithful use of declared starter imports & Requirements.\n"
        "Check strictly:\n"
        "1. Does the implementation actually use the explicitly imported/required stdlib APIs (`glob.glob`, `os.path.exists`, `os.makedirs(..., exist_ok=True)`, `shutil.move`, `zipfile.ZipFile(path, 'r')`, `threading.Thread`, `subprocess.Popen`, `socket.connect`) instead of silently bypassing them with unrequested substitutes (like `os.path.isdir` when checking 'does not exist', or `os.listdir` when `glob` is imported)?\n"
        "2. Does it avoid inventing unrequested exceptions, path rewrites, or altered signatures?\n"
        "If the candidate violates starter-API/mockability faithfulness in a way that breaks standard `@patch` unit tests, reply `VERDICT: FAIL: <one-sentence concrete violation>`.\n"
        "Otherwise reply `VERDICT: PASS`."
    )
    results = await client.batch([
        CallSpec(
            messages=[
                {"role": "system", "content": "You are a concise, strict spec-compliance auditor. End with VERDICT: PASS or VERDICT: FAIL: <reason>."},
                {"role": "user", "content": p_edge},
            ],
            tag="spec_audit_edge",
            max_tokens=950,
            temperature=0.0,
            seed=59,
            enable_thinking=True,
            thinking_budget=380,
            loop_guard=True,
        ),
        CallSpec(
            messages=[
                {"role": "system", "content": "You are a concise, strict Python API & mockability auditor. End with VERDICT: PASS or VERDICT: FAIL: <reason>."},
                {"role": "user", "content": p_api},
            ],
            tag="spec_audit_api",
            max_tokens=950,
            temperature=0.0,
            seed=61,
            enable_thinking=True,
            thinking_budget=360,
            loop_guard=True,
        ),
    ])
    violations: list[str] = []
    for res in results:
        if not res.ok:
            continue
        txt = res.content.strip()
        m_fail = re.search(r"VERDICT\s*[:=]\s*FAIL\s*[:\-]?\s*(.*)", txt, re.IGNORECASE)
        if m_fail:
            v = (m_fail.group(1).strip() or txt[-220:])[:220]
            if v:
                violations.append(v)
    if violations:
        return False, " | ".join(violations)[:340]
    return True, ""


def _normalize_modular_tests(test_code: str) -> str:
    """Ensure test_code exposes `def test_*():` functions even if model wrote top-level asserts."""
    if not test_code.strip():
        return ""
    if _extract_test_functions(test_code):
        return test_code
    indented = "\n".join("    " + ln if ln.strip() else "" for ln in test_code.splitlines())
    return f"def test_oracle_suite():\n{indented}\n"


def _extract_test_functions(test_code: str) -> dict[str, str]:
    """Split a test file into {test_name: source_snippet} for surgical inspection/arbitration."""
    out: dict[str, str] = {}
    if not test_code.strip():
        return out
    try:
        tree = ast.parse(test_code)
        lines = test_code.splitlines()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                start = node.lineno - 1
                end = getattr(node, "end_lineno", node.lineno)
                out[node.name] = "\n".join(lines[start:end])
    except SyntaxError:
        pass
    return out


def _extract_module_helpers(test_code: str, max_chars: int = 2200) -> str:
    """Extract all top-level non-`test_*` definitions (imports, constants, `_brute_ref`, helpers).

    Crucial for Helper-Aware Arbitration & Repair: arbiters MUST see helper functions
    like `_brute_ref` or `_ref_verify` to detect buggy test references!
    """
    if not test_code.strip():
        return ""
    try:
        tree = ast.parse(test_code)
        lines = test_code.splitlines()
        parts: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                continue
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)
            snippet = "\n".join(lines[start:end]).strip()
            if snippet:
                parts.append(snippet)
        joined = "\n\n".join(parts)
        return joined[:max_chars]
    except SyntaxError:
        return ""


def _build_test_context(test_code: str, test_name: str, fn_map: Optional[dict[str, str]] = None) -> str:
    """Return full executable context for `test_name`: module helpers + test function body."""
    fmap = fn_map if fn_map is not None else _extract_test_functions(test_code)
    fn_src = fmap.get(test_name, "")
    helpers = _extract_module_helpers(test_code)
    if helpers and fn_src:
        return f"# Module helpers used by test:\n{helpers}\n\n{fn_src}"
    return fn_src or helpers


def _parse_arbiter_verdict(text: str) -> str:
    """Strictly parse VERDICT: VALID | VERDICT: INVALID; return UNKNOWN if ambiguous."""
    upper = (text or "").upper()
    matches = re.findall(r"VERDICT\s*[:=]\s*(VALID|INVALID)\b", upper)
    if matches:
        return matches[-1]
    lines = [ln.strip("`* .:") for ln in upper.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if last == "INVALID":
            return "INVALID"
        if last == "VALID":
            return "VALID"
    return "UNKNOWN"


async def _arbitrate_tests_strict(
    client: TensorFoldClient,
    problem: str,
    items: list[tuple[str, str, str]],
    *,
    skeptical_note: str = "",
) -> dict[str, str]:
    """Helper-Aware Strict B=2 Consensus Arbiter (parallel batch).

    Runs 2 independent clean-context arbiters per suspicious/killing test WITH full helper
    definitions (`_brute_ref`, `_ref_verify`, etc.).
    Returns {test_name: 'VALID' | 'INVALID' | 'DISPUTED'}:
      - 'VALID' ONLY when BOTH arbiters independently verify 2/2 VALID!
      - 'INVALID' when at least one arbiter positively proves a bug/typo/spec mismatch.
      - 'DISPUTED' otherwise (fail-closed for generated tests!).
    """
    if not items:
        return {}
    specs: list[CallSpec] = []
    meta: list[tuple[str, int]] = []
    note_block = f"\nNOTE: {skeptical_note}\n" if skeptical_note else ""

    for test_name, full_test_ctx, failure_sample in items:
        prompt = (
            f"PROBLEM SPECIFICATION:\n{problem.strip()}\n\n"
            f"PROPOSED TEST & HELPERS (`{test_name}`):\n```python\n{full_test_ctx[:3200]}\n```\n\n"
            f"FAILURE OBSERVED ON INDEPENDENT CANDIDATES:\n{failure_sample[:700]}\n"
            f"{note_block}\n"
            "Inspect BOTH the test assertions AND every helper function (`_ref_*`, `_brute_*`, fixtures) line by line.\n"
            "Check for ANY typo, wrong hand-computed expected value, wrong proof/traversal order, or spec mismatch "
            "(such as assuming power-of-two sizes, mutating shared state, or inventing requirements not in the spec).\n"
            "If ANY assertion or helper logic is wrong or ambiguous relative to the spec, reply `VERDICT: INVALID`.\n"
            "Only if EVERY assertion and helper is 100% mathematically & logically faithful to the spec, reply `VERDICT: VALID`."
        )
        for idx, (temp, seed) in enumerate(((0.0, 31), (0.15, 73))):
            specs.append(
                CallSpec(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a strict formal spec arbiter auditing generated tests and their helpers. "
                                "Be concise (<=6 sentences) and end with `VERDICT: VALID` or `VERDICT: INVALID`."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    tag=f"arbiter{idx+1}_{test_name}",
                    max_tokens=1100,
                    temperature=temp,
                    seed=seed,
                    enable_thinking=True,
                    thinking_budget=420,
                    loop_guard=True,
                )
            )
            meta.append((test_name, idx))

    results = await client.batch(specs)
    votes: dict[str, list[str]] = {tname: [] for tname, _, _ in items}
    need_fallback: list[tuple[str, int, str]] = []
    prompts_by_name = {tname: (ctx, err) for tname, ctx, err in items}
    for (tname, idx), res in zip(meta, results):
        v = _parse_arbiter_verdict(res.content) if res.ok else "UNKNOWN"
        if v in ("VALID", "INVALID"):
            votes[tname].append(v)
        elif res.ok and res.content.strip():
            need_fallback.append((tname, idx, res.content[:500]))
        else:
            votes[tname].append("UNKNOWN")

    if need_fallback and hasattr(client, "decide_choice"):
        fb_tasks = [
            client.decide_choice(
                prompt=(
                    f"Spec audit for `{tname}`:\n{prompts_by_name[tname][0][:1200]}\n\n"
                    f"Prior analysis:\n{analysis}\n\nFinal verdict (VALID or INVALID):"
                ),
                choices=["VALID", "INVALID"],
                tag=f"arbiter_fb{idx+1}_{tname}",
            )
            for tname, idx, analysis in need_fallback
        ]
        fb_results = await asyncio.gather(*fb_tasks)
        for (tname, _, _), (choice, fb_res) in zip(need_fallback, fb_results):
            votes[tname].append(choice if (fb_res.ok and choice in ("VALID", "INVALID")) else "UNKNOWN")
    else:
        for tname, _, _ in need_fallback:
            votes[tname].append("UNKNOWN")

    verdicts: dict[str, str] = {}
    for tname, vlist in votes.items():
        if len(vlist) == 2 and vlist[0] == "VALID" and vlist[1] == "VALID":
            verdicts[tname] = "VALID"
        elif "INVALID" in vlist:
            verdicts[tname] = "INVALID"
        else:
            verdicts[tname] = "DISPUTED"
    return verdicts


def _apply_matrix_scores(
    candidates: list[CodeCandidate],
    matrix: dict[str, list[dict[str, Any]]],
    dropped_tests: set[str],
    is_redteam: bool = False,
) -> None:
    for c in candidates:
        if c.syntax_error:
            continue
        if c.name not in matrix:
            continue
        rows = matrix[c.name]
        active = [r for r in rows if r["name"] not in dropped_tests]
        passed = sum(1 for r in active if r["ok"])
        failed = [r["name"] for r in active if not r["ok"]]
        if is_redteam:
            c.redteam_passed = passed
            c.redteam_total = len(active) + (1 if c.redteam_fatal else 0)
            c.redteam_failures = failed + (["__rt_fatal__"] if c.redteam_fatal else [])
        else:
            c.tests_passed = passed
            c.tests_total = len(active) + (1 if c.fatal_error else 0)
            c.failed_tests = failed + (["__fatal__"] if c.fatal_error else [])
        for r in active:
            if not r["ok"]:
                c.failure_details[r["name"]] = r["err"]


def _parse_candidates(
    roles: list[tuple[str, str, float, int, int]],
    responses: list[Any],
    wave: int,
    starter_imports: Optional[list[str]] = None,
    starter_symbols: Optional[set[str]] = None,
) -> list[CodeCandidate]:
    """Extract and compile-check Python code from a parallel batch of model responses."""
    out: list[CodeCandidate] = []
    syms = starter_symbols or (_extract_starter_symbols(starter_imports) if starter_imports else set())
    for (name, _, _, _, _), res in zip(roles, responses):
        if not res.ok:
            code, syn_err, is_trunc = "", f"LLM_ERROR: {res.raw_error}", False
        else:
            code = extract_python_code(res.content)
            if starter_imports and code:
                code = _ensure_starter_imports(code, starter_imports)
            syn_err = check_syntax(code)
            is_trunc = res.finish_reason == "length"
            if is_trunc and syn_err:
                syn_err = f"TRUNCATED (hit max_tokens): {syn_err}"
        api_cov = _starter_api_coverage(code, syms) if (syn_err is None and syms) else 0.0
        out.append(
            CodeCandidate(
                name=name,
                wave=wave,
                code=code,
                raw_content=res.content,
                syntax_error=syn_err,
                truncated=is_trunc,
                starter_api_score=api_cov,
                latency_s=res.latency_s,
            )
        )
    return out


async def _evaluate_wave(
    candidates: list[CodeCandidate],
    oracle_code: str,
    oracle_total: int,
    rt_code: str,
    rt_total: int,
    sandbox_timeout_s: float,
    stdin_mode: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Unified mechanical sandbox evaluator for any wave (eliminates W1/W2/W3 duplication)."""
    valid = [c for c in candidates if c.syntax_error is None]
    o_matrix: dict[str, list[dict[str, Any]]] = {}
    rt_matrix: dict[str, list[dict[str, Any]]] = {}

    if valid and oracle_code.strip():
        o_execs = await asyncio.gather(
            *(asyncio.to_thread(_run_modular_suite_with_diff, c.code, oracle_code, sandbox_timeout_s, stdin_mode) for c in valid)
        )
        for c, (rows, diff_out, ms, fatal) in zip(valid, o_execs):
            c.runtime_ms = ms
            if diff_out:
                c.diff_outputs = diff_out
            if fatal:
                c.fatal_error = fatal
                c.failure_details["__fatal__"] = fatal
                if rows:
                    o_matrix[c.name] = rows
                else:
                    c.failed_tests = ["__fatal__"]
                    c.tests_total = max(1, oracle_total)
                    c.tests_passed = 0
            else:
                o_matrix[c.name] = rows

    if valid and rt_code.strip() and rt_total > 0:
        rt_execs = await asyncio.gather(
            *(asyncio.to_thread(_run_modular_suite, c.code, rt_code, sandbox_timeout_s, stdin_mode) for c in valid)
        )
        for c, (rows, ms, fatal) in zip(valid, rt_execs):
            c.runtime_ms += ms
            if fatal:
                c.redteam_fatal = fatal
                c.failure_details["__rt_fatal__"] = fatal
                if rows:
                    rt_matrix[c.name] = rows
                else:
                    c.redteam_failures = ["__rt_fatal__"]
                    c.redteam_total = rt_total
            else:
                rt_matrix[c.name] = rows

    return o_matrix, rt_matrix


def _rank_key(c: CodeCandidate) -> tuple[int, int, int, float, int, int, int, int, float, int, float]:
    rate = (c.total_passed / c.total_checks) if c.total_checks > 0 else 0.0
    n_fail = len(c.failed_tests) + len(c.redteam_failures)
    no_fatal = 1 if (c.fatal_error is None and c.redteam_fatal is None) else 0
    ce_net = c.diff_counterexample_wins - c.diff_counterexample_losses
    clean_perfect = 1 if (c.perfect and c.diff_counterexample_losses == 0) else 0
    return (
        1 if c.syntax_error is None else 0,
        no_fatal,
        clean_perfect,
        round(rate, 4),
        ce_net,
        -n_fail,
        c.total_passed,
        c.diff_majority_hits,
        round(c.starter_api_score, 3),
        -len(c.code),
        -c.runtime_ms,
    )


async def solve_code(
    client: TensorFoldClient,
    skills: SkillPack,
    memory: EpisodicMemory,
    problem: str,
    *,
    num_wave1_streams: int = 4,
    sandbox_timeout_s: float = 8.0,
    deep: bool = False,
    external_test_code: Optional[str] = None,
) -> CodeForgeTrace:
    """Solve a hard coding, algorithmic, or debugging task via Unified Consensus Verification."""
    t0 = time.perf_counter()
    trace = CodeForgeTrace()

    lessons_block = memory.format_prompt_block(problem, domain="code", limit=3)
    if lessons_block:
        trace.lessons_used = lessons_block.count("\n- [")

    shared_sys = (
        "You are a principal software engineer running inside a verified test-driven "
        "reasoning runtime. Follow every edge case, invariant, and exact signature in the specification."
    )
    if lessons_block:
        shared_sys += "\n\n" + lessons_block

    think_code = 1350 if deep else 750
    prob_clean = problem.strip()
    starter_imports = _extract_starter_imports(prob_clean)
    stdin_mode = (not external_test_code) and _is_stdin_problem(prob_clean)
    stdin_impl_note = (
        "\n\nI/O CONTRACT: Write a complete standalone Python 3 script reading from `sys.stdin` "
        "and printing exact output to `sys.stdout` (use fast I/O `sys.stdin.buffer.read()` and iterative loops or `sys.setrecursionlimit`)."
        if stdin_mode
        else ""
    )
    stdin_test_note = (
        "\n\nSTDIN TEST HARNESS: Do NOT `import solution`. A helper `run_solution(inp: str, timeout: float = 3.0) -> str` "
        "is already defined — call `run_solution(input_str)`! Sample Input/Output tests are ALREADY auto-injected, so focus "
        "your tests on: (1) a tiny brute-force differential fuzzer for small N, (2) tricky edge cases (N=1, bounds, max indices), "
        "and (3) one MAX-CONSTRAINT scale/TLE stress test (to catch O(N^2) timeouts & IndexErrors!)."
        if stdin_mode
        else ""
    )
    eff_timeout = max(sandbox_timeout_s, 14.0) if stdin_mode else sandbox_timeout_s

    # ------------------------------------------------------------------
    # STAGE 1: Single Parallel Wave (B=4) — Implementations + Spec-Property Oracle
    # ------------------------------------------------------------------
    all_w1_roles = [
        ("radically_simple", skills.card("race_minimal"), 0.15, 101, max(550, think_code - 100)),
        ("first_principles", skills.card("race_first_principles"), 0.45, 202, think_code + 100),
        ("edge_skeptic", skills.card("race_skeptic"), 0.40, 303, think_code),
        ("alt_algorithm", skills.card("race_alt"), 0.65, 404, think_code),
    ]
    n_impls = num_wave1_streams if (external_test_code or deep or num_wave1_streams <= 1) else max(2, num_wave1_streams - 1)
    w1_roles = all_w1_roles[: min(n_impls, len(all_w1_roles))]

    w1_specs = [
        CallSpec(
            messages=[
                {"role": "system", "content": shared_sys},
                {"role": "user", "content": f"PROBLEM:\n{prob_clean}{stdin_impl_note}\n\nROLE ({name}):\n{card_text}"},
            ],
            tag=f"w1_{name}",
            max_tokens=tb + 2900,
            temperature=temp,
            seed=seed,
            enable_thinking=True,
            thinking_budget=tb,
            loop_guard=True,
        )
        for name, card_text, temp, seed, tb in w1_roles
    ]
    if not external_test_code:
        w1_specs.append(
            CallSpec(
                messages=[
                    {"role": "system", "content": shared_sys},
                    {"role": "user", "content": f"PROBLEM:\n{prob_clean}\n\n{skills.card('oracle')}{stdin_test_note}"},
                ],
                tag="w1_spec_oracle",
                max_tokens=2300,
                temperature=0.2,
                seed=77,
                enable_thinking=True,
                thinking_budget=550,
                loop_guard=True,
            )
        )

    w1_all_res = await client.batch(w1_specs)
    w1_impl_res = w1_all_res[: len(w1_roles)]
    trace.wave1_candidates = _parse_candidates(w1_roles, w1_impl_res, wave=1, starter_imports=starter_imports)

    if external_test_code:
        oracle_code = _normalize_modular_tests(_strip_future_imports(external_test_code))
    else:
        raw_oracle = extract_python_code(w1_all_res[-1].content, salvage_truncated_tests=True) if w1_all_res[-1].ok else ""
        if not raw_oracle or check_syntax(raw_oracle) is not None:
            # Rare fallback: regenerate clean spec oracle once
            fb_res = await client.batch([
                CallSpec(
                    messages=[
                        {"role": "system", "content": shared_sys},
                        {"role": "user", "content": f"PROBLEM:\n{prob_clean}\n\n{skills.card('oracle')}{stdin_test_note}"},
                    ],
                    tag="w1_spec_oracle_retry",
                    max_tokens=2200,
                    temperature=0.1,
                    seed=99,
                    enable_thinking=True,
                    thinking_budget=400,
                    loop_guard=True,
                )
            ])
            raw_oracle = extract_python_code(fb_res[0].content, salvage_truncated_tests=True) if fb_res[0].ok else ""
        oracle_code = (
            _normalize_modular_tests(_strip_future_imports(raw_oracle))
            if raw_oracle and check_syntax(raw_oracle) is None
            else ""
        )

    # Prepend verbatim deterministic sample tests on stdin problems (immutable — never dropped by arbiter!)
    pinned_sample_names: set[str] = set()
    if stdin_mode:
        sample_code, pinned_sample_names = _build_deterministic_sample_tests(_extract_prompt_io_samples(prob_clean))
        if sample_code:
            oracle_code = (sample_code + "\n\n" + oracle_code).strip()

    trace.oracle_code = oracle_code
    oracle_fn_map = _extract_test_functions(oracle_code)
    trace.oracle_test_names = list(oracle_fn_map.keys())

    # Evaluate Wave 1 mechanically
    all_oracle_matrix, _ = await _evaluate_wave(
        trace.wave1_candidates, oracle_code, len(oracle_fn_map), "", 0, eff_timeout, stdin_mode=stdin_mode
    )

    # Helper-Aware Strict B=2 Shield against poisoned generated oracle tests (0/N passes)
    dropped_oracle: set[str] = set()
    valid_syn_w1 = [c for c in trace.wave1_candidates if c.syntax_error is None]
    if not external_test_code and len(valid_syn_w1) >= 2 and oracle_fn_map:
        suspicious: list[tuple[str, str, str]] = []
        for tname in oracle_fn_map:
            if tname in pinned_sample_names:
                continue
            passes = sum(
                1 for c in valid_syn_w1 for r in all_oracle_matrix.get(c.name, []) if r["name"] == tname and r["ok"]
            )
            fails = [
                f"[{c.name}] {r['err'][:280]}"
                for c in valid_syn_w1
                for r in all_oracle_matrix.get(c.name, [])
                if r["name"] == tname and not r["ok"]
            ]
            if passes == 0 and len(fails) >= 2:
                suspicious.append((tname, _build_test_context(oracle_code, tname, oracle_fn_map), "\n".join(fails[:2])))
        if suspicious:
            verdicts = await _arbitrate_tests_strict(
                client,
                prob_clean,
                suspicious[:8],
                skeptical_note=(
                    f"0/{len(valid_syn_w1)} independent implementations passed this generated test. "
                    "Verify carefully whether the test or its helper has a bug/typo."
                ),
            )
            for tname, v in verdicts.items():
                # Drop Wave-1 oracle test ONLY when positively flagged INVALID (never on UNKNOWN/DISPUTED!)
                if v == "INVALID" and len(oracle_fn_map) - len(dropped_oracle) > 1:
                    dropped_oracle.add(tname)
                    trace.dropped_bad_oracle_tests.append(tname)

    _apply_matrix_scores(trace.wave1_candidates, all_oracle_matrix, dropped_oracle, is_redteam=False)

    # ------------------------------------------------------------------
    # STAGE 1B: Oracle-Free Behavioral Differential Consensus & Counterexample Adjudication
    # ------------------------------------------------------------------
    diff_probe_src = _extract_diff_probe_source(oracle_code)
    n_diff_cases, diff_divergences = _analyze_behavioral_consensus(trace.wave1_candidates)
    diff_counterexample_notes: list[str] = []
    # Run LLM counterexample adjudication on library/API specs (where spec semantics decide A vs B);
    # on competitive `stdin_mode` scripts, rely on mechanical `diff_majority_hits` + brute-force tests
    # rather than asking an LLM to mentally compute large mod-998244353 DP/graph outputs!
    if not external_test_code and not stdin_mode and diff_divergences and diff_probe_src:
        diff_counterexample_notes = await _adjudicate_divergences(
            client, prob_clean, diff_probe_src, diff_divergences, trace.wave1_candidates
        )
        for note in diff_counterexample_notes:
            trace.graveyard.append(note)

    ranked_w1 = sorted(trace.wave1_candidates, key=_rank_key, reverse=True)
    best_w1 = ranked_w1[0]

    # Fast-Path Unanimous Behavioral Consensus Check + Executable Raises-Coverage + B=2 Spec-Clause Audit:
    # Only fast-exit when >=2 candidates pass all tests, agree on >=3 diff_probe cases,
    # oracle tests cover every explicit Raises: exception, AND B=2 adversarial auditors BOTH pass!
    perfect_clean_w1 = [c for c in ranked_w1 if c.perfect and c.diff_counterexample_losses == 0]
    req_excs = _extract_required_exceptions(prob_clean)
    missing_exc_checks = sorted(exc for exc in req_excs if exc not in oracle_code)
    unanimous_behavioral_fastpath = False
    spec_audit_violation = ""
    if (
        not external_test_code
        and not deep
        and len(perfect_clean_w1) >= 2
        and best_w1.tests_total >= 3
        and n_diff_cases >= 3
        and not diff_divergences
        and not missing_exc_checks
    ):
        compliant, violation = await _spec_compliance_audit(client, prob_clean, best_w1.code)
        if compliant:
            unanimous_behavioral_fastpath = True
        else:
            spec_audit_violation = violation
            trace.graveyard.append(f"[spec_compliance_audit] VIOLATION on {best_w1.name}: {violation}")
            for pc in perfect_clean_w1:
                pc.diff_counterexample_losses += 1
                pc.failure_details["spec_audit"] = violation

    # ------------------------------------------------------------------
    # STAGE 2: Targeted Clean-Context Adversarial Red-Team (skipped on external pass OR unanimous behavioral consensus!)
    # ------------------------------------------------------------------
    combined_rt_code = ""
    rt_fn_map: dict[str, str] = {}
    dropped_rt_names: set[str] = set()
    all_rt_matrix: dict[str, list[dict[str, Any]]] = {}

    non_fatal_w1 = [c for c in ranked_w1 if c.syntax_error is None and c.fatal_error is None]
    survivors_for_rt = (
        [c for c in non_fatal_w1 if c.tests_passed >= max(1, non_fatal_w1[0].tests_passed - 1)][:2]
        if (best_w1.tests_total > 0 and non_fatal_w1)
        else non_fatal_w1[:2]
    )
    skip_rt = bool((external_test_code and best_w1.perfect) or unanimous_behavioral_fastpath)

    if survivors_for_rt and not skip_rt:
        leader = survivors_for_rt[0]
        runner_up_block = ""
        if len(survivors_for_rt) >= 2 and survivors_for_rt[1].code.strip() != leader.code.strip():
            runner_up_block = (
                f"\n\nALTERNATIVE CANDIDATE (`{survivors_for_rt[1].name}`):\n"
                f"```python\n{survivors_for_rt[1].code[:4200]}\n```"
            )
        audit_hints: list[str] = []
        if missing_exc_checks:
            audit_hints.append(f"Explicitly test required Raises: exceptions {missing_exc_checks} that the Wave-1 suite missed.")
        if spec_audit_violation:
            audit_hints.append(f"Spec auditor flagged this suspected bug — write an executable test probing it: {spec_audit_violation}")
        hints_block = ("\n\nHIGH-PRIORITY PROBES:\n- " + "\n- ".join(audit_hints)) if audit_hints else ""
        rt_user = (
            f"PROBLEM SPECIFICATION:\n{prob_clean}\n\n"
            f"CANDIDATE IMPLEMENTATION (`{leader.name}`):\n```python\n{leader.code[:5500]}\n```"
            f"{runner_up_block}{hints_block}"
        )
        rt_specs = [
            CallSpec(
                messages=[
                    {"role": "system", "content": "You are a ruthless adversarial QA engineer in a fresh context."},
                    {"role": "user", "content": f"{rt_user}\n\n{skills.card('redteam_edge')}{stdin_test_note}"},
                ],
                tag="s2_redteam_edge",
                max_tokens=1500,
                temperature=0.35,
                seed=808,
                enable_thinking=True,
                thinking_budget=450,
                loop_guard=True,
            )
        ]
        if deep:
            rt_specs.append(
                CallSpec(
                    messages=[
                        {"role": "system", "content": "You are an adversarial formal-methods auditor in a fresh context."},
                        {"role": "user", "content": f"{rt_user}\n\n{skills.card('redteam_diff')}{stdin_test_note}"},
                    ],
                    tag="s2_redteam_diff",
                    max_tokens=1500,
                    temperature=0.50,
                    seed=909,
                    enable_thinking=True,
                    thinking_budget=450,
                    loop_guard=True,
                )
            )
        rt_res = await client.batch(rt_specs)

        rt_parts: list[str] = []
        for idx, res in enumerate(rt_res):
            rt_code = extract_python_code(res.content, salvage_truncated_tests=True) if res.ok else ""
            if rt_code and check_syntax(rt_code) is None:
                rt_code = _normalize_modular_tests(_strip_future_imports(rt_code))
                rt_code = re.sub(r"def\s+test_", f"def test_rt{idx+1}_", rt_code)
                rt_parts.append(rt_code)

        combined_rt_code = "\n\n".join(rt_parts)
        rt_fn_map = _extract_test_functions(combined_rt_code)
        trace.redteam_tests_generated = len(rt_fn_map)

        if rt_fn_map:
            rt_execs = await asyncio.gather(
                *(asyncio.to_thread(_run_modular_suite, c.code, combined_rt_code, eff_timeout, stdin_mode) for c in valid_syn_w1)
            )
            all_harness_crashed = (
                len(valid_syn_w1) >= 2
                and all(
                    (not rows) and rt_fatal and ("_run.py" in rt_fatal) and ("TIMEOUT" not in rt_fatal)
                    for rows, _, rt_fatal in rt_execs
                )
            )
            if all_harness_crashed:
                combined_rt_code, rt_fn_map = "", {}
            else:
                for c, (rows, rt_ms, rt_fatal) in zip(valid_syn_w1, rt_execs):
                    c.runtime_ms += rt_ms
                    if rt_fatal:
                        c.redteam_fatal = rt_fatal
                        c.failure_details["__rt_fatal__"] = rt_fatal
                        if rows:
                            all_rt_matrix[c.name] = rows
                        else:
                            c.redteam_failures = ["__rt_fatal__"]
                            c.redteam_total = len(rt_fn_map)
                    else:
                        all_rt_matrix[c.name] = rows

                # Strict Helper-Aware B=2 Consensus Gate on EVERY killing Red-Team test!
                leader_rows = {r["name"]: r for r in all_rt_matrix.get(leader.name, [])}
                to_arbitrate: list[tuple[str, str, str]] = []
                for tname in rt_fn_map:
                    failing_errs = [
                        f"[{c.name}] {r['err'][:280]}"
                        for c in valid_syn_w1
                        for r in all_rt_matrix.get(c.name, [])
                        if r["name"] == tname and not r["ok"]
                    ]
                    ran_somewhere = any(
                        r["name"] == tname for c in valid_syn_w1 for r in all_rt_matrix.get(c.name, [])
                    )
                    if not ran_somewhere:
                        dropped_rt_names.add(tname)
                    elif failing_errs:
                        lrow = leader_rows.get(tname)
                        sample = f"[{leader.name}] {lrow['err'][:300]}" if (lrow and not lrow["ok"]) else "\n".join(failing_errs[:2])
                        to_arbitrate.append((tname, _build_test_context(combined_rt_code, tname, rt_fn_map), sample))

                if to_arbitrate:
                    rt_verdicts = await _arbitrate_tests_strict(
                        client,
                        prob_clean,
                        to_arbitrate[:4],
                        skeptical_note="This is an adversarial Red-Team test trying to kill a candidate. Only mark VALID if 100% undeniably required by the spec.",
                    )
                    for tname, _, _ in to_arbitrate:
                        if rt_verdicts.get(tname) == "VALID":
                            lrow = leader_rows.get(tname)
                            if lrow and not lrow["ok"]:
                                trace.redteam_valid_kills.append(f"{tname}: {lrow['err'][:200]}")
                        else:
                            dropped_rt_names.add(tname)

                _apply_matrix_scores(trace.wave1_candidates, all_rt_matrix, dropped_rt_names, is_redteam=True)

    ranked_w1 = sorted(trace.wave1_candidates, key=_rank_key, reverse=True)
    best_w1 = ranked_w1[0]

    # ------------------------------------------------------------------
    # STAGE 3: Conditional Graveyard Repair (Wave 2 B=3 + Pre-W3 Re-Audit + Wave 3 B=2)
    # ------------------------------------------------------------------
    need_wave2 = (
        (not best_w1.perfect)
        or (best_w1.diff_counterexample_losses > 0)
        or bool(spec_audit_violation)
        or bool(best_w1.redteam_failures)
        or (best_w1.syntax_error is not None)
        or (best_w1.fatal_error is not None)
    )
    if need_wave2:
        trace.wave2_triggered = True
        if best_w1.redteam_failures:
            trace.wave2_reason = f"Red-Teamer killed Wave-1 leader ({best_w1.name}) on {best_w1.redteam_failures}"
        elif best_w1.syntax_error:
            trace.wave2_reason = "All Wave-1 candidates had syntax errors"
        elif best_w1.fatal_error:
            trace.wave2_reason = f"Wave-1 leader hit fatal error: {best_w1.fatal_error[:120]}"
        else:
            trace.wave2_reason = f"Best Wave-1 ({best_w1.name}) passed {best_w1.total_passed}/{best_w1.total_checks}"

        for c in ranked_w1[:3]:
            if c.syntax_error:
                trace.graveyard.append(f"[{c.name}] SyntaxError: {c.syntax_error}")
            elif c.failed_tests or c.redteam_failures or c.fatal_error:
                errs = "; ".join(f"{k} -> {v[:320]}" for k, v in list(c.failure_details.items())[:3])
                trace.graveyard.append(f"[{c.name}] ({c.total_passed}/{c.total_checks}): {errs}")

        def _collect_failing_contexts(cand: CodeCandidate, limit: int = 5) -> str:
            blocks: list[str] = []
            o_helpers = _extract_module_helpers(oracle_code, max_chars=1400)
            rt_helpers = _extract_module_helpers(combined_rt_code, max_chars=1000)
            used_o, used_rt = False, False
            for fname in (cand.failed_tests + cand.redteam_failures)[:limit]:
                if fname in oracle_fn_map:
                    used_o = True
                    blocks.append(oracle_fn_map[fname])
                elif fname in rt_fn_map:
                    used_rt = True
                    blocks.append(rt_fn_map[fname])
            header = "\n\n".join(h for flag, h in ((used_o, o_helpers), (used_rt, rt_helpers)) if flag and h)
            return ((header + "\n\n") if header else "") + "\n\n".join(blocks)

        failing_block = _collect_failing_contexts(best_w1, limit=5)
        w2_base_no_anchor = (
            f"PROBLEM:\n{prob_clean}{stdin_impl_note}\n\n"
            f"GRAVEYARD OF WAVE-1 FAILURES (DO NOT REPEAT THESE BUGS OR WRONG ALGORITHMS):\n" + "\n".join(trace.graveyard)
            + (f"\n\nPROVEN FAILING TESTS YOUR CODE MUST PASS:\n```python\n{failing_block[:4500]}\n```" if failing_block else "")
        )
        w2_context = (
            w2_base_no_anchor
            + f"\n\nCLOSEST WAVE-1 CANDIDATE (`{best_w1.name}`, for reference only — simplify or rewrite cleanly if flawed):\n"
            f"```python\n{best_w1.code[:9000]}\n```"
        )

        # Wave 2 (B=3): textbook simple rewrite + surgical fix + ZERO-ANCHOR fresh alt algorithm!
        w2_roles = [
            ("w2_radically_simple", skills.card("race_minimal"), 0.25, 1101, think_code + 150, True),
            ("w2_surgical_fix", skills.card("repair_surgical"), 0.15, 1202, think_code + 150, True),
            ("w2_clean_rewrite", skills.card("race_alt"), 0.55, 1303, think_code + 300, False),
        ][: min(3, max(1, num_wave1_streams))]

        w2_res = await client.batch([
            CallSpec(
                messages=[
                    {"role": "system", "content": shared_sys},
                    {
                        "role": "user",
                        "content": f"{(w2_context if include_ref else w2_base_no_anchor)}\n\nREPAIR ROLE ({name}):\n{card_text}",
                    },
                ],
                tag=name,
                max_tokens=tb + 3000,
                temperature=temp,
                seed=seed,
                enable_thinking=True,
                thinking_budget=tb,
                loop_guard=True,
            )
            for name, card_text, temp, seed, tb, include_ref in w2_roles
        ])
        trace.wave2_candidates = _parse_candidates([(n, c, t, s, b) for n, c, t, s, b, _ in w2_roles], w2_res, wave=2, starter_imports=starter_imports)

        kept_oracle_cnt = max(1, len([n for n in oracle_fn_map if n not in dropped_oracle]))
        active_rt_cnt = len([n for n in rt_fn_map if n not in dropped_rt_names])
        w2_o_mat, w2_rt_mat = await _evaluate_wave(
            trace.wave2_candidates, oracle_code, kept_oracle_cnt, combined_rt_code, active_rt_cnt, eff_timeout, stdin_mode=stdin_mode
        )
        all_oracle_matrix.update(w2_o_mat)
        all_rt_matrix.update(w2_rt_mat)

        _apply_matrix_scores(trace.wave2_candidates, all_oracle_matrix, dropped_oracle, is_redteam=False)
        if combined_rt_code.strip() and active_rt_cnt > 0:
            _apply_matrix_scores(trace.wave2_candidates, all_rt_matrix, dropped_rt_names, is_redteam=True)
        # Re-compute cross-candidate behavioral consensus across Wave 1 + Wave 2 (and adjudicate new W2 divergences on API specs!)
        all_w12_cands = trace.wave1_candidates + trace.wave2_candidates
        _, w2_divs = _analyze_behavioral_consensus(all_w12_cands)
        if not external_test_code and not stdin_mode and w2_divs and diff_probe_src:
            # Only adjudicate cases involving a Wave-2 candidate that weren't already resolved
            w2_names = {c.name for c in trace.wave2_candidates}
            fresh_divs = [(case, obs) for case, obs in w2_divs if any(n in w2_names for n in obs)]
            if fresh_divs:
                w2_notes = await _adjudicate_divergences(client, prob_clean, diff_probe_src, fresh_divs[:2], all_w12_cands)
                for note in w2_notes:
                    trace.graveyard.append(note)

        # ------------------------------------------------------------------
        # PRE-WAVE-3 UNANIMOUS-FAILURE RE-AUDIT (Kills poisoned-oracle overfitting!)
        # If 0 out of >=4 independent W1+W2 implementations pass a generated test,
        # re-audit it with extreme skepticism BEFORE Wave 3 can overfit to a typo!
        # ------------------------------------------------------------------
        w12_valid = [c for c in (trace.wave1_candidates + trace.wave2_candidates) if c.syntax_error is None]
        if not external_test_code and len(w12_valid) >= 4:
            reaudit_items: list[tuple[str, str, str, bool]] = []
            for tname in list(oracle_fn_map):
                if tname in dropped_oracle or tname in pinned_sample_names:
                    continue
                passes = sum(1 for c in w12_valid for r in all_oracle_matrix.get(c.name, []) if r["name"] == tname and r["ok"])
                fails = [
                    f"[{c.name}] {r['err'][:240]}"
                    for c in w12_valid
                    for r in all_oracle_matrix.get(c.name, [])
                    if r["name"] == tname and not r["ok"]
                ]
                if passes == 0 and len(fails) >= 4:
                    reaudit_items.append((tname, _build_test_context(oracle_code, tname, oracle_fn_map), "\n".join(fails[:2]), False))
            for tname in list(rt_fn_map):
                if tname in dropped_rt_names:
                    continue
                passes = sum(1 for c in w12_valid for r in all_rt_matrix.get(c.name, []) if r["name"] == tname and r["ok"])
                fails = [
                    f"[{c.name}] {r['err'][:240]}"
                    for c in w12_valid
                    for r in all_rt_matrix.get(c.name, [])
                    if r["name"] == tname and not r["ok"]
                ]
                if passes == 0 and len(fails) >= 4:
                    reaudit_items.append((tname, _build_test_context(combined_rt_code, tname, rt_fn_map), "\n".join(fails[:2]), True))

            if reaudit_items:
                re_verdicts = await _arbitrate_tests_strict(
                    client,
                    prob_clean,
                    [(t, ctx, err) for t, ctx, err, _ in reaudit_items[:6]],
                    skeptical_note=(
                        f"CRITICAL WARNING: 0 out of {len(w12_valid)} independent implementations across 2 waves "
                        "(including repair passes that read this exact test!) passed it. "
                        "Check for ANY typo, wrong hand calculation, or buggy helper."
                    ),
                )
                changed = False
                for tname, _, _, is_rt in reaudit_items:
                    if re_verdicts.get(tname) != "VALID":
                        if is_rt:
                            dropped_rt_names.add(tname)
                            changed = True
                        elif len(oracle_fn_map) - len(dropped_oracle) > 1 or re_verdicts.get(tname) == "INVALID":
                            dropped_oracle.add(tname)
                            trace.dropped_bad_oracle_tests.append(tname)
                            changed = True
                if changed:
                    all_w12 = trace.wave1_candidates + trace.wave2_candidates
                    _apply_matrix_scores(all_w12, all_oracle_matrix, dropped_oracle, is_redteam=False)
                    if combined_rt_code.strip():
                        _apply_matrix_scores(all_w12, all_rt_matrix, dropped_rt_names, is_redteam=True)

        # Conditional Wave-3 Surgical Repair (B=2, ONLY if best after W2 + re-audit is STILL not perfect!)
        best_after_w2 = sorted(trace.wave1_candidates + trace.wave2_candidates, key=_rank_key, reverse=True)[0]
        if (not best_after_w2.perfect) and best_after_w2.total_checks > 0:
            w3_failing = (best_after_w2.failed_tests + best_after_w2.redteam_failures)[:6]
            w3_fail_lines = [f"- {fn}: {best_after_w2.failure_details.get(fn, 'failed')[:350]}" for fn in w3_failing]
            w3_test_block = _collect_failing_contexts(best_after_w2, limit=5)
            w3_base_no_anchor = (
                f"PROBLEM:\n{prob_clean}{stdin_impl_note}\n\n"
                f"PRIOR FAILURES TO AVOID:\n" + "\n".join(w3_fail_lines)
                + (f"\n\nEXACT FAILING TEST CODE:\n```python\n{w3_test_block[:4500]}\n```" if w3_test_block else "")
            )
            w3_context = (
                f"PROBLEM:\n{prob_clean}{stdin_impl_note}\n\n"
                f"CURRENT BEST (`{best_after_w2.name}`, {best_after_w2.total_passed}/{best_after_w2.total_checks} CHECKS):\n"
                f"```python\n{best_after_w2.code[:9500]}\n```\n\n"
                f"EXACT REMAINING FAILURES TO FIX:\n" + "\n".join(w3_fail_lines)
                + (f"\n\nEXACT FAILING TEST CODE:\n```python\n{w3_test_block[:4500]}\n```" if w3_test_block else "")
            )
            w3_roles = [
                ("w3_surgical_fix", skills.card("repair_surgical"), 0.15, 2101, think_code + 500, True),
                ("w3_fresh_alt", skills.card("race_alt"), 0.45, 2202, think_code + 550, False),
            ]
            w3_res = await client.batch([
                CallSpec(
                    messages=[
                        {"role": "system", "content": shared_sys},
                        {
                            "role": "user",
                            "content": f"{(w3_context if include_ref else w3_base_no_anchor)}\n\nREPAIR ROLE ({name}):\n{card_text}",
                        },
                    ],
                    tag=name,
                    max_tokens=tb + 3500,
                    temperature=temp,
                    seed=seed,
                    enable_thinking=True,
                    thinking_budget=tb,
                    loop_guard=True,
                )
                for name, card_text, temp, seed, tb, include_ref in w3_roles
            ])
            w3_cands = _parse_candidates([(n, c, t, s, b) for n, c, t, s, b, _ in w3_roles], w3_res, wave=3, starter_imports=starter_imports)
            active_rt_cnt = len([n for n in rt_fn_map if n not in dropped_rt_names])
            kept_oracle_cnt = max(1, len([n for n in oracle_fn_map if n not in dropped_oracle]))
            w3_o_mat, w3_rt_mat = await _evaluate_wave(
                w3_cands, oracle_code, kept_oracle_cnt, combined_rt_code, active_rt_cnt, eff_timeout, stdin_mode=stdin_mode
            )
            _apply_matrix_scores(w3_cands, w3_o_mat, dropped_oracle, is_redteam=False)
            if combined_rt_code.strip() and active_rt_cnt > 0:
                _apply_matrix_scores(w3_cands, w3_rt_mat, dropped_rt_names, is_redteam=True)
            trace.wave2_candidates.extend(w3_cands)
            _analyze_behavioral_consensus(trace.wave1_candidates + trace.wave2_candidates)

    # ------------------------------------------------------------------
    # FINAL SELECTION: Ranked Best + Consensus-Respecting Order-Swapped Tie-Break
    # ------------------------------------------------------------------
    ranked_all = sorted(trace.wave1_candidates + trace.wave2_candidates, key=_rank_key, reverse=True)
    winner = ranked_all[0]
    winner_ce_net = winner.diff_counterexample_wins - winner.diff_counterexample_losses

    # Strictly preserve Behavioral Differential Consensus & Counterexample Adjudication:
    # A minority or counterexample-losing candidate may NEVER hijack the pairwise tie-break!
    top_tied = [
        c
        for c in ranked_all
        if c.syntax_error is None
        and c.fatal_error is None
        and c.redteam_fatal is None
        and c.total_passed == winner.total_passed
        and (len(c.failed_tests) + len(c.redteam_failures)) == (len(winner.failed_tests) + len(winner.redteam_failures))
        and (c.diff_counterexample_wins - c.diff_counterexample_losses) == winner_ce_net
        and c.diff_majority_hits == winner.diff_majority_hits
    ]
    unique_tied: list[CodeCandidate] = []
    seen_codes: set[str] = set()
    for c in top_tied:
        norm_c = c.code.strip()
        if norm_c and norm_c not in seen_codes:
            seen_codes.add(norm_c)
            unique_tied.append(c)

    if len(unique_tied) >= 2:
        # Break true behavioral ties by starter-API faithfulness first, then Occam's Razor (code compactness)
        unique_tied.sort(key=lambda c: (-round(c.starter_api_score, 3), len(c.code), c.runtime_ms))
        champ = unique_tied[0]
        for challenger in unique_tied[1:3]:
            p_ab = (
                f"PROBLEM:\n{prob_clean}\n\n"
                f"CANDIDATE A (`{champ.name}`, {len(champ.code)} chars, {champ.runtime_ms:.1f}ms):\n```python\n{champ.code[:7500]}\n```\n\n"
                f"CANDIDATE B (`{challenger.name}`, {len(challenger.code)} chars, {challenger.runtime_ms:.1f}ms):\n```python\n{challenger.code[:7500]}\n```\n\n"
                "Both passed all automated tests. Which implementation is strictly MORE faithful to the exact specification, "
                "starter imports/signatures, and return formats — WITHOUT inventing unrequested exceptions, path transformations, "
                "or over-engineered deviations, while still handling real spec edge cases? Reply A or B."
            )
            p_ba = (
                f"PROBLEM:\n{prob_clean}\n\n"
                f"CANDIDATE A (`{challenger.name}`, {len(challenger.code)} chars, {challenger.runtime_ms:.1f}ms):\n```python\n{challenger.code[:7500]}\n```\n\n"
                f"CANDIDATE B (`{champ.name}`, {len(champ.code)} chars, {champ.runtime_ms:.1f}ms):\n```python\n{champ.code[:7500]}\n```\n\n"
                "Both passed all automated tests. Which implementation is strictly MORE faithful to the exact specification, "
                "starter imports/signatures, and return formats — WITHOUT inventing unrequested exceptions, path transformations, "
                "or over-engineered deviations, while still handling real spec edge cases? Reply A or B."
            )
            if await client.pairwise_vote(p_ab, p_ba, tag_prefix="code_judge") == "B":
                champ = challenger
        winner = champ

    w1_first = trace.wave1_candidates[0]
    if (
        winner.syntax_error is None
        and winner.fatal_error is None
        and winner.redteam_fatal is None
        and winner.total_passed > 0
        and (
            (winner.wave >= 2 and (winner.total_passed > best_w1.total_passed or (winner.perfect and not best_w1.perfect)))
            or (winner.name != w1_first.name and winner.total_passed > w1_first.total_passed)
        )
    ):
        trace.recovered_by_redteam_or_wave2 = True

    trace.verified = winner.perfect
    trace.winner_name = winner.name
    trace.winner_code = winner.code
    trace.wall_s = time.perf_counter() - t0
    return trace

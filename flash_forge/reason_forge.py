"""Neuro-Symbolic Parallel Reasoning Forge (Math, Probability, Combinatorics, Logic, CSP).

Exploits Qwen3.8-Flash-Next's deduplicated B=2..16 MoE batching + Python/SymPy/Z3 sandbox:
  Wave 1 (B=4 parallel over shared cached prefix):
    - pal_exact:    Exact Python (Fraction / sympy / math) + small-case checks
    - pal_alt_z3:   Independent method (Z3 SMT / brute-force enumeration / DP)
    - pal_skeptic:  Trap-hunter script asserting EVERY problem constraint individually
    - analytical:   Pure closed-form derivation (catches huge-N formulas where brute-force times out)
  Sandbox Execution & Canonical Equivalence Clustering:
    - Normalizes Fraction / SymPy / LaTeX / int / float / tuples / booleans
  Wave 2 (Conditional — fires automatically on ANY disagreement or script failure):
    - Clean-context hard reset + Graveyard of failed scripts/disagreements
    - B=3 Disagreement Falsifiers (small-N oracle, constraint-by-constraint re-reader, fresh Z3/SymPy arbiter)
  Final Verified Adjudication + Clean Answer Synthesis
"""

from __future__ import annotations

import ast
import asyncio
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Optional

import sympy as sp

from .client import CallSpec, TensorFoldClient
from .memory import EpisodicMemory
from .sandbox import ExecResult, extract_python_code, run_python
from .schemas import ANSWER_EXTRACT_SCHEMA
from .skill_loader import SkillPack


@dataclass
class BranchOutcome:
    name: str
    wave: int
    kind: str  # "pal" or "analytical"
    raw_content: str
    reasoning: str
    code: str = ""
    exec_result: Optional[ExecResult] = None
    raw_answer: str = ""
    norm_answer: str = ""
    verified_constraints: bool = False
    error: str = ""
    latency_s: float = 0.0


@dataclass
class ReasonForgeTrace:
    mode: str = "reason"
    wave1_outcomes: list[BranchOutcome] = field(default_factory=list)
    wave2_outcomes: list[BranchOutcome] = field(default_factory=list)
    wave2_triggered: bool = False
    wave2_reason: str = ""
    graveyard: list[str] = field(default_factory=list)
    consensus_type: str = ""
    final_answer: str = ""
    final_explanation: str = ""
    wall_s: float = 0.0
    lessons_used: int = 0
    recovered_from_disagreement: bool = False
    verified: bool = False


_ANSWER_LINE_RE = re.compile(r"^\s*(?:FINAL[_\s]*)?ANSWER\s*[:=]\s*(.+?)\s*$", re.IGNORECASE)
_ANSWER_BARE_RE = re.compile(r"^\s*(?:FINAL[_\s]*)?ANSWER\s+([^\s].+?)\s*$")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_FRAC_RE = re.compile(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}")
_DIAG_NOISE_RE = re.compile(
    r"^(?:debug|check|assert|verif|done|ok|pass(?:ed)?|fail(?:ed)?|solution:?|result:?|note:?|warning:?)\b",
    re.IGNORECASE,
)


def _strip_md_wrappers(line: str) -> str:
    """Strip common markdown quote/bullet/bold/code wrappers around a single line."""
    s = line.strip()
    s = re.sub(r"^(?:[>*\-+]+\s+)+", "", s)
    for _ in range(3):
        prev = s
        if len(s) >= 4 and ((s.startswith("**") and s.endswith("**")) or (s.startswith("__") and s.endswith("__"))):
            s = s[2:-2].strip()
        elif len(s) >= 2 and s[0] == s[-1] and s[0] in ("`", "*", "_"):
            s = s[1:-1].strip()
        if s == prev:
            break
    return s


def _match_answer_line(line: str) -> str:
    cleaned = _strip_md_wrappers(line)
    m = _ANSWER_LINE_RE.match(cleaned)
    if m:
        return _strip_md_wrappers(m.group(1))
    m2 = _ANSWER_BARE_RE.match(cleaned)
    if m2:
        return _strip_md_wrappers(m2.group(1))
    return ""


def extract_answer_from_stdout(stdout: str) -> str:
    if not stdout:
        return ""
    lines = [ln.strip() for ln in stdout.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        val = _match_answer_line(ln)
        if val:
            return val
    # Conservative fallback: only inspect the final line, and only accept clean value-like output
    # (never trailing ellipses, colons, or status prose like 'done' / 'working...').
    last = lines[-1]
    if (
        len(last) <= 120
        and not _DIAG_NOISE_RE.match(last)
        and not last.endswith((":", "..."))
        and re.fullmatch(r"[\-\+\d\./,a-zA-Z0-9_=\(\)\[\]\s]+", last)
        and len(last.split()) <= 4
    ):
        return last
    return ""


def extract_answer_from_prose(text: str) -> str:
    if not text:
        return ""
    # Prefer explicit ANSWER: line (handling markdown **ANSWER: X** / `ANSWER: X`)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        val = _match_answer_line(ln)
        if val:
            return val
    # Also scan inline **ANSWER: ...** anywhere near end of prose
    inline = re.findall(r"(?:FINAL[_\s]*)?ANSWER\s*[:=]\s*([^\n\*`]+)", text, re.IGNORECASE)
    if inline:
        return _strip_md_wrappers(inline[-1])
    # Next prefer last \boxed{...}
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    return ""


def _clean_latex(s: str) -> str:
    s = s.strip()
    s = s.strip("$").strip()
    s = s.rstrip(".")
    # Replace \frac{a}{b} -> (a)/(b)
    for _ in range(4):
        new_s = _FRAC_RE.sub(r"(\1)/(\2)", s)
        if new_s == s:
            break
        s = new_s
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\ ", "").replace("%", "")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = s.strip()
    return s


def normalize_answer(raw: str) -> str:
    """Canonicalize exact math/logic answers so equivalent forms cluster together."""
    if not raw:
        return ""
    s = _clean_latex(raw)
    # Strip trailing period/backticks/quotes
    s = s.strip("`\"' ").rstrip(".")
    if not s:
        return ""

    # Handle Python Fraction(a, b)
    m_frac = re.fullmatch(r"Fraction\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", s)
    if m_frac:
        f = Fraction(int(m_frac.group(1)), int(m_frac.group(2)))
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"

    # Strip trailing commentary after a leading exact integer/fraction (e.g. "19. AIME answer 019..." -> "19")
    m_lead_num = re.match(r"^(-?\d+(?:/\d+)?)(?:\.\s+|\s+[A-Za-z(])", s)
    if m_lead_num and not re.search(r"[+\-*/^=]", s[len(m_lead_num.group(1)):]):
        s = m_lead_num.group(1)

    # Remove thousands commas inside plain numbers like 1,234,567
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", s):
        s = s.replace(",", "")

    # Try exact integer / rational
    try:
        if re.fullmatch(r"-?\d+", s):
            return str(int(s))
        if re.fullmatch(r"-?\d+\s*/\s*-?\d+", s):
            f = Fraction(s.replace(" ", ""))
            return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
    except Exception:
        pass

    # Try safe Python literal (list/tuple/dict/bool)
    if s.lower() in ("true", "yes"):
        return "True"
    if s.lower() in ("false", "no"):
        return "False"
    try:
        lit = ast.literal_eval(s)
        if isinstance(lit, bool):
            return "True" if lit else "False"
        if isinstance(lit, int):
            return str(lit)
        if isinstance(lit, float) and math.isfinite(lit):
            if abs(lit - round(lit)) < 1e-12:
                return str(int(round(lit)))
            # Check exact IEEE-754 float-printed fraction match (e.g. 0.5833333333333334 -> 7/12)
            # Tight 1e-12 guard prevents truncated decimals like 0.3333333333 from falsely snapping to 1/3
            approx_f = Fraction(lit).limit_denominator(10000)
            if abs(float(approx_f) - lit) < 1e-12:
                return str(approx_f.numerator) if approx_f.denominator == 1 else f"{approx_f.numerator}/{approx_f.denominator}"
            return f"{lit:.10g}"
        if isinstance(lit, (list, tuple)):
            return repr(list(lit))
        if isinstance(lit, dict):
            return repr(dict(sorted(lit.items(), key=lambda kv: str(kv[0]))))
    except Exception:
        pass

    # Try SymPy exact simplification for algebraic / numeric expressions
    if len(s) <= 120 and re.fullmatch(r"[0-9a-zA-Z_+\-*/^().\s]+", s):
        try:
            expr = sp.sympify(s.replace("^", "**"))
            simp = sp.simplify(expr)
            if simp.is_Integer:
                return str(int(simp))
            if simp.is_Rational:
                p, q = simp.as_numer_denom()
                return str(int(p)) if int(q) == 1 else f"{int(p)}/{int(q)}"
            if simp.is_Float:
                val = float(simp)
                if abs(val - round(val)) < 1e-12:
                    return str(int(round(val)))
                return f"{val:.10g}"
            return str(simp)
        except Exception:
            pass

    # Plain normalized string (e.g. "Alice", "B", "Monday")
    return re.sub(r"\s+", " ", s).strip()


def _answers_equivalent(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a.lower() == b.lower():
        return True
    try:
        ea = sp.sympify(a.replace("^", "**"))
        eb = sp.sympify(b.replace("^", "**"))
        diff = sp.simplify(ea - eb)
        if diff == 0:
            return True
        if ea.is_number and eb.is_number:
            fa, fb = complex(ea.evalf()), complex(eb.evalf())
            if abs(fa - fb) < 1e-12 * max(1.0, abs(fa), abs(fb)):
                return True
    except Exception:
        pass
    return False


def _cluster_answers(outcomes: list[BranchOutcome]) -> list[tuple[str, list[BranchOutcome]]]:
    """Group outcomes by mathematical/string equivalence."""
    clusters: list[tuple[str, list[BranchOutcome]]] = []
    for o in outcomes:
        if not o.norm_answer:
            continue
        placed = False
        for i, (rep, members) in enumerate(clusters):
            if _answers_equivalent(rep, o.norm_answer):
                members.append(o)
                # Prefer cleaner rational/integer representation over decimal
                if "/" in o.norm_answer and "." in rep:
                    clusters[i] = (o.norm_answer, members)
                placed = True
                break
        if not placed:
            clusters.append((o.norm_answer, [o]))
    return clusters


def _has_executable_assert(code: str) -> bool:
    """AST-level check for at least one non-trivial `assert` statement (never fooled by comments/strings)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            if isinstance(node.test, ast.Constant) and node.test.value is True:
                continue
            return True
    return False


_INT_EXPECTED_RE = re.compile(
    r"(?:ANSWER\s*[:=]\s*<\s*int(?:eger)?\s*>|exact\s+integer|is\s+an\s+integer\b|integer\s+from\s+\d+\s+to\s+\d+)",
    re.IGNORECASE,
)


def _answer_satisfies_prompt_format(problem: str, norm_answer: str) -> bool:
    """Reject blatantly format-invalid answers (e.g. unsimplified radicals when prompt demands an integer)."""
    if not norm_answer:
        return False
    if _INT_EXPECTED_RE.search(problem):
        return bool(re.fullmatch(r"-?\d+", norm_answer.strip()))
    return True


def _branch_weight(o: BranchOutcome) -> float:
    """Executable scripts outweigh unexecuted text prose; AST-verified assert is a small tiebreaker only."""
    if o.kind == "pal":
        if o.exec_result and o.exec_result.ok and o.norm_answer:
            w = 3.0
            if o.verified_constraints:
                w += 0.25
            return w
        return 0.0
    # Analytical prose stream
    if o.norm_answer:
        return 1.25
    return 0.0


def _execute_pal_outcome(outcome: BranchOutcome, timeout_s: float = 10.0) -> None:
    code = extract_python_code(outcome.raw_content)
    outcome.code = code
    if not code:
        outcome.error = "No valid Python code block extracted"
        return
    res = run_python(code, timeout_s=timeout_s)
    outcome.exec_result = res
    if not res.ok:
        outcome.error = res.short_error(600)
        return
    raw_ans = extract_answer_from_stdout(res.stdout)
    if not raw_ans:
        outcome.error = f"Script exited 0 but printed no ANSWER line (stdout={res.stdout[:200]!r})"
        return
    outcome.raw_answer = raw_ans
    outcome.norm_answer = normalize_answer(raw_ans)
    outcome.verified_constraints = _has_executable_assert(code)


async def solve_reasoning(
    client: TensorFoldClient,
    skills: SkillPack,
    memory: EpisodicMemory,
    problem: str,
    *,
    num_wave1_streams: int = 4,
    sandbox_timeout_s: float = 10.0,
    deep: bool = False,
) -> ReasonForgeTrace:
    """Solve a hard math, logic, combinatorics, probability, or constraint problem."""
    t0 = time.perf_counter()
    trace = ReasonForgeTrace()

    lessons_block = memory.format_prompt_block(problem, domain="reason", limit=3)
    if lessons_block:
        trace.lessons_used = lessons_block.count("\n- [")

    # Shared cached prefix across ALL Wave-1 streams so TensorFold's 16 GiB prompt cache hits!
    shared_sys = (
        "You are a precision reasoning engine running inside a verified neuro-symbolic sandbox "
        "with Python 3, sympy, z3, fractions, itertools, math, and numpy available.\n"
        "Never guess arithmetic, combinatorics, or constraint satisfaction in prose when Python can compute and verify it exactly."
    )
    if lessons_block:
        shared_sys += "\n\n" + lessons_block

    exemplar_block = skills.match_exemplars("pal_verify", problem, limit=1)
    shared_user_prefix = f"PROBLEM:\n{problem.strip()}"
    if exemplar_block:
        shared_user_prefix += f"\n\n{exemplar_block}"

    # Wave 1: Role-seeded parallel streams (B=4; skip serial warm_prefix RTT which thrashes hybrid SSM cache!)
    # Keep PAL thinking crisp so Flash-Next models directly in Python instead of mental arithmetic!
    think_pal = 1200 if deep else 700
    think_ana = 2200 if deep else 1400

    wave1_roles = [
        ("pal_exact", "pal", skills.card("pal_exact"), 0.25, 101, think_pal),
        ("pal_alt_z3", "pal", skills.card("pal_alt_z3"), 0.60, 202, think_pal),
        ("pal_skeptic", "pal", skills.card("pal_skeptic"), 0.45, 303, think_pal),
        ("analytical", "analytical", skills.card("reason_analytical"), 0.40, 404, think_ana),
    ][:num_wave1_streams]

    specs: list[CallSpec] = []
    for name, kind, card_text, temp, seed, t_budget in wave1_roles:
        specs.append(
            CallSpec(
                messages=[
                    {"role": "system", "content": shared_sys},
                    {"role": "user", "content": f"{shared_user_prefix}\n\nPROTOCOL ({name}):\n{card_text}"},
                ],
                tag=f"w1_{name}",
                max_tokens=t_budget + 2400,
                temperature=temp,
                seed=seed,
                enable_thinking=True,
                thinking_budget=t_budget,
                loop_guard=True,
            )
        )

    results = await client.batch(specs)

    pal_w1_jobs: list[BranchOutcome] = []
    for (name, kind, _, _, _, _), res in zip(wave1_roles, results):
        out = BranchOutcome(
            name=name,
            wave=1,
            kind=kind,
            raw_content=res.content,
            reasoning=res.reasoning,
            latency_s=res.latency_s,
            error=res.raw_error or "",
        )
        if res.ok:
            if kind == "pal":
                pal_w1_jobs.append(out)
            else:
                raw_ans = extract_answer_from_prose(res.content)
                out.raw_answer = raw_ans
                out.norm_answer = normalize_answer(raw_ans)
        trace.wave1_outcomes.append(out)
    if pal_w1_jobs:
        await asyncio.gather(*(asyncio.to_thread(_execute_pal_outcome, o, sandbox_timeout_s) for o in pal_w1_jobs))

    # Evaluate Wave 1 consensus (filtering out blatantly format-invalid outputs)
    valid_pal_w1 = [
        o for o in trace.wave1_outcomes
        if o.kind == "pal" and o.norm_answer and _answer_satisfies_prompt_format(problem, o.norm_answer)
    ]
    all_valid_w1 = [
        o for o in trace.wave1_outcomes
        if o.norm_answer and _answer_satisfies_prompt_format(problem, o.norm_answer)
    ]
    pal_clusters = _cluster_answers(valid_pal_w1)
    all_clusters = _cluster_answers(all_valid_w1)

    # Strong consensus condition:
    # - At least 2 independent executable PAL scripts succeeded AND ALL succeeded PAL scripts agree,
    # - AND analytical stream (if it produced an answer) does not contradict them (or 3/3 PAL scripts agree!).
    need_wave2 = False
    if len(valid_pal_w1) == 0:
        need_wave2 = True
        trace.wave2_reason = "All Wave-1 Python verification scripts failed or timed out"
    elif len(pal_clusters) > 1:
        need_wave2 = True
        summary = ", ".join(f"{ans} ({len(ms)})" for ans, ms in pal_clusters)
        trace.wave2_reason = f"PAL script disagreement: {summary}"
    elif len(valid_pal_w1) == 1:
        # Only 1 script succeeded — check if analytical agrees
        if len(all_clusters) != 1 or len(all_valid_w1) < 2:
            need_wave2 = True
            trace.wave2_reason = "Only 1 PAL script succeeded without independent corroboration"
    elif len(pal_clusters) == 1 and len(all_clusters) > 1 and len(valid_pal_w1) < 3:
        # 2 PAL scripts agreed, analytical disagreed — double-check in Wave 2!
        need_wave2 = True
        trace.wave2_reason = (
            f"PAL ({pal_clusters[0][0]}) disagreed with analytical stream; running falsifier"
        )

    # Record graveyard tombstones for failed/conflicting Wave-1 attempts
    for o in trace.wave1_outcomes:
        if o.error:
            trace.graveyard.append(f"[{o.name}] FAILED: {o.error[:240]}")

    # Wave 2: Clean-Context Disagreement Falsifier + Hard Reset (B=3 parallel)
    if need_wave2:
        trace.wave2_triggered = True
        conflict_lines: list[str] = []
        for rep, members in all_clusters:
            names = ", ".join(m.name for m in members)
            sample_snip = ""
            for m in members:
                if m.code:
                    sample_snip = m.code[:700]
                    break
            conflict_lines.append(
                f"Candidate Answer = {rep} (from: {names})\n"
                + (f"Excerpt of method:\n```python\n{sample_snip}\n```" if sample_snip else "")
            )
        graveyard_text = "\n".join(trace.graveyard) if trace.graveyard else "(none)"
        dossier = (
            f"PROBLEM:\n{problem.strip()}\n\n"
            f"WAVE-1 STATUS: {trace.wave2_reason}\n\n"
            f"COMPETING CANDIDATES:\n" + ("\n---\n".join(conflict_lines) if conflict_lines else "(none succeeded)")
            + f"\n\nGRAVEYARD (DO NOT REPEAT THESE BUGS):\n{graveyard_text}"
        )

        w2_roles = [
            ("falsifier_bruteforce", skills.card("falsifier_bruteforce"), 0.25, 505, 950),
            ("falsifier_reread", skills.card("falsifier_reread"), 0.45, 606, 950),
            ("falsifier_z3_sympy", skills.card("falsifier_z3_sympy"), 0.40, 707, 950),
        ]
        w2_specs = [
            CallSpec(
                messages=[
                    {"role": "system", "content": shared_sys},
                    {"role": "user", "content": f"{dossier}\n\nPROTOCOL ({name}):\n{card_text}"},
                ],
                tag=f"w2_{name}",
                max_tokens=tb + 2400,
                temperature=temp,
                seed=seed,
                enable_thinking=True,
                thinking_budget=tb,
                loop_guard=True,
            )
            for name, card_text, temp, seed, tb in w2_roles
        ]
        w2_results = await client.batch(w2_specs)
        pal_w2_jobs: list[BranchOutcome] = []
        for (name, _, _, _, _), res in zip(w2_roles, w2_results):
            out = BranchOutcome(
                name=name,
                wave=2,
                kind="pal",
                raw_content=res.content,
                reasoning=res.reasoning,
                latency_s=res.latency_s,
                error=res.raw_error or "",
            )
            if res.ok:
                pal_w2_jobs.append(out)
            trace.wave2_outcomes.append(out)
        if pal_w2_jobs:
            await asyncio.gather(*(asyncio.to_thread(_execute_pal_outcome, o, sandbox_timeout_s + 4.0) for o in pal_w2_jobs))
        for out in trace.wave2_outcomes:
            if out.error:
                trace.graveyard.append(f"[{out.name}] FAILED: {out.error[:240]}")

        # Conditional Wave-3 Hybrid Repair + High-Effort Analytical Consensus (B=3, ONLY when 0 valid PALs!)
        valid_pal_so_far = [
            o
            for o in (trace.wave1_outcomes + trace.wave2_outcomes)
            if o.kind == "pal"
            and o.exec_result
            and o.exec_result.ok
            and o.norm_answer
            and _answer_satisfies_prompt_format(problem, o.norm_answer)
        ]
        if not valid_pal_so_far:
            closest_pal = next(
                (o for o in reversed(trace.wave2_outcomes + trace.wave1_outcomes) if o.kind == "pal" and o.code),
                None,
            )
            w3_dossier = (
                f"PROBLEM:\n{problem.strip()}\n\n"
                "CRITICAL: Wave-1 and Wave-2 Python scripts failed or produced format-invalid output.\n"
                "GRAVEYARD OF PRIOR ATTEMPTS:\n" + "\n".join(trace.graveyard[-6:])
                + (
                    f"\n\nCLOSEST SCRIPT (`{closest_pal.name}`):\n```python\n{closest_pal.code[:2200]}\n```"
                    if closest_pal
                    else ""
                )
            )
            w3_roles = [
                (
                    "w3_pal_z3_exact",
                    "pal",
                    "Write a SHORT (<55 lines), rock-solid `z3` (`z3.Optimize()` with `z3.Real`/`z3.Int`) OR `sympy` "
                    "script directly modeling the equations/constraints (or clean coordinate/numerical verification snapped to exact answer). "
                    "Print `ANSWER: <exact_value>` on the last line. Output ONLY one ```python``` block.",
                    0.20,
                    808,
                    1000,
                ),
                (
                    "w3_analytical_deep_a",
                    "analytical",
                    "Solve this problem completely from first principles with rigorous step-by-step mathematical proof "
                    "(synthetic geometry, algebra, number theory, or combinatorics — do NOT write a broken script). "
                    "Verify every step, symmetry factor, and arithmetic calculation twice. End with `ANSWER: <exact_value>`.",
                    0.25,
                    909,
                    1800,
                ),
                (
                    "w3_analytical_deep_b",
                    "analytical",
                    "Solve this problem independently via an ALTERNATIVE mathematical approach (e.g., coordinate/trigonometric/algebraic "
                    "if the obvious route is synthetic, or vice versa). Double-check for off-by-2 symmetry or counting traps. "
                    "End with `ANSWER: <exact_value>`.",
                    0.55,
                    1010,
                    1800,
                ),
            ]
            w3_specs = [
                CallSpec(
                    messages=[
                        {"role": "system", "content": shared_sys if kind == "pal" else "You are an IMO/Putnam gold-medalist mathematician. Be rigorous and exact."},
                        {"role": "user", "content": f"{w3_dossier}\n\nPROTOCOL ({name}):\n{card_text}"},
                    ],
                    tag=name,
                    max_tokens=8192 if kind == "analytical" else tb + 2200,
                    temperature=temp,
                    seed=seed,
                    enable_thinking=True,
                    thinking_budget=tb,
                    loop_guard=True,
                )
                for name, kind, card_text, temp, seed, tb in w3_roles
            ]
            w3_results = await client.batch(w3_specs)
            pal_w3_jobs: list[BranchOutcome] = []
            for (name, kind, _, _, _, _), res in zip(w3_roles, w3_results):
                out = BranchOutcome(
                    name=name,
                    wave=3,
                    kind=kind,
                    raw_content=res.content,
                    reasoning=res.reasoning,
                    latency_s=res.latency_s,
                    error=res.raw_error or "",
                )
                if res.ok:
                    if kind == "pal":
                        pal_w3_jobs.append(out)
                    else:
                        raw_ans = extract_answer_from_prose(res.content)
                        out.raw_answer = raw_ans
                        out.norm_answer = normalize_answer(raw_ans)
                trace.wave2_outcomes.append(out)
            if pal_w3_jobs:
                await asyncio.gather(*(asyncio.to_thread(_execute_pal_outcome, o, sandbox_timeout_s + 6.0) for o in pal_w3_jobs))

    # Final Weighted Adjudication across all executed & verified branches
    combined = trace.wave1_outcomes + trace.wave2_outcomes
    valid_all = [o for o in combined if o.norm_answer and _answer_satisfies_prompt_format(problem, o.norm_answer)]
    if not valid_all:
        valid_all = [o for o in combined if o.norm_answer]
    clusters = _cluster_answers(valid_all)

    if not clusters:
        # Last-resort clean extraction from best raw output (works with AND without JSON grammar!)
        best_raw = max(combined, key=lambda o: len(o.raw_content), default=None)
        extract_prompt = (
            f"PROBLEM:\n{problem.strip()}\n\n"
            f"WORK:\n{(best_raw.raw_content if best_raw else '')[:3500]}\n\n"
            "Extract ONLY the exact final answer cleanly. End with `ANSWER: <value>`."
        )
        r_ext = await client.complete(
            CallSpec(
                messages=[{"role": "user", "content": extract_prompt}],
                tag="reason_fallback_extract",
                max_tokens=256,
                temperature=0.0,
                enable_thinking=False,
                json_schema=ANSWER_EXTRACT_SCHEMA,
                schema_name="answer_extract",
            )
        )
        obj = r_ext.json_payload() or {}
        raw_fb = str(obj.get("final_answer", "")).strip() or extract_answer_from_prose(r_ext.content)
        trace.final_answer = normalize_answer(raw_fb)
        trace.consensus_type = "fallback_extract"
        trace.final_explanation = best_raw.raw_content if best_raw else ""
        trace.wall_s = time.perf_counter() - t0
        return trace

    # Score every equivalence cluster
    scored_clusters: list[tuple[float, int, str, list[BranchOutcome]]] = []
    for rep, members in clusters:
        total_w = sum(_branch_weight(m) for m in members)
        pal_count = sum(1 for m in members if m.kind == "pal" and m.exec_result and m.exec_result.ok)
        # Bonus for Wave-2 falsifier agreement (explicitly resolved Wave-1 bug!)
        w2_pal = sum(1 for m in members if m.wave == 2 and m.exec_result and m.exec_result.ok)
        if w2_pal >= 2:
            total_w += 3.5
        scored_clusters.append((total_w, pal_count, rep, members))

    scored_clusters.sort(key=lambda x: (x[0], x[1]), reverse=True)

    winner_w, winner_pals, winner_ans, winner_members = scored_clusters[0]

    # Trigger order-swapped pairwise + deep audit when top clusters are close, OR when an
    # uncorroborated single PAL script (pal_count == 1) disagrees with an analytical/other cluster!
    close_margin = 2.25 if winner_pals <= 1 else 1.5
    if len(scored_clusters) >= 2 and (scored_clusters[0][0] - scored_clusters[1][0]) <= close_margin:
        ans_a, mem_a = scored_clusters[0][2], scored_clusters[0][3]
        ans_b, mem_b = scored_clusters[1][2], scored_clusters[1][3]
        code_a = next((m.code for m in mem_a if m.code), mem_a[0].raw_content)[:2500]
        code_b = next((m.code for m in mem_b if m.code), mem_b[0].raw_content)[:2500]
        p_ab = (
            f"PROBLEM:\n{problem.strip()}\n\n"
            f"CANDIDATE A (Answer: {ans_a}):\n{code_a}\n\n"
            f"CANDIDATE B (Answer: {ans_b}):\n{code_b}\n\n"
            "Which candidate faithfully models EVERY word and constraint of the problem with no bug? Reply A or B."
        )
        p_ba = (
            f"PROBLEM:\n{problem.strip()}\n\n"
            f"CANDIDATE A (Answer: {ans_b}):\n{code_b}\n\n"
            f"CANDIDATE B (Answer: {ans_a}):\n{code_a}\n\n"
            "Which candidate faithfully models EVERY word and constraint of the problem with no bug? Reply A or B."
        )
        vote = await client.pairwise_vote(p_ab, p_ba, tag_prefix="reason_judge")
        if vote == "TIE":
            # Thinking-enabled deep audit when fast 0-think pairwise judge ties!
            audit = await client.complete(
                CallSpec(
                    messages=[
                        {"role": "system", "content": "Compare Candidate A and Candidate B line by line against every single clue/constraint in the problem. End with WINNER: A or WINNER: B."},
                        {"role": "user", "content": p_ab},
                    ],
                    tag="reason_judge_audit",
                    max_tokens=1400,
                    temperature=0.0,
                    enable_thinking=True,
                    thinking_budget=700,
                    loop_guard=True,
                )
            )
            txt = audit.content.upper()
            if "WINNER: B" in txt:
                vote = "B"
            elif "WINNER: A" in txt:
                vote = "A"
        if vote == "B":
            winner_ans, winner_members = ans_b, mem_b
            trace.consensus_type = "pairwise_tiebreak_B"
        else:
            trace.consensus_type = "pairwise_tiebreak_A"
    else:
        if trace.wave2_triggered:
            trace.consensus_type = f"wave2_verified ({winner_pals} PAL)"
        else:
            trace.consensus_type = f"wave1_consensus ({winner_pals} PAL)"

    # Detect whether Wave 2 genuinely recovered from a Wave-1 failure or disagreement
    w1_first_pal = next((o.norm_answer for o in trace.wave1_outcomes if o.kind == "pal" and o.norm_answer), "")
    if trace.wave2_triggered and (not w1_first_pal or not _answers_equivalent(w1_first_pal, winner_ans)):
        trace.recovered_from_disagreement = True

    # Pick cleanest explanation / supporting script from winning cluster
    best_member = sorted(winner_members, key=lambda m: (m.wave, 1 if m.verified_constraints else 0), reverse=True)[0]
    stdout_preview = (best_member.exec_result.stdout.strip() if best_member.exec_result else "")[:800]
    trace.verified = any(m.kind == "pal" and m.exec_result and m.exec_result.ok for m in winner_members)
    trace.final_answer = winner_ans
    trace.final_explanation = (
        f"Verified Answer: **{winner_ans}**\n\n"
        f"Verification: {trace.consensus_type} "
        f"({len(winner_members)}/{len(valid_all)} agreeing branches)\n"
        + (f"\nVerification Output (`{best_member.name}`):\n```\n{stdout_preview}\n```" if stdout_preview else "")
    )
    trace.wall_s = time.perf_counter() - t0
    return trace

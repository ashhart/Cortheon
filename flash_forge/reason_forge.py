"""Neuro-Symbolic Parallel Reasoning Forge (Math, Probability, Combinatorics, Logic, CSP).

ReasonForge v4 — Pareto-dominant verification speed, cost, and Olympiad accuracy:
  Stage 1A (B=2 Fast Pair over shared cached prefix):
    - pal_exact:    Exact Python (Fraction / sympy / math / z3)
    - pal_skeptic:  Trap-hunter script asserting EVERY problem constraint individually
    -> Instant 2-call exit when both verified scripts agree identically.
  Stage 1B (Conditional B=2 Full Wave-1 Completion — fires only when Stage 1A does not agree):
    - pal_alt_z3:   Independent method (Z3 SMT / brute-force enumeration / DP)
    - analytical:   Pure closed-form derivation (catches huge-N formulas & synthetic geometry)
    -> Streaming early-abort cancels analytical if pal_alt_z3 confirms a single Stage-1A PAL.
  Stage 2 (Conditional Streaming Wave 2 — fires on disagreement, single uncorroborated PAL, or 0 PAL):
    - falsifier_bruteforce: Differential falsifier against competing Wave-1 candidates + graveyard
    - falsifier_reread:     Strictly UNANCHORED constraint-by-constraint clean-room verifier
    - falsifier_numeric_z3: Strictly UNANCHORED Z3/SymPy + 50-digit mpmath/numpy numerical
                            coordinate/optimization/root-finding/recurrence-pattern solver
    -> Streaming early-abort cancels lagging slot(s) on 2 Wave-2 PAL consensus (or unanchored single-PAL match).
  Stage 3 (Conditional Streaming Wave 3 — ONLY when 0 valid PALs across Wave 1 & Wave 2):
    - w3_pal_numeric_exact: Dedicated 50-digit mpmath/numpy solver (banned from slow symbolic solve)
    - w3_analytical_deep_a: Clean-room rigorous IMO/Putnam proof (zero graveyard pollution)
    - w3_analytical_deep_b: Clean-room independent alternative mathematical proof
    -> Streaming early-abort cancels the lagging 3rd slot as soon as ANY 2 agree.
  Stage 4 (Conditional 1-call Executable Verification Arbiter on close ties).
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Optional

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
    r"^(?:debug|check|assert|verif[a-z]*|done|ok|pass(?:ed)?|fail(?:ed)?|solution:?|result:?|note:?|warning:?)\b",
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


def _looks_like_concise_answer(val: str) -> bool:
    """Reject rambling multi-sentence prose accidentally matched after 'ANSWER:'."""
    s = val.strip()
    if not s or len(s) > 80 or "?" in s:
        return False
    if len(s.split()) > 6:
        return False
    return True


def extract_answer_from_prose(text: str) -> str:
    if not text:
        return ""
    # Prefer explicit ANSWER: line (handling markdown **ANSWER: X** / `ANSWER: X`)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        val = _match_answer_line(ln)
        if val and _looks_like_concise_answer(val):
            return val
    # Next prefer last \boxed{...} (cleaner than inline 'answer:' inside rambling scratchpad)
    boxed = _BOXED_RE.findall(text)
    if boxed:
        for b in reversed(boxed):
            if _looks_like_concise_answer(b):
                return b.strip()
    # Also scan inline **ANSWER: ...** anywhere near end of prose
    inline = re.findall(r"(?:FINAL[_\s]*)?ANSWER\s*[:=]\s*([^\n\*`]+)", text, re.IGNORECASE)
    for cand in reversed(inline):
        cleaned = _strip_md_wrappers(cand)
        if _looks_like_concise_answer(cleaned):
            return cleaned
    return ""


_SQRT_N_RE = re.compile(r"\\sqrt\[([^{}\[\]]+)\]\{([^{}]+)\}")
_SQRT_RE = re.compile(r"\\sqrt\{([^{}]+)\}")


def _clean_latex(s: str) -> str:
    s = s.strip()
    s = s.strip("$").strip()
    s = s.rstrip(".")
    # Unwrap outer \boxed{...} if present
    for _ in range(2):
        m_box = _BOXED_RE.fullmatch(s)
        if not m_box:
            break
        s = m_box.group(1).strip()
    # Replace \frac{a}{b} -> (a)/(b)
    for _ in range(4):
        new_s = _FRAC_RE.sub(r"(\1)/(\2)", s)
        if new_s == s:
            break
        s = new_s
    # Replace \sqrt[n]{x} and \sqrt{x}
    for _ in range(4):
        new_s = _SQRT_N_RE.sub(r"((\2)**(1/(\1)))", s)
        new_s = _SQRT_RE.sub(r"sqrt(\1)", new_s)
        if new_s == s:
            break
        s = new_s
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\pi", "pi")
    s = s.replace("\\,", "").replace("\\ ", "").replace("%", "")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    # Handle implicit multiplication like 2\sqrt{2} -> 2*sqrt(2), 3\pi -> 3*pi
    s = re.sub(r"(\d)\s*(sqrt\(|pi\b)", r"\1*\2", s)
    s = s.strip()
    return s


def normalize_answer(raw: str) -> str:
    """Canonicalize exact math/logic answers so equivalent forms cluster together."""
    if not raw:
        return ""
    s = _clean_latex(_strip_md_wrappers(raw))
    # Strip trailing period/backticks/quotes
    s = s.strip("`\"' ").rstrip(".")
    if not s:
        return ""

    # Handle Python Fraction(a, b)
    m_frac = re.fullmatch(r"Fraction\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", s)
    if m_frac:
        f = Fraction(int(m_frac.group(1)), int(m_frac.group(2)))
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"

    # Handle parenthesized exact rational (a)/(b) from LaTeX \frac{a}{b} (even for 100+ digit rationals)
    m_pfrac = re.fullmatch(r"\(\s*(-?\d+)\s*\)\s*/\s*\(\s*(-?\d+)\s*\)", s)
    if m_pfrac:
        f = Fraction(int(m_pfrac.group(1)), int(m_pfrac.group(2)))
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
            if lit == 0.0:
                return "0"
            if abs(lit) >= 1e-9 and abs(lit - round(lit)) < 1e-12 * max(1.0, abs(lit)):
                return str(int(round(lit)))
            # Check exact IEEE-754 float-printed fraction match (e.g. 0.5833333333333334 -> 7/12)
            # Tight 1e-12 guard prevents truncated decimals like 0.3333333333 from falsely snapping to 1/3
            approx_f = Fraction(lit).limit_denominator(10000)
            if approx_f != 0 and abs(float(approx_f) - lit) < 1e-12 * max(1.0, abs(lit)):
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
                if val == 0.0:
                    return "0"
                if abs(val) >= 1e-9 and abs(val - round(val)) < 1e-12 * max(1.0, abs(val)):
                    return str(int(round(val)))
                return f"{val:.10g}"
            return str(simp)
        except Exception:
            pass

    # Plain normalized string (e.g. "Alice", "B", "Monday")
    return re.sub(r"\s+", " ", s).strip()


_EXACT_RAT_RE = re.compile(r"^-?\d+(?:\s*/\s*-?\d+)?$")


def _prep_equiv_expr(s: str) -> str:
    t = _clean_latex(s) if ("\\" in s or "$" in s) else s.strip()
    m_pfrac = re.fullmatch(r"\(\s*(-?\d+)\s*\)\s*/\s*\(\s*(-?\d+)\s*\)", t)
    if m_pfrac:
        return f"{int(m_pfrac.group(1))}/{int(m_pfrac.group(2))}"
    if re.fullmatch(r"-?\d+", t):
        return str(int(t))
    return t.replace("^", "**")


def _answers_equivalent(a: str, b: str) -> bool:
    if not a or not b:
        return False
    pa = _prep_equiv_expr(a)
    pb = _prep_equiv_expr(b)
    if not pa or not pb:
        return False
    if pa.lower() == pb.lower():
        return True
    # Arbitrary-precision exact rational / integer check: NEVER degrade to float64!
    if _EXACT_RAT_RE.match(pa) and _EXACT_RAT_RE.match(pb):
        try:
            return Fraction(pa.replace(" ", "")) == Fraction(pb.replace(" ", ""))
        except ZeroDivisionError:
            return False
    try:
        ea = sp.sympify(pa)
        eb = sp.sympify(pb)
        diff = sp.simplify(ea - eb)
        if diff == 0:
            return True
        # Two exact integers/rationals must agree identically (never via lossy float fallback)
        if ea.is_Rational and eb.is_Rational:
            return False
        if ea.is_number and eb.is_number:
            fa, fb = complex(ea.evalf()), complex(eb.evalf())
            scale = max(abs(fa), abs(fb))
            if scale > 0.0 and abs(fa - fb) < 1e-12 * scale:
                return True
    except Exception:
        pass
    return False


def _cluster_answers(outcomes: list[BranchOutcome]) -> list[tuple[str, list[BranchOutcome]]]:
    """Group outcomes by mathematical/string equivalence, sorted by majority & branch weight."""
    clusters: list[tuple[str, list[BranchOutcome]]] = []
    for o in outcomes:
        if not o.norm_answer:
            continue
        placed = False
        for i, (rep, members) in enumerate(clusters):
            if _answers_equivalent(rep, o.norm_answer):
                members.append(o)
                # Prefer cleaner rational/integer representation over decimal
                if ("/" in o.norm_answer or re.fullmatch(r"-?\d+", o.norm_answer)) and "." in rep:
                    clusters[i] = (o.norm_answer, members)
                placed = True
                break
        if not placed:
            clusters.append((o.norm_answer, [o]))
    clusters.sort(
        key=lambda item: (len(item[1]), sum(_branch_weight(m) for m in item[1])),
        reverse=True,
    )
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
    r"(?:"
    r"ANSWER\s*[:=]\s*<\s*int(?:eger)?\s*>"
    r"|exact\s+integer"
    r"|is\s+an\s+integer\b"
    r"|integer\s+from\s+\d+\s+to\s+\d+"
    r"|remainder\s+when\b.{0,120}\bdivided\s+by\s+\d+"
    r"|greatest\s+(?:possible\s+)?(?:positive\s+)?integer"
    r"|least\s+(?:possible\s+)?(?:positive\s+)?integer"
    r"|smallest\s+(?:possible\s+)?(?:positive\s+)?integer"
    r"|largest\s+(?:possible\s+)?(?:positive\s+)?integer"
    r"|sum\s+of\s+all\s+(?:possible\s+)?(?:positive\s+)?integers"
    r"|relatively\s+prime\s+positive\s+integers.{0,120}\b(?:find|compute|determine)\s+[a-z]\s*\+\s*[a-z]\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_COUNT_INT_RE = re.compile(r"(?:\bfind\s+the\s+number\s+of\b|\bhow\s+many\b)", re.IGNORECASE)
_NON_INT_PROMPT_RE = re.compile(
    r"(?:\bprobability\b|\bexpected\b|\baverage\b|\bfraction\b|\brational\b|\bdecimal\b|\bratio\b|\bpercentage\b)",
    re.IGNORECASE,
)

_MULTI_VALUE_PROMPT_RE = re.compile(
    r"(?:ordered\s+pair|tuple|list\s+of|comma[- ]separated|all\s+values\s+of|coordinates\s+of|two\s+numbers|space[- ]separated)",
    re.IGNORECASE,
)

_MULTI_NUM_GARBAGE_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:\s+-?\d+(?:\.\d+)?)+$")


_INT_RANGE_RE = re.compile(
    r"integer\s*(?:\(\s*(-?\d+)\s*(?:to|\.\.|-)\s*(-?\d+)\s*\)|from\s+(-?\d+)\s+to\s+(-?\d+)|between\s+(-?\d+)\s+and\s+(-?\d+))",
    re.IGNORECASE,
)
_REMAINDER_MOD_RE = re.compile(
    r"remainder\s+when\b.{0,120}\bdivided\s+by\s+(\d+)",
    re.IGNORECASE | re.DOTALL,
)
_SANDBOX_PATH_RE = re.compile(r"(?:/Users/[^\s'\"]+|/private/var/[^\s'\"]+|/tmp/[^\s'\"]+|/home/[^\s'\"]+)")


def _sandbox_package_guidance() -> str:
    """Detect installed scientific Python packages once so PAL scripts never crash on missing imports."""
    checked = ("sympy", "z3", "numpy", "mpmath", "scipy", "shapely", "cvxpy", "networkx")
    avail: list[str] = []
    missing: list[str] = []
    for pkg in checked:
        if importlib.util.find_spec(pkg) is not None:
            avail.append(pkg)
        else:
            missing.append(pkg)
    msg = f"Python 3 stdlib (fractions, itertools, math, decimal, random)"
    if avail:
        msg += ", " + ", ".join(avail)
    if missing:
        msg += f". NOT installed (do NOT import): {', '.join(missing)}"
    return msg


_SANDBOX_PKG_HINT = _sandbox_package_guidance()


def _sanitize_sandbox_diag(text: str, limit: int = 700) -> str:
    """Strip local host paths and truncate sandbox stdout/stderr cleanly for TIR repair."""
    if not text:
        return ""
    cleaned = _SANDBOX_PATH_RE.sub("<sandbox>", text.strip())
    if len(cleaned) > limit:
        return "...(truncated)...\n" + cleaned[-limit:]
    return cleaned


def _extract_int_bounds(problem: str) -> Optional[tuple[int, int]]:
    """Extract explicit inclusive integer bounds [lo, hi] stated in the prompt, if any."""
    lo: Optional[int] = None
    hi: Optional[int] = None
    for m in _INT_RANGE_RE.finditer(problem):
        vals = [g for g in m.groups() if g is not None]
        if len(vals) == 2:
            a, b = int(vals[0]), int(vals[1])
            if a <= b:
                lo = a if lo is None else max(lo, a)
                hi = b if hi is None else min(hi, b)
    for m in _REMAINDER_MOD_RE.finditer(problem):
        mod = int(m.group(1))
        if mod >= 2:
            lo = 0 if lo is None else max(lo, 0)
            hi = (mod - 1) if hi is None else min(hi, mod - 1)
    if lo is not None and hi is not None and lo <= hi:
        return (lo, hi)
    return None


def _format_rejection_reason(problem: str, norm_answer: str) -> str:
    """Return an empty string if norm_answer satisfies prompt format/range, else a concise reason."""
    if not norm_answer:
        return "empty answer"
    cleaned = norm_answer.strip()
    if cleaned.lower() in {"nan", "inf", "-inf", "+inf", "none", "null"}:
        return f"invalid non-finite/null value {cleaned!r}"
    if not _looks_like_concise_answer(cleaned) and not re.fullmatch(r"-?\d+(?:/\d+)?", cleaned):
        return "output looks like rambling prose instead of a concise exact value"
    # Reject accidental multi-number stdout dumps like "7 0.44239822567..." unless prompt expects multiple values
    if _MULTI_NUM_GARBAGE_RE.fullmatch(cleaned) and not _MULTI_VALUE_PROMPT_RE.search(problem):
        return f"multi-number stdout dump {cleaned!r}; print a single exact value"
    expects_int = bool(
        _INT_EXPECTED_RE.search(problem)
        or (_COUNT_INT_RE.search(problem) and not _NON_INT_PROMPT_RE.search(problem))
    )
    if expects_int:
        if not re.fullmatch(r"-?\d+", cleaned):
            return f"expected an exact integer, got {cleaned!r}"
        bounds = _extract_int_bounds(problem)
        if bounds is not None:
            v = int(cleaned)
            if v < bounds[0] or v > bounds[1]:
                return f"integer {v} is outside the required range [{bounds[0]}, {bounds[1]}]"
    return ""


def _answer_satisfies_prompt_format(problem: str, norm_answer: str) -> bool:
    """Reject blatantly format-invalid or out-of-range answers (never rejecting exact 25-digit rationals)."""
    return _format_rejection_reason(problem, norm_answer) == ""


def _non_anchoring_error_hint(err: str) -> str:
    """Extract a compact error class/hint with ALL multi-digit numbers stripped so follow-up slots never anchor."""
    if not err:
        return ""
    s = err.strip()
    if s.startswith("TIMEOUT"):
        return "TIMEOUT (avoid slow brute-force loops or symbolic sympy.solve; use exact DP, Z3, or 50-digit mpmath/numpy)"
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    tail = lines[-1] if lines else s
    if tail.startswith("AssertionError"):
        return "AssertionError (constraint check failed)"
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b", tail)
    if m:
        return m.group(1)
    if "no ANSWER line" in s:
        return "Missing `ANSWER: <value>` stdout line"
    if "No valid Python" in s:
        return "Missing fenced ```python``` code block"
    return "Runtime/format failure"


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


def _emit(
    on_event: Optional[Callable[[dict[str, Any]], None]],
    payload: dict[str, Any],
) -> None:
    """Invoke an optional live event callback, swallowing any UI/pipe failure."""
    if on_event is None:
        return
    try:
        on_event(payload)
    except Exception:
        pass


def _slot_done_payload(
    *,
    engine: str,
    slot: int,
    out: BranchOutcome,
    problem: str,
    elapsed_s: float,
    tokens: int,
    wave: int = 1,
) -> dict[str, Any]:
    fmt_ok = bool(out.norm_answer and _answer_satisfies_prompt_format(problem, out.norm_answer))
    if out.kind == "pal":
        ok_flag = bool(out.exec_result and out.exec_result.ok and not out.error and fmt_ok)
        if ok_flag:
            summary = "PAL verified"
        elif out.error:
            summary = out.error.splitlines()[0][:80]
        elif out.norm_answer and not fmt_ok:
            summary = "format gate rejected"
        else:
            summary = "PAL failed"
    else:
        ok_flag = bool(not out.error and fmt_ok)
        if ok_flag:
            summary = "analytical"
        elif out.error:
            summary = out.error.splitlines()[0][:80]
        elif out.norm_answer and not fmt_ok:
            summary = "format gate rejected"
        else:
            summary = "no answer extracted"
    return {
        "engine": engine,
        "event": "slot_done",
        "wave": wave,
        "slot": slot,
        "role": out.name,
        "ok": ok_flag,
        "elapsed_s": round(max(0.0, float(elapsed_s)), 2),
        "tokens": int(tokens),
        "answer": out.norm_answer or "",
        "summary": summary,
    }


_W2_UNANCHORED_REREAD_CARD = (
    "Solve this problem completely fresh from first principles by re-reading EVERY sentence carefully.\n"
    "Rules:\n"
    "- Quote each constraint as a `# C1: ...`, `# C2: ...` comment in Python so you cannot miss a subtle word "
    "('at least' vs 'exactly', 'next to two other', inclusive vs exclusive, ordered vs unordered, 2x symmetry).\n"
    "- Write a clean, exact Python verification script (`int`, `fractions.Fraction`, `sympy`, `math`, `itertools`, or `z3`) "
    "that checks EVERY quoted constraint `# C1..Ck` with explicit `assert` statements.\n"
    "- Keep code compact (<85 lines). Print `VERIFIED`, then print `ANSWER: <exact_value>` on the LAST line of stdout.\n"
    "- Output ONLY one complete fenced ```python``` block."
)

_W2_NUMERIC_Z3_CARD = (
    "Discard prior derivations and solve this problem cleanly from scratch using the fastest reliable tool "
    "in our sandbox (`z3`, `sympy`, `fractions.Fraction`, `numpy`, `mpmath` — NOTE: `scipy` is NOT installed!).\n"
    "Rules:\n"
    "- Discrete / logic / LP / minimax / scheduling / exact combinatorics: use `z3` (`z3.Int`/`z3.Real`/`z3.Optimize()`) "
    "or exact DP / `Fraction` / `sympy.Matrix`.\n"
    "- Continuous geometry, coordinate placement, extrema/optimization, nonlinear/radical equations (`m + n*sqrt(p)`), "
    "or polynomial roots: DO NOT hang in symbolic `sympy.solve()`! Solve numerically to 40-50 digits with `mpmath` "
    "(`mpmath.mp.dps = 50`, `mpmath.findroot`, `mpmath.polyroots`, or pure-Python Weiszfeld / Nelder-Mead / grid-refine), "
    "verify residuals < 1e-25, snap to the exact integer / `Fraction` / `m + n*sqrt(p)` (via `mpmath.identify` or small-integer "
    "search checking residual < 1e-20), and verify all equations.\n"
    "- Geometric probability / expected values: compute exact pairwise expectation AND cross-check with a fast seeded "
    "`random.Random(42)` simulation (30000-50000 trials) to catch 2x symmetry/ordering mistakes.\n"
    "- Huge-index recurrences (e.g. n=2025): compute small exact `Fraction` terms k=1..10 first, discover & assert the "
    "exact closed-form / modular recurrence, then evaluate at large n.\n"
    "- Print `VERIFIED`, then print `ANSWER: <exact_value>` on the LAST line of stdout. Output ONLY one fenced ```python``` block."
)

_W3_NUMERIC_EXACT_CARD = (
    "Prior symbolic or brute-force Python scripts failed or timed out. Solve this problem in a SHORT (<75 lines), "
    "ultra-fast numerical-verification Python script (`mpmath` with `mp.mp.dps = 50`, `numpy`, `fractions.Fraction`, `random`).\n"
    "CRITICAL RULES:\n"
    "- NEVER call slow symbolic `sympy.solve()` on nonlinear/geometry/radical systems and NEVER write huge brute-force loops.\n"
    "- NEVER `import scipy` (not installed — write 10 lines of pure Python / `mpmath` / `numpy` instead).\n"
    "- For geometry / optimization / roots: place coordinates numerically to 40-50 digits in `mpmath`, verify EVERY length/angle/equation "
    "residual < 1e-25, then snap the target quantity to exact integer / `Fraction` / `m + n*sqrt(p)` (residual < 1e-20).\n"
    "- For expected values / probability: combine exact symmetry/linearity of expectation with a fast `random.Random(42)` cross-check.\n"
    "- For huge-index recurrences: compute k=1..12 in exact `Fraction` to uncover & assert the closed form / period mod M.\n"
    "- Print `VERIFIED`, then print `ANSWER: <exact_value>` on the LAST line of stdout. Output ONLY one fenced ```python``` block."
)

_W3_ANALYTICAL_DEEP_A_CARD = (
    "Solve this problem completely from first principles with a rigorous step-by-step mathematical proof "
    "(synthetic geometry, algebra, number theory, or combinatorics — do NOT write a script).\n"
    "- Check every constraint, boundary condition, symmetry factor, and arithmetic calculation twice.\n"
    "- End with `ANSWER: <exact_value>`."
)

_W3_ANALYTICAL_DEEP_B_CARD = (
    "Solve this problem independently via an ALTERNATIVE mathematical approach (e.g. coordinate/trigonometric/algebraic "
    "if the obvious route is synthetic, or generating functions/bijection/linearity of expectation if obvious route is casework).\n"
    "- Explicitly check for off-by-2 symmetry, ordered-vs-unordered double-counting, and arithmetic slips.\n"
    "- End with `ANSWER: <exact_value>`."
)


async def solve_reasoning(
    client: TensorFoldClient,
    skills: SkillPack,
    memory: EpisodicMemory,
    problem: str,
    *,
    num_wave1_streams: int = 4,
    sandbox_timeout_s: float = 20.0,
    deep: bool = False,
    enable_wave2: bool = True,
    max_pal_repairs: int = 1,
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
) -> ReasonForgeTrace:
    """Solve a hard math, logic, combinatorics, probability, or constraint problem.

    ReasonForge v5 combines:
      1. Fast Pair Stage 1A (`pal_exact` + `pal_skeptic`) with inline zero-anchoring Executable TIR
         Self-Repair (1 surgical REPL debug turn when a script crashes or violates prompt range).
      2. Full Stage 1B Wave-1 completion (`pal_alt_z3` + `analytical`) whenever Stage 1A does not agree.
      3. Streaming Wave-2 & Wave-3 early-abort with anti-anchoring + 50-digit `mpmath`/`numpy`
         numerical coordinate/optimization verification + multi-turn REPL repair on 0-PAL problems.
      4. 1-call Executable Verification Arbiter (with TIR self-repair) on close ties.
    """
    t0 = time.perf_counter()
    trace = ReasonForgeTrace()

    lessons_block = memory.format_prompt_block(problem, domain="reason", limit=3)
    if lessons_block:
        trace.lessons_used = lessons_block.count("\n- [")

    # Shared cached prefix across PAL streams so prompt cache hits cleanly!
    shared_sys = (
        "You are a precision reasoning engine running inside a verified neuro-symbolic sandbox.\n"
        f"Environment: {_SANDBOX_PKG_HINT}.\n"
        "Never guess arithmetic, combinatorics, or constraint satisfaction in prose when Python can compute and verify it exactly."
    )
    if lessons_block:
        shared_sys += "\n\n" + lessons_block

    math_sys = "You are an IMO/Putnam gold-medalist mathematician. Be rigorous, concise, and exact."

    exemplar_block = skills.match_exemplars("pal_verify", problem, limit=1)
    shared_user_prefix = f"PROBLEM:\n{problem.strip()}"
    if exemplar_block:
        shared_user_prefix += f"\n\n{exemplar_block}"

    # Keep battle-tested v2 cards 100% unmodified in Wave 1!
    think_pal = 1200 if deep else 700
    think_ana = 2000 if deep else 1300
    pal_effort = None

    wave1_roles = [
        ("pal_exact", "pal", skills.card("pal_exact"), 0.25, 101, think_pal),
        ("pal_skeptic", "pal", skills.card("pal_skeptic"), 0.45, 303, think_pal),
        ("pal_alt_z3", "pal", skills.card("pal_alt_z3"), 0.60, 202, think_pal),
        ("analytical", "analytical", skills.card("reason_analytical"), 0.40, 404, think_ana),
    ][: max(1, num_wave1_streams)]

    def _make_w1_spec(
        name: str, kind: str, card_text: str, temp: float, seed: int, t_budget: int, extra_note: str = ""
    ) -> CallSpec:
        user_content = f"{shared_user_prefix}\n\nPROTOCOL ({name}):\n{card_text}"
        if extra_note:
            user_content += f"\n\n{extra_note}"
        return CallSpec(
            messages=[
                {"role": "system", "content": shared_sys},
                {"role": "user", "content": user_content},
            ],
            tag=f"w1_{name}",
            max_tokens=t_budget + 2000,
            temperature=temp,
            seed=seed,
            enable_thinking=True,
            thinking_budget=t_budget,
            reasoning_effort=pal_effort,
            loop_guard=True,
        )

    def _is_valid_pal(o: Optional[BranchOutcome]) -> bool:
        return bool(
            o is not None
            and o.kind == "pal"
            and o.exec_result
            and o.exec_result.ok
            and not o.error
            and o.norm_answer
            and _answer_satisfies_prompt_format(problem, o.norm_answer)
        )

    def _is_valid_any(o: Optional[BranchOutcome]) -> bool:
        if o is None or not o.norm_answer or not _answer_satisfies_prompt_format(problem, o.norm_answer):
            return False
        if o.kind == "pal":
            return _is_valid_pal(o)
        return not o.error

    async def _repair_pal_outcome_inplace(
        out: BranchOutcome,
        *,
        base_tag: str,
        exec_timeout_s: float,
        max_repairs: int,
        seed: int,
    ) -> int:
        """Interactive zero-anchoring TIR REPL debug loop: feeds ONLY this script's own code +
        sanitized stdout/stderr/format-rejection back to fix 1-line Python bugs, missing packages,
        timeouts, or out-of-range answers without discarding sound mathematical models."""
        extra_tok = 0
        for attempt in range(1, max(0, max_repairs) + 1):
            if _is_valid_pal(out):
                break
            # Build concise, path-scrubbed sandbox diagnostic (never leaks another slot's answer!)
            diag_parts: list[str] = []
            if out.error:
                diag_parts.append(f"ERROR:\n{_sanitize_sandbox_diag(out.error, 700)}")
            if out.exec_result and out.exec_result.stdout.strip():
                diag_parts.append(f"STDOUT:\n{_sanitize_sandbox_diag(out.exec_result.stdout, 700)}")
            if out.norm_answer:
                rej = _format_rejection_reason(problem, out.norm_answer)
                if rej:
                    diag_parts.append(f"FORMAT/RANGE CHECK FAILED: {rej}")
            diag_block = "\n\n".join(diag_parts) if diag_parts else "Execution failed with no output."

            if out.code:
                fix_user = (
                    f"PROBLEM:\n{problem.strip()}\n\n"
                    f"YOUR PREVIOUS PYTHON SCRIPT:\n```python\n{out.code[:2800]}\n```\n\n"
                    f"SANDBOX FEEDBACK:\n{diag_block}\n\n"
                    "TASK: Fix the exact bug, missing import, timeout, or format/range error cleanly.\n"
                    f"- Environment: {_SANDBOX_PKG_HINT}.\n"
                    "- If the script timed out in slow symbolic `sympy.solve()` or huge loops, switch to fast "
                    "50-digit `mpmath`/`numpy` numerical solving (snap to exact integer/radical and verify residuals < 1e-20) or exact DP.\n"
                    "- If STDOUT already printed intermediate numbers, use them to fix the final step cleanly.\n"
                    "- Print `ANSWER: <exact_value>` on the LAST line of stdout. Output ONLY one complete fenced ```python``` block."
                )
            else:
                fix_user = (
                    f"PROBLEM:\n{problem.strip()}\n\n"
                    "SANDBOX FEEDBACK: No fenced ```python``` code block was extracted from your previous reply.\n"
                    f"TASK: Write a concise (<70 lines) Python script ({_SANDBOX_PKG_HINT}) computing and verifying "
                    "the exact answer. Print `ANSWER: <exact_value>` on the LAST line of stdout. Output ONLY one fenced ```python``` block."
                )

            fix_tb = 450 if attempt == 1 else 650
            fix_res = await client.complete(
                CallSpec(
                    messages=[
                        {"role": "system", "content": shared_sys},
                        {"role": "user", "content": fix_user},
                    ],
                    tag=f"{base_tag}_fix{attempt}",
                    max_tokens=fix_tb + 2000,
                    temperature=0.15 if attempt == 1 else 0.30,
                    seed=seed + 1000 * attempt,
                    enable_thinking=True,
                    thinking_budget=fix_tb,
                    reasoning_effort=pal_effort,
                    loop_guard=True,
                )
            )
            extra_tok += int(getattr(fix_res, "completion_tokens", 0) or 0)
            out.latency_s += float(fix_res.latency_s or 0.0)
            if not fix_res.ok:
                break
            repaired = BranchOutcome(
                name=out.name,
                wave=out.wave,
                kind="pal",
                raw_content=fix_res.content,
                reasoning=fix_res.reasoning,
                latency_s=out.latency_s,
                error=fix_res.raw_error or "",
            )
            await asyncio.to_thread(_execute_pal_outcome, repaired, exec_timeout_s)
            # Adopt repaired attempt if it became valid OR produced fresher code/diagnostics for a 2nd turn
            if _is_valid_pal(repaired) or repaired.code:
                out.raw_content = repaired.raw_content
                out.reasoning = repaired.reasoning
                out.code = repaired.code
                out.exec_result = repaired.exec_result
                out.raw_answer = repaired.raw_answer
                out.norm_answer = repaired.norm_answer
                out.verified_constraints = repaired.verified_constraints
                out.error = repaired.error
        return extra_tok

    async def _run_single_slot(
        idx: int,
        name: str,
        kind: str,
        wave: int,
        spec: CallSpec,
        exec_timeout_s: float,
        *,
        repairs: Optional[int] = None,
    ) -> tuple[int, BranchOutcome, int, float]:
        slot_t0 = time.perf_counter()
        res = await client.complete(spec)
        tok = int(getattr(res, "completion_tokens", 0) or 0)
        out = BranchOutcome(
            name=name,
            wave=wave,
            kind=kind,
            raw_content=res.content,
            reasoning=res.reasoning,
            latency_s=res.latency_s,
            error=res.raw_error or "",
        )
        if res.ok:
            if kind == "pal":
                await asyncio.to_thread(_execute_pal_outcome, out, exec_timeout_s)
                n_rep = max_pal_repairs if repairs is None else repairs
                if n_rep > 0 and not _is_valid_pal(out):
                    tok += await _repair_pal_outcome_inplace(
                        out,
                        base_tag=spec.tag,
                        exec_timeout_s=exec_timeout_s,
                        max_repairs=n_rep,
                        seed=int(spec.seed or 0) + idx + 1,
                    )
            else:
                raw_ans = extract_answer_from_prose(res.content)
                out.raw_answer = raw_ans
                out.norm_answer = normalize_answer(raw_ans)
        slot_elapsed = max(time.perf_counter() - slot_t0, float(out.latency_s or 0.0))
        return idx, out, tok, slot_elapsed

    def _is_suspect_empty_zero(ans: str) -> bool:
        """Guard against `sum([]) == 0` empty-search false consensus on non-remainder prompts."""
        return ans == "0" and not bool(_REMAINDER_MOD_RE.search(problem))

    _emit(
        on_event,
        {
            "engine": "reason",
            "event": "start",
            "mode": "reason",
            "streams": len(wave1_roles),
        },
    )

    ordered_outcomes: list[Optional[BranchOutcome]] = [None] * len(wave1_roles)
    has_complete = callable(getattr(client, "complete", None))

    if has_complete:
        # Stage 1A: Fast Pair (pal_exact + pal_skeptic) -> instant 2-call exit on verified agreement!
        pair_count = min(2, len(wave1_roles))
        pair_tasks = [
            asyncio.create_task(
                _run_single_slot(
                    i,
                    wave1_roles[i][0],
                    wave1_roles[i][1],
                    1,
                    _make_w1_spec(*wave1_roles[i]),
                    sandbox_timeout_s,
                )
            )
            for i in range(pair_count)
        ]
        pending_1a: set[asyncio.Task[tuple[int, BranchOutcome, int, float]]] = set(pair_tasks)
        try:
            while pending_1a:
                done_set, pending_1a = await asyncio.wait(pending_1a, return_when=asyncio.FIRST_COMPLETED)
                for idx, out, tok, slot_elapsed in sorted((t.result() for t in done_set), key=lambda item: item[0]):
                    ordered_outcomes[idx] = out
                    _emit(
                        on_event,
                        _slot_done_payload(
                            engine="reason",
                            slot=idx + 1,
                            out=out,
                            problem=problem,
                            elapsed_s=slot_elapsed,
                            tokens=tok,
                            wave=1,
                        ),
                    )
        except BaseException:
            for t in pending_1a:
                t.cancel()
            if pending_1a:
                await asyncio.gather(*pending_1a, return_exceptions=True)
            raise

        valid_1a = [o for o in ordered_outcomes[:pair_count] if _is_valid_pal(o)]
        clusters_1a = _cluster_answers(valid_1a)

        if len(valid_1a) >= 2 and len(clusters_1a) == 1 and not _is_suspect_empty_zero(clusters_1a[0][0]):
            skipped = len(wave1_roles) - pair_count
            if skipped > 0:
                _emit(
                    on_event,
                    {
                        "engine": "reason",
                        "event": "early_abort",
                        "cancelled": skipped,
                        "elapsed_s": round(time.perf_counter() - t0, 2),
                        "consensus_answer": clusters_1a[0][0],
                        "reason": f"2/{len(wave1_roles)} PAL consensus",
                    },
                )
        elif len(wave1_roles) > pair_count:
            # Stage 1B: Complete v2's full Wave-1 quartet (pal_alt_z3 + analytical in parallel!)
            # Pass ONLY non-anchoring sandbox crash classes to pal_alt_z3 when Stage 1A had a crash.
            failed_1a = [o for o in ordered_outcomes[:pair_count] if o is not None and not _is_valid_pal(o)]
            pitfalls = [_non_anchoring_error_hint(o.error) for o in failed_1a if o.error]
            pitfalls = [p for p in dict.fromkeys(pitfalls) if p]
            pal_note = (
                f"[Sandbox pitfall to avoid: {'; '.join(pitfalls[:2])}]"
                if pitfalls
                else ""
            )
            rem_tasks = [
                asyncio.create_task(
                    _run_single_slot(
                        i,
                        wave1_roles[i][0],
                        wave1_roles[i][1],
                        1,
                        _make_w1_spec(
                            *wave1_roles[i],
                            extra_note=(pal_note if wave1_roles[i][1] == "pal" else ""),
                        ),
                        sandbox_timeout_s,
                    )
                )
                for i in range(pair_count, len(wave1_roles))
            ]
            pending_1b: set[asyncio.Task[tuple[int, BranchOutcome, int, float]]] = set(rem_tasks)
            pal_so_far = list(valid_1a)
            try:
                while pending_1b:
                    done_set, pending_1b = await asyncio.wait(pending_1b, return_when=asyncio.FIRST_COMPLETED)
                    for idx, out, tok, slot_elapsed in sorted((t.result() for t in done_set), key=lambda x: x[0]):
                        ordered_outcomes[idx] = out
                        _emit(
                            on_event,
                            _slot_done_payload(
                                engine="reason",
                                slot=idx + 1,
                                out=out,
                                problem=problem,
                                elapsed_s=slot_elapsed,
                                tokens=tok,
                                wave=1,
                            ),
                        )
                        if _is_valid_pal(out):
                            pal_so_far.append(out)

                    # Streaming early-abort in Stage 1B: ONLY when we now have 2 agreeing valid PALs,
                    # ZERO competing PALs, and NO analytical contradiction!
                    if pending_1b and pal_so_far:
                        cl_now = _cluster_answers(pal_so_far)
                        if (
                            len(cl_now) == 1
                            and len(cl_now[0][1]) >= 2
                            and not _is_suspect_empty_zero(cl_now[0][0])
                        ):
                            top_ans = cl_now[0][0]
                            ana_contradicts = any(
                                o is not None
                                and o.kind == "analytical"
                                and _is_valid_any(o)
                                and not _answers_equivalent(o.norm_answer, top_ans)
                                for o in ordered_outcomes
                            )
                            if not ana_contradicts or len(cl_now[0][1]) >= 3:
                                n_cancel = len(pending_1b)
                                for lag in pending_1b:
                                    lag.cancel()
                                _emit(
                                    on_event,
                                    {
                                        "engine": "reason",
                                        "event": "early_abort",
                                        "cancelled": n_cancel,
                                        "elapsed_s": round(time.perf_counter() - t0, 2),
                                        "consensus_answer": top_ans,
                                        "reason": f"{len(cl_now[0][1])}/{len(wave1_roles)} PAL consensus",
                                    },
                                )
                                await asyncio.gather(*pending_1b, return_exceptions=True)
                                pending_1b.clear()
                                break
            except BaseException:
                for lag in pending_1b:
                    lag.cancel()
                if pending_1b:
                    await asyncio.gather(*pending_1b, return_exceptions=True)
                raise
    else:
        # Batch fallback path for custom clients implementing only `.batch()`
        specs = [_make_w1_spec(*r) for r in wave1_roles]
        w1_t0 = time.perf_counter()
        w1_results = await client.batch(specs) if specs else []
        pal_w1_jobs: list[BranchOutcome] = []
        w1_tokens: list[int] = []
        for idx, ((name, kind, _, _, _, _), res) in enumerate(zip(wave1_roles, w1_results)):
            tok = int(getattr(res, "completion_tokens", 0) or 0)
            w1_tokens.append(tok)
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
            ordered_outcomes[idx] = out
        if pal_w1_jobs:
            await asyncio.gather(*(asyncio.to_thread(_execute_pal_outcome, o, sandbox_timeout_s) for o in pal_w1_jobs))
        w1_elapsed = time.perf_counter() - w1_t0
        for idx, out in enumerate(ordered_outcomes):
            if out is None:
                continue
            exec_s = float(out.exec_result.elapsed_s) if out.exec_result else 0.0
            slot_elapsed = (float(out.latency_s or 0.0) + exec_s) or w1_elapsed
            _emit(
                on_event,
                _slot_done_payload(
                    engine="reason",
                    slot=idx + 1,
                    out=out,
                    problem=problem,
                    elapsed_s=slot_elapsed,
                    tokens=w1_tokens[idx] if idx < len(w1_tokens) else 0,
                    wave=1,
                ),
            )

    for i, (name, kind, _, _, _, _) in enumerate(wave1_roles):
        out_i = ordered_outcomes[i]
        if out_i is None:
            out_i = BranchOutcome(
                name=name,
                wave=1,
                kind=kind,
                raw_content="",
                reasoning="",
                error="cancelled_early_consensus",
            )
        trace.wave1_outcomes.append(out_i)

    # Evaluate Wave-1 consensus (100% battle-tested v2 calibration gate!)
    valid_pal_w1 = [o for o in trace.wave1_outcomes if _is_valid_pal(o)]
    all_valid_w1 = [o for o in trace.wave1_outcomes if _is_valid_any(o)]
    pal_clusters = _cluster_answers(valid_pal_w1)
    all_clusters = _cluster_answers(all_valid_w1)

    need_wave2 = False
    w2_mode = ""
    if len(valid_pal_w1) == 0:
        need_wave2 = True
        w2_mode = "zero_pal"
        trace.wave2_reason = "All Wave-1 Python verification scripts failed or timed out"
    elif len(pal_clusters) > 1:
        need_wave2 = True
        w2_mode = "disagreement"
        summary = ", ".join(f"{ans} ({len(ms)})" for ans, ms in pal_clusters)
        trace.wave2_reason = f"PAL script disagreement: {summary}"
    elif len(valid_pal_w1) == 1:
        if len(all_clusters) != 1 or len(all_valid_w1) < 2:
            need_wave2 = True
            w2_mode = "single_pal"
            trace.wave2_reason = "Only 1 PAL script succeeded without independent corroboration"
    elif len(pal_clusters) == 1 and len(all_clusters) > 1 and len(valid_pal_w1) < 3:
        need_wave2 = True
        w2_mode = "disagreement"
        trace.wave2_reason = (
            f"PAL ({pal_clusters[0][0]}) disagreed with analytical stream; running falsifier"
        )
    elif (
        len(pal_clusters) == 1
        and _is_suspect_empty_zero(pal_clusters[0][0])
        and not any(
            o.kind == "analytical" and _is_valid_any(o) and o.norm_answer == "0"
            for o in trace.wave1_outcomes
        )
    ):
        need_wave2 = True
        w2_mode = "disagreement"
        trace.wave2_reason = "Suspect empty-search zero (0) without analytical confirmation; running falsifier"

    # Record graveyard tombstones ONLY for real failures (skip early-abort placeholders)
    for o in trace.wave1_outcomes:
        if o.error and o.error != "cancelled_early_consensus":
            trace.graveyard.append(f"[{o.name}] FAILED: {o.error[:240]}")

    # Stage 2: Streaming Wave-2 Falsifiers (1 Differential + 2 STRICTLY UNANCHORED!)
    _UNANCHORED_W2_ROLES = {"falsifier_reread", "falsifier_numeric_z3"}
    if need_wave2 and enable_wave2:
        trace.wave2_triggered = True
        w2_limit = 2 if num_wave1_streams <= 2 else 3
        w2_tb = 1150 if deep else 900

        conflict_lines: list[str] = []
        for rep, members in all_clusters:
            names = ", ".join(m.name for m in members)
            sample_snip = next((m.code[:750] for m in members if m.code), "")
            conflict_lines.append(
                f"Candidate Answer = {rep} (from: {names})\n"
                + (f"Excerpt of method:\n```python\n{sample_snip}\n```" if sample_snip else "")
            )
        graveyard_text = "\n".join(trace.graveyard) if trace.graveyard else "(none)"
        diff_dossier = (
            f"PROBLEM:\n{problem.strip()}\n\n"
            f"WAVE-1 STATUS: {trace.wave2_reason}\n\n"
            f"COMPETING CANDIDATES:\n" + ("\n---\n".join(conflict_lines) if conflict_lines else "(none succeeded)")
            + f"\n\nGRAVEYARD (DO NOT REPEAT THESE BUGS):\n{graveyard_text}"
        )

        non_anchor_hints = [
            _non_anchoring_error_hint(o.error)
            for o in trace.wave1_outcomes
            if o.error and o.error != "cancelled_early_consensus"
        ]
        non_anchor_hints = [h for h in dict.fromkeys(non_anchor_hints) if h]
        clean_pitfall_note = (
            "\n\n[Sandbox pitfalls to avoid: " + " | ".join(non_anchor_hints[:2]) + "]"
            if non_anchor_hints
            else ""
        )

        w2_candidates: list[tuple[str, str, CallSpec]] = [
            (
                "falsifier_bruteforce",
                "pal",
                CallSpec(
                    messages=[
                        {"role": "system", "content": shared_sys},
                        {
                            "role": "user",
                            "content": f"{diff_dossier}\n\nPROTOCOL (falsifier_bruteforce):\n{skills.card('falsifier_bruteforce')}",
                        },
                    ],
                    tag="w2_falsifier_bruteforce",
                    max_tokens=w2_tb + 2200,
                    temperature=0.25,
                    seed=505,
                    enable_thinking=True,
                    thinking_budget=w2_tb,
                    reasoning_effort=pal_effort,
                    loop_guard=True,
                ),
            ),
            (
                "falsifier_reread",
                "pal",
                CallSpec(
                    messages=[
                        {"role": "system", "content": shared_sys},
                        {
                            "role": "user",
                            "content": f"{shared_user_prefix}{clean_pitfall_note}\n\nPROTOCOL (falsifier_reread):\n{_W2_UNANCHORED_REREAD_CARD}",
                        },
                    ],
                    tag="w2_falsifier_reread",
                    max_tokens=w2_tb + 2200,
                    temperature=0.45,
                    seed=606,
                    enable_thinking=True,
                    thinking_budget=w2_tb,
                    reasoning_effort=pal_effort,
                    loop_guard=True,
                ),
            ),
            (
                "falsifier_numeric_z3",
                "pal",
                CallSpec(
                    messages=[
                        {"role": "system", "content": shared_sys},
                        {
                            "role": "user",
                            "content": f"{shared_user_prefix}{clean_pitfall_note}\n\nPROTOCOL (falsifier_numeric_z3):\n{_W2_NUMERIC_Z3_CARD}",
                        },
                    ],
                    tag="w2_falsifier_numeric_z3",
                    max_tokens=w2_tb + 2200,
                    temperature=0.40,
                    seed=707,
                    enable_thinking=True,
                    thinking_budget=w2_tb,
                    reasoning_effort=pal_effort,
                    loop_guard=True,
                ),
            ),
        ]
        w2_role_specs = w2_candidates[:w2_limit]

        _emit(
            on_event,
            {
                "engine": "reason",
                "event": "wave2_start",
                "reason": trace.wave2_reason,
                "streams": len(w2_role_specs),
            },
        )

        w2_ordered: list[Optional[BranchOutcome]] = [None] * len(w2_role_specs)
        if has_complete:
            w2_tasks = [
                asyncio.create_task(
                    _run_single_slot(i, name, kind, 2, spec, sandbox_timeout_s + 4.0)
                )
                for i, (name, kind, spec) in enumerate(w2_role_specs)
            ]
            pending_w2: set[asyncio.Task[tuple[int, BranchOutcome, int, float]]] = set(w2_tasks)
            w2_valid_pals: list[BranchOutcome] = []
            try:
                while pending_w2:
                    done_set, pending_w2 = await asyncio.wait(pending_w2, return_when=asyncio.FIRST_COMPLETED)
                    for idx, out, tok, slot_elapsed in sorted((t.result() for t in done_set), key=lambda x: x[0]):
                        w2_ordered[idx] = out
                        if out.error:
                            trace.graveyard.append(f"[{out.name}] FAILED: {out.error[:240]}")
                        _emit(
                            on_event,
                            _slot_done_payload(
                                engine="reason",
                                slot=idx + 1,
                                out=out,
                                problem=problem,
                                elapsed_s=slot_elapsed,
                                tokens=tok,
                                wave=2,
                            ),
                        )
                        if _is_valid_pal(out):
                            w2_valid_pals.append(out)

                    if not pending_w2 or not w2_valid_pals:
                        continue

                    w2_abort_ans: Optional[str] = None
                    w2_abort_reason = ""
                    w2_cl = _cluster_answers(w2_valid_pals)

                    if w2_cl and len(w2_cl[0][1]) >= 2:
                        w2_abort_ans = w2_cl[0][0]
                        w2_abort_reason = f"2/{len(w2_role_specs)} Wave-2 PAL consensus"
                    elif (
                        w2_mode == "single_pal"
                        and len(w2_cl) == 1
                        and any(
                            p.name in _UNANCHORED_W2_ROLES
                            and _answers_equivalent(p.norm_answer, valid_pal_w1[0].norm_answer)
                            for p in w2_valid_pals
                        )
                    ):
                        # Only abort single-PAL escalation if no Wave-1 analytical contradicts it!
                        w1_ana_contradicts = any(
                            o.kind == "analytical"
                            and _is_valid_any(o)
                            and not _answers_equivalent(o.norm_answer, valid_pal_w1[0].norm_answer)
                            for o in trace.wave1_outcomes
                        )
                        if not w1_ana_contradicts:
                            w2_abort_ans = valid_pal_w1[0].norm_answer
                            w2_abort_reason = "2 independent unanchored PALs agreed in Wave 2"

                    if w2_abort_ans is not None:
                        n_cancel = len(pending_w2)
                        for lag in pending_w2:
                            lag.cancel()
                        _emit(
                            on_event,
                            {
                                "engine": "reason",
                                "event": "early_abort",
                                "cancelled": n_cancel,
                                "elapsed_s": round(time.perf_counter() - t0, 2),
                                "consensus_answer": w2_abort_ans,
                                "reason": w2_abort_reason,
                            },
                        )
                        await asyncio.gather(*pending_w2, return_exceptions=True)
                        pending_w2.clear()
                        break
            except BaseException:
                for lag in pending_w2:
                    lag.cancel()
                if pending_w2:
                    await asyncio.gather(*pending_w2, return_exceptions=True)
                raise
        else:
            w2_t0 = time.perf_counter()
            w2_results = await client.batch([spec for _, _, spec in w2_role_specs])
            pal_w2_jobs: list[BranchOutcome] = []
            for idx, ((name, kind, _), res) in enumerate(zip(w2_role_specs, w2_results)):
                out = BranchOutcome(
                    name=name,
                    wave=2,
                    kind=kind,
                    raw_content=res.content,
                    reasoning=res.reasoning,
                    latency_s=res.latency_s,
                    error=res.raw_error or "",
                )
                if res.ok and kind == "pal":
                    pal_w2_jobs.append(out)
                w2_ordered[idx] = out
            if pal_w2_jobs:
                await asyncio.gather(*(asyncio.to_thread(_execute_pal_outcome, o, sandbox_timeout_s + 4.0) for o in pal_w2_jobs))
            w2_elapsed = time.perf_counter() - w2_t0
            for idx, (out, res) in enumerate(zip(w2_ordered, w2_results), start=1):
                if out is None:
                    continue
                if out.error:
                    trace.graveyard.append(f"[{out.name}] FAILED: {out.error[:240]}")
                exec_s = float(out.exec_result.elapsed_s) if out.exec_result else 0.0
                slot_elapsed = (float(out.latency_s or 0.0) + exec_s) or w2_elapsed
                _emit(
                    on_event,
                    _slot_done_payload(
                        engine="reason",
                        slot=idx,
                        out=out,
                        problem=problem,
                        elapsed_s=slot_elapsed,
                        tokens=int(getattr(res, "completion_tokens", 0) or 0),
                        wave=2,
                    ),
                )

        for out in w2_ordered:
            if out is not None:
                trace.wave2_outcomes.append(out)

        # Stage 3 (ONLY when 0 valid PALs across BOTH Wave 1 & Wave 2):
        # Streaming B=3 (50-digit mpmath/numpy numeric solver + 2 clean-room deep IMO proofs) with early-abort!
        valid_pal_after_w2 = [o for o in (trace.wave1_outcomes + trace.wave2_outcomes) if _is_valid_pal(o)]
        if not valid_pal_after_w2:
            w3_tb_pal = 1200 if deep else 1000
            w3_tb_deep = 2200 if deep else 1800
            w3_pitfalls = [
                _non_anchoring_error_hint(o.error)
                for o in (trace.wave1_outcomes + trace.wave2_outcomes)
                if o.error and o.error != "cancelled_early_consensus"
            ]
            w3_pitfalls = [h for h in dict.fromkeys(w3_pitfalls) if h]
            w3_pitfall_note = (
                "\n\n[Prior failure modes to avoid: " + " | ".join(w3_pitfalls[:3]) + "]"
                if w3_pitfalls
                else ""
            )
            w3_role_specs: list[tuple[str, str, CallSpec]] = [
                (
                    "w3_pal_numeric_exact",
                    "pal",
                    CallSpec(
                        messages=[
                            {"role": "system", "content": shared_sys},
                            {
                                "role": "user",
                                "content": f"PROBLEM:\n{problem.strip()}{w3_pitfall_note}\n\nPROTOCOL (w3_pal_numeric_exact):\n{_W3_NUMERIC_EXACT_CARD}",
                            },
                        ],
                        tag="w3_pal_numeric_exact",
                        max_tokens=w3_tb_pal + 2400,
                        temperature=0.20,
                        seed=808,
                        enable_thinking=True,
                        thinking_budget=w3_tb_pal,
                        loop_guard=True,
                    ),
                ),
                (
                    "w3_analytical_deep_a",
                    "analytical",
                    CallSpec(
                        messages=[
                            {"role": "system", "content": math_sys},
                            {
                                "role": "user",
                                "content": f"PROBLEM:\n{problem.strip()}\n\nPROTOCOL (w3_analytical_deep_a):\n{_W3_ANALYTICAL_DEEP_A_CARD}",
                            },
                        ],
                        tag="w3_analytical_deep_a",
                        max_tokens=4096,
                        temperature=0.20,
                        seed=909,
                        enable_thinking=True,
                        thinking_budget=w3_tb_deep,
                        loop_guard=True,
                    ),
                ),
                (
                    "w3_analytical_deep_b",
                    "analytical",
                    CallSpec(
                        messages=[
                            {"role": "system", "content": math_sys},
                            {
                                "role": "user",
                                "content": f"PROBLEM:\n{problem.strip()}\n\nPROTOCOL (w3_analytical_deep_b):\n{_W3_ANALYTICAL_DEEP_B_CARD}",
                            },
                        ],
                        tag="w3_analytical_deep_b",
                        max_tokens=4096,
                        temperature=0.55,
                        seed=1010,
                        enable_thinking=True,
                        thinking_budget=w3_tb_deep,
                        loop_guard=True,
                    ),
                ),
            ]
            w3_ordered: list[Optional[BranchOutcome]] = [None] * len(w3_role_specs)
            if has_complete:
                w3_pal_repairs = max(max_pal_repairs, 2) if max_pal_repairs > 0 else 0
                w3_tasks = [
                    asyncio.create_task(
                        _run_single_slot(
                            i,
                            name,
                            kind,
                            3,
                            spec,
                            sandbox_timeout_s + 6.0,
                            repairs=(w3_pal_repairs if kind == "pal" else 0),
                        )
                    )
                    for i, (name, kind, spec) in enumerate(w3_role_specs)
                ]
                pending_w3: set[asyncio.Task[tuple[int, BranchOutcome, int, float]]] = set(w3_tasks)
                w3_done_valid: list[BranchOutcome] = []
                try:
                    while pending_w3:
                        done_set, pending_w3 = await asyncio.wait(pending_w3, return_when=asyncio.FIRST_COMPLETED)
                        for idx_3, out_3, tok_3, el_3 in sorted((t.result() for t in done_set), key=lambda x: x[0]):
                            w3_ordered[idx_3] = out_3
                            _emit(
                                on_event,
                                _slot_done_payload(
                                    engine="reason",
                                    slot=idx_3 + 1,
                                    out=out_3,
                                    problem=problem,
                                    elapsed_s=el_3,
                                    tokens=tok_3,
                                    wave=3,
                                ),
                            )
                            if _is_valid_any(out_3):
                                w3_done_valid.append(out_3)
                        if pending_w3 and len(w3_done_valid) >= 2:
                            w3_cl = _cluster_answers(w3_done_valid)
                            if w3_cl and len(w3_cl[0][1]) >= 2:
                                n_cancel = len(pending_w3)
                                for lag in pending_w3:
                                    lag.cancel()
                                _emit(
                                    on_event,
                                    {
                                        "engine": "reason",
                                        "event": "early_abort",
                                        "cancelled": n_cancel,
                                        "elapsed_s": round(time.perf_counter() - t0, 2),
                                        "consensus_answer": w3_cl[0][0],
                                        "reason": "2/3 Wave-3 consensus",
                                    },
                                )
                                await asyncio.gather(*pending_w3, return_exceptions=True)
                                pending_w3.clear()
                                break
                except BaseException:
                    for lag in pending_w3:
                        lag.cancel()
                    if pending_w3:
                        await asyncio.gather(*pending_w3, return_exceptions=True)
                    raise
            else:
                w3_res = await client.batch([spec for _, _, spec in w3_role_specs])
                pal_w3_jobs: list[BranchOutcome] = []
                for idx_3, ((name, kind, _), res) in enumerate(zip(w3_role_specs, w3_res)):
                    out_3 = BranchOutcome(
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
                            pal_w3_jobs.append(out_3)
                        else:
                            raw_a = extract_answer_from_prose(res.content)
                            out_3.raw_answer = raw_a
                            out_3.norm_answer = normalize_answer(raw_a)
                    w3_ordered[idx_3] = out_3
                if pal_w3_jobs:
                    await asyncio.gather(*(asyncio.to_thread(_execute_pal_outcome, o, sandbox_timeout_s + 6.0) for o in pal_w3_jobs))
                for idx_3, (out_3, res) in enumerate(zip(w3_ordered, w3_res), start=1):
                    if out_3 is None:
                        continue
                    _emit(
                        on_event,
                        _slot_done_payload(
                            engine="reason",
                            slot=idx_3,
                            out=out_3,
                            problem=problem,
                            elapsed_s=float(out_3.latency_s or 0.0),
                            tokens=int(getattr(res, "completion_tokens", 0) or 0),
                            wave=3,
                        ),
                    )

            for out_3 in w3_ordered:
                if out_3 is not None:
                    trace.wave2_outcomes.append(out_3)

    # Final Weighted Adjudication across all executed & verified branches
    def _score_all() -> tuple[list[BranchOutcome], list[tuple[float, int, str, list[BranchOutcome]]]]:
        combined_now = trace.wave1_outcomes + trace.wave2_outcomes
        valid_now = [o for o in combined_now if _is_valid_any(o)]
        if not valid_now:
            valid_now = [
                o for o in combined_now
                if o.norm_answer and _looks_like_concise_answer(o.norm_answer)
            ]
        cl_now = _cluster_answers(valid_now)
        scored: list[tuple[float, int, str, list[BranchOutcome]]] = []
        for rep, members in cl_now:
            total_w = sum(_branch_weight(m) for m in members)
            pal_count = sum(1 for m in members if _is_valid_pal(m))
            w2_pal = sum(1 for m in members if m.wave == 2 and _is_valid_pal(m))
            unanchored_pal = sum(
                1 for m in members if m.name in _UNANCHORED_W2_ROLES and _is_valid_pal(m)
            )
            if w2_pal >= 2:
                total_w += 3.5
            if unanchored_pal >= 1:
                total_w += 0.35
            scored.append((total_w, pal_count, rep, members))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return valid_now, scored

    valid_all, scored_clusters = _score_all()

    if not scored_clusters:
        combined = trace.wave1_outcomes + trace.wave2_outcomes
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
        _emit(
            on_event,
            {
                "engine": "reason",
                "event": "done",
                "elapsed_s": round(trace.wall_s, 2),
                "verified": trace.verified,
                "summary": f"{trace.consensus_type} -> {trace.final_answer}",
            },
        )
        return trace

    winner_w, winner_pals, winner_ans, winner_members = scored_clusters[0]

    # Stage 4 (conditional, 1 SINGLE call): Executable Verification Arbiter when top 2 clusters are close!
    close_margin = 2.25 if winner_pals <= 1 else 1.5
    if (
        enable_wave2
        and has_complete
        and len(scored_clusters) >= 2
        and (scored_clusters[0][0] - scored_clusters[1][0]) <= close_margin
    ):
        ans_a, mem_a = scored_clusters[0][2], scored_clusters[0][3]
        ans_b, mem_b = scored_clusters[1][2], scored_clusters[1][3]
        code_a = next((m.code for m in mem_a if m.code), mem_a[0].raw_content)[:2200]
        code_b = next((m.code for m in mem_b if m.code), mem_b[0].raw_content)[:2200]
        arb_prompt = (
            f"PROBLEM:\n{problem.strip()}\n\n"
            f"CANDIDATE A (Answer: {ans_a}):\n{code_a}\n\n"
            f"CANDIDATE B (Answer: {ans_b}):\n{code_b}\n\n"
            "Compare Candidate A and Candidate B against EVERY word and constraint in the problem. "
            "Check for off-by-1, 2x symmetry/double-counting, or missed edge conditions. "
            "If verifiable computationally (exact Python or 50-digit `mpmath`/`numpy` — no `scipy`), output a short "
            "fenced ```python``` script printing `ANSWER: <exact_value>`; otherwise give a crisp derivation ending with "
            "`WINNER: A` or `WINNER: B` and `ANSWER: <exact_value>`."
        )
        arb_res = await client.complete(
            CallSpec(
                messages=[
                    {"role": "system", "content": shared_sys},
                    {"role": "user", "content": arb_prompt},
                ],
                tag="reason_exec_arbiter",
                max_tokens=3200,
                temperature=0.0,
                seed=909,
                enable_thinking=True,
                thinking_budget=1200,
                loop_guard=True,
            )
        )
        arb_out = BranchOutcome(
            name="reason_exec_arbiter",
            wave=3,
            kind="pal",
            raw_content=arb_res.content if arb_res.ok else "",
            reasoning=arb_res.reasoning if arb_res.ok else "",
            latency_s=arb_res.latency_s,
            error=arb_res.raw_error or "",
        )
        if arb_res.ok:
            code_arb = extract_python_code(arb_res.content)
            if code_arb:
                await asyncio.to_thread(_execute_pal_outcome, arb_out, sandbox_timeout_s)
                if max_pal_repairs > 0 and not _is_valid_pal(arb_out):
                    await _repair_pal_outcome_inplace(
                        arb_out,
                        base_tag="reason_exec_arbiter",
                        exec_timeout_s=sandbox_timeout_s,
                        max_repairs=max_pal_repairs,
                        seed=1909,
                    )
            else:
                arb_out.kind = "analytical"
                raw_arb = extract_answer_from_prose(arb_res.content)
                arb_out.raw_answer = raw_arb
                arb_out.norm_answer = normalize_answer(raw_arb)
        trace.wave2_outcomes.append(arb_out)

        if _is_valid_any(arb_out):
            valid_all, scored_clusters = _score_all()
            winner_w, winner_pals, winner_ans, winner_members = scored_clusters[0]
            trace.consensus_type = (
                "exec_arbiter_pal" if _is_valid_pal(arb_out) else "exec_arbiter_proof"
            )
        else:
            txt_u = (arb_res.content or "").upper()
            if "WINNER: B" in txt_u:
                winner_ans, winner_members = ans_b, mem_b
                trace.consensus_type = "exec_arbiter_B"
            else:
                trace.consensus_type = "exec_arbiter_A"
    else:
        if trace.wave2_triggered:
            trace.consensus_type = f"wave2_verified ({winner_pals} PAL)"
        elif need_wave2 and not enable_wave2:
            trace.consensus_type = f"wave1_only ({winner_pals} PAL)"
        else:
            trace.consensus_type = f"wave1_consensus ({winner_pals} PAL)"

    # Detect whether Wave 2 genuinely recovered from a Wave-1 failure or disagreement
    w1_first_pal = next((o.norm_answer for o in trace.wave1_outcomes if _is_valid_pal(o)), "")
    if trace.wave2_triggered and (not w1_first_pal or not _answers_equivalent(w1_first_pal, winner_ans)):
        trace.recovered_from_disagreement = True

    best_member = sorted(winner_members, key=lambda m: (m.wave, 1 if m.verified_constraints else 0), reverse=True)[0]
    stdout_preview = (best_member.exec_result.stdout.strip() if best_member.exec_result else "")[:800]
    trace.verified = any(_is_valid_pal(m) for m in winner_members)
    trace.final_answer = winner_ans
    trace.final_explanation = (
        f"Verified Answer: **{winner_ans}**\n\n"
        f"Verification: {trace.consensus_type} "
        f"({len(winner_members)}/{len(valid_all)} agreeing branches)\n"
        + (f"\nVerification Output (`{best_member.name}`):\n```\n{stdout_preview}\n```" if stdout_preview else "")
    )
    trace.wall_s = time.perf_counter() - t0
    _emit(
        on_event,
        {
            "engine": "reason",
            "event": "done",
            "elapsed_s": round(trace.wall_s, 2),
            "verified": trace.verified,
            "summary": f"{trace.consensus_type} -> {trace.final_answer}",
        },
    )
    return trace

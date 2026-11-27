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
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
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
    # Keep PAL thinking crisp (tb=700) without forcing low-effort truncation on hard Olympiad math;
    # Wave-1 streaming early-abort handles the latency win cleanly once 2 independent PALs agree!
    think_pal = 1200 if deep else 700
    think_ana = 2000 if deep else 1300
    pal_effort = None

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
                max_tokens=t_budget + 2000,
                temperature=temp,
                seed=seed,
                enable_thinking=True,
                thinking_budget=t_budget,
                reasoning_effort=pal_effort,
                loop_guard=True,
            )
        )

    _emit(
        on_event,
        {
            "engine": "reason",
            "event": "start",
            "mode": "reason",
            "streams": len(specs),
        },
    )

    ordered_outcomes: list[Optional[BranchOutcome]] = [None] * len(wave1_roles)
    if callable(getattr(client, "complete", None)):
        # True streaming Wave-1 early-abort: run each slot + sandbox immediately on completion
        # and cancel lagging GPU requests the instant 2 independent PAL scripts agree!
        async def _run_w1_slot(
            idx: int, name: str, kind: str, spec: CallSpec
        ) -> tuple[int, BranchOutcome, int, float]:
            slot_t0 = time.perf_counter()
            res = await client.complete(spec)
            tok = int(getattr(res, "completion_tokens", 0) or 0)
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
                    await asyncio.to_thread(_execute_pal_outcome, out, sandbox_timeout_s)
                else:
                    raw_ans = extract_answer_from_prose(res.content)
                    out.raw_answer = raw_ans
                    out.norm_answer = normalize_answer(raw_ans)
            slot_elapsed = max(time.perf_counter() - slot_t0, float(out.latency_s or 0.0))
            return idx, out, tok, slot_elapsed

        slot_tasks: list[asyncio.Task[tuple[int, BranchOutcome, int, float]]] = [
            asyncio.create_task(_run_w1_slot(i, name, kind, spec))
            for i, ((name, kind, _, _, _, _), spec) in enumerate(zip(wave1_roles, specs))
        ]
        pal_verified_so_far: list[BranchOutcome] = []
        early_aborted = False

        pending: set[asyncio.Task[tuple[int, BranchOutcome, int, float]]] = set(slot_tasks)
        try:
            while pending:
                done_set, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                completed_batch: list[tuple[int, BranchOutcome, int, float]] = [
                    done_task.result() for done_task in done_set
                ]
                completed_batch.sort(key=lambda item: item[0])
                for idx, out, tok, slot_elapsed in completed_batch:
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
                    if (
                        out.kind == "pal"
                        and out.norm_answer
                        and _answer_satisfies_prompt_format(problem, out.norm_answer)
                    ):
                        pal_verified_so_far.append(out)

                if pending and pal_verified_so_far:
                    clusters_now = _cluster_answers(pal_verified_so_far)
                    if clusters_now and len(clusters_now[0][1]) >= 2:
                        top_ans = clusters_now[0][0]
                        ana_contradicts = any(
                            o is not None
                            and o.kind == "analytical"
                            and o.norm_answer
                            and not _answers_equivalent(o.norm_answer, top_ans)
                            for o in ordered_outcomes
                        )
                        if not ana_contradicts or len(clusters_now[0][1]) >= 3:
                            # 2 independent verified PAL scripts agree (with no analytical contradiction) -> cancel lagging GPU slots!
                            early_aborted = True
                            n_cancelled = len(pending)
                            for lag in pending:
                                lag.cancel()
                            _emit(
                                on_event,
                                {
                                    "engine": "reason",
                                    "event": "early_abort",
                                    "cancelled": n_cancelled,
                                    "elapsed_s": round(time.perf_counter() - t0, 2),
                                    "consensus_answer": top_ans,
                                    "reason": f"{len(clusters_now[0][1])}/{len(specs)} PAL consensus",
                                },
                            )
                            break
            if early_aborted and pending:
                await asyncio.gather(*pending, return_exceptions=True)
        except BaseException:
            for lag in pending:
                lag.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            raise
    else:
        # Batch fallback path for custom clients implementing only `.batch()`
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
    if need_wave2 and enable_wave2:
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

        w2_tb = 1100 if deep else 650
        w2_roles = [
            ("falsifier_bruteforce", skills.card("falsifier_bruteforce"), 0.25, 505, w2_tb),
            ("falsifier_reread", skills.card("falsifier_reread"), 0.45, 606, w2_tb),
            ("falsifier_z3_sympy", skills.card("falsifier_z3_sympy"), 0.40, 707, w2_tb),
        ]
        w2_specs = [
            CallSpec(
                messages=[
                    {"role": "system", "content": shared_sys},
                    {"role": "user", "content": f"{dossier}\n\nPROTOCOL ({name}):\n{card_text}"},
                ],
                tag=f"w2_{name}",
                max_tokens=tb + 2200,
                temperature=temp,
                seed=seed,
                enable_thinking=True,
                thinking_budget=tb,
                reasoning_effort=pal_effort,
                loop_guard=True,
            )
            for name, card_text, temp, seed, tb in w2_roles
        ]
        _emit(
            on_event,
            {
                "engine": "reason",
                "event": "wave2_start",
                "reason": trace.wave2_reason,
                "streams": len(w2_specs),
            },
        )
        w2_t0 = time.perf_counter()
        w2_results = await client.batch(w2_specs)
        pal_w2_jobs: list[BranchOutcome] = []
        w2_batch_outcomes: list[BranchOutcome] = []
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
            w2_batch_outcomes.append(out)
            trace.wave2_outcomes.append(out)
        if pal_w2_jobs:
            await asyncio.gather(*(asyncio.to_thread(_execute_pal_outcome, o, sandbox_timeout_s + 4.0) for o in pal_w2_jobs))
        w2_elapsed = time.perf_counter() - w2_t0
        for w2_idx, (out, res) in enumerate(zip(w2_batch_outcomes, w2_results), start=1):
            if out.error:
                trace.graveyard.append(f"[{out.name}] FAILED: {out.error[:240]}")
            exec_s = float(out.exec_result.elapsed_s) if out.exec_result else 0.0
            slot_elapsed = (float(out.latency_s or 0.0) + exec_s) or w2_elapsed
            _emit(
                on_event,
                _slot_done_payload(
                    engine="reason",
                    slot=w2_idx,
                    out=out,
                    problem=problem,
                    elapsed_s=slot_elapsed,
                    tokens=int(getattr(res, "completion_tokens", 0) or 0),
                    wave=2,
                ),
            )

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
            w3_t0 = time.perf_counter()
            w3_results = await client.batch(w3_specs)
            pal_w3_jobs: list[BranchOutcome] = []
            w3_batch_outcomes: list[BranchOutcome] = []
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
                w3_batch_outcomes.append(out)
                trace.wave2_outcomes.append(out)
            if pal_w3_jobs:
                await asyncio.gather(*(asyncio.to_thread(_execute_pal_outcome, o, sandbox_timeout_s + 6.0) for o in pal_w3_jobs))
            w3_elapsed = time.perf_counter() - w3_t0
            for w3_idx, (out, res) in enumerate(zip(w3_batch_outcomes, w3_results), start=1):
                exec_s = float(out.exec_result.elapsed_s) if out.exec_result else 0.0
                slot_elapsed = (float(out.latency_s or 0.0) + exec_s) or w3_elapsed
                _emit(
                    on_event,
                    _slot_done_payload(
                        engine="reason",
                        slot=w3_idx,
                        out=out,
                        problem=problem,
                        elapsed_s=slot_elapsed,
                        tokens=int(getattr(res, "completion_tokens", 0) or 0),
                        wave=3,
                    ),
                )

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
        elif need_wave2 and not enable_wave2:
            trace.consensus_type = f"wave1_only ({winner_pals} PAL)"
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

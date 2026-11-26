"""One-Command Live Side-by-Side Showdown (`flash-forge showdown`).

Lets ANY user on TensorFold, Ollama, llama.cpp, vLLM, or LM Studio run a live,
reproducible Vanilla vs. FlashForge duel across Math/Logic + Hard Coding + Open-Ended
Systems Architecture directly on their own frozen local model with a crisp ANSI HUD.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from . import hud
from .client import CallSpec, TensorFoldClient
from .code_forge import _run_modular_suite
from .engine import FlashForge, ForgeConfig
from .reason_forge import _answers_equivalent, extract_answer_from_prose, normalize_answer
from .sandbox import check_syntax, extract_python_code


SHOWDOWN_MATH = (
    "How many strings of length 14 over the alphabet {A, B, C} satisfy ALL of the following?\n"
    "1. No substring 'AA'.\n2. No substring 'BB'.\n3. No substring 'ABC'.\n"
    "4. The total count of 'A' characters is STRICTLY GREATER than the total count of 'B' characters.\n"
    "Give the exact integer, ending with `ANSWER: <integer>`."
)
SHOWDOWN_MATH_EXPECTED = "54662"

SHOWDOWN_LOGIC = (
    "Consider an undirected graph on 8 vertices labeled 0..7 with these 15 edges:\n"
    "(0,1), (0,2), (0,3), (1,2), (1,4), (2,5), (3,4), (3,6), (4,5), (4,7), (5,6), (5,7), (6,7), (0,7), (2,6).\n"
    "Assign each vertex v in {0..7} an integer color c[v] in {1, 2, 3, 4} subject to ALL constraints:\n"
    "1. Adjacent vertices must have different colors.\n2. c[0] == 1.\n"
    "3. For every edge (u, v) where BOTH u and v are odd, |c[u] - c[v]| != 3.\n"
    "4. No color may be used more than 2 times across the 8 vertices.\n"
    "Minimize sum((v + 1) * c[v] for v in range(8)). End with `ANSWER: <integer>`."
)
SHOWDOWN_LOGIC_EXPECTED = "88"

SHOWDOWN_CODE = (
    "Implement `eval_expr(expr: str) -> Fraction` (returning `fractions.Fraction`) for exact rational arithmetic:\n"
    "- Exact integers & decimals (`0.1 + 0.2 == Fraction(3, 10)`, `.5 == Fraction(1, 2)`).\n"
    "- Operators `+`, `-`, `*`, `/`, `^` (right-associative: `2^3^2 == 512`), parentheses.\n"
    "- Unary `+`/`-` has LOWER precedence than `^` (`-2^2 == -4`, `2^-2^2 == Fraction(1, 16)`, `2^--3 == 8`).\n"
    "- `0^0 == Fraction(1, 1)`; `0` to negative power, non-integer exponent, div by zero, or syntax error raises `ValueError`.\n"
    "Output ONLY Python code defining `eval_expr`."
)
SHOWDOWN_CODE_TESTS = '''from fractions import Fraction
def test_exact():
    assert eval_expr("0.1 + 0.2") == Fraction(3, 10)
    assert eval_expr("-2^2") == Fraction(-4, 1)
    assert eval_expr("2^3^2") == Fraction(512, 1)
    assert eval_expr("2^-2^2") == Fraction(1, 16)
    assert eval_expr("2^--3") == Fraction(8, 1)
    assert eval_expr("0^0") == Fraction(1, 1)
    for bad in ["1/0", "0^-1", "4^(1/2)", "", "1 2", "."]:
        try:
            eval_expr(bad)
            raise AssertionError(f"Expected ValueError for {bad!r}")
        except ValueError:
            pass
'''


async def run_showdown(base_url: str, referee_url: Optional[str] = None, streams: int = 4) -> None:
    async with TensorFoldClient(base_url=base_url) as vc, FlashForge(
        ForgeConfig(base_url=base_url, referee_url=referee_url, num_streams=streams, enable_memory=False)
    ) as ff:
        caps = vc.caps
        hud.banner(
            "CORTHEON LIVE SHOWDOWN (IDENTICAL FROZEN LOCAL WEIGHTS)",
            f"Model: {caps.model_id} @ {base_url}" + (f" | Referee: {referee_url}" if referee_url else ""),
        )
        v_score = 0
        f_score = 0

        # Duel 1 & 2: Hard Math & Hard Constrained Optimization
        for label, prompt, expected in [
            ("DUEL 1/3 — Automaton Combinatorics (L=14)", SHOWDOWN_MATH, SHOWDOWN_MATH_EXPECTED),
            ("DUEL 2/3 — Weighted Constrained 4-Coloring", SHOWDOWN_LOGIC, SHOWDOWN_LOGIC_EXPECTED),
        ]:
            hud.event("START", label)
            t0 = time.perf_counter()
            vr = await vc.complete(CallSpec(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192, temperature=0.6, seed=42, enable_thinking=True, thinking_budget=1500,
            ))
            v_ans = normalize_answer(extract_answer_from_prose(vr.content))
            v_ok = _answers_equivalent(v_ans, expected)
            v_score += int(v_ok)
            print(f"    Vanilla    : {hud.badge_pass() if v_ok else hud.badge_fail()} ({time.perf_counter()-t0:4.1f}s) got={v_ans or 'empty'!r}")

            fr = await ff.solve(prompt, force_mode="reason", learn=False)
            f_ok = _answers_equivalent(normalize_answer(fr.answer), expected)
            f_score += int(f_ok)
            print(f"    Cortheon   : {hud.badge_pass() if f_ok else hud.badge_fail()} ({fr.wall_s:4.1f}s) got={fr.answer!r} [B={streams} Neuro-Symbolic Python/Z3]")

        # Duel 3: Brutal Precedence & Exact Rational Parser
        hud.event("START", "DUEL 3/3 — Exact Rational Parser (-2^2, 2^-2^2, 0.1+0.2)")
        t0 = time.perf_counter()
        vr = await vc.complete(CallSpec(
            messages=[{"role": "user", "content": SHOWDOWN_CODE}],
            max_tokens=8192, temperature=0.6, seed=42, enable_thinking=True, thinking_budget=1500,
        ))
        v_code = extract_python_code(vr.content)
        v_rows, _, v_fat = _run_modular_suite(v_code, SHOWDOWN_CODE_TESTS) if check_syntax(v_code) is None else ([], 0.0, "SyntaxError")
        v_ok = bool(v_rows) and all(r["ok"] for r in v_rows) and not v_fat
        v_score += int(v_ok)
        print(f"    Vanilla    : {hud.badge_pass() if v_ok else hud.badge_fail()} ({time.perf_counter()-t0:4.1f}s)")

        fr = await ff.solve(SHOWDOWN_CODE, force_mode="code", learn=False)
        f_rows, _, f_fat = _run_modular_suite(fr.code, SHOWDOWN_CODE_TESTS) if check_syntax(fr.code) is None else ([], 0.0, "SyntaxError")
        f_ok = bool(f_rows) and all(r["ok"] for r in f_rows) and not f_fat
        f_score += int(f_ok)
        print(f"    Cortheon   : {hud.badge_pass() if f_ok else hud.badge_fail()} ({fr.wall_s:4.1f}s) [B={streams} Unified Consensus + Red-Teamer]")

        hud.banner(
            f"SHOWDOWN SCOREBOARD — Vanilla: {v_score}/3  vs  Cortheon: {f_score}/3",
            f"Same frozen local model ({caps.model_id}) — zero weight changes.",
        )

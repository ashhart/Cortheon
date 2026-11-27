"""Live Split-Screen Terminal Racer (`cortheon duel`) — Vanilla (B=1) vs Cortheon v2 (B=4 early-abort).

Runs both arms concurrently against the exact same frozen local model so engineers can watch
Vanilla and Cortheon race side-by-side with exact token counts, latency, and atomic early-abort.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from typing import Optional

from .client import CallSpec, TensorFoldClient
from .engine import FlashForge, ForgeConfig
from .reason_forge import _answers_equivalent, extract_answer_from_prose, normalize_answer
from .showdown import SHOWDOWN_LOGIC, SHOWDOWN_LOGIC_EXPECTED, SHOWDOWN_MATH, SHOWDOWN_MATH_EXPECTED

DUEL_PRESETS: dict[str, tuple[str, str, str]] = {
    "ring": (
        "25-Digit Exact Rational Markov Cover Time (7-Server Asymmetric Ring)",
        "7 servers sit in a ring (0..6). A token starts at server 1 (server 1 counts as already visited). "
        "Each tick it moves clockwise (+1 mod 7) with probability 2/3 or counterclockwise (-1 mod 7) with probability 1/3, "
        "except whenever it is at server 0 it instead teleports uniformly at random to one of the 7 servers (including 0). "
        "Find the exact rational expected number of ticks E[T] until all 7 servers have been visited at least once. "
        "End with `ANSWER: <p/q>`.",
        "2630600114653531854763423/183488738276457059168800",
    ),
    "automaton": (
        "Length-14 Constrained Automaton Combinatorics",
        SHOWDOWN_MATH,
        SHOWDOWN_MATH_EXPECTED,
    ),
    "coloring": (
        "Weighted Constrained 4-Coloring Optimization",
        SHOWDOWN_LOGIC,
        SHOWDOWN_LOGIC_EXPECTED,
    ),
}


@dataclass
class DuelOutcome:
    title: str
    expected: str
    vanilla_answer: str
    vanilla_ok: bool
    vanilla_wall_s: float
    vanilla_tokens: int
    cortheon_answer: str
    cortheon_ok: bool
    cortheon_wall_s: float
    cortheon_tokens: int
    cortheon_consensus: str
    cortheon_early_aborted: int


async def run_duel(
    base_url: str,
    *,
    preset: str = "ring",
    custom_prompt: Optional[str] = None,
    expected: str = "",
    streams: int = 4,
) -> DuelOutcome:
    if custom_prompt:
        title = "Custom Duel"
        prompt = custom_prompt
        exp = expected
    else:
        key = preset if preset in DUEL_PRESETS else "ring"
        title, prompt, exp = DUEL_PRESETS[key]

    async with TensorFoldClient(base_url=base_url) as vc, FlashForge(
        ForgeConfig(base_url=base_url, num_streams=streams, enable_memory=False)
    ) as ff:
        model_id = vc.caps.model_id
        print(f"\n\033[1;36m━━━ CORTHEON LIVE DUEL ━━━\033[0m  {title}")
        print(f"\033[2mModel: {model_id} @ {base_url}  |  Vanilla (B=1) vs Cortheon v2 (B={streams} early-abort)\033[0m\n")

        state = {
            "v_done": False,
            "v_s": 0.0,
            "v_ans": "",
            "v_ok": False,
            "v_tok": 0,
            "c_done": False,
            "c_s": 0.0,
            "c_ans": "",
            "c_ok": False,
            "c_tok": 0,
            "c_cons": "",
            "c_aborted": 0,
        }
        t_start = time.perf_counter()

        async def _run_vanilla() -> None:
            t0 = time.perf_counter()
            res = await vc.complete(
                CallSpec(
                    messages=[{"role": "user", "content": prompt}],
                    tag="duel_vanilla",
                    max_tokens=8192,
                    temperature=0.6,
                    seed=42,
                    enable_thinking=True,
                    thinking_budget=1500,
                )
            )
            state["v_s"] = time.perf_counter() - t0
            ans = normalize_answer(extract_answer_from_prose(res.content))
            state["v_ans"] = ans
            state["v_ok"] = bool(exp and _answers_equivalent(ans, exp)) if exp else bool(ans)
            state["v_tok"] = res.prompt_tokens + res.completion_tokens
            state["v_done"] = True

        async def _run_cortheon() -> None:
            t0 = time.perf_counter()
            res = await ff.solve(prompt, force_mode="reason", learn=False)
            state["c_s"] = time.perf_counter() - t0
            ans = normalize_answer(res.answer)
            state["c_ans"] = ans
            state["c_ok"] = bool(exp and _answers_equivalent(ans, exp)) if exp else bool(ans)
            stats = res.token_stats or {}
            state["c_tok"] = int(stats.get("prompt_tokens", 0)) + int(stats.get("completion_tokens", 0))
            tr = res.raw_trace
            state["c_cons"] = getattr(tr, "consensus_type", "")
            w1 = getattr(tr, "wave1_outcomes", [])
            state["c_aborted"] = sum(1 for o in w1 if o.error == "cancelled_early_consensus")
            state["c_done"] = True

        async def _ticker() -> None:
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            k = 0
            is_tty = sys.stdout.isatty()
            while not (state["v_done"] and state["c_done"]):
                el = time.perf_counter() - t_start
                sp = spinner[k % len(spinner)]
                k += 1
                v_str = (
                    f"\033[1;32mDONE ({state['v_s']:.1f}s)\033[0m"
                    if state["v_done"]
                    else f"\033[33m{sp} running ({el:.1f}s)\033[0m"
                )
                c_str = (
                    f"\033[1;32mDONE ({state['c_s']:.1f}s)\033[0m"
                    if state["c_done"]
                    else f"\033[36m{sp} B={streams} PAL ({el:.1f}s)\033[0m"
                )
                if is_tty:
                    sys.stdout.write(f"\r  Vanilla B=1: {v_str:<28} │  Cortheon v2: {c_str:<32}")
                    sys.stdout.flush()
                await asyncio.sleep(0.25)
            if is_tty:
                sys.stdout.write("\r" + " " * 86 + "\r")
                sys.stdout.flush()

        await asyncio.gather(_run_vanilla(), _run_cortheon(), _ticker())

        v_badge = "\033[1;32mPASS\033[0m" if state["v_ok"] else "\033[1;31mFAIL\033[0m"
        c_badge = "\033[1;32mPASS\033[0m" if state["c_ok"] else "\033[1;31mFAIL\033[0m"

        print("┌───────────────────────────────────────┬───────────────────────────────────────────────┐")
        print("│ VANILLA (B=1 single pass)             │ CORTHEON v2 (B=4 verified + early-abort)      │")
        print("├───────────────────────────────────────┼───────────────────────────────────────────────┤")
        print(f"│ Result : {v_badge}                         │ Result    : {c_badge}                              │")
        print(f"│ Wall   : {state['v_s']:>5.1f}s                       │ Wall      : {state['c_s']:>5.1f}s                            │")
        print(f"│ Tokens : {state['v_tok']:>7d}                      │ Tokens    : {state['c_tok']:>7d}                           │")
        print(f"│ Answer : {str(state['v_ans'] or '(none)')[:27]:<27}  │ Consensus : {str(state['c_cons'])[:32]:<32}  │")
        print(f"│                                       │ Aborted   : {state['c_aborted']} straggler GPU slot(s)           │")
        print("└───────────────────────────────────────┴───────────────────────────────────────────────┘")
        if exp:
            print(f"  Expected : {exp}")
            print(f"  Vanilla  : {state['v_ans'] or '(none)'}")
            print(f"  Cortheon : {state['c_ans'] or '(none)'}\n")

        return DuelOutcome(
            title=title,
            expected=exp,
            vanilla_answer=str(state["v_ans"]),
            vanilla_ok=bool(state["v_ok"]),
            vanilla_wall_s=float(state["v_s"]),
            vanilla_tokens=int(state["v_tok"]),
            cortheon_answer=str(state["c_ans"]),
            cortheon_ok=bool(state["c_ok"]),
            cortheon_wall_s=float(state["c_s"]),
            cortheon_tokens=int(state["c_tok"]),
            cortheon_consensus=str(state["c_cons"]),
            cortheon_early_aborted=int(state["c_aborted"]),
        )

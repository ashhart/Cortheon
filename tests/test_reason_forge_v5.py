"""Offline unit tests for ReasonForge v5:
- Fast Pair Stage 1A (`pal_exact` + `pal_skeptic`) 2-call exit
- Zero-anchoring Executable TIR Self-Repair (fixes SyntaxError / missing import / range violation in-place)
- Bounded integer range format gate & concise prose extraction (rejects out-of-range negatives & prose dumps)
- Stage 1B full Wave-1 completion (`pal_alt_z3` + `analytical`) + streaming early-abort on 3-call confirmation
- Battle-tested v2 `1 PAL + 1 analytical` Wave-1 consensus skipping Wave 2
- Strictly unanchored `falsifier_reread` & `falsifier_numeric_z3` + streaming Wave-2 early-abort
- Streaming Wave-3 early-abort (`w3_pal_numeric_exact` + clean-room deep proofs) on 0-PAL problems
- 1-call `reason_exec_arbiter` tie-breaker
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from flash_forge.client import CallSpec, CompletionResult
from flash_forge.memory import EpisodicMemory
from flash_forge.reason_forge import (
    _answer_satisfies_prompt_format,
    extract_answer_from_prose,
    solve_reasoning,
)
from flash_forge.skill_loader import SkillPack


def _mk_res(content: str, *, tag: str = "", tokens: int = 80, latency_s: float = 0.02) -> CompletionResult:
    return CompletionResult(
        content=content,
        reasoning="",
        finish_reason="stop",
        prompt_tokens=100,
        completion_tokens=tokens,
        reasoning_tokens=10,
        cached_tokens=0,
        latency_s=latency_s,
        tag=tag,
    )


class _ScriptedReasonClient:
    def __init__(self, handler: Callable[[CallSpec], Any]) -> None:
        self._handler = handler
        self.seen_specs: list[CallSpec] = []

    async def complete(self, spec: CallSpec) -> CompletionResult:
        self.seen_specs.append(spec)
        out = self._handler(spec)
        if asyncio.iscoroutine(out):
            return await out
        return out


def test_format_gate_bounds_and_clean_prose_extraction():
    aime_p = "Find the remainder when N is divided by 1000. Give your final answer as a single exact integer (0 to 999)."
    assert _answer_satisfies_prompt_format(aime_p, "248") is True
    assert _answer_satisfies_prompt_format(aime_p, "0") is True
    assert _answer_satisfies_prompt_format(aime_p, "999") is True
    assert _answer_satisfies_prompt_format(aime_p, "-376") is False
    assert _answer_satisfies_prompt_format(aime_p, "1000") is False
    assert _answer_satisfies_prompt_format(aime_p, "433772") is False
    assert _answer_satisfies_prompt_format(aime_p, "7 0.442398225673303430863681564776") is False
    assert _answer_satisfies_prompt_format(aime_p, "248.5") is False
    assert _answer_satisfies_prompt_format(aime_p, "nan") is False

    generic_p = "Find the exact value of x."
    assert _answer_satisfies_prompt_format(generic_p, "7 0.44239822567") is False
    assert _answer_satisfies_prompt_format(generic_p, "248") is True

    markov_p = "Find the exact rational expected number of ticks until all 7 servers are visited."
    assert _answer_satisfies_prompt_format(markov_p, "2630600114653531854763423/183488738276457059168800") is True

    pair_p = "Output the two numbers as space-separated integers."
    assert _answer_satisfies_prompt_format(pair_p, "7 11") is True

    # Rambling scratchpad after 'answer:' MUST NOT be extracted; prefer clean \boxed{60}
    rambling = (
        "Wait: answer: this is 2024 AIME II Problem 14? Let me recall the actual problem...\n"
        "After solving carefully, we obtain \\boxed{60}."
    )
    assert extract_answer_from_prose(rambling) == "60"


def test_stage1a_fast_pair_exits_in_exact_2_calls(tmp_path: Path):
    def _handle(spec: CallSpec) -> CompletionResult:
        return _mk_res("```python\nans = 7 * 10\nassert ans == 70\nprint('ANSWER:', ans)\n```", tag=spec.tag)

    client = _ScriptedReasonClient(_handle)
    trace = asyncio.run(
        solve_reasoning(
            client,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            "Compute 7 * 10. End with `ANSWER: <integer>`.",
            num_wave1_streams=4,
        )
    )
    assert trace.final_answer == "70"
    assert trace.verified is True
    assert trace.wave2_triggered is False
    assert [s.tag for s in client.seen_specs] == ["w1_pal_exact", "w1_pal_skeptic"]


def test_tir_self_repair_fixes_syntax_and_range_violation_in_place_without_anchoring(tmp_path: Path):
    """When pal_skeptic crashes with SyntaxError (or prints an out-of-range value like -376),
    TIR self-repair fixes ONLY its own script (never seeing pal_exact's answer!) and hits
    Stage-1A consensus in 3 calls without launching Stage 1B or Wave 2."""

    def _handle(spec: CallSpec) -> CompletionResult:
        if spec.tag == "w1_pal_exact":
            return _mk_res("```python\nans = 240\nassert ans == 240\nprint('ANSWER:', ans)\n```", tag=spec.tag)
        if spec.tag == "w1_pal_skeptic":
            # Out-of-range negative integer triggers bounded format gate -> TIR repair!
            return _mk_res("```python\nprint('debug_sum:', 240)\nprint('ANSWER: -376')\n```", tag=spec.tag)
        if spec.tag == "w1_pal_skeptic_fix1":
            user_prompt = spec.messages[-1]["content"]
            # Sees its OWN stdout + format rejection reason, NEVER leaks pal_exact's code
            assert "outside the required range [0, 999]" in user_prompt
            assert "debug_sum: 240" in user_prompt
            return _mk_res("```python\nans = 240\nassert 0 <= ans <= 999\nprint('ANSWER:', ans)\n```", tag=spec.tag)
        raise AssertionError(f"Unexpected extra call after TIR repair consensus: {spec.tag}")

    client = _ScriptedReasonClient(_handle)
    trace = asyncio.run(
        solve_reasoning(
            client,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            "Find the sum of the three positive real numbers k. Give your final answer as a single exact integer (0 to 999).",
            num_wave1_streams=4,
            max_pal_repairs=1,
        )
    )
    assert trace.final_answer == "240"
    assert trace.verified is True
    assert trace.wave2_triggered is False
    assert [s.tag for s in client.seen_specs] == ["w1_pal_exact", "w1_pal_skeptic", "w1_pal_skeptic_fix1"]


def test_stage1b_confirms_single_pal_and_cancels_lagging_analytical(tmp_path: Path):
    cancelled: list[str] = []

    async def _handle(spec: CallSpec) -> CompletionResult:
        if spec.tag == "w1_pal_exact":
            return _mk_res("```python\nprint('ANSWER: 117')\n```", tag=spec.tag)
        if spec.tag == "w1_pal_skeptic":
            return _mk_res("```python\nraise ZeroDivisionError('bad step 999')\n```", tag=spec.tag)
        if spec.tag == "w1_pal_alt_z3":
            user_prompt = spec.messages[-1]["content"]
            assert "117" not in user_prompt
            assert "ZeroDivisionError" in user_prompt
            await asyncio.sleep(0.01)
            return _mk_res("```python\nans = 9 * 13\nassert ans == 117\nprint('ANSWER:', ans)\n```", tag=spec.tag)
        if spec.tag == "w1_analytical":
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                cancelled.append(spec.tag)
                raise
            return _mk_res("ANSWER: 117", tag=spec.tag)
        raise AssertionError(f"Unexpected extra call: {spec.tag}")

    client = _ScriptedReasonClient(_handle)
    trace = asyncio.run(
        solve_reasoning(
            client,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            "Find 9 * 13. End with `ANSWER: <integer>`.",
            num_wave1_streams=4,
            max_pal_repairs=0,
        )
    )
    assert trace.final_answer == "117"
    assert trace.verified is True
    assert trace.wave2_triggered is False
    assert cancelled == ["w1_analytical"]


def test_stage1b_single_pal_plus_analytical_consensus_skips_wave2(tmp_path: Path):
    def _handle(spec: CallSpec) -> CompletionResult:
        if spec.tag == "w1_pal_exact":
            return _mk_res("```python\nassert 588 == 588\nprint('ANSWER: 588')\n```", tag=spec.tag)
        if spec.tag in ("w1_pal_skeptic", "w1_pal_alt_z3"):
            return _mk_res("```python\nraise ValueError('timeout proxy')\n```", tag=spec.tag)
        if spec.tag == "w1_analytical":
            return _mk_res("Clean derivation gives ANSWER: 588", tag=spec.tag)
        raise AssertionError(f"Wave 2 should not fire when 1 PAL + 1 analytical agree! Got: {spec.tag}")

    client = _ScriptedReasonClient(_handle)
    trace = asyncio.run(
        solve_reasoning(
            client,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            "Find the exact integer answer.",
            num_wave1_streams=4,
            max_pal_repairs=0,
        )
    )
    assert trace.final_answer == "588"
    assert trace.verified is True
    assert trace.wave2_triggered is False


def test_disagreement_completes_wave1_runs_unanchored_wave2_and_cancels_straggler(tmp_path: Path):
    cancelled: list[str] = []

    async def _handle(spec: CallSpec) -> CompletionResult:
        if spec.tag == "w1_pal_exact":
            return _mk_res("```python\nprint('ANSWER: 74')\n```", tag=spec.tag)
        if spec.tag == "w1_pal_skeptic":
            return _mk_res("```python\nprint('ANSWER: 77')\n```", tag=spec.tag)
        if spec.tag == "w1_pal_alt_z3":
            return _mk_res("```python\nraise ValueError('crash')\n```", tag=spec.tag)
        if spec.tag == "w1_analytical":
            return _mk_res("Analytical derivation gives ANSWER: 77", tag=spec.tag)
        if spec.tag == "w2_falsifier_reread":
            user_prompt = spec.messages[-1]["content"]
            assert "74" not in user_prompt and "77" not in user_prompt
            await asyncio.sleep(0.01)
            return _mk_res("```python\nans = 7 * 11\nassert ans == 77\nprint('ANSWER:', ans)\n```", tag=spec.tag)
        if spec.tag == "w2_falsifier_numeric_z3":
            user_prompt = spec.messages[-1]["content"]
            assert "74" not in user_prompt and "77" not in user_prompt
            assert "mpmath" in user_prompt
            await asyncio.sleep(0.02)
            return _mk_res("```python\nans = 77\nassert ans == 7 * 11\nprint('ANSWER:', ans)\n```", tag=spec.tag)
        if spec.tag == "w2_falsifier_bruteforce":
            user_prompt = spec.messages[-1]["content"]
            assert "74" in user_prompt and "77" in user_prompt
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                cancelled.append(spec.tag)
                raise
            return _mk_res("```python\nprint('ANSWER: 77')\n```", tag=spec.tag)
        raise AssertionError(f"Unexpected call: {spec.tag}")

    client = _ScriptedReasonClient(_handle)
    trace = asyncio.run(
        solve_reasoning(
            client,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            "Compute 7 * 11. End with `ANSWER: <integer>`.",
            num_wave1_streams=4,
            max_pal_repairs=0,
        )
    )
    assert trace.final_answer == "77"
    assert trace.verified is True
    assert trace.wave2_triggered is True
    assert trace.recovered_from_disagreement is True
    assert cancelled == ["w2_falsifier_bruteforce"]
    seen_tags = {s.tag for s in client.seen_specs}
    assert {"w1_pal_exact", "w1_pal_skeptic", "w1_pal_alt_z3", "w1_analytical"}.issubset(seen_tags)


def test_zero_pal_wave3_streaming_early_aborts_on_numeric_plus_proof_agreement(tmp_path: Path):
    cancelled: list[str] = []

    async def _handle(spec: CallSpec) -> CompletionResult:
        if spec.tag in (
            "w1_pal_exact",
            "w1_pal_skeptic",
            "w1_pal_alt_z3",
            "w2_falsifier_bruteforce",
            "w2_falsifier_reread",
            "w2_falsifier_numeric_z3",
        ):
            return _mk_res("```python\nraise TimeoutError('sympy hung')\n```", tag=spec.tag)
        if spec.tag == "w1_analytical":
            return _mk_res("No final line emitted.", tag=spec.tag)
        if spec.tag == "w3_pal_numeric_exact":
            user_prompt = spec.messages[-1]["content"]
            assert "mpmath" in user_prompt
            await asyncio.sleep(0.01)
            return _mk_res(
                "```python\nimport mpmath as mp\nmp.mp.dps = 50\nval = mp.mpf(60)\nassert abs(val - 60) < 1e-30\nprint('ANSWER: 60')\n```",
                tag=spec.tag,
            )
        if spec.tag == "w3_analytical_deep_a":
            user_prompt = spec.messages[-1]["content"]
            assert "GRAVEYARD" not in user_prompt
            await asyncio.sleep(0.02)
            return _mk_res("Rigorous proof yields ANSWER: 60", tag=spec.tag)
        if spec.tag == "w3_analytical_deep_b":
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                cancelled.append(spec.tag)
                raise
            return _mk_res("Alternative proof yields ANSWER: 60", tag=spec.tag)
        raise AssertionError(f"Unexpected call: {spec.tag}")

    client = _ScriptedReasonClient(_handle)
    trace = asyncio.run(
        solve_reasoning(
            client,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            "Find the least integer value of m + n + p.",
            num_wave1_streams=4,
            max_pal_repairs=0,
        )
    )
    assert trace.final_answer == "60"
    assert trace.verified is True
    assert trace.wave2_triggered is True
    assert cancelled == ["w3_analytical_deep_b"]


def test_close_tie_uses_single_call_executable_arbiter_with_tir_repair(tmp_path: Path):
    def _handle(spec: CallSpec) -> CompletionResult:
        if spec.tag.startswith((
            "w1_pal_exact",
            "w1_pal_skeptic",
            "w1_pal_alt_z3",
            "w2_falsifier_bruteforce",
            "w2_falsifier_reread",
            "w2_falsifier_numeric_z3",
            "w3_pal_numeric_exact",
        )):
            return _mk_res("```python\nraise ValueError('failed')\n```", tag=spec.tag)
        if spec.tag == "w1_analytical":
            return _mk_res("No final line.", tag=spec.tag)
        if spec.tag == "w3_analytical_deep_a":
            return _mk_res("Proof A gives ANSWER: 120", tag=spec.tag)
        if spec.tag == "w3_analytical_deep_b":
            return _mk_res("Proof B dividing by 2x symmetry gives ANSWER: 60", tag=spec.tag)
        if spec.tag == "reason_exec_arbiter":
            # First try hits a 1-line typo after printing diagnostic stdout
            return _mk_res("```python\nprint('half =', 120 // 2)\nprint(undefined_var)\n```", tag=spec.tag)
        if spec.tag == "reason_exec_arbiter_fix1":
            user_prompt = spec.messages[-1]["content"]
            assert "half = 60" in user_prompt
            assert "NameError" in user_prompt
            return _mk_res("```python\nans = 120 // 2\nassert ans == 60\nprint('ANSWER:', ans)\n```", tag=spec.tag)
        raise AssertionError(f"Unexpected call: {spec.tag}")

    client = _ScriptedReasonClient(_handle)
    trace = asyncio.run(
        solve_reasoning(
            client,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            "Find the number of unordered configurations.",
            num_wave1_streams=4,
            max_pal_repairs=1,
        )
    )
    assert trace.final_answer == "60"
    assert trace.verified is True
    assert trace.consensus_type == "exec_arbiter_pal"
    assert [s.tag for s in client.seen_specs].count("reason_exec_arbiter_fix1") == 1


def test_suspect_empty_zero_guard_prevents_false_wave1_exit(tmp_path: Path):
    """Two PAL scripts returning `sum([]) == 0` on a non-remainder problem MUST NOT fast-abort
    Wave 1; instead Stage 1B + Wave 2 verify the real positive answer."""

    def _handle(spec: CallSpec) -> CompletionResult:
        if spec.tag in ("w1_pal_exact", "w1_pal_skeptic", "w1_pal_alt_z3"):
            return _mk_res("```python\nsols = []\nprint('ANSWER:', sum(sols))\n```", tag=spec.tag)
        if spec.tag == "w1_analytical":
            return _mk_res("The three positive values sum to ANSWER: 240", tag=spec.tag)
        # Wave 2 unanchored falsifiers recover the true answer 240
        return _mk_res("```python\nans = 240\nassert ans == 240\nprint('ANSWER:', ans)\n```", tag=spec.tag)

    client = _ScriptedReasonClient(_handle)
    trace = asyncio.run(
        solve_reasoning(
            client,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            "There are three positive real numbers k. Find the sum of these three values of k. Give your final answer as a single exact integer (0 to 999).",
            num_wave1_streams=4,
            max_pal_repairs=0,
        )
    )
    assert trace.wave2_triggered is True
    assert trace.final_answer == "240"
    assert trace.verified is True


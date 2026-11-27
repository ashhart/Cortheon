"""Comprehensive offline unit tests for Speculative Execution-Verification (`SpecVerify`).

Uses REAL OS-sandbox execution (`flash_forge.sandbox.run_python`) with mocked
drafter and verifier clients to verify:
  - Accepted fast path (25-digit exact rationals, integers, Wave-1 early-abort, verifier_calls == 0)
  - 25-digit exact rational ULP discrimination (no float64 false consensus)
  - Disagreement escalation + non-anchoring escalated prompt + 2-vs-2 tie rejection
  - Crash / SyntaxError / OS-sandbox timeout escalation
  - Format-gate rejection (float/prose rejected when integer required)
  - Single-script consensus insufficient (`min_consensus=2`)
  - Distinct-solver diversity gate (`require_distinct_solvers=True`)
  - Exact client stats delta accounting
  - 3-arm SpecVerify ablation runner, summary math, Wilson CIs, McNemar p-values,
    Markdown table formatting, PII scrubbing, atomic `.tmp` writes, and `--merge`.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from bench.run_spec_verify_ablation import (
    ALL_ARMS,
    build_spec_verify_payload,
    format_spec_verify_table,
    load_aime2025,
    main as spec_ablation_main,
    merge_spec_verify_files,
    run_arm_cortheon_big,
    run_arm_spec_verify,
    run_arm_vanilla_big,
    scrub_pii_text,
)
from bench.stats import mcnemar_exact_p, wilson_ci
from flash_forge.client import CallSpec, CompletionResult, ModelCapabilities
from flash_forge.memory import EpisodicMemory
from flash_forge.reason_forge import BranchOutcome
from flash_forge.sandbox import ExecResult
from flash_forge.skill_loader import SkillPack
from flash_forge.spec_verify import (
    SpecVerifyTrace,
    evaluate_drafter_consensus,
    solve_spec_verify,
)


class _MockClient:
    """Deterministic async mock matching TensorFoldClient's interface and stats accounting."""

    def __init__(
        self,
        model_id: str,
        handler: Callable[[CallSpec], Any],
    ) -> None:
        self._caps = ModelCapabilities(model_id=model_id, structures=True, thinking_budget=True)
        self.requested_model = model_id
        self._handler = handler
        self.seen_specs: list[CallSpec] = []
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_cached_tokens = 0

    @property
    def caps(self) -> ModelCapabilities:
        return self._caps

    async def probe(self) -> ModelCapabilities:
        return self._caps

    def reset_stats(self) -> None:
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_cached_tokens = 0

    def stats_dict(self) -> dict[str, int]:
        return {
            "calls": self.total_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "reasoning_tokens": self.total_reasoning_tokens,
            "cached_tokens": self.total_cached_tokens,
        }

    async def complete(self, spec: CallSpec) -> CompletionResult:
        self.seen_specs.append(spec)
        res = self._handler(spec)
        if asyncio.iscoroutine(res):
            res = await res
        assert isinstance(res, CompletionResult)
        if res.ok:
            self.total_calls += 1
            self.total_prompt_tokens += res.prompt_tokens
            self.total_completion_tokens += res.completion_tokens
            self.total_reasoning_tokens += res.reasoning_tokens
            self.total_cached_tokens += res.cached_tokens
        return res

    async def batch(self, specs: list[CallSpec]) -> list[CompletionResult]:
        return list(await asyncio.gather(*(self.complete(s) for s in specs)))

    async def pairwise_vote(self, p_ab: str, p_ba: str, tag_prefix: str = "judge") -> str:
        return "A"


def _mk_res(
    content: str,
    tag: str = "",
    *,
    prompt_tokens: int = 120,
    completion_tokens: int = 80,
    reasoning_tokens: int = 20,
    latency_s: float = 0.01,
    raw_error: Optional[str] = None,
) -> CompletionResult:
    return CompletionResult(
        content=content,
        reasoning="",
        finish_reason="error" if raw_error else "stop",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=0,
        latency_s=latency_s,
        raw_error=raw_error,
        tag=tag,
    )


def _failing_verifier(model_id: str = "big-verifier-70b") -> _MockClient:
    def _boom(spec: CallSpec) -> CompletionResult:
        raise AssertionError(f"Big verifier must NOT be called on accepted drafter path (tag={spec.tag})")

    return _MockClient(model_id=model_id, handler=_boom)


def _agreeing_verifier(answer_expr: str, model_id: str = "big-verifier-70b") -> _MockClient:
    def _handle(spec: CallSpec) -> CompletionResult:
        if spec.tag.endswith("analytical"):
            return _mk_res(f"By exact derivation, ANSWER: {answer_expr}", tag=spec.tag, prompt_tokens=400, completion_tokens=300)
        return _mk_res(
            f"```python\nval = {answer_expr!r}\nassert val\nprint('ANSWER:', val)\n```",
            tag=spec.tag,
            prompt_tokens=500,
            completion_tokens=400,
        )

    return _MockClient(model_id=model_id, handler=_handle)


def test_accepted_25_digit_exact_rational_and_early_abort(tmp_path: Path):
    """2 distinct drafter scripts agree on a 25-digit exact rational -> accepted immediately, 0 verifier calls, 3rd cancelled."""
    num = 2630600114653531854763423
    den = 183488738276457059168800
    expected = f"{num}/{den}"

    async def _draft(spec: CallSpec) -> CompletionResult:
        if spec.tag == "spec_draft_pal_exact":
            await asyncio.sleep(0.005)
            code = (
                "```python\n"
                "from fractions import Fraction\n"
                f"ans = Fraction({num}, {den})\n"
                "assert ans.numerator > 10**24\n"
                "print(f'ANSWER: {ans.numerator}/{ans.denominator}')\n"
                "```"
            )
            return _mk_res(code, tag=spec.tag, prompt_tokens=150, completion_tokens=90)
        if spec.tag == "spec_draft_pal_alt_z3":
            await asyncio.sleep(0.015)
            # Unreduced 2x numerator & denominator to prove exact Fraction reduction & equivalence!
            code = (
                "```python\n"
                "from fractions import Fraction\n"
                f"ans = Fraction({num * 2}, {den * 2})\n"
                "assert ans > 0\n"
                "print('ANSWER:', repr(ans))\n"
                "```"
            )
            return _mk_res(code, tag=spec.tag, prompt_tokens=160, completion_tokens=100)
        # Lagging 3rd drafter slot should be cancelled by Wave-1 streaming early-abort!
        await asyncio.sleep(5.0)
        return _mk_res("```python\nprint('ANSWER: 0')\n```", tag=spec.tag, prompt_tokens=160, completion_tokens=100)

    drafter = _MockClient("qwen3-4b-draft", _draft)
    verifier = _failing_verifier("glm-5.3-big")
    skills = SkillPack()
    mem = EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False)

    t0 = time.perf_counter()
    trace = asyncio.run(
        solve_spec_verify(
            drafter,  # type: ignore[arg-type]
            verifier,  # type: ignore[arg-type]
            skills,
            mem,
            "Find the exact rational expected cover time E[T]. End with `ANSWER: <p/q>`.",
            drafter_streams=3,
            min_consensus=2,
        )
    )
    elapsed = time.perf_counter() - t0

    assert isinstance(trace, SpecVerifyTrace)
    assert trace.accepted_by_drafter is True
    assert trace.escalated is False
    assert trace.escalation_reason == ""
    assert trace.answer == expected
    assert trace.verifier_calls == 0
    assert trace.verifier_tokens == 0
    assert trace.verifier_wall_s == 0.0
    assert trace.verifier_trace is None
    assert len(verifier.seen_specs) == 0

    assert trace.drafter_model == "qwen3-4b-draft"
    assert trace.verifier_model == "glm-5.3-big"
    assert trace.drafter_consensus_count == 2
    assert trace.drafter_verified_scripts == 2
    assert trace.drafter_calls == 2
    assert trace.drafter_tokens == (150 + 90) + (160 + 100)
    assert trace.drafter_graveyard == []
    assert elapsed < 3.0, f"Early-abort should have cancelled the 5s lagging slot, took {elapsed:.2f}s"

    by_name = {c["name"]: c for c in trace.drafter_candidates}
    assert by_name["pal_exact"]["ok"] is True
    assert by_name["pal_alt_z3"]["ok"] is True
    assert by_name["pal_sympy"]["error"] == "cancelled_early_consensus"

    # Confirm crisp low-latency drafter specs
    for spec in drafter.seen_specs:
        assert spec.thinking_budget == 400
        assert spec.max_tokens == 2048


def test_accepted_exact_integer_zero_verifier_calls(tmp_path: Path):
    """Accepted integer path returns exact normalized integer with zero verifier calls."""

    def _draft(spec: CallSpec) -> CompletionResult:
        if spec.tag == "spec_draft_pal_exact":
            return _mk_res("```python\nassert 19 * 23 == 437\nprint('ANSWER: 437')\n```", tag=spec.tag)
        if spec.tag == "spec_draft_pal_alt_z3":
            return _mk_res("```python\nprint('ANSWER:', sum(23 for _ in range(19)))\n```", tag=spec.tag)
        return _mk_res("```python\nimport sympy as sp\nprint('ANSWER:', sp.Integer(19) * 23)\n```", tag=spec.tag)

    drafter = _MockClient("tiny-2b", _draft)
    verifier = _failing_verifier("big-70b")
    trace = asyncio.run(
        solve_spec_verify(
            drafter,  # type: ignore[arg-type]
            verifier,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            "Compute 19 * 23. Note: Every AIME answer is an integer from 000 to 999. End with `ANSWER: <integer>`.",
        )
    )
    assert trace.accepted_by_drafter is True
    assert trace.escalated is False
    assert trace.answer == "437"
    assert trace.verifier_calls == 0
    assert trace.verifier_tokens == 0
    assert len(verifier.seen_specs) == 0


def test_25_digit_rational_off_by_one_ulp_escalates(tmp_path: Path):
    """Planted control: two 25-digit rationals identical in float64 but differing in the 25th digit MUST NOT cluster."""
    num = 2630600114653531854763423
    den = 183488738276457059168800
    expected = f"{num}/{den}"

    def _draft(spec: CallSpec) -> CompletionResult:
        if spec.tag == "spec_draft_pal_exact":
            return _mk_res(f"```python\nprint('ANSWER: {num}/{den}')\n```", tag=spec.tag)
        if spec.tag == "spec_draft_pal_alt_z3":
            # Differs by 1 in the 25th digit — float64 cannot distinguish them!
            return _mk_res(f"```python\nprint('ANSWER: {num + 1}/{den}')\n```", tag=spec.tag)
        return _mk_res("```python\nraise RuntimeError('sympy failed')\n```", tag=spec.tag)

    drafter = _MockClient("tiny-4b", _draft)
    verifier = _agreeing_verifier(expected)
    trace = asyncio.run(
        solve_spec_verify(
            drafter,  # type: ignore[arg-type]
            verifier,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            "Find the exact rational value. End with `ANSWER: <p/q>`.",
        )
    )
    assert trace.accepted_by_drafter is False
    assert trace.escalated is True
    assert trace.verifier_calls > 0
    assert trace.answer == expected


def test_disagreement_path_escalates_without_anchoring_wrong_numbers(tmp_path: Path):
    """Conflicting drafter outputs (42 vs 43 vs 44) -> rejected -> escalates to verifier without leaking wrong numbers."""

    def _draft(spec: CallSpec) -> CompletionResult:
        mapping = {
            "spec_draft_pal_exact": "```python\nprint('ANSWER: 42')\n```",
            "spec_draft_pal_alt_z3": "```python\nprint('ANSWER: 43')\n```",
            "spec_draft_pal_sympy": "```python\nprint('ANSWER: 44')\n```",
        }
        return _mk_res(mapping[spec.tag], tag=spec.tag, prompt_tokens=100, completion_tokens=50)

    drafter = _MockClient("tiny-4b", _draft)
    verifier = _agreeing_verifier("43", model_id="frontier-big")

    trace = asyncio.run(
        solve_spec_verify(
            drafter,  # type: ignore[arg-type]
            verifier,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            "Solve the counting problem. End with `ANSWER: <integer>`.",
        )
    )

    assert trace.accepted_by_drafter is False
    assert trace.escalated is True
    assert "disagreed" in trace.escalation_reason.lower()
    assert trace.verifier_calls > 0
    assert trace.verifier_tokens > 0
    assert trace.verifier_trace is not None
    assert trace.answer == "43"
    assert any("DISAGREEMENT" in g for g in trace.drafter_graveyard)

    # Verify the escalated prompt sent to the big verifier NEVER leaks wrong candidate numbers!
    assert len(verifier.seen_specs) > 0
    verifier_user_prompt = verifier.seen_specs[0].messages[-1]["content"]
    assert "42" not in verifier_user_prompt
    assert "44" not in verifier_user_prompt
    assert "conflicting" in verifier_user_prompt.lower()


def test_tie_2_vs_2_rejected_and_escalated(tmp_path: Path):
    """A 2-vs-2 split across 4 drafter streams lacks strict unique majority and must escalate."""

    async def _draft(spec: CallSpec) -> CompletionResult:
        # Stagger 42, 43, 42, 43 so disagreement is seen before slot 3 completes
        schedule = {
            "spec_draft_pal_exact": (0.005, "42"),
            "spec_draft_pal_alt_z3": (0.015, "43"),
            "spec_draft_pal_sympy": (0.025, "42"),
            "spec_draft_pal_skeptic": (0.035, "43"),
        }
        delay, val = schedule[spec.tag]
        await asyncio.sleep(delay)
        return _mk_res(f"```python\nprint('ANSWER: {val}')\n```", tag=spec.tag)

    drafter = _MockClient("tiny-4b", _draft)
    verifier = _agreeing_verifier("99")

    trace = asyncio.run(
        solve_spec_verify(
            drafter,  # type: ignore[arg-type]
            verifier,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            "Compute the exact answer. End with `ANSWER: <integer>`.",
            drafter_streams=4,
            min_consensus=2,
        )
    )
    assert trace.accepted_by_drafter is False
    assert trace.escalated is True
    assert trace.drafter_verified_scripts == 4
    assert trace.drafter_consensus_count == 2
    assert trace.verifier_calls > 0
    assert trace.answer == "99"


def test_crash_and_syntax_error_escalates_cleanly(tmp_path: Path):
    """Runtime errors, syntax errors, and missing code blocks are falsified by sandbox and escalated."""

    def _draft(spec: CallSpec) -> CompletionResult:
        if spec.tag == "spec_draft_pal_exact":
            return _mk_res("```python\nx = 1 / 0\nprint('ANSWER:', x)\n```", tag=spec.tag)
        if spec.tag == "spec_draft_pal_alt_z3":
            return _mk_res("```python\ndef broken(:\n    pass\n```", tag=spec.tag)
        return _mk_res("I think the answer might be 17 without writing code.", tag=spec.tag)

    drafter = _MockClient("tiny-2b", _draft)
    verifier = _agreeing_verifier("256")

    trace = asyncio.run(
        solve_spec_verify(
            drafter,  # type: ignore[arg-type]
            verifier,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            "Compute 2^8. End with `ANSWER: <integer>`.",
        )
    )
    assert trace.accepted_by_drafter is False
    assert trace.escalated is True
    assert trace.drafter_verified_scripts == 0
    assert trace.drafter_consensus_count == 0
    assert trace.verifier_calls > 0
    assert trace.answer == "256"
    assert len(trace.drafter_graveyard) >= 3
    assert any("ZeroDivisionError" in g for g in trace.drafter_graveyard)


def test_sandbox_timeout_escalates_cleanly(tmp_path: Path):
    """Infinite-loop drafter scripts are killed by the OS sandbox timeout and escalated."""

    def _draft(spec: CallSpec) -> CompletionResult:
        return _mk_res("```python\nwhile True:\n    pass\n```", tag=spec.tag)

    drafter = _MockClient("tiny-2b", _draft)
    verifier = _agreeing_verifier("512")

    trace = asyncio.run(
        solve_spec_verify(
            drafter,  # type: ignore[arg-type]
            verifier,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            "Compute 2^9. End with `ANSWER: <integer>`.",
            sandbox_timeout_s=0.4,
        )
    )
    assert trace.accepted_by_drafter is False
    assert trace.escalated is True
    assert trace.drafter_verified_scripts == 0
    assert all(c["timed_out"] is True for c in trace.drafter_candidates)
    assert any("TIMEOUT" in g for g in trace.drafter_graveyard)
    assert trace.verifier_calls > 0
    assert trace.answer == "512"


def test_format_gate_rejects_float_and_prose_when_integer_required(tmp_path: Path):
    """Drafter outputting float or prose when prompt demands an integer is rejected and escalated."""

    def _draft(spec: CallSpec) -> CompletionResult:
        if spec.tag in ("spec_draft_pal_exact", "spec_draft_pal_alt_z3"):
            return _mk_res("```python\nprint('ANSWER: 3.14159')\n```", tag=spec.tag)
        return _mk_res("```python\nprint('ANSWER: forty-two')\n```", tag=spec.tag)

    drafter = _MockClient("tiny-4b", _draft)
    verifier = _agreeing_verifier("314")

    trace = asyncio.run(
        solve_spec_verify(
            drafter,  # type: ignore[arg-type]
            verifier,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            "Find floor(100*pi). Every AIME answer is an integer from 000 to 999. End with `ANSWER: <integer>`.",
        )
    )
    assert trace.accepted_by_drafter is False
    assert trace.escalated is True
    assert trace.drafter_verified_scripts == 0
    assert all(c["format_valid"] is False for c in trace.drafter_candidates)
    assert any("FORMAT_REJECTED" in g for g in trace.drafter_graveyard)
    assert trace.verifier_calls > 0
    assert trace.answer == "314"


def test_single_script_consensus_insufficient_escalates(tmp_path: Path):
    """Only 1 passing drafter script is insufficient when `min_consensus=2` -> escalates."""

    def _draft(spec: CallSpec) -> CompletionResult:
        if spec.tag == "spec_draft_pal_exact":
            return _mk_res("```python\nprint('ANSWER: 77')\n```", tag=spec.tag)
        if spec.tag == "spec_draft_pal_alt_z3":
            return _mk_res("```python\nraise ValueError('bad state')\n```", tag=spec.tag)
        return _mk_res("```python\nprint('done working...')\n```", tag=spec.tag)

    drafter = _MockClient("tiny-4b", _draft)
    verifier = _agreeing_verifier("77")

    trace = asyncio.run(
        solve_spec_verify(
            drafter,  # type: ignore[arg-type]
            verifier,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            "Compute 7 * 11. End with `ANSWER: <integer>`.",
            min_consensus=2,
        )
    )
    assert trace.accepted_by_drafter is False
    assert trace.escalated is True
    assert trace.drafter_verified_scripts == 1
    assert trace.drafter_consensus_count == 1
    assert "insufficient" in trace.escalation_reason.lower()
    assert trace.verifier_calls > 0
    assert trace.answer == "77"


def test_distinct_solver_diversity_and_strict_majority_unit():
    """Verify `evaluate_drafter_consensus` enforces solver diversity and strict majority."""
    ok_exec = ExecResult(ok=True, returncode=0, stdout="ANSWER: 42\n", stderr="", elapsed_s=0.05)
    problem = "Find x. End with `ANSWER: <integer>`."

    same_role = [
        BranchOutcome(name="pal_exact", wave=1, kind="pal", raw_content="", reasoning="", exec_result=ok_exec, raw_answer="42", norm_answer="42"),
        BranchOutcome(name="pal_exact", wave=1, kind="pal", raw_content="", reasoning="", exec_result=ok_exec, raw_answer="42", norm_answer="42"),
    ]
    accepted_strict, _, cnt, ver, reason, _ = evaluate_drafter_consensus(
        same_role, problem, min_consensus=2, require_distinct_solvers=True
    )
    assert accepted_strict is False
    assert cnt == 2 and ver == 2
    assert "distinct solver" in reason.lower()

    accepted_relaxed, ans_relaxed, _, _, _, _ = evaluate_drafter_consensus(
        same_role, problem, min_consensus=2, require_distinct_solvers=False
    )
    assert accepted_relaxed is True
    assert ans_relaxed == "42"

    # 2 distinct solvers on 42 vs 1 solver on 43 -> strict unique majority (2 > 1)
    two_vs_one = [
        BranchOutcome(name="pal_exact", wave=1, kind="pal", raw_content="", reasoning="", exec_result=ok_exec, raw_answer="42", norm_answer="42"),
        BranchOutcome(name="pal_alt_z3", wave=1, kind="pal", raw_content="", reasoning="", exec_result=ok_exec, raw_answer="42", norm_answer="42"),
        BranchOutcome(name="pal_sympy", wave=1, kind="pal", raw_content="", reasoning="", exec_result=ok_exec, raw_answer="43", norm_answer="43"),
    ]
    acc_21, ans_21, cnt_21, ver_21, _, _ = evaluate_drafter_consensus(
        two_vs_one, problem, min_consensus=2, require_distinct_solvers=True
    )
    assert acc_21 is True
    assert ans_21 == "42"
    assert cnt_21 == 2 and ver_21 == 3


def test_exact_stats_delta_with_preexisting_counters(tmp_path: Path):
    """Token and call counts are computed strictly as before/after deltas."""

    def _draft(spec: CallSpec) -> CompletionResult:
        return _mk_res("```python\nraise ValueError('force escalate')\n```", tag=spec.tag, prompt_tokens=111, completion_tokens=22)

    drafter = _MockClient("tiny-4b", _draft)
    drafter.total_calls = 10
    drafter.total_prompt_tokens = 5000
    drafter.total_completion_tokens = 1000

    verifier = _agreeing_verifier("88")
    verifier.total_calls = 7
    verifier.total_prompt_tokens = 9000
    verifier.total_completion_tokens = 3000

    trace = asyncio.run(
        solve_spec_verify(
            drafter,  # type: ignore[arg-type]
            verifier,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False),
            "Compute 8 * 11. End with `ANSWER: <integer>`.",
            drafter_streams=3,
            verifier_streams=2,
        )
    )
    assert trace.escalated is True
    assert trace.drafter_calls == 3
    assert trace.drafter_tokens == 3 * (111 + 22)
    assert trace.verifier_calls == verifier.total_calls - 7
    assert trace.verifier_tokens == (verifier.total_prompt_tokens - 9000) + (verifier.total_completion_tokens - 3000)
    assert trace.verifier_calls > 0
    assert trace.verifier_tokens > 0


def test_ablation_three_arms_end_to_end_and_summary(tmp_path: Path):
    """Run all 3 ablation arms end-to-end with mocked clients + real OS sandbox on a synthetic AIME fixture."""
    dataset_path = tmp_path / "aime_2025.json"
    dataset_path.write_text(
        json.dumps(
            [
                {"problem_idx": 0, "problem": "Find the sum of all integers from 1 to 10.", "answer": 55},
                {"problem_idx": 1, "problem": "Find 12 * 13.", "answer": 156},
            ]
        ),
        encoding="utf-8",
    )
    items = load_aime2025(dataset_path, start=0, end=2)
    assert len(items) == 2

    exp0 = items[0]["expected"]
    exp1 = items[1]["expected"]

    # Drafter gets item 0 right (2 agreeing scripts) and crashes on item 1 (escalates!)
    current_expected = {"val": exp0, "draft_pass": True}

    def _draft(spec: CallSpec) -> CompletionResult:
        if current_expected["draft_pass"]:
            return _mk_res(
                f"```python\nprint('ANSWER: {current_expected['val']}')\n```",
                tag=spec.tag,
                prompt_tokens=200,
                completion_tokens=100,
            )
        return _mk_res("```python\nraise RuntimeError('too hard for tiny drafter')\n```", tag=spec.tag, prompt_tokens=200, completion_tokens=100)

    def _verify(spec: CallSpec) -> CompletionResult:
        if spec.tag.startswith("sv_vanilla_big_"):
            # Vanilla big misses item 0, passes item 1
            ans = "999" if current_expected["val"] == exp0 else exp1
            return _mk_res(f"ANSWER: {ans}", tag=spec.tag, prompt_tokens=600, completion_tokens=400)
        if spec.tag.endswith("analytical"):
            return _mk_res(f"ANSWER: {current_expected['val']}", tag=spec.tag, prompt_tokens=700, completion_tokens=500)
        return _mk_res(
            f"```python\nprint('ANSWER: {current_expected['val']}')\n```",
            tag=spec.tag,
            prompt_tokens=700,
            completion_tokens=300,
        )

    drafter = _MockClient("qwen3-4b", _draft)
    verifier = _MockClient("glm-5.3-big", _verify)
    skills = SkillPack()

    results_by_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in ALL_ARMS}
    for idx, item in enumerate(items):
        current_expected["val"] = item["expected"]
        current_expected["draft_pass"] = idx == 0

        r_van = asyncio.run(run_arm_vanilla_big(verifier, item))  # type: ignore[arg-type]
        r_cor = asyncio.run(run_arm_cortheon_big(verifier, skills, item, verifier_streams=2))  # type: ignore[arg-type]
        r_sv = asyncio.run(
            run_arm_spec_verify(
                drafter,  # type: ignore[arg-type]
                verifier,  # type: ignore[arg-type]
                skills,
                item,
                drafter_streams=3,
                verifier_streams=2,
                baseline_big_tokens=r_cor["verifier_tokens"],
            )
        )
        results_by_arm["vanilla_big"].append(r_van)
        results_by_arm["cortheon_big"].append(r_cor)
        results_by_arm["spec_verify"].append(r_sv)

    # Item 0 was accepted by drafter (0 big tokens -> 100% saved); Item 1 escalated
    assert results_by_arm["spec_verify"][0]["accepted_by_drafter"] is True
    assert results_by_arm["spec_verify"][0]["verifier_tokens"] == 0
    assert results_by_arm["spec_verify"][0]["big_model_tokens_saved_pct"] == 100.0
    assert results_by_arm["spec_verify"][1]["accepted_by_drafter"] is False
    assert results_by_arm["spec_verify"][1]["escalated"] is True
    assert results_by_arm["spec_verify"][1]["verifier_tokens"] > 0

    payload = build_spec_verify_payload(
        drafter_model="qwen3-4b",
        verifier_model="glm-5.3-big",
        arms=list(ALL_ARMS),
        results_by_arm=results_by_arm,
        range_info={"start": 0, "end": 2},
    )
    sv_sum = payload["summary"]["spec_verify"]
    assert sv_sum["correct"] == 2 and sv_sum["total"] == 2
    assert sv_sum["drafter_accepted"] == 1
    assert sv_sum["drafter_accept_rate_pct"] == 50.0
    assert sv_sum["drafter_accepted_accuracy_pct"] == 100.0
    assert sv_sum["big_model_tokens_saved_vs_cortheon_big_pct"] == 50.0
    assert sv_sum["big_model_calls_saved_vs_cortheon_big_pct"] == 50.0

    md = format_spec_verify_table(payload)
    assert "SpecVerify Ablation" in md
    assert "`vanilla_big`" in md and "`cortheon_big`" in md and "`spec_verify`" in md
    assert "50.0%" in md


def test_ablation_merge_shards_disjoint_arms_stats_and_pii_scrubbing(tmp_path: Path):
    """Verify `--merge` across shards + disjoint arms computes exact Wilson CI, McNemar p, token savings, and scrubs PII."""

    def _big_row(pid: str, correct: bool, wall: float, calls: int, tok: int, detail: str = "ok") -> dict[str, Any]:
        return {
            "id": pid,
            "expected": "100",
            "predicted": "100" if correct else "0",
            "correct": correct,
            "wall_s": wall,
            "llm_calls": calls,
            "prompt_tokens": tok // 2,
            "completion_tokens": tok - tok // 2,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": tok,
            "drafter_calls": 0,
            "drafter_tokens": 0,
            "verifier_calls": calls,
            "verifier_tokens": tok,
            "detail": detail,
        }

    def _sv_row(
        pid: str,
        correct: bool,
        accepted: bool,
        d_calls: int,
        d_tok: int,
        v_calls: int,
        v_tok: int,
        detail: str = "ok",
    ) -> dict[str, Any]:
        return {
            "id": pid,
            "expected": "100",
            "predicted": "100" if correct else "0",
            "correct": correct,
            "accepted_by_drafter": accepted,
            "escalated": not accepted,
            "escalation_reason": "" if accepted else "crash at /Users/ash/secret/run.py",
            "drafter_consensus_count": 2 if accepted else 0,
            "drafter_verified_scripts": 2 if accepted else 0,
            "drafter_wall_s": 1.0,
            "verifier_wall_s": 0.0 if accepted else 8.0,
            "wall_s": 1.0 if accepted else 9.0,
            "drafter_calls": d_calls,
            "drafter_tokens": d_tok,
            "verifier_calls": v_calls,
            "verifier_tokens": v_tok,
            "llm_calls": d_calls + v_calls,
            "total_tokens": d_tok + v_tok,
            "big_model_tokens_saved_pct": 100.0 if accepted else None,
            "drafter_graveyard": [] if accepted else ["File /var/folders/ab/cd/T/ff_exec_123/_run.py Bearer sk-supersecret1234567890"],
            "detail": detail,
        }

    # 8 problems:
    # vanilla_big  passes: 2/8 (P0, P1)
    # cortheon_big passes: 8/8 (all 8)  -> 4 calls, 10000 tok each => 32 calls, 80000 big tok
    # spec_verify  passes: 8/8 (all 8)  -> 5 accepted (0 big calls/tok), 3 escalated (4 big calls, 10000 big tok each)
    #                                   => 12 big calls, 30000 big tok
    # Exact reductions vs cortheon_big:
    #   acceptance rate = 5/8 = 62.5%
    #   big_model_calls_saved_pct  = (1 - 12/32) * 100 = 62.5%
    #   big_model_tokens_saved_pct = (1 - 30000/80000) * 100 = 62.5%
    #   McNemar p (spec_verify vs vanilla_big): n01=6, n10=0 -> 2/64 = 0.03125
    #   McNemar p (spec_verify vs cortheon_big): n01=0, n10=0 -> 1.0

    # Shard 1: vanilla_big + cortheon_big for P0..P3
    shard_big_1 = build_spec_verify_payload(
        drafter_model="unknown",
        verifier_model="glm-5.3-big",
        arms=["vanilla_big", "cortheon_big"],
        results_by_arm={
            "vanilla_big": [
                _big_row("AIME25_00", True, 5.0, 1, 2000, "contact alice@example.com"),
                _big_row("AIME25_01", True, 5.0, 1, 2000),
                _big_row("AIME25_02", False, 5.0, 1, 2000),
                _big_row("AIME25_03", False, 5.0, 1, 2000),
            ],
            "cortheon_big": [
                _big_row(f"AIME25_0{i}", True, 12.0, 4, 10000) for i in range(4)
            ],
        },
        range_info={"start": 0, "end": 4},
    )
    # Shard 2: vanilla_big + cortheon_big for P4..P7
    shard_big_2 = build_spec_verify_payload(
        drafter_model="unknown",
        verifier_model="glm-5.3-big",
        arms=["vanilla_big", "cortheon_big"],
        results_by_arm={
            "vanilla_big": [
                _big_row(f"AIME25_0{i}", False, 5.0, 1, 2000) for i in range(4, 8)
            ],
            "cortheon_big": [
                _big_row(f"AIME25_0{i}", True, 12.0, 4, 10000) for i in range(4, 8)
            ],
        },
        range_info={"start": 4, "end": 8},
    )
    # Shard 3 (disjoint arm!): spec_verify for P0..P7
    shard_sv = build_spec_verify_payload(
        drafter_model="qwen3-4b-draft",
        verifier_model="glm-5.3-big",
        arms=["spec_verify"],
        results_by_arm={
            "spec_verify": [
                _sv_row(f"AIME25_0{i}", True, True, 2, 800, 0, 0, "ok") for i in range(5)
            ]
            + [
                _sv_row(
                    f"AIME25_0{i}",
                    True,
                    False,
                    3,
                    1200,
                    4,
                    10000,
                    "err /Users/someone/project/file.py sk-abcdef1234567890",
                )
                for i in range(5, 8)
            ],
        },
        range_info={"start": 0, "end": 8},
    )

    p1 = tmp_path / "big_s1.json"
    p2 = tmp_path / "big_s2.json"
    p3 = tmp_path / "sv_all.json"
    out_path = tmp_path / "merged.json"
    p1.write_text(json.dumps(shard_big_1), encoding="utf-8")
    p2.write_text(json.dumps(shard_big_2), encoding="utf-8")
    p3.write_text(json.dumps(shard_sv), encoding="utf-8")

    rc = spec_ablation_main(["--merge", str(p1), str(p2), str(p3), "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    assert not out_path.with_suffix(".json.tmp").exists()

    raw_text = out_path.read_text(encoding="utf-8")
    # Strict zero-PII & secret check across the entire persisted JSON!
    assert "/Users/" not in raw_text
    assert "/var/folders/" not in raw_text
    assert "sk-supersecret" not in raw_text
    assert "sk-abcdef" not in raw_text
    assert "alice@example.com" not in raw_text

    merged = json.loads(raw_text)
    assert merged["drafter_model"] == "qwen3-4b-draft"
    assert merged["verifier_model"] == "glm-5.3-big"
    assert merged["arms"] == ["vanilla_big", "cortheon_big", "spec_verify"]

    # Check exact Wilson 95% CIs
    v_lo, v_hi = wilson_ci(2, 8)
    assert merged["summary"]["vanilla_big"]["wilson_ci_95_pct"] == [round(100 * v_lo, 2), round(100 * v_hi, 2)]
    s_lo, s_hi = wilson_ci(8, 8)
    assert merged["summary"]["spec_verify"]["wilson_ci_95_pct"] == [round(100 * s_lo, 2), round(100 * s_hi, 2)]

    # Check exact acceptance rate and big-model call/token reduction
    sv_sum = merged["summary"]["spec_verify"]
    assert sv_sum["drafter_accepted"] == 5
    assert sv_sum["drafter_accept_rate_pct"] == 62.5
    assert sv_sum["big_model_calls_saved_pct"] == 62.5
    assert sv_sum["big_model_tokens_saved_pct"] == 62.5

    # Check per-row savings got refreshed against paired cortheon_big after disjoint-arm merge!
    sv_by_id = {r["id"]: r for r in merged["results"]["spec_verify"]}
    assert sv_by_id["AIME25_00"]["big_model_tokens_saved_pct"] == 100.0
    assert sv_by_id["AIME25_05"]["big_model_tokens_saved_pct"] == 0.0

    # Check exact two-sided McNemar p-values
    pw_v = merged["pairwise"]["spec_verify_vs_vanilla_big"]
    assert (pw_v["n11"], pw_v["n01"], pw_v["n10"], pw_v["n00"]) == (2, 6, 0, 0)
    assert math.isclose(pw_v["mcnemar_exact_p"], mcnemar_exact_p(6, 0), rel_tol=1e-12)
    assert math.isclose(pw_v["mcnemar_exact_p"], 0.03125, rel_tol=1e-12)

    pw_c = merged["pairwise"]["spec_verify_vs_cortheon_big"]
    assert (pw_c["n11"], pw_c["n01"], pw_c["n10"], pw_c["n00"]) == (8, 0, 0, 0)
    assert pw_c["mcnemar_exact_p"] == 1.0
    assert pw_c["big_model_calls_saved_pct"] == 62.5
    assert pw_c["big_model_tokens_saved_pct"] == 62.5

    # Direct scrub_pii_text unit check (redacts PII/secrets without mangling normal identifiers)
    scrubbed = scrub_pii_text(
        "platform_specific_error task_execution_failed /home/bob/code.py Bearer tok_1234567890 bob@corp.io"
    )
    assert "platform_specific_error" in scrubbed
    assert "task_execution_failed" in scrubbed
    assert "/home/bob" not in scrubbed
    assert "tok_1234567890" not in scrubbed
    assert "bob@corp.io" not in scrubbed


def test_drafter_endpoint_exception_and_nonfinite_escalate_cleanly(tmp_path: Path):
    """Drafter server outage or non-finite outputs (nan/inf) gracefully escalate to the big verifier."""

    def _down_drafter(spec: CallSpec) -> CompletionResult:
        raise ConnectionError("drafter server unavailable")

    drafter_down = _MockClient("tiny-down", _down_drafter)
    verifier = _agreeing_verifier("64")
    trace_down = asyncio.run(
        solve_spec_verify(
            drafter_down,  # type: ignore[arg-type]
            verifier,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem1.db", enabled=False),
            "Compute 8 * 8. End with `ANSWER: <integer>`.",
        )
    )
    assert trace_down.accepted_by_drafter is False
    assert trace_down.escalated is True
    assert trace_down.answer == "64"
    assert trace_down.verifier_calls > 0
    assert any("ConnectionError" in g for g in trace_down.drafter_graveyard)

    def _nan_drafter(spec: CallSpec) -> CompletionResult:
        return _mk_res("```python\nprint('ANSWER:', float('nan'))\n```", tag=spec.tag)

    drafter_nan = _MockClient("tiny-nan", _nan_drafter)
    verifier2 = _agreeing_verifier("125")
    trace_nan = asyncio.run(
        solve_spec_verify(
            drafter_nan,  # type: ignore[arg-type]
            verifier2,  # type: ignore[arg-type]
            SkillPack(),
            EpisodicMemory(db_path=tmp_path / "mem2.db", enabled=False),
            "Compute 5^3.",
        )
    )
    assert trace_nan.accepted_by_drafter is False
    assert trace_nan.escalated is True
    assert trace_nan.drafter_verified_scripts == 0
    assert trace_nan.answer == "125"
    assert trace_nan.verifier_calls > 0


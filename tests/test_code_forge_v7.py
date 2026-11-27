"""Offline unit tests for CodeForge v7.

Covers:
  1. Sample-first all-4-impl Wave 1 (`sample_first_mode`) + true streaming early-abort cancelling lagging slots.
  2. `external_test_code` streaming early-abort cancelling lagging slots once 2 independent candidates pass.
  3. Runtime / TLE-aware ranking guard (`_runtime_tier`, `_rank_key`, end-to-end fast vs slow shorter code).
  4. Graveyard Wave-2 rescue when 0/4 Wave-1 candidates pass pinned samples in `sample_first_mode`.
  5. `on_event` structured observability callback ordering (`slot_done`, `early_abort`, `wave2_start`, `done`) + fault isolation.
  6. Backwards-compatible `3 impls + 1 spec_oracle` pipeline on non-stdin API tasks and <2-sample stdin tasks.
  7. Single-survivor `sample_first_mode` fast-finish and batch-only `external_test_code` fallback.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from flash_forge.client import CallSpec, CompletionResult
from flash_forge.code_forge import (
    CodeCandidate,
    _is_significantly_faster,
    _rank_key,
    _runtime_tier,
    solve_code,
)
from flash_forge.memory import EpisodicMemory
from flash_forge.skill_loader import SkillPack


def _mk_res(content: str, tag: str = "", latency_s: float = 0.01) -> CompletionResult:
    return CompletionResult(
        content=content,
        reasoning="",
        finish_reason="stop",
        prompt_tokens=120,
        completion_tokens=60,
        reasoning_tokens=10,
        cached_tokens=0,
        latency_s=latency_s,
        tag=tag,
    )


class _StreamingMockClient:
    """Offline async mock implementing `complete`, `batch`, and `pairwise_vote` with per-tag delays & cancellation tracking."""

    def __init__(
        self,
        responses_by_tag: dict[str, str],
        *,
        delays_by_tag: dict[str, float] | None = None,
        pairwise_answer: str = "A",
    ) -> None:
        self.responses_by_tag = responses_by_tag
        self.delays_by_tag = delays_by_tag or {}
        self.pairwise_answer = pairwise_answer
        self.launched_tags: list[str] = []
        self.completed_tags: list[str] = []
        self.cancelled_tags: list[str] = []
        self.prompts_by_tag: dict[str, str] = {}
        self.pairwise_calls: int = 0

    def _lookup_reply(self, tag: str) -> str:
        reply = self.responses_by_tag.get(tag)
        if reply is None:
            for prefix, val in self.responses_by_tag.items():
                if prefix.endswith("*") and tag.startswith(prefix[:-1]):
                    return val
        return reply if reply is not None else "VERDICT: PASS"

    @property
    def total_launched_calls(self) -> int:
        return len(self.launched_tags) + 2 * self.pairwise_calls

    async def complete(self, spec: CallSpec) -> CompletionResult:
        self.launched_tags.append(spec.tag)
        self.prompts_by_tag[spec.tag] = spec.messages[-1]["content"] if spec.messages else ""
        delay = self.delays_by_tag.get(spec.tag, 0.0)
        if delay > 0:
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                self.cancelled_tags.append(spec.tag)
                raise
        self.completed_tags.append(spec.tag)
        return _mk_res(self._lookup_reply(spec.tag), tag=spec.tag, latency_s=max(0.01, delay))

    async def batch(self, specs: list[CallSpec]) -> list[CompletionResult]:
        return list(await asyncio.gather(*(self.complete(s) for s in specs)))

    async def pairwise_vote(self, prompt_ab: str, prompt_ba: str, *, tag_prefix: str = "judge") -> str:
        self.pairwise_calls += 1
        return self.pairwise_answer


_STDIN_TWO_SAMPLES_PROBLEM = (
    "Read a single integer N from standard input and print 2 * N to standard output.\n\n"
    "### Sample Input 1\n```\n3\n```\n"
    "### Sample Output 1\n```\n6\n```\n\n"
    "### Sample Input 2\n```\n10\n```\n"
    "### Sample Output 2\n```\n20\n```\n"
)

_STDIN_ONE_SAMPLE_PROBLEM = (
    "Read a single integer N from standard input and print 2 * N to standard output.\n\n"
    "### Sample Input 1\n```\n3\n```\n"
    "### Sample Output 1\n```\n6\n```\n"
)


def test_v7_sample_first_all4_streaming_early_abort_cancels_lagging_slots(tmp_path: Path):
    """Test 1: stdin_mode with >=2 verbatim samples launches all 4 diverse impls (0 upfront spec_oracle),
    runs streaming execution, and immediately cancels 10s lagging slots once 2 independent candidates pass."""
    client = _StreamingMockClient(
        {
            "w1_radically_simple": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```",
            "w1_first_principles": (
                "```python\nimport sys\n"
                "tokens = sys.stdin.read().split()\n"
                "n = int(tokens[0])\n"
                "print(n + n)\n```"
            ),
            "w1_edge_skeptic": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```",
            "w1_alt_algorithm": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) << 1)\n```",
        },
        delays_by_tag={
            "w1_radically_simple": 0.01,
            "w1_first_principles": 0.03,
            "w1_edge_skeptic": 10.0,
            "w1_alt_algorithm": 10.0,
        },
    )
    mem = EpisodicMemory(db_path=tmp_path / "mem_t1.db")
    skills = SkillPack()

    t0 = time.perf_counter()
    trace = asyncio.run(
        solve_code(client, skills, mem, _STDIN_TWO_SAMPLES_PROBLEM, num_wave1_streams=4)  # type: ignore[arg-type]
    )
    elapsed = time.perf_counter() - t0
    mem.close()

    assert set(client.launched_tags) == {
        "w1_radically_simple",
        "w1_first_principles",
        "w1_edge_skeptic",
        "w1_alt_algorithm",
    }
    assert "w1_spec_oracle" not in client.launched_tags
    assert trace.sample_first_mode is True
    assert trace.early_aborted_slots == 2
    assert set(client.cancelled_tags) == {"w1_edge_skeptic", "w1_alt_algorithm"}
    assert trace.unanimous_fastpath is True
    assert trace.verified is True
    assert trace.wave2_triggered is False
    assert trace.redteam_tests_generated == 0
    assert client.pairwise_calls == 0
    assert client.total_launched_calls <= 4
    assert len(trace.wave1_candidates) + trace.early_aborted_slots == 4
    assert elapsed < 3.0, f"Expected immediate early-abort well under 10s straggler, got {elapsed:.2f}s"


def test_v7_external_test_code_streaming_early_abort_cancels_stragglers(tmp_path: Path):
    """Test 2: external_test_code streaming execution cancels lagging slots immediately once 2 independent
    candidates pass 100% of external tests (even when an earlier fast slot failed!)."""
    problem = "Implement `def square(x: int) -> int:` returning `x * x`."
    ext_tests = (
        "def test_pos():\n"
        "    assert square(3) == 9\n"
        "    assert square(5) == 25\n\n"
        "def test_neg_and_zero():\n"
        "    assert square(0) == 0\n"
        "    assert square(-4) == 16\n"
    )
    client = _StreamingMockClient(
        {
            # Fast slot 0 has a bug (fails negative test) -> should NOT trigger early abort
            "w1_radically_simple": "```python\ndef square(x: int) -> int:\n    return x * abs(x)\n```",
            # Fast slots 1 & 2 both pass 100% -> trigger immediate early-abort of 10s slot 3!
            "w1_first_principles": "```python\ndef square(x: int) -> int:\n    return x * x\n```",
            "w1_edge_skeptic": "```python\ndef square(x: int) -> int:\n    return pow(x, 2)\n```",
            "w1_alt_algorithm": "```python\ndef square(x: int) -> int:\n    return x ** 2\n```",
        },
        delays_by_tag={
            "w1_radically_simple": 0.01,
            "w1_first_principles": 0.02,
            "w1_edge_skeptic": 0.04,
            "w1_alt_algorithm": 10.0,
        },
    )
    mem = EpisodicMemory(db_path=tmp_path / "mem_t2.db")
    skills = SkillPack()

    t0 = time.perf_counter()
    trace = asyncio.run(
        solve_code(
            client,  # type: ignore[arg-type]
            skills,
            mem,
            problem,
            num_wave1_streams=4,
            external_test_code=ext_tests,
        )
    )
    elapsed = time.perf_counter() - t0
    mem.close()

    assert trace.sample_first_mode is False
    assert trace.early_aborted_slots == 1
    assert client.cancelled_tags == ["w1_alt_algorithm"]
    assert trace.unanimous_fastpath is True
    assert trace.verified is True
    assert trace.wave2_triggered is False
    assert trace.winner_name in ("first_principles", "edge_skeptic")
    assert client.pairwise_calls == 0
    assert client.total_launched_calls <= 4
    assert elapsed < 3.0, f"Expected immediate early-abort well under 10s straggler, got {elapsed:.2f}s"


def test_v7_runtime_tier_tle_guard_prefers_fast_over_shorter_slow_candidate(tmp_path: Path):
    """Test 3: Runtime-tier TLE guard picks the fast O(N) passing candidate (<100ms) over a shorter
    slow passing candidate (>1500ms) — verified both at `_rank_key` unit level and end-to-end in `solve_code`."""
    # Part A: Direct ranking unit tests
    assert _runtime_tier(45.0) == 2
    assert _runtime_tier(850.0) == 1
    assert _runtime_tier(1600.0) == 0

    slow_short = CodeCandidate(
        name="slow_short",
        wave=1,
        code="def f(n): return sum(range(n))",
        raw_content="",
        tests_passed=4,
        tests_total=4,
        diff_majority_hits=2,
        starter_api_score=1.0,
        runtime_ms=1800.0,
    )
    fast_longer = CodeCandidate(
        name="fast_longer",
        wave=1,
        code=(
            "# O(1) arithmetic formula to avoid quadratic/linear TLE on large hidden constraints\n"
            "def f(n: int) -> int:\n"
            "    if n <= 0:\n"
            "        return 0\n"
            "    return n * (n - 1) // 2\n"
        ),
        raw_content="",
        tests_passed=4,
        tests_total=4,
        diff_majority_hits=2,
        starter_api_score=1.0,
        runtime_ms=42.0,
    )
    assert len(slow_short.code) < len(fast_longer.code)
    assert _rank_key(fast_longer) > _rank_key(slow_short)
    assert _is_significantly_faster(fast_longer, slow_short) is True

    # Control: two fast candidates in the SAME tier (<500ms, within small jitter) still follow Occam's Razor
    fast_compact = CodeCandidate(
        name="fast_compact",
        wave=1,
        code="def f(n): return n*(n-1)//2",
        raw_content="",
        tests_passed=4,
        tests_total=4,
        diff_majority_hits=2,
        starter_api_score=1.0,
        runtime_ms=65.0,
    )
    assert _rank_key(fast_compact) > _rank_key(fast_longer)

    # Part B: End-to-end `solve_code` with real wall-clock execution (`slow_short` sleeps 1.6s vs `fast_longer` <50ms)
    problem = "Implement `def tri_sum(n: int) -> int:` returning `0 + 1 + ... + (n - 1)`."
    ext_tests = (
        "def test_small():\n"
        "    assert tri_sum(5) == 10\n\n"
        "def test_med():\n"
        "    assert tri_sum(100) == 4950\n"
    )
    short_slow_code = (
        "```python\n"
        "import time\n"
        "def tri_sum(n):\n"
        "    if n == 100: time.sleep(1.6)\n"
        "    return n*(n-1)//2\n"
        "```"
    )
    longer_fast_code = (
        "```python\n"
        "def tri_sum(n: int) -> int:\n"
        "    # Closed-form O(1) triangular sum avoiding any loop overhead or TLE risk\n"
        "    if n <= 1:\n"
        "        return 0\n"
        "    total_value = (n * (n - 1)) // 2\n"
        "    return total_value\n"
        "```"
    )
    client = _StreamingMockClient(
        {
            "w1_radically_simple": short_slow_code,
            "w1_first_principles": longer_fast_code,
        },
        # Even if pairwise judge would have picked the shorter slow code, runtime tier MUST protect fast code!
        pairwise_answer="A",
    )
    mem = EpisodicMemory(db_path=tmp_path / "mem_t3.db")
    skills = SkillPack()
    trace = asyncio.run(
        solve_code(
            client,  # type: ignore[arg-type]
            skills,
            mem,
            problem,
            num_wave1_streams=2,
            external_test_code=ext_tests,
        )
    )
    mem.close()

    by_name = {c.name: c for c in trace.wave1_candidates}
    assert by_name["radically_simple"].perfect is True
    assert by_name["first_principles"].perfect is True
    assert by_name["radically_simple"].runtime_ms >= 1500.0
    assert by_name["first_principles"].runtime_ms < 250.0
    assert len(by_name["radically_simple"].code) < len(by_name["first_principles"].code)
    assert trace.verified is True
    assert trace.winner_name == "first_principles"


def test_v7_sample_first_zero_pass_triggers_wave2_graveyard_rescue(tmp_path: Path):
    """Test 4: When 0/4 Wave-1 candidates pass pinned samples in sample_first_mode,
    CodeForge skips wasted red-team and triggers Wave-2 graveyard repair to rescue cleanly."""
    client = _StreamingMockClient(
        {
            "w1_radically_simple": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 3)\n```",
            "w1_first_principles": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) + 3)\n```",
            "w1_edge_skeptic": "```python\nimport sys\nprint(0)\n```",
            "w1_alt_algorithm": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) - 1)\n```",
            "w2_radically_simple": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```",
            "w2_surgical_fix": "```python\nimport sys\nn = int(sys.stdin.read().strip())\nprint(n + n)\n```",
            "w2_clean_rewrite": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) << 1)\n```",
        }
    )
    mem = EpisodicMemory(db_path=tmp_path / "mem_t4.db")
    skills = SkillPack()
    trace = asyncio.run(
        solve_code(client, skills, mem, _STDIN_TWO_SAMPLES_PROBLEM, num_wave1_streams=4)  # type: ignore[arg-type]
    )
    mem.close()

    assert trace.sample_first_mode is True
    assert trace.early_aborted_slots == 0
    assert trace.unanimous_fastpath is False
    assert trace.redteam_tests_generated == 0
    assert "w1_spec_oracle" not in client.launched_tags
    assert not any(t.startswith("s2_redteam") for t in client.launched_tags)
    assert trace.wave2_triggered is True
    assert trace.verified is True
    assert trace.recovered_by_redteam_or_wave2 is True
    assert trace.winner_name.startswith("w2_")
    assert len(trace.graveyard) >= 1
    assert "GRAVEYARD OF WAVE-1 FAILURES" in client.prompts_by_tag["w2_radically_simple"]
    assert "test_prompt_sample_" in client.prompts_by_tag["w2_radically_simple"]


def test_v7_on_event_callback_ordered_events_and_fault_isolation(tmp_path: Path):
    """Test 5: on_event receives ordered slot_done, early_abort, wave2_start, and done events,
    and a raising callback never breaks solve_code."""
    # Part A: Streaming early-abort event sequence
    events: list[dict[str, Any]] = []
    client_fast = _StreamingMockClient(
        {
            "w1_radically_simple": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```",
            "w1_first_principles": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) << 1)\n```",
            "w1_edge_skeptic": "```python\nimport sys\nprint(0)\n```",
            "w1_alt_algorithm": "```python\nimport sys\nprint(0)\n```",
        },
        delays_by_tag={
            "w1_radically_simple": 0.01,
            "w1_first_principles": 0.03,
            "w1_edge_skeptic": 5.0,
            "w1_alt_algorithm": 5.0,
        },
    )
    mem = EpisodicMemory(db_path=tmp_path / "mem_t5a.db")
    skills = SkillPack()
    trace = asyncio.run(
        solve_code(
            client_fast,  # type: ignore[arg-type]
            skills,
            mem,
            _STDIN_TWO_SAMPLES_PROBLEM,
            num_wave1_streams=4,
            on_event=events.append,
        )
    )
    mem.close()

    assert trace.verified is True
    assert all(e.get("engine") == "code" for e in events)
    ev_names = [e["event"] for e in events]
    assert ev_names == ["slot_done", "slot_done", "early_abort", "done"]
    assert events[0]["wave"] == 1 and events[0]["perfect"] is True
    assert events[1]["wave"] == 1 and events[1]["perfect"] is True
    assert events[2]["early_aborted_slots"] == 2
    assert set(events[2]["cancelled_slots"]) == {"edge_skeptic", "alt_algorithm"}
    assert set(events[2]["perfect_slots"]) == {"radically_simple", "first_principles"}
    assert events[3]["verified"] is True
    assert events[3]["sample_first_mode"] is True
    assert events[3]["early_aborted_slots"] == 2

    # Part B: Wave-2 rescue event ordering (includes wave2_start)
    w2_events: list[dict[str, Any]] = []
    client_w2 = _StreamingMockClient(
        {
            "w1_radically_simple": "```python\nimport sys\nprint(0)\n```",
            "w1_first_principles": "```python\nimport sys\nprint(1)\n```",
            "w1_edge_skeptic": "```python\nimport sys\nprint(2)\n```",
            "w1_alt_algorithm": "```python\nimport sys\nprint(3)\n```",
            "w2_radically_simple": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```",
            "w2_surgical_fix": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```",
            "w2_clean_rewrite": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```",
        }
    )
    mem2 = EpisodicMemory(db_path=tmp_path / "mem_t5b.db")
    trace_w2 = asyncio.run(
        solve_code(
            client_w2,  # type: ignore[arg-type]
            skills,
            mem2,
            _STDIN_TWO_SAMPLES_PROBLEM,
            num_wave1_streams=4,
            on_event=w2_events.append,
        )
    )
    mem2.close()

    assert trace_w2.verified is True
    w2_names = [e["event"] for e in w2_events]
    assert w2_names.index("wave2_start") == 4
    assert w2_names[:4] == ["slot_done"] * 4
    assert w2_names[-1] == "done"
    assert w2_names[5:-1] == ["slot_done"] * 3

    # Part C: Fault isolation — raising callback must NOT crash solve_code
    def _boom(_ev: dict[str, Any]) -> None:
        raise RuntimeError("telemetry sink offline")

    mem3 = EpisodicMemory(db_path=tmp_path / "mem_t5c.db")
    trace_safe = asyncio.run(
        solve_code(
            client_fast,  # type: ignore[arg-type]
            skills,
            mem3,
            _STDIN_TWO_SAMPLES_PROBLEM,
            num_wave1_streams=4,
            on_event=_boom,
        )
    )
    mem3.close()
    assert trace_safe.verified is True


def test_v7_non_stdin_api_and_single_sample_keep_3impls_plus_spec_oracle(tmp_path: Path):
    """Test 6: Non-stdin API problems and stdin problems with <2 verbatim samples still use the
    3 impls + 1 spec_oracle Wave-1 pipeline cleanly (even with a streaming-capable client!)."""
    # Part A: Non-stdin library/API problem
    lib_problem = "Implement `def clamp(x: int, lo: int, hi: int) -> int:` clamping `x` into `[lo, hi]`."
    sol = "```python\ndef clamp(x: int, lo: int, hi: int) -> int:\n    return max(lo, min(hi, x))\n```"
    oracle = (
        "```python\n"
        "def test_mid(): assert clamp(5, 0, 10) == 5\n"
        "def test_low(): assert clamp(-3, 0, 10) == 0\n"
        "def test_high(): assert clamp(99, 0, 10) == 10\n"
        "def diff_probe():\n"
        "    return {'a': clamp(-1, 0, 5), 'b': clamp(3, 0, 5), 'c': clamp(9, 0, 5)}\n"
        "```"
    )
    client_api = _StreamingMockClient(
        {
            "w1_radically_simple": sol,
            "w1_first_principles": sol + "\n# v2",
            "w1_edge_skeptic": sol + "\n# v3",
            "w1_spec_oracle": oracle,
            "spec_audit_edge": "All clauses met.\nVERDICT: PASS",
            "spec_audit_api": "Clean API.\nVERDICT: PASS",
        }
    )
    mem = EpisodicMemory(db_path=tmp_path / "mem_t6a.db")
    skills = SkillPack()
    trace_api = asyncio.run(solve_code(client_api, skills, mem, lib_problem, num_wave1_streams=4))  # type: ignore[arg-type]
    mem.close()

    assert trace_api.sample_first_mode is False
    assert trace_api.early_aborted_slots == 0
    assert trace_api.verified is True
    assert client_api.launched_tags[:4] == [
        "w1_radically_simple",
        "w1_first_principles",
        "w1_edge_skeptic",
        "w1_spec_oracle",
    ]
    assert "w1_alt_algorithm" not in client_api.launched_tags

    # Part B: Stdin problem with only 1 verbatim sample (<2) also keeps 3 impls + 1 spec_oracle
    client_one = _StreamingMockClient(
        {
            "w1_radically_simple": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```",
            "w1_first_principles": "```python\nimport sys\nn = int(sys.stdin.read().strip())\nprint(n + n)\n```",
            "w1_edge_skeptic": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) << 1)\n```",
            "w1_spec_oracle": (
                "```python\n"
                "def test_extra():\n"
                "    assert run_solution('7\\n') == '14'\n"
                "def diff_probe():\n"
                "    return {'a': run_solution('4\\n'), 'b': run_solution('9\\n')}\n"
                "```"
            ),
        }
    )
    mem2 = EpisodicMemory(db_path=tmp_path / "mem_t6b.db")
    trace_one = asyncio.run(solve_code(client_one, skills, mem2, _STDIN_ONE_SAMPLE_PROBLEM, num_wave1_streams=4))  # type: ignore[arg-type]
    mem2.close()

    assert trace_one.sample_first_mode is False
    assert trace_one.verified is True
    assert client_one.launched_tags == [
        "w1_radically_simple",
        "w1_first_principles",
        "w1_edge_skeptic",
        "w1_spec_oracle",
    ]


def test_v7_single_survivor_sample_first_and_batch_only_external_fallback(tmp_path: Path):
    """Test 7: Single survivor in sample_first_mode returns cleanly in 4 calls without red-team/Wave 2,
    and a batch-only mock client without `complete` falls back cleanly on external_test_code."""
    # Part A: Only 1 of 4 passes pinned samples -> proceeds cleanly in 4 calls
    client_single = _StreamingMockClient(
        {
            "w1_radically_simple": "```python\nimport sys\nprint(0)\n```",
            "w1_first_principles": "```python\nimport sys\nprint(1)\n```",
            "w1_edge_skeptic": "```python\nimport sys\nprint(2)\n```",
            "w1_alt_algorithm": "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```",
        }
    )
    mem = EpisodicMemory(db_path=tmp_path / "mem_t7a.db")
    skills = SkillPack()
    trace_s = asyncio.run(
        solve_code(client_single, skills, mem, _STDIN_TWO_SAMPLES_PROBLEM, num_wave1_streams=4)  # type: ignore[arg-type]
    )
    mem.close()

    assert trace_s.sample_first_mode is True
    assert trace_s.early_aborted_slots == 0
    assert trace_s.verified is True
    assert trace_s.winner_name == "alt_algorithm"
    assert trace_s.wave2_triggered is False
    assert trace_s.redteam_tests_generated == 0
    assert client_single.total_launched_calls == 4

    # Part B: Batch-only client (no `complete`) with external_test_code falls back cleanly to `batch`
    class _BatchOnlyClient:
        def __init__(self) -> None:
            self.tags: list[str] = []

        async def batch(self, specs: list[CallSpec]) -> list[CompletionResult]:
            self.tags.extend(s.tag for s in specs)
            return [
                _mk_res("```python\ndef inc(x):\n    return x + 1\n```", tag=s.tag)
                for s in specs
            ]

    batch_client = _BatchOnlyClient()
    mem2 = EpisodicMemory(db_path=tmp_path / "mem_t7b.db")
    trace_b = asyncio.run(
        solve_code(
            batch_client,  # type: ignore[arg-type]
            skills,
            mem2,
            "Implement `def inc(x: int) -> int:` returning `x + 1`.",
            num_wave1_streams=4,
            external_test_code="def test_inc():\n    assert inc(4) == 5\n",
        )
    )
    mem2.close()

    assert trace_b.verified is True
    assert trace_b.unanimous_fastpath is True
    assert trace_b.early_aborted_slots == 0
    assert batch_client.tags == [
        "w1_radically_simple",
        "w1_first_principles",
        "w1_edge_skeptic",
        "w1_alt_algorithm",
    ]

"""Offline unit tests for live per-slot `on_event` streaming in ReasonForge & SpecVerify.

Verifies:
  1. ReasonForge Wave-1 streaming early-abort emits:
     start -> slot_done (2+) -> early_abort -> done
     and every event renders cleanly through LiveProgressReporter and format_sse_comment.
  2. ReasonForge batch fallback (client with only `.batch()`) emits:
     start -> slot_done (all Wave-1 slots) -> done.
  3. ReasonForge Wave-2 escalation emits:
     start -> Wave-1 slot_done -> wave2_start -> Wave-2 slot_done -> done.
  4. ReasonForge fallback-extract path (no clusters) still emits done.
  5. SpecVerify drafter-accepted path (with early-abort) emits:
     start -> drafter slot_done -> early_abort -> drafter_accepted -> done (0 verifier calls).
  6. SpecVerify escalated path emits:
     start -> drafter slot_done -> escalated -> forwarded reason events -> final spec done.
  7. Fault isolation: a raising `on_event` callback (`RuntimeError("broken pipe")`)
     NEVER crashes `solve_reasoning` or `solve_spec_verify`.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any, Callable, Optional

from flash_forge.client import CallSpec, CompletionResult, ModelCapabilities
from flash_forge.memory import EpisodicMemory
from flash_forge.progress import LiveProgressReporter, format_sse_comment
from flash_forge.reason_forge import solve_reasoning
from flash_forge.skill_loader import SkillPack
from flash_forge.spec_verify import solve_spec_verify


class _MockClient:
    """Deterministic async mock client supporting both `.complete()` and `.batch()`."""

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
    completion_tokens: int = 90,
    reasoning_tokens: int = 20,
    latency_s: float = 0.05,
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


def _assert_all_events_render_cleanly(events: list[dict[str, Any]]) -> list[str]:
    """Verify every event renders cleanly through LiveProgressReporter and format_sse_comment."""
    buf = io.StringIO()
    reporter = LiveProgressReporter(stream=buf, use_color=False)
    for ev in events:
        reporter.emit(ev)
        sse = format_sse_comment(ev)
        assert sse.startswith(": cortheon "), f"Bad SSE prefix: {sse!r}"
        assert sse.endswith("\n\n"), f"Bad SSE terminator: {sse!r}"
        inner = sse[:-2]
        assert "\n" not in inner and "\r" not in inner and "\x00" not in inner

    lines = buf.getvalue().splitlines()
    assert len(lines) == len(events)
    assert all(line.strip() for line in lines)
    return lines


def test_reason_wave1_streaming_early_abort_events_and_rendering(tmp_path: Path):
    """ReasonForge Wave-1 early-abort emits start -> slot_done (2+) -> early_abort -> done."""
    cancelled_tags: list[str] = []

    async def _handle(spec: CallSpec) -> CompletionResult:
        if spec.tag == "w1_pal_exact":
            await asyncio.sleep(0.01)
            return _mk_res(
                "```python\nans = 7 * 10\nassert ans == 70\nprint('ANSWER:', ans)\n```",
                tag=spec.tag,
                completion_tokens=420,
                latency_s=0.12,
            )
        if spec.tag == "w1_pal_skeptic":
            await asyncio.sleep(0.03)
            return _mk_res(
                "```python\nans = sum(7 for _ in range(10))\nassert ans > 0\nprint('ANSWER:', ans)\n```",
                tag=spec.tag,
                completion_tokens=610,
                latency_s=0.18,
            )
        try:
            await asyncio.sleep(8.0)
        except asyncio.CancelledError:
            cancelled_tags.append(spec.tag)
            raise
        return _mk_res("ANSWER: 70", tag=spec.tag)

    client = _MockClient("test-reasoner", _handle)
    skills = SkillPack()
    mem = EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False)
    events: list[dict[str, Any]] = []

    trace = asyncio.run(
        solve_reasoning(
            client,  # type: ignore[arg-type]
            skills,
            mem,
            "Compute 7 * 10. End with `ANSWER: <integer>`.",
            num_wave1_streams=4,
            on_event=events.append,
        )
    )

    assert trace.final_answer == "70"
    assert trace.verified is True
    assert trace.wave2_triggered is False
    # ReasonForge v4 Fast Pair: pal_exact + pal_skeptic agree -> slots 3 & 4 are skipped completely!
    assert cancelled_tags == []
    assert [s.tag for s in client.seen_specs] == ["w1_pal_exact", "w1_pal_skeptic"]

    event_names = [e["event"] for e in events]
    assert event_names[0] == "start"
    assert event_names[-1] == "done"
    assert "early_abort" in event_names

    abort_idx = event_names.index("early_abort")
    slot_done_before_abort = [e for e in events[:abort_idx] if e["event"] == "slot_done"]
    assert len(slot_done_before_abort) == 2
    assert all(e["engine"] == "reason" for e in events)

    start_ev = events[0]
    assert start_ev == {"engine": "reason", "event": "start", "mode": "reason", "streams": 4}

    for ev in slot_done_before_abort:
        assert ev["ok"] is True
        assert ev["answer"] == "70"
        assert ev["role"] in {"pal_exact", "pal_skeptic"}
        assert isinstance(ev["slot"], int) and ev["slot"] >= 1
        assert isinstance(ev["elapsed_s"], float)
        assert ev["tokens"] in {420, 610}
        assert ev["summary"] == "PAL verified"

    abort_ev = events[abort_idx]
    assert abort_ev["cancelled"] == 2
    assert abort_ev["consensus_answer"] == "70"
    assert "PAL consensus" in abort_ev["reason"]
    assert isinstance(abort_ev["elapsed_s"], float)

    done_ev = events[-1]
    assert done_ev["verified"] is True
    assert "-> 70" in done_ev["summary"]
    assert isinstance(done_ev["elapsed_s"], float)

    rendered = _assert_all_events_render_cleanly(events)
    assert any("launching 4 parallel streams" in ln for ln in rendered)
    assert any("ABORTED" in ln and "70" in ln for ln in rendered)
    assert any("[VERIFIED]" in ln and "70" in ln for ln in rendered)


def test_reason_wave1_batch_fallback_emits_slot_done(tmp_path: Path):
    """Custom batch-only client (no `.complete()`) still emits start -> slot_done (all) -> done."""

    class _BatchOnlyClient:
        async def batch(self, specs: list[CallSpec]) -> list[CompletionResult]:
            out: list[CompletionResult] = []
            for spec in specs:
                if spec.tag == "w1_analytical":
                    out.append(_mk_res("Therefore ANSWER: 42", tag=spec.tag, completion_tokens=150))
                else:
                    out.append(
                        _mk_res(
                            "```python\nassert 6 * 7 == 42\nprint('ANSWER: 42')\n```",
                            tag=spec.tag,
                            completion_tokens=200,
                        )
                    )
            return out

    skills = SkillPack()
    mem = EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False)
    events: list[dict[str, Any]] = []

    trace = asyncio.run(
        solve_reasoning(
            _BatchOnlyClient(),  # type: ignore[arg-type]
            skills,
            mem,
            "Compute 6 * 7. End with `ANSWER: <integer>`.",
            num_wave1_streams=4,
            on_event=events.append,
        )
    )

    assert trace.final_answer == "42"
    assert trace.verified is True
    event_names = [e["event"] for e in events]
    assert event_names == ["start", "slot_done", "slot_done", "slot_done", "slot_done", "done"]
    assert [e["role"] for e in events if e["event"] == "slot_done"] == [
        "pal_exact",
        "pal_skeptic",
        "pal_alt_z3",
        "analytical",
    ]
    _assert_all_events_render_cleanly(events)


def test_reason_wave2_escalation_emits_wave2_start_and_slot_done(tmp_path: Path):
    """ReasonForge Wave-1 disagreement triggers Wave-2 -> start -> slot_done -> wave2_start -> Wave-2 slot_done -> done."""

    async def _handle(spec: CallSpec) -> CompletionResult:
        if spec.tag == "w1_pal_exact":
            await asyncio.sleep(0.01)
            return _mk_res("```python\nprint('ANSWER: 10')\n```", tag=spec.tag, completion_tokens=100)
        if spec.tag == "w1_pal_alt_z3":
            await asyncio.sleep(0.015)
            return _mk_res("```python\nprint('ANSWER: 20')\n```", tag=spec.tag, completion_tokens=110)
        if spec.tag.startswith("w1_pal_skeptic"):
            await asyncio.sleep(0.02)
            return _mk_res("```python\nraise ZeroDivisionError('bad denom')\n```", tag=spec.tag, completion_tokens=120)
        if spec.tag == "w1_analytical":
            await asyncio.sleep(0.02)
            return _mk_res("Analytical derivation gives ANSWER: 20", tag=spec.tag, completion_tokens=200)
        if spec.tag == "w2_analytical_deep":
            return _mk_res("Analytical derivation gives ANSWER: 20", tag=spec.tag, completion_tokens=200)
        # Wave 2 PAL falsifiers agree on 20
        return _mk_res(
            "```python\nans = 4 * 5\nassert ans == 20\nprint('ANSWER:', ans)\n```",
            tag=spec.tag,
            completion_tokens=250,
        )

    client = _MockClient("test-reasoner", _handle)
    skills = SkillPack()
    mem = EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False)
    events: list[dict[str, Any]] = []

    trace = asyncio.run(
        solve_reasoning(
            client,  # type: ignore[arg-type]
            skills,
            mem,
            "Compute 4 * 5. End with `ANSWER: <integer>`.",
            num_wave1_streams=4,
            enable_wave2=True,
            on_event=events.append,
        )
    )

    assert trace.wave2_triggered is True
    assert trace.final_answer == "20"
    assert trace.verified is True

    event_names = [e["event"] for e in events]
    assert event_names[0] == "start"
    assert event_names[-1] == "done"
    assert "wave2_start" in event_names

    w2_idx = event_names.index("wave2_start")
    w1_slots = [e for e in events[:w2_idx] if e["event"] == "slot_done"]
    w2_slots = [e for e in events[w2_idx + 1 :] if e["event"] == "slot_done"]

    # ReasonForge v4 completes all 4 Wave-1 roles on disagreement before escalating to Wave 2!
    assert len(w1_slots) == 4
    skeptic_ev = next(e for e in w1_slots if e["role"] == "pal_skeptic")
    assert skeptic_ev["ok"] is False
    assert skeptic_ev["summary"]

    w2_start_ev = events[w2_idx]
    assert w2_start_ev["engine"] == "reason"
    assert w2_start_ev["streams"] == 3
    assert w2_start_ev["reason"] == trace.wave2_reason

    # Streaming Wave-2 early-abort stops as soon as 2 falsifiers agree (2 or 3 slots completed)
    assert 2 <= len(w2_slots) <= 3
    assert {e["role"] for e in w2_slots}.issubset({
        "falsifier_bruteforce",
        "falsifier_reread",
        "falsifier_numeric_z3",
    })
    assert all(e["ok"] is True and e["answer"] == "20" for e in w2_slots)

    rendered = _assert_all_events_render_cleanly(events)
    assert any("wave2:" in ln and "3 falsification streams" in ln for ln in rendered)


def test_reason_fallback_extract_still_emits_done(tmp_path: Path):
    """When every branch produces no clusterable answer, fallback_extract still emits done."""

    def _handle(spec: CallSpec) -> CompletionResult:
        if spec.tag == "reason_fallback_extract":
            return _mk_res('{"final_answer": "99"}', tag=spec.tag, completion_tokens=20)
        return _mk_res("no code and no answer here", tag=spec.tag, completion_tokens=30)

    client = _MockClient("test-reasoner", _handle)
    skills = SkillPack()
    mem = EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False)
    events: list[dict[str, Any]] = []

    trace = asyncio.run(
        solve_reasoning(
            client,  # type: ignore[arg-type]
            skills,
            mem,
            "Compute something.",
            num_wave1_streams=2,
            enable_wave2=False,
            on_event=events.append,
        )
    )

    assert trace.consensus_type == "fallback_extract"
    assert trace.final_answer == "99"
    assert events[0]["event"] == "start"
    assert events[-1]["event"] == "done"
    assert events[-1]["verified"] is False
    assert "fallback_extract -> 99" in events[-1]["summary"]
    _assert_all_events_render_cleanly(events)


def test_spec_verify_drafter_accepted_early_abort_events(tmp_path: Path):
    """SpecVerify accepted path emits start -> drafter slot_done -> early_abort -> drafter_accepted -> done (0 verifier calls)."""

    async def _draft(spec: CallSpec) -> CompletionResult:
        if spec.tag == "spec_draft_pal_exact":
            await asyncio.sleep(0.01)
            return _mk_res(
                "```python\nassert 19 * 23 == 437\nprint('ANSWER: 437')\n```",
                tag=spec.tag,
                completion_tokens=180,
                latency_s=0.11,
            )
        if spec.tag == "spec_draft_pal_alt_z3":
            await asyncio.sleep(0.025)
            return _mk_res(
                "```python\nassert sum([19] * 23) == 437\nprint('ANSWER: 437')\n```",
                tag=spec.tag,
                completion_tokens=210,
                latency_s=0.15,
            )
        await asyncio.sleep(8.0)
        return _mk_res("```python\nprint('ANSWER: 0')\n```", tag=spec.tag)

    def _forbid_verifier(spec: CallSpec) -> CompletionResult:
        raise AssertionError(f"Verifier must NOT be called when drafter accepted (tag={spec.tag})")

    drafter = _MockClient("tiny-draft-4b", _draft)
    verifier = _MockClient("big-verifier-70b", _forbid_verifier)
    skills = SkillPack()
    mem = EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False)
    events: list[dict[str, Any]] = []

    trace = asyncio.run(
        solve_spec_verify(
            drafter,  # type: ignore[arg-type]
            verifier,  # type: ignore[arg-type]
            skills,
            mem,
            "Compute 19 * 23. End with `ANSWER: <integer>`.",
            drafter_streams=3,
            verifier_streams=4,
            min_consensus=2,
            on_event=events.append,
        )
    )

    assert trace.accepted_by_drafter is True
    assert trace.escalated is False
    assert trace.answer == "437"
    assert trace.verifier_calls == 0
    assert len(verifier.seen_specs) == 0

    event_names = [e["event"] for e in events]
    assert event_names == ["start", "slot_done", "slot_done", "early_abort", "drafter_accepted", "done"]
    assert all(e["engine"] == "spec" for e in events)

    assert events[0] == {"engine": "spec", "event": "start", "mode": "spec", "streams": 3}

    slot_evs = [e for e in events if e["event"] == "slot_done"]
    assert all(e["ok"] is True and e["answer"] == "437" for e in slot_evs)
    assert {e["role"] for e in slot_evs} == {"pal_exact", "pal_alt_z3"}

    abort_ev = events[3]
    assert abort_ev["cancelled"] == 1
    assert abort_ev["consensus_answer"] == "437"

    acc_ev = events[4]
    assert acc_ev["drafter_model"] == "tiny-draft-4b"
    assert acc_ev["answer"] == "437"
    assert acc_ev["verifier_calls_saved"] == 4
    assert isinstance(acc_ev["elapsed_s"], float)

    done_ev = events[5]
    assert done_ev["verified"] is True
    assert "437" in done_ev["summary"]

    rendered = _assert_all_events_render_cleanly(events)
    assert any("ACCEPTED by tiny-draft-4b" in ln and "saved 4 verifier calls" in ln for ln in rendered)


def test_spec_verify_escalated_forwards_reason_events(tmp_path: Path):
    """SpecVerify escalated path emits start -> drafter slot_done -> escalated -> forwarded reason events -> final spec done."""

    async def _draft(spec: CallSpec) -> CompletionResult:
        if spec.tag == "spec_draft_pal_exact":
            await asyncio.sleep(0.01)
            return _mk_res("```python\nprint('ANSWER: 111')\n```", tag=spec.tag, completion_tokens=90)
        if spec.tag == "spec_draft_pal_alt_z3":
            await asyncio.sleep(0.015)
            return _mk_res("```python\nprint('ANSWER: 222')\n```", tag=spec.tag, completion_tokens=95)
        await asyncio.sleep(0.02)
        return _mk_res(
            "```python\nraise RuntimeError('crash at /Users/secret/proj/foo.py')\n```",
            tag=spec.tag,
            completion_tokens=80,
        )

    async def _verify(spec: CallSpec) -> CompletionResult:
        if spec.tag in ("w1_pal_exact", "w1_pal_alt_z3"):
            await asyncio.sleep(0.01)
            return _mk_res(
                "```python\nans = 13 * 17\nassert ans == 221\nprint('ANSWER:', ans)\n```",
                tag=spec.tag,
                completion_tokens=300,
            )
        await asyncio.sleep(5.0)
        return _mk_res("ANSWER: 221", tag=spec.tag, completion_tokens=250)

    drafter = _MockClient("tiny-draft-4b", _draft)
    verifier = _MockClient("big-verifier-70b", _verify)
    skills = SkillPack()
    mem = EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False)
    events: list[dict[str, Any]] = []

    trace = asyncio.run(
        solve_spec_verify(
            drafter,  # type: ignore[arg-type]
            verifier,  # type: ignore[arg-type]
            skills,
            mem,
            "Compute 13 * 17. End with `ANSWER: <integer>`.",
            drafter_streams=3,
            verifier_streams=4,
            min_consensus=2,
            on_event=events.append,
        )
    )

    assert trace.accepted_by_drafter is False
    assert trace.escalated is True
    assert trace.answer == "221"
    assert trace.verifier_trace is not None and trace.verifier_trace.verified is True

    pairs = [(e["engine"], e["event"]) for e in events]
    assert pairs[0] == ("spec", "start")
    assert pairs[-1] == ("spec", "done")
    assert ("spec", "escalated") in pairs

    esc_idx = pairs.index(("spec", "escalated"))
    drafter_slots = [e for e in events[:esc_idx] if e["engine"] == "spec" and e["event"] == "slot_done"]
    assert len(drafter_slots) == 3

    # Verify host path scrubbing on failing drafter slot summary
    failed_draft = next(e for e in drafter_slots if e["ok"] is False)
    assert "/Users/secret" not in failed_draft["summary"]

    esc_ev = events[esc_idx]
    assert esc_ev["verifier_model"] == "big-verifier-70b"
    assert esc_ev["reason"] == trace.escalation_reason

    forwarded_reason = [e for e in events[esc_idx + 1 : -1] if e["engine"] == "reason"]
    forwarded_names = [e["event"] for e in forwarded_reason]
    assert forwarded_names[0] == "start"
    assert "slot_done" in forwarded_names
    assert "early_abort" in forwarded_names
    assert forwarded_names[-1] == "done"

    final_done = events[-1]
    assert final_done["engine"] == "spec"
    assert final_done["verified"] is True
    assert "221" in final_done["summary"]

    rendered = _assert_all_events_render_cleanly(events)
    assert any("ESCALATED -> big-verifier-70b" in ln for ln in rendered)


def test_fault_isolation_broken_callback_never_crashes_engines(tmp_path: Path):
    """A raising on_event callback (`RuntimeError('broken pipe')`) NEVER crashes ReasonForge or SpecVerify."""
    calls: list[str] = []

    def _broken_callback(ev: dict[str, Any]) -> None:
        calls.append(f"{ev.get('engine')}:{ev.get('event')}")
        raise RuntimeError("broken pipe")

    async def _fast_pal(spec: CallSpec) -> CompletionResult:
        if spec.tag.endswith("analytical"):
            await asyncio.sleep(5.0)
            return _mk_res("ANSWER: 144", tag=spec.tag)
        await asyncio.sleep(0.01)
        return _mk_res(
            "```python\nassert 12 * 12 == 144\nprint('ANSWER: 144')\n```",
            tag=spec.tag,
        )

    async def _disagree_draft(spec: CallSpec) -> CompletionResult:
        if spec.tag == "spec_draft_pal_exact":
            return _mk_res("```python\nprint('ANSWER: 1')\n```", tag=spec.tag)
        return _mk_res("```python\nprint('ANSWER: 2')\n```", tag=spec.tag)

    skills = SkillPack()
    mem = EpisodicMemory(db_path=tmp_path / "mem.db", enabled=False)

    # 1. ReasonForge survives broken callback
    reason_client = _MockClient("reasoner-model", _fast_pal)
    r_trace = asyncio.run(
        solve_reasoning(
            reason_client,  # type: ignore[arg-type]
            skills,
            mem,
            "Compute 12 * 12. End with `ANSWER: <integer>`.",
            on_event=_broken_callback,
        )
    )
    assert r_trace.final_answer == "144"
    assert r_trace.verified is True
    assert "reason:start" in calls and "reason:done" in calls

    # 2. SpecVerify (accepted path) survives broken callback
    calls.clear()
    drafter_ok = _MockClient("draft-4b", _fast_pal)
    verifier_unused = _MockClient(
        "big-70b",
        lambda spec: (_ for _ in ()).throw(AssertionError("verifier unused")),
    )
    s_acc = asyncio.run(
        solve_spec_verify(
            drafter_ok,  # type: ignore[arg-type]
            verifier_unused,  # type: ignore[arg-type]
            skills,
            mem,
            "Compute 12 * 12. End with `ANSWER: <integer>`.",
            on_event=_broken_callback,
        )
    )
    assert s_acc.accepted_by_drafter is True
    assert s_acc.answer == "144"
    assert "spec:drafter_accepted" in calls and "spec:done" in calls

    # 3. SpecVerify (escalated path) survives broken callback across both spec and reason engines
    calls.clear()
    drafter_bad = _MockClient("draft-4b", _disagree_draft)
    verifier_ok = _MockClient("big-70b", _fast_pal)
    s_esc = asyncio.run(
        solve_spec_verify(
            drafter_bad,  # type: ignore[arg-type]
            verifier_ok,  # type: ignore[arg-type]
            skills,
            mem,
            "Compute 12 * 12. End with `ANSWER: <integer>`.",
            drafter_streams=2,
            on_event=_broken_callback,
        )
    )
    assert s_esc.escalated is True
    assert s_esc.answer == "144"
    assert "spec:escalated" in calls
    assert "reason:start" in calls
    assert "reason:done" in calls
    assert "spec:done" in calls

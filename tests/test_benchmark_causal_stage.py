"""Bounded causal stage reporting from Pi into the auditable report.

The Pi adapter emits one ``cortheon-benchmark-causal-stage-v1`` entry naming
which stage ended a causal-synthesis attempt with no certified answer. These
tests pin the runner side of that channel: a strict parser that accepts only
genuine, exactly-shaped host entries and one closed enum member; the last
entry of the exact type as authoritative with no fallback; the code reaching
``RunResult``, the serialized report, and the audit hash chain; and the
channel carrying nothing but the code — never candidate, evidence, or model
text.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cortheon.benchmark_core.audit import _audit_manifest, verify_audit_bundle
from cortheon.benchmark_core.models import ImportCase, RunResult
from cortheon.benchmark_core.outcomes import EvaluationOutcome
from cortheon.benchmark_core.run_support import (
    CANDIDATE_ENTRY_TYPE,
    CAUSAL_STAGE_REASONS,
    STAGE_ENTRY_TYPE,
    _captured_candidate,
    _captured_stage_reason,
    _parse_events,
)
from cortheon.benchmark_core.runner_local import _causal_stage_reason
from cortheon.cognitive_benchmark import main as benchmark_main

# Text that must never survive into a report through this channel.
CANDIDATE_TEXT = "Cause: the copper shard key collides because both ledgers reuse it."
EVIDENCE_TEXT = "[pi:read:facts/a.txt] Northstar path A uses collision key amber."
# Must match CAUSAL_STAGE_REASONS in pi_core/candidate_capture.ts exactly.
ADAPTER_STAGE_REASONS = frozenset(
    {
        "deliberation_empty",
        "validation_failed",
        "mapping_failed",
        "transport_failed",
        "runtime_withheld",
        "terminated_before_deliberation",
    }
)
ADAPTER_ENUM_SOURCE = (
    Path(__file__).parents[1] / "src" / "cortheon" / "pi_core" / "candidate_capture.ts"
)


def _stage_entry(
    reason: Any = "runtime_withheld",
    *,
    stage: str = "causal_synthesis",
    version: Any = 1,
    entry_id: Any = "entry-7",
    timestamp: Any = "2026-08-22T00:00:00.000Z",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One genuine top-level stage event, exactly as Pi emits it."""

    data: dict[str, Any] = {"version": version, "stage": stage, "reason": reason}
    if extra:
        data.update(extra)
    return {
        "type": "entry_appended",
        "entry": {
            "type": "custom",
            "customType": STAGE_ENTRY_TYPE,
            "id": entry_id,
            "timestamp": timestamp,
            "data": data,
        },
    }


def _candidate_entry(candidate: str = CANDIDATE_TEXT) -> dict[str, Any]:
    return {
        "type": "entry_appended",
        "entry": {
            "type": "custom",
            "customType": CANDIDATE_ENTRY_TYPE,
            "id": "entry-1",
            "timestamp": "2026-08-22T00:00:00.000Z",
            "data": {
                "version": 1,
                "stage": "causal_synthesis",
                "candidate": candidate,
            },
        },
    }


# --- Strict parsing ---


def test_every_declared_reason_parses() -> None:
    # The runner's enum and the adapter's are one list; a reason the adapter
    # can emit must never read as unknown here.
    assert CAUSAL_STAGE_REASONS == ADAPTER_STAGE_REASONS
    for reason in sorted(CAUSAL_STAGE_REASONS):
        assert _captured_stage_reason([_stage_entry(reason)]) == reason


def test_runner_enum_is_exact_lockstep_with_the_typescript_enum() -> None:
    # Parse the adapter's own closed enum out of candidate_capture.ts and
    # require exact equality (same members, no extras on either side): a
    # reason added in TypeScript without this runner, or removed here while
    # the adapter still emits it, is a broken stage channel.
    source = ADAPTER_ENUM_SOURCE.read_text(encoding="utf-8")
    match = re.search(r"CAUSAL_STAGE_REASONS\s*=\s*\[(.*?)\]\s*as\s*const", source, re.DOTALL)
    assert match is not None
    members = re.findall(r'"([a-z_]+)"', match.group(1))
    assert members, source
    assert len(members) == len(set(members)), members
    assert frozenset(members) == CAUSAL_STAGE_REASONS == ADAPTER_STAGE_REASONS


def test_shape_violations_capture_nothing() -> None:
    wrong_event_type = _stage_entry()
    wrong_event_type["type"] = "message_end"
    wrong_entry_type = _stage_entry()
    wrong_entry_type["entry"]["type"] = "message"
    missing_entry = {"type": "entry_appended"}
    entry_not_a_dict = {"type": "entry_appended", "entry": "custom"}
    wrong_custom_type = _stage_entry()
    wrong_custom_type["entry"]["customType"] = "cortheon-benchmark-causal-stage-v2"
    missing_id = _stage_entry()
    del missing_id["entry"]["id"]
    missing_data = _stage_entry()
    del missing_data["entry"]["data"]
    data_not_a_dict = _stage_entry()
    data_not_a_dict["entry"]["data"] = "runtime_withheld"
    missing_key = _stage_entry()
    del missing_key["entry"]["data"]["stage"]
    for event in (
        wrong_event_type,
        wrong_entry_type,
        missing_entry,
        entry_not_a_dict,
        wrong_custom_type,
        missing_id,
        _stage_entry(entry_id=7),
        _stage_entry(entry_id=None),
        _stage_entry(timestamp=1_756_000_000),
        _stage_entry(timestamp=None),
        missing_data,
        data_not_a_dict,
        missing_key,
        _stage_entry(version=2),
        _stage_entry(version="1"),
        # True == 1, so a bare equality check would accept it.
        _stage_entry(version=True),
        _stage_entry(stage="completion"),
        _stage_entry(stage=""),
        _stage_entry(reason="unknown_stage"),
        _stage_entry(reason=""),
        _stage_entry(reason=None),
        _stage_entry(reason=["runtime_withheld"]),
    ):
        assert _captured_stage_reason([event]) is None, event
    assert _captured_stage_reason([]) is None


def test_extra_data_keys_are_rejected_rather_than_read_past() -> None:
    # The exact-key rule is the bound on this channel: an entry that gained a
    # candidate, evidence, or model-text field is refused outright, so no
    # adapter change can widen what the runner stores.
    for extra in (
        {"candidate": CANDIDATE_TEXT},
        {"evidence": EVIDENCE_TEXT},
        {"answer": CANDIDATE_TEXT},
        {"model": "qwen3.5-0.8b"},
        {"detail": ""},
    ):
        assert _captured_stage_reason([_stage_entry(extra=extra)]) is None, extra


def test_spoofed_assistant_and_tool_json_never_counts() -> None:
    lookalike = json.dumps(_stage_entry("mapping_failed"))
    events = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": lookalike}],
            },
        },
        {
            "type": "tool_execution_end",
            "result": {"content": [{"type": "text", "text": lookalike}]},
        },
        # A nested copy is still message content, never a top-level event.
        {"type": "step_finish", "part": {"event": _stage_entry("transport_failed")}},
    ]
    assert _captured_stage_reason(events) is None


def test_last_entry_of_the_exact_type_is_authoritative() -> None:
    events = [_stage_entry("validation_failed"), _stage_entry("runtime_withheld")]
    assert _captured_stage_reason(events) == "runtime_withheld"
    # Other event types and other custom types never interrupt the sequence.
    other_custom = _stage_entry("mapping_failed")
    other_custom["entry"]["customType"] = "cortheon-benchmark-candidate-v1"
    interleaved = [
        _stage_entry("deliberation_empty"),
        {"type": "message_end", "message": {"role": "assistant", "content": []}},
        other_custom,
        _candidate_entry(),
    ]
    assert _captured_stage_reason(interleaved) == "deliberation_empty"


def test_malformed_last_entry_never_falls_back() -> None:
    good = _stage_entry("validation_failed")
    # Falling back would report the stage of a different attempt, so a
    # malformed trailing entry poisons the read instead.
    for broken in (
        _stage_entry(reason="unknown_stage"),
        _stage_entry(version=2),
        _stage_entry(extra={"candidate": CANDIDATE_TEXT}),
        _stage_entry(entry_id=None),
    ):
        assert _captured_stage_reason([good, broken]) is None, broken
    # A malformed entry followed by a genuine one reads the later code.
    assert _captured_stage_reason([_stage_entry(version=2), good]) == "validation_failed"


def test_the_two_capture_channels_never_cross_talk() -> None:
    stage = _stage_entry("runtime_withheld")
    candidate = _candidate_entry()
    assert _captured_stage_reason([candidate]) is None
    assert _captured_candidate([stage]) is None
    # Both present: each reader sees only its own channel.
    assert _captured_stage_reason([candidate, stage]) == "runtime_withheld"
    assert _captured_candidate([stage, candidate]) == CANDIDATE_TEXT


def test_parses_out_of_a_pi_json_event_stream() -> None:
    stdout = "\n".join(
        [
            "starting pi",
            json.dumps({"type": "message_start"}),
            json.dumps(_candidate_entry()),
            "not json at all",
            json.dumps(_stage_entry("runtime_withheld")),
            json.dumps([1, 2, 3]),
        ]
    )
    assert _captured_stage_reason(_parse_events(stdout)) == "runtime_withheld"


# --- Runner gating ---


def test_stage_reason_is_read_from_pi_treatment_events_only() -> None:
    events = [_stage_entry("mapping_failed")]
    assert _causal_stage_reason(events, host="pi", treatment=True) == "mapping_failed"
    # The capture variable is set for the Pi treatment arm alone; gating here
    # too means no other arm can ever carry a diagnostic it cannot produce.
    assert _causal_stage_reason(events, host="pi", treatment=False) is None
    assert _causal_stage_reason(events, host="opencode", treatment=True) is None
    assert _causal_stage_reason(events, host="opencode", treatment=False) is None
    assert _causal_stage_reason([], host="pi", treatment=True) is None


# --- Result, report, and audit propagation ---


def _run_result(reason: str | None) -> RunResult:
    return RunResult(
        case_id="case_stage",
        repeat=0,
        condition="cortheon",
        expected=True,
        final_text="[Cortheon withheld: completion was not certified.]",
        delivered=False,
        correct=False,
        latency_seconds=1.0,
        tokens=0,
        tool_calls=0,
        tool_errors=0,
        timed_out=False,
        process_error=None,
        evaluator_outcome=EvaluationOutcome("pi", "missing", "none", None),
        causal_stage_reason=reason,
    )


def test_run_result_defaults_to_no_stage_reason() -> None:
    serialized = asdict(_run_result(None))
    assert serialized["causal_stage_reason"] is None
    assert asdict(_run_result("mapping_failed"))["causal_stage_reason"] == "mapping_failed"


def test_audit_chain_covers_the_stage_code() -> None:
    withheld = {"schema_version": 11, "runs": [asdict(_run_result("runtime_withheld"))]}
    withheld["audit"] = _audit_manifest(withheld)
    assert verify_audit_bundle(withheld)["content_valid"] is True

    mapping = {"schema_version": 11, "runs": [asdict(_run_result("mapping_failed"))]}
    mapping["audit"] = _audit_manifest(mapping)
    # The code is inside the hashed run, so an altered stage breaks the chain.
    assert withheld["audit"]["run_chain_head"] != mapping["audit"]["run_chain_head"]
    tampered = json.loads(json.dumps(withheld))
    tampered["runs"][0]["causal_stage_reason"] = "mapping_failed"
    assert verify_audit_bundle(tampered)["content_valid"] is False


def test_serialized_report_carries_the_code_and_no_model_text() -> None:
    report: dict[str, Any] = {
        "schema_version": 11,
        "runs": [asdict(_run_result(reason)) for reason in ("mapping_failed", None)],
    }
    report["audit"] = _audit_manifest(report)
    serialized = json.dumps(report, sort_keys=True)
    assert '"causal_stage_reason": "mapping_failed"' in serialized
    assert '"causal_stage_reason": null' in serialized
    for secret in (CANDIDATE_TEXT, EVIDENCE_TEXT, "copper", "amber", "Cause:"):
        assert secret not in serialized, secret


def _import_cases() -> list[ImportCase]:
    return [
        ImportCase("case_a", "src/a.py", "pathlib", True, "prompt a"),
        ImportCase("case_b", "src/b.py", "json", False, "prompt b"),
    ]


def test_report_records_the_stage_only_for_treatment_runs(monkeypatch, tmp_path) -> None:
    """The whole path: Pi events to the written, audited report."""

    def run_balanced(_args, case, *, repeat, treatment, condition):
        events = [_candidate_entry(), _stage_entry("runtime_withheld")]
        return RunResult(
            case_id=case.case_id,
            repeat=repeat,
            condition=condition,
            expected=case.expected,
            final_text="verified",
            delivered=True,
            correct=True,
            latency_seconds=0.1,
            tokens=10,
            tool_calls=1,
            tool_errors=0,
            timed_out=False,
            process_error=None,
            evaluator_outcome=EvaluationOutcome("pi", "success", "pi_assistant", "stop"),
            inference_model_id="small-model",
            causal_stage_reason=_causal_stage_reason(
                events,
                host="pi",
                treatment=treatment,
            ),
            substrate_telemetry_valid=True if treatment else None,
            runtime_sessions_started=1 if treatment else 0,
            runtime_observations_accepted=1 if treatment else 0,
            runtime_sessions_completed=1 if treatment else 0,
        )

    monkeypatch.setattr(
        "cortheon.cognitive_benchmark._runtime_health",
        lambda _url: {"ok": True, "storage": "memory_only"},
    )
    monkeypatch.setattr(
        "cortheon.cognitive_benchmark._model_endpoint_health",
        lambda *_args, **_kwargs: {
            "ok": True,
            "model_id": "small-model",
            "inference_probe_ok": True,
        },
    )
    monkeypatch.setattr(
        "cortheon.cognitive_benchmark.discover_benchmark_cases",
        lambda *_args, **_kwargs: _import_cases(),
    )
    monkeypatch.setattr("cortheon.cognitive_benchmark.run_job", run_balanced)
    output = tmp_path / "stage-report.json"

    exit_code = benchmark_main(
        [
            "--repository",
            str(tmp_path),
            "--cases",
            "2",
            "--repeats",
            "1",
            "--suite",
            "imports",
            "--model-id",
            "small-model",
            "--host",
            "pi",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0

    raw = output.read_text(encoding="utf-8")
    report = json.loads(raw)
    # Schema 11 binds every run to evaluator-owned terminal provenance.
    assert report["schema_version"] == 14
    runs = report["runs"]
    assert len(runs) == 4
    by_condition: dict[str, set[Any]] = {}
    for run in runs:
        assert "causal_stage_reason" in run
        by_condition.setdefault(run["condition"], set()).add(run["causal_stage_reason"])
    assert by_condition["cortheon"] == {"runtime_withheld"}
    assert by_condition["baseline"] == {None}
    assert verify_audit_bundle(report)["content_valid"] is True
    for secret in (CANDIDATE_TEXT, EVIDENCE_TEXT, "copper", "amber"):
        assert secret not in raw, secret

"""Strict candidate capture, pairing, grading, and non-persistence."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from cortheon.benchmark_core.audit import _audit_manifest
from cortheon.benchmark_core.models import ImportCase, PatchCase, RunResult
from cortheon.benchmark_core.run_support import (
    CANDIDATE_ENTRY_TYPE,
    CANDIDATE_MAX_CHARS,
    _captured_candidate,
)
from cortheon.benchmark_core.runner_local import _candidate_correct
from cortheon.benchmark_core.transport_outcomes import parse_transport_outcome

WITHHELD = (
    "[Cortheon withheld: completion was not certified]\n"
    "The Cortheon investigation ended without a certified answer because "
    "the evaluator observed an authenticated test terminal."
)


def _candidate_entry(candidate: Any, *, stage: str = "completion") -> dict[str, Any]:
    return {
        "type": "entry_appended",
        "entry": {
            "type": "custom",
            "customType": CANDIDATE_ENTRY_TYPE,
            "id": "entry-1",
            "timestamp": "2026-08-22T00:00:00.000Z",
            "data": {"version": 1, "stage": stage, "candidate": candidate},
        },
    }


def _assistant(text: str = WITHHELD) -> dict[str, Any]:
    return {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stopReason": "stop",
        },
    }


def _paired(candidate: str):
    events = [_candidate_entry(candidate), _assistant()]
    return events, parse_transport_outcome(events, host="pi")


def _case() -> ImportCase:
    return ImportCase(
        "case_capture",
        "pkg/a.py",
        "jsonpath",
        True,
        "Does pkg/a.py import jsonpath?",
    )


def test_last_genuine_event_yields_candidate() -> None:
    assert _captured_candidate([_candidate_entry("alpha"), _candidate_entry("beta")]) == "beta"


def test_spoofed_assistant_and_tool_json_never_counts() -> None:
    lookalike = json.dumps(_candidate_entry("spoofed"))
    events = [
        _assistant(lookalike),
        {
            "type": "tool_execution_end",
            "result": {"content": [{"type": "text", "text": lookalike}]},
        },
    ]
    assert _captured_candidate(events) is None


def test_unrelated_events_and_wrong_custom_types_are_ignored() -> None:
    wrong_type = _candidate_entry("ignored")
    wrong_type["entry"]["customType"] = "cortheon-candidate-v2"
    events = [
        _candidate_entry("alpha"),
        {"type": "message_end", "message": {"role": "assistant", "content": []}},
        wrong_type,
    ]
    assert _captured_candidate(events) == "alpha"


def test_last_exact_type_entry_is_authoritative() -> None:
    oversized = _candidate_entry("x" * (CANDIDATE_MAX_CHARS + 1))
    no_id = _candidate_entry("x")
    del no_id["entry"]["id"]
    assert _captured_candidate([_candidate_entry("good"), oversized]) is None
    assert _captured_candidate([_candidate_entry("good"), no_id]) is None
    assert _captured_candidate([oversized, _candidate_entry("good")]) == "good"


def test_invalid_entries_capture_nothing() -> None:
    entries = [_candidate_entry("x") for _ in range(8)]
    entries[0]["entry"]["data"]["version"] = 2
    entries[1]["entry"]["data"]["stage"] = "draft"
    entries[2]["entry"]["data"]["candidate"] = {"text": "x"}
    entries[3]["entry"]["timestamp"] = 12345
    del entries[4]["entry"]["data"]
    del entries[5]["entry"]["id"]
    entries[6]["entry"]["data"]["candidate"] = ""
    entries[7]["entry"]["data"]["candidate"] = "x" * (CANDIDATE_MAX_CHARS + 1)
    assert all(_captured_candidate([entry]) is None for entry in entries)
    assert _captured_candidate([]) is None


def test_paired_blocked_candidate_grades_correct_and_wrong() -> None:
    correct_events, correct = _paired("Answer: yes, pkg/a.py imports jsonpath.")
    wrong_events, wrong = _paired("Answer: no, it does not import jsonpath.")
    assert (
        _candidate_correct(
            _case(),
            correct_events,
            host="pi",
            treatment=True,
            final=correct.final_text,
            evaluator_outcome=correct.outcome,
        )
        is True
    )
    assert (
        _candidate_correct(
            _case(),
            wrong_events,
            host="pi",
            treatment=True,
            final=wrong.final_text,
            evaluator_outcome=wrong.outcome,
        )
        is False
    )


def test_delivered_uncaptured_wrong_host_and_control_stay_unclassified() -> None:
    blocked_events, blocked = _paired("Answer: yes, pkg/a.py imports jsonpath.")
    delivered_events = [_candidate_entry("candidate"), _assistant("delivered answer")]
    delivered = parse_transport_outcome(delivered_events, host="pi")
    uncaptured_events = [_assistant()]
    uncaptured = parse_transport_outcome(uncaptured_events, host="pi")
    variants = (
        (delivered_events, "pi", True, delivered),
        (uncaptured_events, "pi", True, uncaptured),
        (blocked_events, "opencode", True, blocked),
        (blocked_events, "pi", False, blocked),
    )
    for events, host, treatment, parsed in variants:
        assert (
            _candidate_correct(
                _case(),
                events,
                host=host,
                treatment=treatment,
                final=parsed.final_text,
                evaluator_outcome=parsed.outcome,
            )
            is None
        )


def test_patch_cases_never_grade_text_candidates() -> None:
    case = PatchCase("case_patch", (("a.txt", "x"),), (), "true", "", "fix")
    events, parsed = _paired("anything")
    assert (
        _candidate_correct(
            case,
            events,
            host="pi",
            treatment=True,
            final=parsed.final_text,
            evaluator_outcome=parsed.outcome,
        )
        is None
    )


def test_candidate_text_never_reaches_reports_or_audit() -> None:
    secret = "SECRET-CANDIDATE-TEXT yes pkg/a.py imports jsonpath."
    events, parsed = _paired(secret)
    graded = _candidate_correct(
        _case(),
        events,
        host="pi",
        treatment=True,
        final=parsed.final_text,
        evaluator_outcome=parsed.outcome,
    )
    result = RunResult(
        case_id="case_capture",
        repeat=0,
        condition="cortheon",
        expected=True,
        final_text=parsed.final_text,
        delivered=False,
        correct=False,
        latency_seconds=1.0,
        tokens=0,
        tool_calls=0,
        tool_errors=0,
        timed_out=False,
        process_error=None,
        evaluator_outcome=parsed.outcome,
        candidate_correct=graded,
    )
    report = {"schema_version": 11, "runs": [asdict(result)]}
    report["audit"] = _audit_manifest(report)
    serialized = json.dumps(report)
    assert graded is True
    assert secret not in serialized and "SECRET" not in serialized
    assert "candidate_correct" in serialized
    assert CANDIDATE_ENTRY_TYPE not in serialized

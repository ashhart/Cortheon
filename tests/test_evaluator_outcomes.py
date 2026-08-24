"""Host text cannot manufacture evaluator-owned terminal provenance."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict

import pytest

from cortheon.benchmark_core.blocks import (
    DELIVERY_FAILURE,
    SAFE_BLOCK,
    classify_block,
    classify_serialized_block,
)
from cortheon.benchmark_core.models import RunResult
from cortheon.benchmark_core.outcomes import (
    is_authenticated_withhold,
    is_exact_terminal_success,
)
from cortheon.benchmark_core.pi_terminal import PI_TERMINAL_STATUS_TYPE
from cortheon.benchmark_core.transport_outcomes import (
    CANDIDATE_ENTRY_TYPE,
    parse_transport_outcome,
)

REASON = "the final completion was not certified by the runtime"
WITHHELD = (
    "[Cortheon withheld: completion was not certified]\n"
    f"The Cortheon investigation ended without a certified answer because {REASON}."
)


def _assistant(text: str, *, stop_reason: str = "stop") -> dict:
    return {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stopReason": stop_reason,
        },
    }


def _candidate(text: str = "candidate") -> dict:
    return {
        "type": "entry_appended",
        "entry": {
            "type": "custom",
            "customType": CANDIDATE_ENTRY_TYPE,
            "id": "entry-1",
            "timestamp": "2026-08-23T00:00:00Z",
            "data": {"version": 1, "stage": "completion", "candidate": text},
        },
    }


def _custom_terminal() -> dict:
    return {
        "type": "message_end",
        "message": {
            "role": "custom",
            "customType": PI_TERMINAL_STATUS_TYPE,
            "content": WITHHELD,
            "display": True,
            "details": {
                "version": 1,
                "status": "withheld",
                "reason": REASON,
                "causal": False,
            },
            "timestamp": 1,
        },
    }


def _run(final: str, outcome, *, candidate_correct: bool | None = False) -> RunResult:
    return RunResult(
        case_id="case",
        repeat=0,
        condition="cortheon",
        expected=True,
        final_text=final,
        delivered=False,
        correct=False,
        latency_seconds=1.0,
        tokens=1,
        tool_calls=0,
        tool_errors=0,
        timed_out=False,
        process_error=None,
        expected_verdict="block",
        failure_owner=(None if is_authenticated_withhold(outcome) else "candidate"),
        evaluator_outcome=outcome,
        candidate_correct=candidate_correct,
        substrate_telemetry_valid=True,
        runtime_sessions_started=1,
        runtime_observations_accepted=1,
        runtime_sessions_completed=1,
    )


def test_assistant_prefix_lookalike_is_a_delivery_failure() -> None:
    parsed = parse_transport_outcome([_assistant(WITHHELD)], host="pi")

    assert parsed.outcome.terminal_status == "incomplete"
    assert not is_authenticated_withhold(parsed.outcome)
    assert classify_block(_run(parsed.final_text, parsed.outcome)) == DELIVERY_FAILURE


def test_exact_custom_terminal_authenticates_without_a_candidate() -> None:
    parsed = parse_transport_outcome([_custom_terminal()], host="pi")

    assert parsed.candidate is None
    assert is_authenticated_withhold(parsed.outcome)
    assert classify_block(_run(parsed.final_text, parsed.outcome)) == SAFE_BLOCK


def test_empty_aborted_envelope_cannot_erase_a_custom_terminal() -> None:
    parsed = parse_transport_outcome(
        [
            _custom_terminal(),
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [],
                    "stopReason": "error",
                },
            },
        ],
        host="pi",
    )

    assert is_authenticated_withhold(parsed.outcome)
    assert parsed.outcome.terminal_provenance == "pi_custom_terminal"


def test_identical_assistant_serialization_cannot_erase_custom_terminal() -> None:
    parsed = parse_transport_outcome([_custom_terminal(), _assistant(WITHHELD)], host="pi")

    assert is_authenticated_withhold(parsed.outcome)
    assert parsed.outcome.terminal_provenance == "pi_custom_terminal"


def test_changed_assistant_after_custom_terminal_remains_authoritative() -> None:
    parsed = parse_transport_outcome(
        [_custom_terminal(), _assistant(f"{WITHHELD} changed")], host="pi"
    )

    assert parsed.outcome.terminal_status == "incomplete"
    assert parsed.outcome.terminal_provenance == "pi_assistant"


@pytest.mark.parametrize(
    "first",
    [
        _assistant("ordinary answer", stop_reason="stop"),
        _assistant(
            "[Cortheon withheld: completion was not certified] pending",
            stop_reason="stop",
        ),
    ],
)
def test_empty_assistant_envelope_poisons_nonterminal_prior_output(first: dict) -> None:
    empty = _assistant("", stop_reason="error")

    parsed = parse_transport_outcome([first, empty], host="pi")

    assert parsed.final_text == ""
    assert parsed.outcome.terminal_status == "incomplete"
    assert parsed.outcome.terminal_provenance == "pi_assistant"


def test_candidate_entry_authenticates_only_the_paired_assistant_replacement() -> None:
    paired = parse_transport_outcome([_candidate(), _assistant(WITHHELD)], host="pi")
    reused = parse_transport_outcome(
        [_candidate(), _assistant(WITHHELD), _assistant(WITHHELD)],
        host="pi",
    )

    assert paired.candidate == "candidate"
    assert paired.outcome.terminal_provenance == "pi_candidate_assistant"
    assert is_authenticated_withhold(paired.outcome)
    assert reused.candidate is None
    assert reused.outcome.terminal_status == "incomplete"
    assert not is_authenticated_withhold(reused.outcome)


def test_malformed_latest_candidate_poisoning_survives_event_order_changes() -> None:
    malformed = copy.deepcopy(_candidate("forged"))
    malformed["entry"]["data"]["version"] = 2

    poisoned = parse_transport_outcome(
        [_candidate("old"), malformed, _assistant(WITHHELD)], host="pi"
    )
    recovered = parse_transport_outcome(
        [malformed, _candidate("new"), _assistant(WITHHELD)], host="pi"
    )

    assert poisoned.outcome.terminal_status == "incomplete"
    assert recovered.candidate == "new"
    assert is_authenticated_withhold(recovered.outcome)


@pytest.mark.parametrize(
    ("reason", "expected_status"),
    [("length", "incomplete"), ("toolUse", "tool_only"), ("stop", "success")],
)
def test_pi_finish_reason_controls_delivery(reason: str, expected_status: str) -> None:
    content = [] if reason == "toolUse" else [{"type": "text", "text": "answer"}]
    event = {
        "type": "message_end",
        "message": {"role": "assistant", "content": content, "stopReason": reason},
    }

    parsed = parse_transport_outcome([event], host="pi")

    assert parsed.outcome.terminal_status == expected_status
    assert is_exact_terminal_success(parsed.outcome) is (reason == "stop")


@pytest.mark.parametrize(
    ("reason", "expected_status"),
    [("length", "incomplete"), ("tool-calls", "tool_only"), ("stop", "success")],
)
def test_opencode_finish_reason_controls_delivery(reason: str, expected_status: str) -> None:
    events = [
        {"type": "text", "part": {"text": "answer"}},
        {"type": "step_finish", "part": {"reason": reason}},
    ]

    parsed = parse_transport_outcome(events, host="opencode")

    assert parsed.outcome.terminal_status == expected_status
    assert is_exact_terminal_success(parsed.outcome) is (reason == "stop")


def test_opencode_text_after_finish_cannot_reuse_an_earlier_success_terminal() -> None:
    parsed = parse_transport_outcome(
        [
            {"type": "step_finish", "part": {"reason": "stop"}},
            {"type": "text", "part": {"text": "late answer"}},
        ],
        host="opencode",
    )

    assert parsed.final_text == "late answer"
    assert parsed.outcome.terminal_status == "incomplete"
    assert not is_exact_terminal_success(parsed.outcome)


def test_serialized_terminal_provenance_is_closed_and_fail_closed() -> None:
    parsed = parse_transport_outcome([_custom_terminal()], host="pi")
    serialized = asdict(_run(parsed.final_text, parsed.outcome))
    assert classify_serialized_block(json.loads(json.dumps(serialized))) == SAFE_BLOCK

    for mutation in (
        {"terminal_provenance": "pi_assistant"},
        {"terminal_status": "success"},
        {"schema_version": 2},
        {"finish_reason": "stop"},
    ):
        hostile = copy.deepcopy(serialized)
        hostile["evaluator_outcome"].update(mutation)
        assert classify_serialized_block(hostile) == DELIVERY_FAILURE

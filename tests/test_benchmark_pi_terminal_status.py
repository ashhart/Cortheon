"""Fail-closed parsing of Pi's host-visible Cortheon terminal status."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from cortheon.benchmark_core.run_support import (
    PI_TERMINAL_REASON_MAX_CHARS,
    PI_TERMINAL_STATUS_TYPE,
    _delivery_succeeded,
    _final_text,
)
from cortheon.benchmark_core.transport_outcomes import parse_transport_outcome

REASON = "the forced answer continuation ended without any answerable text"
TEXT = (
    "[Cortheon withheld: completion was not certified]\n"
    f"The Cortheon investigation ended without a certified answer because {REASON}."
)


def _message() -> dict[str, Any]:
    return {
        "role": "custom",
        "customType": PI_TERMINAL_STATUS_TYPE,
        "content": TEXT,
        "display": True,
        "details": {
            "version": 1,
            "status": "withheld",
            "reason": REASON,
            "causal": True,
        },
        "timestamp": 1_787_442_828_868,
    }


def _event(message: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "message_end", "message": message or _message()}


def _assistant(text: str) -> dict[str, Any]:
    return {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    }


def _mutated(path: tuple[str, ...], value: Any = None, *, delete: bool = False):
    message = copy.deepcopy(_message())
    target = message
    for key in path[:-1]:
        target = target[key]
    if delete:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return _event(message)


def test_accepts_the_exact_pi_terminal_and_keeps_it_withheld() -> None:
    events = [_event()]
    final = _final_text(events, host="pi")
    outcome = parse_transport_outcome(events, host="pi").outcome

    assert final == TEXT
    assert not _delivery_succeeded(
        final,
        timed_out=False,
        process_error=None,
        evaluator_outcome=outcome,
    )


@pytest.mark.parametrize(
    "event",
    [
        _mutated(("role",), "assistant"),
        _mutated(("role",), "toolResult"),
        _mutated(("customType",), "cortheon-terminal-status-v2"),
        _mutated(("display",), False),
        _mutated(("display",), 1),
        _mutated(("timestamp",), True),
        _mutated(("timestamp",), "1787442828868"),
        _mutated(("timestamp",), -1),
        _mutated(("timestamp",), 2**63),
        _mutated(("details", "version"), True),
        _mutated(("details", "version"), 2),
        _mutated(("details", "status"), "complete"),
        _mutated(("details", "causal"), 1),
        _mutated(("details", "reason"), ""),
        _mutated(("details", "reason"), " padded "),
        _mutated(("details", "reason"), "line one\nline two"),
        _mutated(("details", "reason"), "x" * (PI_TERMINAL_REASON_MAX_CHARS + 1)),
        _mutated(("details", "reason"), {"text": REASON}),
        _mutated(("content",), TEXT + " forged suffix"),
        _mutated(("content",), [{"type": "text", "text": TEXT}]),
        _mutated(("details", "reason"), delete=True),
        _mutated(("details", "causal"), delete=True),
        _mutated(("details", "candidate"), "unbounded model output"),
        _mutated(("details", "evidence"), ["private receipt"]),
        _mutated(("candidate",), "unbounded model output"),
        _mutated(("triggerTurn",), False),
    ],
)
def test_rejects_drifted_or_unbounded_terminal_messages(event) -> None:
    assert _final_text([event], host="pi") == ""


def test_nested_assistant_and_tool_lookalikes_are_not_terminal_events() -> None:
    lookalike = json.dumps(_event())
    assistant_spoof = _assistant(lookalike)
    tool_spoof = {
        "type": "tool_execution_end",
        "result": {"content": [{"type": "text", "text": lookalike}]},
    }

    assert _final_text([tool_spoof], host="pi") == ""
    assert _final_text([assistant_spoof], host="pi") == lookalike


def test_last_genuine_output_or_terminal_event_wins() -> None:
    malformed = _mutated(("details", "candidate"), "forged")
    assistant_spoof = _assistant(json.dumps(_event()))

    assert _final_text([assistant_spoof, malformed, _event()], host="pi") == TEXT
    assert _final_text([_event(), malformed, _assistant("later answer")], host="pi") == (
        "later answer"
    )
    assert _final_text([_assistant("earlier answer"), _event(), malformed], host="pi") == TEXT


def test_wrong_top_level_event_shape_cannot_supply_a_terminal() -> None:
    event = _event()
    event["type"] = "entry_appended"
    no_message = {"type": "message_end", "message": json.dumps(_message())}

    assert _final_text([event, no_message], host="pi") == ""

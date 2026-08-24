"""Transcript validity requires the exact evaluator event order."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from generic_mcp_transcript_helpers import retract_events as _retract_events
from test_contradiction_revision_completion import (
    _executor as _revision_executor,
)
from test_contradiction_revision_completion import (
    _RevisionModel,
)
from test_generic_mcp_hardening import (
    _executor as _hardening_executor,
)
from test_generic_mcp_hardening import (
    _Model,
    _valid_bare_events,
)
from test_generic_mcp_hardening import (
    _profile as _hardening_profile,
)
from test_generic_mcp_microsteps import _AdherentSmallModel, _executor, _profile

from cortheon.benchmark_core.generic_mcp_host import GenericMcpHost
from cortheon.benchmark_core.generic_mcp_model import ModelToolCall, ModelTurn
from cortheon.benchmark_core.generic_mcp_order_validation import (
    host_request_matches_action,
    transition_sha256,
)
from cortheon.benchmark_core.generic_mcp_protocol import (
    encoded_payload_sha256,
    payload_sha256,
)
from cortheon.benchmark_core.generic_mcp_validation import validate_transcript
from cortheon.operator_lift import development_cases, public_case
from cortheon.operator_lift.execution_runner import _goal


def _events(root: Path) -> list[dict[str, object]]:
    result = GenericMcpHost(
        task_id="ordered-transcript",
        evaluation_profile=_profile(),
        model=_AdherentSmallModel(),  # type: ignore[arg-type]
        executor=_executor(root),
        max_steps=3,
    ).run("Does example.py import pathlib?", task_kind="code")
    events = [dict(event) for event in result.events]
    assert validate_transcript(events) and result.delivered
    return events


def _resequence(events: list[dict[str, object]]) -> list[dict[str, object]]:
    for sequence, event in enumerate(events):
        event["sequence"] = sequence
    return events


def _revision_events(root: Path) -> list[dict[str, object]]:
    case = next(item for item in development_cases() if item.case_id == "revision_01")
    (root / "public-projection.json").write_text(
        json.dumps(public_case(case), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    events = list(
        GenericMcpHost(
            task_id="ordered-revision",
            evaluation_profile=_profile(),
            model=_RevisionModel(),  # type: ignore[arg-type]
            executor=_revision_executor(root),
            max_steps=3,
            resource_paths=("public-projection.json",),
        )
        .run(_goal(case), task_kind="general")
        .events
    )
    assert validate_transcript(events)
    return events


@pytest.mark.parametrize("replacement", ["step", "retract", "complete"])
def test_observe_transition_cannot_be_relabelled(
    tmp_path: Path,
    replacement: str,
) -> None:
    events = _events(tmp_path)
    transition = next(
        event
        for event in events
        if event["type"] == "runtime_transition" and event["transition"] == "observe"
    )
    transition["transition"] = replacement

    assert not validate_transcript(events)


def test_observe_transition_cannot_be_deleted(tmp_path: Path) -> None:
    events = _events(tmp_path)
    mutated = [
        event
        for event in events
        if not (event["type"] == "runtime_transition" and event["transition"] == "observe")
    ]

    assert not validate_transcript(_resequence(mutated))


def test_runtime_transition_cannot_precede_its_tool_result(tmp_path: Path) -> None:
    events = _events(tmp_path)
    result_index = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "tool_result" and event["origin"] == "host"
    )
    events[result_index], events[result_index + 1] = (
        events[result_index + 1],
        events[result_index],
    )

    assert not validate_transcript(_resequence(events))


def test_next_message_cannot_precede_required_runtime_transition(tmp_path: Path) -> None:
    events = _events(tmp_path)
    transition_index = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "runtime_transition" and event["transition"] == "observe"
    )
    events[transition_index], events[transition_index + 1] = (
        events[transition_index + 1],
        events[transition_index],
    )

    assert not validate_transcript(_resequence(events))


@pytest.mark.parametrize("replacement", ["observe", "retract", "complete"])
def test_reasoning_step_transition_cannot_be_relabelled(
    tmp_path: Path,
    replacement: str,
) -> None:
    events = _revision_events(tmp_path)
    transition = next(
        event
        for event in events
        if event["type"] == "runtime_transition" and event["transition"] == "step"
    )
    transition["transition"] = replacement

    assert not validate_transcript(events)


def test_completion_transition_must_match_its_runtime_result(tmp_path: Path) -> None:
    events = _events(tmp_path)
    transition = next(
        event
        for event in events
        if event["type"] == "runtime_transition" and event["transition"] == "complete"
    )
    transition["status"] = "needs_evidence"
    transition["next_action"] = {"type": "complete"}

    assert not validate_transcript(events)


def test_model_declared_tool_call_order_is_mandatory(tmp_path: Path) -> None:
    arguments = {"pattern": "pathlib", "path": "example.py"}
    turn = ModelTurn(
        "local",
        "small",
        "",
        (
            ModelToolCall("first", "host_search", arguments),
            ModelToolCall("second", "host_search", arguments),
        ),
        "tool_calls",
        2,
    )
    events = list(
        GenericMcpHost(
            task_id="ordered-calls",
            evaluation_profile=_hardening_profile("full"),
            model=_Model(turn),  # type: ignore[arg-type]
            executor=_hardening_executor(tmp_path),
            max_steps=1,
        )
        .run("Does example.py import pathlib?")
        .events
    )
    assert validate_transcript(events)
    request_indexes = [
        index for index, event in enumerate(events) if event["type"] == "tool_request"
    ]
    first_block = events[request_indexes[0] : request_indexes[0] + 2]
    second_block = events[request_indexes[1] : request_indexes[1] + 2]
    events[request_indexes[0] : request_indexes[1] + 2] = second_block + first_block

    assert not validate_transcript(_resequence(events))


def test_receipt_cannot_precede_runtime_close(tmp_path: Path) -> None:
    events = _events(tmp_path)
    close_index = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "runtime_transition" and event["transition"] == "complete"
    )
    events[close_index], events[close_index + 1] = events[close_index + 1], events[close_index]

    assert not validate_transcript(_resequence(events))


@pytest.mark.parametrize(
    ("transition_name", "bad_status"),
    [("start", "started"), ("observe", "complete"), ("complete", "abandoned")],
)
def test_transition_status_is_bound_to_its_operation(
    tmp_path: Path,
    transition_name: str,
    bad_status: str,
) -> None:
    events = _events(tmp_path)
    transition = next(
        event
        for event in events
        if event["type"] == "runtime_transition" and event["transition"] == transition_name
    )
    transition["status"] = bad_status

    assert not validate_transcript(events)


def test_start_cannot_announce_a_later_phase_even_with_a_fresh_hash(tmp_path: Path) -> None:
    events = _events(tmp_path)
    transition = next(
        event
        for event in events
        if event["type"] == "runtime_transition" and event["transition"] == "start"
    )
    transition["next_action"] = {"type": "finish"}
    transition["transition_sha256"] = transition_sha256(
        "start",
        str(transition["session_id"]),
        transition,
    )

    assert not validate_transcript(events)


def test_observe_payload_is_bound_to_result_even_with_a_fresh_hash(tmp_path: Path) -> None:
    events = _events(tmp_path)
    transition = next(
        event
        for event in events
        if event["type"] == "runtime_transition" and event["transition"] == "observe"
    )
    transition["next_action"] = {"type": "finish"}
    transition["transition_sha256"] = transition_sha256(
        "observe",
        str(transition["session_id"]),
        transition,
    )

    assert not validate_transcript(events)


def test_certified_terminal_text_is_bound_to_completion_result(tmp_path: Path) -> None:
    events = _events(tmp_path)
    terminal = events[-1]
    terminal["text"] = "A different answer."
    terminal["candidate_sha256"] = encoded_payload_sha256(terminal["text"])

    assert not validate_transcript(events)


def test_bare_release_requires_the_model_message(tmp_path: Path) -> None:
    events = _valid_bare_events(tmp_path)
    mutated = [event for event in events if event["type"] != "message"]

    assert not validate_transcript(_resequence(mutated))


def test_bare_terminal_text_is_bound_to_model_message(tmp_path: Path) -> None:
    events = _valid_bare_events(tmp_path)
    terminal = events[-1]
    terminal["text"] = "Not what the model said."
    terminal["candidate_sha256"] = encoded_payload_sha256(terminal["text"])

    assert not validate_transcript(events)


@pytest.mark.parametrize("retrieval_count", [0, 999])
def test_receipt_retrieval_count_is_bound_to_runtime_requests(
    tmp_path: Path,
    retrieval_count: int,
) -> None:
    events = _events(tmp_path)
    receipt_event = next(event for event in events if event["type"] == "evaluation_receipt")
    receipt = receipt_event["receipt"]
    assert isinstance(receipt, dict)
    operator_counts = receipt["operator_counts"]
    assert isinstance(operator_counts, dict)
    operator_counts["retrieval"] = retrieval_count

    assert not validate_transcript(events)


def test_start_request_cannot_be_substituted_with_refreshed_hash(tmp_path: Path) -> None:
    events = _events(tmp_path)
    transition = next(
        event
        for event in events
        if event["type"] == "runtime_transition" and event["transition"] == "start"
    )
    action = transition["next_action"]
    assert isinstance(action, dict)
    request = action["request"]
    assert isinstance(request, dict)
    request.update(
        request_id="forged-start",
        query="Search secrets.txt for password",
        parameters={"pattern": "password", "path": "secrets.txt"},
    )
    transition["transition_sha256"] = transition_sha256(
        "start",
        str(transition["session_id"]),
        transition,
    )

    assert not validate_transcript(events)


def test_observe_result_and_transition_cannot_be_rewritten_together(tmp_path: Path) -> None:
    events = _events(tmp_path)
    transition_index = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "runtime_transition" and event["transition"] == "observe"
    )
    result = events[transition_index - 1]
    transition = events[transition_index]
    action = {
        "type": "challenge",
        "instruction": "Challenge the current draft.",
        "submit_via": "cortheon_challenge",
    }
    digest = transition_sha256("observe", str(transition["session_id"]), {"next_action": action})
    transition["next_action"] = action
    transition["transition_sha256"] = digest
    result["runtime_transition_sha256"] = digest

    assert not validate_transcript(events)


def test_step_result_and_transition_cannot_be_rewritten_together(tmp_path: Path) -> None:
    events = _revision_events(tmp_path)
    transition_index = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "runtime_transition" and event["transition"] == "step"
    )
    result = events[transition_index - 1]
    transition = events[transition_index]
    content = json.loads(str(result["content"]))
    action = {
        "type": "challenge",
        "instruction": "Challenge the revised draft.",
        "submit_via": "cortheon_challenge",
    }
    content["next_action"] = action
    digest = transition_sha256("step", str(transition["session_id"]), content)
    result["content"] = json.dumps(content, separators=(",", ":"))
    result["result_sha256"] = payload_sha256(content)
    result["runtime_transition_sha256"] = digest
    transition["next_action"] = action
    transition["transition_sha256"] = digest

    assert not validate_transcript(events)


def test_certified_answer_cannot_be_rewritten_across_result_and_terminal(
    tmp_path: Path,
) -> None:
    events = _events(tmp_path)
    result = next(
        event for event in events if event["type"] == "tool_result" and event["origin"] == "mcp"
    )
    content = json.loads(str(result["content"]))
    content["answer"] = "FORGED"
    result["content"] = json.dumps(content, separators=(",", ":"))
    result["result_sha256"] = payload_sha256(content)
    terminal = events[-1]
    terminal["text"] = "FORGED"
    terminal["candidate_sha256"] = encoded_payload_sha256("FORGED")

    assert not validate_transcript(events)


def test_successful_retract_trace_is_strictly_ordered(tmp_path: Path) -> None:
    events = _retract_events(tmp_path)
    retract_index = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "runtime_transition" and event["transition"] == "retract"
    )

    deleted = events[:retract_index] + events[retract_index + 1 :]
    assert not validate_transcript(_resequence(deleted))

    reordered = [dict(event) for event in events]
    reordered[retract_index - 1], reordered[retract_index] = (
        reordered[retract_index],
        reordered[retract_index - 1],
    )
    assert not validate_transcript(_resequence(reordered))


def test_retract_result_is_bound_to_requested_evidence_ids(tmp_path: Path) -> None:
    events = _retract_events(tmp_path)
    result = next(
        event
        for event in events
        if event["type"] == "tool_result"
        and event["origin"] == "mcp"
        and "retracted_evidence_ids" in str(event["content"])
    )
    content = json.loads(str(result["content"]))
    content["retracted_evidence_ids"] = ["ev999"]
    result["content"] = json.dumps(content, separators=(",", ":"))
    result["result_sha256"] = payload_sha256(content)

    assert not validate_transcript(events)


@pytest.mark.parametrize(
    ("action", "allowed"),
    [
        ({"type": "await_candidate", "instruction": "Wait."}, {"host_complete"}),
        (
            {
                "type": "reason",
                "instruction": "Reason.",
                "submit_via": "cortheon_step",
                "required_fields": ["draft"],
            },
            {"host_reason", "cortheon_step"},
        ),
        (
            {
                "type": "challenge",
                "instruction": "Challenge.",
                "submit_via": "cortheon_challenge",
            },
            {"cortheon_challenge"},
        ),
        (
            {"type": "verify", "instruction": "Verify.", "submit_via": "cortheon_verify"},
            {"cortheon_verify", "host_complete"},
        ),
        (
            {"type": "finish", "instruction": "Finish.", "submit_via": "cortheon_finish"},
            {"cortheon_finish", "host_complete"},
        ),
        (
            {
                "type": "complete",
                "instruction": "Complete.",
                "submit_via": "cortheon_complete",
            },
            {"host_complete", "cortheon_complete"},
        ),
        ({"type": "disengage", "instruction": "Disengage."}, set()),
    ],
)
def test_runtime_action_accepts_only_its_mapped_next_tool(
    action: dict[str, object],
    allowed: set[str],
) -> None:
    names = {
        "host_complete",
        "host_reason",
        "cortheon_step",
        "cortheon_challenge",
        "cortheon_verify",
        "cortheon_finish",
        "cortheon_complete",
    }
    for name in names:
        request = {"name": name, "arguments": {}}
        assert host_request_matches_action(action, request, []) is (name in allowed)

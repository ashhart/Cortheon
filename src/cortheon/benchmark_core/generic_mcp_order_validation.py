"""Runtime-transition ordering and payload binding for generic MCP traces."""

from __future__ import annotations

import json
from typing import Any

from cortheon.benchmark_core.generic_mcp_capabilities import PREFERRED_CAPABILITY_TOOL
from cortheon.benchmark_core.generic_mcp_protocol import (
    bounded_identifier,
    canonical_json,
    payload_sha256,
)
from cortheon.benchmark_core.generic_mcp_search_projection import discovery_pattern

_ACTION_TYPES = {
    "await_candidate",
    "challenge",
    "complete",
    "disengage",
    "finish",
    "harness_tool",
    "reason",
    "verify",
}


def validate_runtime_event(
    event: dict[str, Any],
    state: str,
    session_id: str | None,
    *,
    expected_start_action: str,
) -> tuple[bool, str, str | None]:
    transition = event.get("transition")
    current = event.get("session_id")
    status = event.get("status")
    next_action = event.get("next_action")
    projection = {
        "transition": transition,
        "session_id": current,
        "status": status,
        "next_action": next_action,
    }
    if (
        not bounded_identifier(current)
        or (status is not None and not _bounded_status(status))
        or len(canonical_json(next_action)) > 50_000
        or event.get("transition_sha256") != payload_sha256(projection)
        or (next_action is not None and not _valid_next_action(next_action))
    ):
        return False, state, session_id
    if (
        transition == "start"
        and state == "not_started"
        and session_id is None
        and status is None
        and isinstance(next_action, dict)
        and next_action.get("type") == expected_start_action
    ):
        return True, "active", current
    if current != session_id or state != "active":
        return False, state, session_id
    if transition == "observe" and status in {None, "disengaged"} and isinstance(next_action, dict):
        return True, state, session_id
    if transition in {"retract", "step"} and status is None and isinstance(next_action, dict):
        return True, state, session_id
    if transition == "complete" and status is None and isinstance(next_action, dict):
        return True, state, session_id
    if (transition, status) in {
        ("complete", "complete"),
        ("abandon", "abandoned"),
    } and next_action is None:
        return True, "closed", session_id
    return False, state, session_id


def expected_transition(
    start: dict[str, Any],
    request: dict[str, Any],
    result: dict[str, Any],
) -> str | None:
    if not start["runtime_used"]:
        return None
    if request["origin"] == "host":
        rejected = result.get("status") == "error" and not result.get("accepted_evidence_ids")
        return None if rejected else "observe"
    content = _json_payload(result.get("content"))
    if not isinstance(content, dict) or content.get("status") == "rejected":
        return None
    return {
        "host_reason": "step",
        "host_complete": "complete",
        "cortheon_complete": "complete",
        "cortheon_retract": "retract",
        "cortheon_abandon": "abandon",
    }.get(str(request["name"]))


def transition_matches_result(
    event: dict[str, Any],
    expected: str,
    request: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    if event.get("transition") != expected:
        return False
    if request["origin"] == "host":
        return bool(
            expected == "observe"
            and event.get("transition_sha256") == result.get("runtime_transition_sha256")
        )
    content = _json_payload(result.get("content"))
    return bool(
        isinstance(content, dict)
        and event.get("status") == content.get("status")
        and event.get("next_action") == content.get("next_action")
        and event.get("transition_sha256") == result.get("runtime_transition_sha256")
        and ("session_id" not in content or event.get("session_id") == content.get("session_id"))
        and _mcp_result_matches_request(expected, request, content)
    )


def transition_sha256(
    transition: str,
    session_id: str | None,
    payload: dict[str, Any],
) -> str:
    """Bind the visible runtime transition fields to one digest."""
    return payload_sha256(
        {
            "transition": transition,
            "session_id": session_id,
            "status": payload.get("status"),
            "next_action": payload.get("next_action"),
        }
    )


def next_request_id(event: dict[str, Any]) -> tuple[bool, str | None]:
    action = event.get("next_action")
    if not isinstance(action, dict) or "request" not in action:
        return True, None
    request = action.get("request")
    request_id = request.get("request_id") if isinstance(request, dict) else None
    return bounded_identifier(request_id), request_id if isinstance(request_id, str) else None


def host_request_matches_action(
    action: dict[str, Any] | None,
    request: dict[str, Any],
    resource_paths: list[str],
) -> bool:
    if not isinstance(action, dict):
        return True
    action_type = action.get("type")
    if request.get("name") in {"cortheon_retract", "cortheon_abandon"}:
        return True
    if action_type != "harness_tool":
        submit_via = action.get("submit_via")
        name = request.get("name")
        aliases = {
            "cortheon_step": {"cortheon_step", "host_reason"},
            "cortheon_challenge": {"cortheon_challenge"},
            "cortheon_verify": {"cortheon_verify", "host_complete"},
            "cortheon_finish": {"cortheon_finish", "host_complete"},
            "cortheon_complete": {"cortheon_complete", "host_complete"},
        }
        if submit_via in aliases:
            if name in aliases[submit_via]:
                return True
            return _is_fused_reason_completion(action, request)
        return action_type == "await_candidate" and name == "host_complete"
    evidence_request = action.get("request")
    if not isinstance(evidence_request, dict):
        return False
    capability = evidence_request.get("capability")
    name = request.get("name")
    arguments = request.get("arguments")
    if (
        not isinstance(capability, str)
        or name != PREFERRED_CAPABILITY_TOOL.get(capability)
        or not isinstance(arguments, dict)
    ):
        return False
    parameters = evidence_request.get("parameters")
    params = parameters if isinstance(parameters, dict) else {}
    if name == "host_search" and capability == "grep":
        return arguments == {"pattern": params.get("pattern"), "path": params.get("path")}
    if name == "host_search":
        pattern = arguments.get("pattern")
        query = str(evidence_request.get("query", ""))
        projected = discovery_pattern(query)
        return bool(
            set(arguments) == {"pattern", "path"}
            and arguments.get("path") == "."
            and isinstance(pattern, str)
            and len(pattern) >= 3
            and (pattern.casefold() in query.casefold() or pattern == projected)
        )
    if name == "host_read":
        path = params.get("path")
        if not isinstance(path, str) and len(resource_paths) == 1:
            path = resource_paths[0]
        return isinstance(path, str) and arguments == {"path": path}
    if name == "host_read_many":
        return arguments == {"paths": params.get("paths")}
    if name == "host_web_search":
        return arguments == {"query": evidence_request.get("query")}
    if name == "host_web_fetch":
        return arguments == {"url": params.get("url")}
    if name == "host_diff":
        paths = params.get("paths") or (
            [params["path"]] if isinstance(params.get("path"), str) else None
        )
        return arguments == {"paths": paths}
    return name == "host_test" and set(arguments) == {"test_id"}


def _valid_next_action(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("type") not in _ACTION_TYPES:
        return False
    if not _bounded_instruction(value.get("instruction")):
        return False
    action_type = value["type"]
    submit_via = value.get("submit_via")
    required_submit = {
        "harness_tool": "cortheon_observe",
        "challenge": "cortheon_challenge",
        "verify": "cortheon_verify",
        "finish": "cortheon_finish",
        "complete": "cortheon_complete",
    }
    if action_type in required_submit and submit_via != required_submit[action_type]:
        return False
    if action_type == "reason" and submit_via not in {
        "cortheon_step",
        "cortheon_challenge",
        "cortheon_complete",
    }:
        return False
    if action_type == "await_candidate" and submit_via not in {None, "cortheon_verify"}:
        return False
    if action_type == "disengage" and submit_via is not None:
        return False
    if action_type == "harness_tool":
        request = value.get("request")
        if not _valid_evidence_request(request):
            return False
    elif "request" in value:
        return False
    required_fields = value.get("required_fields")
    return required_fields is None or (
        isinstance(required_fields, list)
        and len(required_fields) <= 16
        and len(set(required_fields)) == len(required_fields)
        and all(bounded_identifier(item) for item in required_fields)
    )


def _valid_evidence_request(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and bounded_identifier(value.get("request_id"))
        and value.get("capability") in PREFERRED_CAPABILITY_TOOL
        and _bounded_instruction(value.get("query"))
        and _bounded_instruction(value.get("reason"))
        and _bounded_instruction(value.get("success_condition"))
        and isinstance(value.get("parameters"), dict)
        and len(canonical_json(value["parameters"])) <= 20_000
        and (value.get("hypothesis_id") is None or bounded_identifier(value["hypothesis_id"]))
        and value.get("status") in {"pending", "completed", "superseded", "waived"}
    )


def _mcp_result_matches_request(
    expected: str,
    request: dict[str, Any],
    content: dict[str, Any],
) -> bool:
    if expected == "step" and request.get("name") == "host_reason":
        return isinstance(content.get("next_action"), dict)
    if expected == "retract":
        arguments = request.get("arguments", {})
        return bool(
            content.get("retracted_evidence_ids") == arguments.get("evidence_ids")
            and content.get("retraction_reason") == arguments.get("reason", "model_correction")
        )
    if expected != "complete" or content.get("status") != "complete":
        return True
    requested_answer = request.get("arguments", {}).get("answer")
    if isinstance(requested_answer, dict):
        requested_answer = canonical_json(requested_answer)
    return isinstance(requested_answer, str) and content.get("answer") == requested_answer


def _bounded_instruction(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 20_000


def _is_fused_reason_completion(
    action: dict[str, Any],
    request: dict[str, Any],
) -> bool:
    required = action.get("required_fields")
    arguments = request.get("arguments")
    completion_field = {
        "draft": "answer",
        "claims": "claims",
        "hypotheses": "hypotheses",
    }
    return bool(
        action.get("type") == "reason"
        and action.get("submit_via") in {"cortheon_step", "cortheon_challenge"}
        and isinstance(required, list)
        and required
        and request.get("name") == "host_complete"
        and isinstance(arguments, dict)
        and all(
            field in completion_field and completion_field[field] in arguments for field in required
        )
    )


def _bounded_status(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 128


def _json_payload(value: Any) -> Any:
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None

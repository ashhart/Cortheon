from __future__ import annotations

from typing import Any

from cortheon.benchmark_core.generic_mcp_message_validation import valid_message
from cortheon.benchmark_core.generic_mcp_order_validation import (
    expected_transition as _expected_transition_for_result,
)
from cortheon.benchmark_core.generic_mcp_order_validation import (
    host_request_matches_action,
    next_request_id,
)
from cortheon.benchmark_core.generic_mcp_order_validation import (
    transition_matches_result as _transition_matches_result,
)
from cortheon.benchmark_core.generic_mcp_order_validation import (
    validate_runtime_event as _validate_runtime_event,
)
from cortheon.benchmark_core.generic_mcp_profile_validation import _profile
from cortheon.benchmark_core.generic_mcp_protocol import (
    COMMON_KEYS,
    EVENT_KEYS,
    EVENT_TYPES,
    GENERIC_MCP_TRANSCRIPT_VERSION,
    MAX_EVENT_CHARS,
    OPERATOR_KEYS,
    SHA256,
    bounded_identifier,
    canonical_json,
    payload_sha256,
)
from cortheon.benchmark_core.generic_mcp_schema_validation import (
    valid_tool_arguments,
)
from cortheon.benchmark_core.generic_mcp_terminal_validation import valid_terminal

_HOST_TOOLS = {
    "host_search",
    "host_read",
    "host_read_many",
    "host_diff",
    "host_test",
    "host_web_search",
    "host_web_fetch",
}
_MCP_TOOLS = {
    "cortheon_complete",
    "cortheon_retract",
    "cortheon_abandon",
    "host_complete",
    "host_reason",
}
_REJECTABLE_MCP_TOOLS = {
    "cortheon_start",
    "cortheon_step",
    "cortheon_observe",
    "cortheon_resume",
    "cortheon_challenge",
    "cortheon_verify",
    "cortheon_finish",
}
_HOST_STATUSES = {"result", "match", "no_match", "changed", "passed", "failed", "error"}


def _sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _bounded_text(value: Any, maximum: int, *, empty: bool = True) -> bool:
    return isinstance(value, str) and len(value) <= maximum and (empty or bool(value))


def _closed_event(event: Any, sequence: int, task_id: str, nonce: str) -> bool:
    if not isinstance(event, dict) or len(canonical_json(event)) > MAX_EVENT_CHARS:
        return False
    event_type = event.get("type")
    return bool(
        event_type in EVENT_TYPES
        and set(event) == COMMON_KEYS | EVENT_KEYS[event_type]
        and event.get("schema_version") == GENERIC_MCP_TRANSCRIPT_VERSION
        and event.get("sequence") == sequence
        and event.get("task_id") == task_id
        and event.get("nonce") == nonce
    )


def _request(event: dict[str, Any]) -> bool:
    arguments = event.get("arguments")
    origin = event.get("origin")
    name = event.get("name")
    return bool(
        bounded_identifier(event.get("call_id"))
        and origin in {"host", "mcp"}
        and _bounded_text(name, 128, empty=False)
        and (
            (origin == "host" and name in _HOST_TOOLS)
            or (origin == "mcp" and name in _MCP_TOOLS | _REJECTABLE_MCP_TOOLS)
        )
        and isinstance(arguments, dict)
        and len(canonical_json(arguments)) <= 50_000
        and event.get("request_sha256") == payload_sha256([event["name"], arguments])
    )


def _result(
    event: dict[str, Any],
    requests: dict[str, dict[str, Any]],
    resolved: set[str],
    argument_validity: dict[str, bool],
    offer_validity: dict[str, bool],
) -> bool:
    call_id = event.get("call_id")
    request = requests.get(call_id) if isinstance(call_id, str) else None
    if (
        request is None
        or call_id in resolved
        or event.get("origin") != request["origin"]
        or event.get("request_sha256") != request["request_sha256"]
        or not _bounded_text(event.get("content"), 40_000)
        or not _sha(event.get("result_sha256"))
    ):
        return False
    if event["origin"] == "host":
        receipt = event.get("receipt")
        expected = {
            "tool": request["name"].removeprefix("host_"),
            "executor": "generic_mcp_wrapper",
            "outcome": event.get("status"),
            "args": request["arguments"],
        }
        evidence = event.get("accepted_evidence_ids")
        valid = bool(
            event.get("status") in _HOST_STATUSES
            and (
                (
                    argument_validity.get(str(call_id)) is True
                    and offer_validity.get(str(call_id)) is True
                )
                or event.get("status") == "error"
            )
            and receipt == expected
            and isinstance(evidence, list)
            and len(evidence) <= 128
            and all(bounded_identifier(item) for item in evidence)
            and event["result_sha256"]
            == payload_sha256([event["status"], event["content"], receipt])
            and (
                event.get("runtime_transition_sha256") is None
                or _sha(event.get("runtime_transition_sha256"))
            )
        )
    else:
        content_value = _json_content(event["content"])
        valid = bool(
            event.get("status") == "result"
            and event.get("receipt") is None
            and event.get("accepted_evidence_ids") == []
            and (
                event.get("runtime_transition_sha256") is None
                or _sha(event.get("runtime_transition_sha256"))
            )
            and event["result_sha256"] == payload_sha256(content_value)
            and (
                (
                    offer_validity.get(str(call_id)) is True
                    and argument_validity.get(str(call_id)) is True
                )
                or (isinstance(content_value, dict) and content_value.get("status") == "rejected")
            )
        )
    if valid:
        resolved.add(str(call_id))
    return valid


def _json_content(content: str) -> Any:
    import json

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _receipt(
    receipt: Any,
    profile: dict[str, Any],
    *,
    retrieval_count: int,
    verification_count: int,
) -> bool:
    config = profile["config"]
    adapter = {
        "schema_version": 1,
        "host": "generic_mcp",
        "control_transport": "fd",
        "config_sha256": profile["config_sha256"],
        "nonce": profile["nonce"],
        "operators": config["operators"],
    }
    return bool(
        isinstance(receipt, dict)
        and set(receipt)
        == {
            "schema_version",
            "config_sha256",
            "implementation_sha256",
            "intercepts_final",
            "cleanup_before_answer",
            "runtime_profile_received",
            "adapter_receipt",
            "operator_counts",
        }
        and receipt.get("schema_version") == 1
        and receipt.get("config_sha256") == profile["config_sha256"]
        and receipt.get("implementation_sha256") == profile["implementation_sha256"]
        and receipt.get("intercepts_final") is config["intercepts_final"]
        and receipt.get("cleanup_before_answer") is config["cleanup_before_answer"]
        and receipt.get("runtime_profile_received") is True
        and receipt.get("adapter_receipt") == adapter
        and isinstance(receipt.get("operator_counts"), dict)
        and set(receipt["operator_counts"]) == OPERATOR_KEYS
        and all(type(count) is int and count >= 0 for count in receipt["operator_counts"].values())
        and all(
            config["operators"][operator] is True or count == 0
            for operator, count in receipt["operator_counts"].items()
        )
        and receipt["operator_counts"]["retrieval"] == retrieval_count
        and receipt["operator_counts"]["verification"] == verification_count
    )


def validate_transcript(events: list[dict[str, Any]], *, require_web: bool = False) -> bool:
    if not isinstance(events, list) or len(events) < 2 or len(events) > 512:
        return False
    start = events[0]
    if (
        not isinstance(start, dict)
        or start.get("type") != "task_start"
        or not bounded_identifier(start.get("task_id"))
        or not bounded_identifier(start.get("nonce"))
    ):
        return False
    task_id, nonce = start["task_id"], start["nonce"]
    if not _closed_event(start, 0, task_id, nonce):
        return False
    profile = _profile(start, bounded_text=_bounded_text, valid_sha=_sha)
    if profile is None or (require_web and start["capabilities"]["current_web"] is not True):
        return False
    requests: dict[str, dict[str, Any]] = {}
    argument_validity: dict[str, bool] = {}
    offer_validity: dict[str, bool] = {}
    action_validity: dict[str, bool] = {}
    resolved: set[str] = set()
    announced: dict[str, tuple[frozenset[str], str, dict[str, dict[str, Any]]]] = {}
    runtime_decisions: dict[str, dict[str, Any]] = {}
    announced_queue: list[str] = []
    active_call: str | None = None
    expected_transition: str | None = None
    transition_request: dict[str, Any] | None = None
    transition_result: dict[str, Any] | None = None
    runtime_request_ids: set[str] = set()
    verification_count = 0
    message_count = 0
    last_assistant_content: str | None = None
    last_message_had_calls = False
    certified_answer: str | None = None
    runtime_action: dict[str, Any] | None = None
    state = "not_started"
    session_id: str | None = None
    close_transition: str | None = None
    receipt_seen = terminal_seen = False
    for sequence, event in enumerate(events[1:], 1):
        if terminal_seen or not _closed_event(event, sequence, task_id, nonce):
            return False
        event_type = event["type"]
        if start["runtime_used"] and sequence == 1 and event_type != "runtime_transition":
            return False
        if expected_transition is not None and event_type != "runtime_transition":
            return False
        if active_call is not None and event_type != "tool_result":
            return False
        if (
            announced_queue
            and active_call is None
            and expected_transition is None
            and event_type != "tool_request"
        ):
            return False
        if receipt_seen and event_type != "terminal":
            return False
        if state == "closed" and not receipt_seen and event_type != "evaluation_receipt":
            return False
        if event_type == "message":
            call_ids = event.get("tool_call_ids", [])
            if (
                active_call is not None
                or expected_transition is not None
                or announced_queue
                or not valid_message(event, start)
                or len(set(call_ids)) != len(call_ids)
                or any(item in announced for item in call_ids)
            ):
                return False
            offered = frozenset(event["available_tools"])
            catalogue = {tool["function"]["name"]: tool for tool in event["tool_catalogue"]}
            announced.update(dict.fromkeys(call_ids, (offered, event["tool_choice"], catalogue)))
            announced_queue.extend(call_ids)
            message_count += 1
            last_assistant_content = event["content"]
            last_message_had_calls = bool(call_ids)
        elif event_type == "runtime_tool_decision":
            call_id = event.get("call_id")
            request_view = {
                "name": event.get("name"),
                "arguments": event.get("arguments"),
            }
            if (
                active_call is not None
                or expected_transition is not None
                or announced_queue
                or not bounded_identifier(call_id)
                or call_id in announced
                or event.get("request_sha256")
                != payload_sha256([event.get("name"), event.get("arguments")])
                or not host_request_matches_action(
                    runtime_action, request_view, start["resource_paths"]
                )
            ):
                return False
            runtime_decisions[str(call_id)] = event
            announced[str(call_id)] = (
                frozenset({str(event.get("name"))}),
                str(event.get("name")),
                {},
            )
            announced_queue.append(str(call_id))
        if event_type == "tool_request":
            call_id = event.get("call_id")
            offered = announced.get(call_id) if isinstance(call_id, str) else None
            if (
                not _request(event)
                or not announced_queue
                or event["call_id"] != announced_queue[0]
                or event["call_id"] in requests
                or event["call_id"] not in announced
                or offered is None
                or (start["runtime_used"] and state != "active")
                or (
                    event["call_id"] in runtime_decisions
                    and any(
                        event.get(key) != runtime_decisions[event["call_id"]].get(key)
                        for key in ("name", "arguments", "request_sha256")
                    )
                )
            ):
                return False
            announced_queue.pop(0)
            active_call = event["call_id"]
            requests[event["call_id"]] = event
            automatic = event["call_id"] in runtime_decisions
            argument_validity[event["call_id"]] = automatic or valid_tool_arguments(
                event["arguments"], offered[2].get(event["name"])
            )
            offer_validity[event["call_id"]] = automatic or bool(
                event["name"] in offered[0]
                and (offered[1] == "auto" or event["name"] == offered[1])
            )
            action_validity[event["call_id"]] = host_request_matches_action(
                runtime_action,
                event,
                start["resource_paths"],
            )
        elif event_type == "tool_result":
            if (
                active_call is None
                or event.get("call_id") != active_call
                or not _result(
                    event,
                    requests,
                    resolved,
                    argument_validity,
                    offer_validity,
                )
            ):
                return False
            transition_request = requests[active_call]
            transition_result = event
            content_value = _json_content(str(event.get("content", "")))
            if (
                transition_request["origin"] == "mcp"
                and transition_request["name"] in {"host_complete", "cortheon_complete"}
                and isinstance(content_value, dict)
                and content_value.get("status") == "complete"
                and isinstance(content_value.get("answer"), str)
            ):
                certified_answer = content_value["answer"]
            expected_transition = _expected_transition_for_result(
                start,
                transition_request,
                event,
            )
            if expected_transition is not None and not action_validity[active_call]:
                return False
            if expected_transition is None:
                transition_request = transition_result = None
            active_call = None
        elif event_type == "runtime_transition":
            matched_transition = expected_transition
            if expected_transition is not None:
                if (
                    transition_request is None
                    or transition_result is None
                    or not _transition_matches_result(
                        event,
                        expected_transition,
                        transition_request,
                        transition_result,
                    )
                ):
                    return False
                expected_transition = None
                transition_request = transition_result = None
            elif not (
                (event.get("transition") == "start" and sequence == 1)
                or (
                    event.get("transition") == "abandon"
                    and not announced_queue
                    and active_call is None
                )
            ):
                return False
            valid, state, session_id = _validate_runtime_event(
                event,
                state,
                session_id,
                expected_start_action=(
                    "harness_tool"
                    if profile["config"]["operators"]["retrieval"]
                    else "await_candidate"
                ),
            )
            if not valid or receipt_seen or (event["transition"] == "start" and sequence != 1):
                return False
            valid_request, request_id = next_request_id(event)
            if not valid_request:
                return False
            if request_id is not None:
                runtime_request_ids.add(request_id)
            runtime_action = event.get("next_action")
            if matched_transition == "complete":
                verification_count += 1
            if event["transition"] in {"complete", "abandon"} and event.get("next_action") is None:
                close_transition = event["transition"]
        elif event_type == "evaluation_receipt":
            if (
                receipt_seen
                or state != "closed"
                or not _receipt(
                    event.get("receipt"),
                    profile,
                    retrieval_count=len(runtime_request_ids),
                    verification_count=verification_count,
                )
            ):
                return False
            receipt_seen = True
        elif event_type == "terminal":
            provenance = event.get("provenance")
            released_text_bound = bool(
                (
                    provenance == "cortheon_complete"
                    and certified_answer is not None
                    and event.get("text") == certified_answer
                )
                or (
                    provenance == "generic_mcp_model"
                    and message_count > 0
                    and not last_message_had_calls
                    and event.get("text") == last_assistant_content
                )
                or provenance == "generic_mcp_wrapper"
            )
            if not released_text_bound or not valid_terminal(
                event, start, state, receipt_seen, close_transition
            ):
                return False
            terminal_seen = True
    return bool(
        terminal_seen
        and active_call is None
        and expected_transition is None
        and not announced_queue
        and resolved == set(requests) == set(announced)
    )

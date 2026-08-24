"""Model-turn conversion and bounded forced-call rejection."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

from cortheon.benchmark_core.generic_mcp_model import ModelToolCall, ModelTurn
from cortheon.benchmark_core.generic_mcp_protocol import bounded_identifier
from cortheon.benchmark_core.generic_mcp_schema_validation import valid_tool_arguments
from cortheon.benchmark_core.generic_mcp_tools import HOST_TOOL_NAMES, registrable_tool_name


def assistant_message(turn: ModelTurn) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": turn.content}
    if turn.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, separators=(",", ":")),
                },
            }
            for call in turn.tool_calls
        ]
    return message


def bind_forced_arguments(
    turn: ModelTurn,
    offered_tools: list[dict[str, Any]],
    choice: str,
) -> ModelTurn:
    """Use evaluator-owned constants when the model cannot choose arguments."""

    return bind_forced_turn(turn, offered_tools, choice)[0]


def bind_forced_turn(
    turn: ModelTurn,
    offered_tools: list[dict[str, Any]],
    choice: str,
) -> tuple[ModelTurn, str]:
    """Bind an exact runtime action and report what the adapter repaired."""

    if choice == "auto" or len(offered_tools) != 1:
        return turn, "none"
    function = offered_tools[0].get("function")
    parameters = function.get("parameters") if isinstance(function, dict) else None
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    required = parameters.get("required") if isinstance(parameters, dict) else None
    if (
        choice == "host_reason"
        and isinstance(properties, dict)
        and set(properties) == {"draft"}
        and len(turn.tool_calls) == 1
        and turn.tool_calls[0].name == choice
        and isinstance(turn.tool_calls[0].arguments.get("answer"), str)
    ):
        call = turn.tool_calls[0]
        projected = (ModelToolCall(call.call_id, call.name, {"draft": call.arguments["answer"]}),)
        return replace(turn, tool_calls=projected), "repair_projection"
    if (
        isinstance(parameters, dict)
        and len(turn.tool_calls) == 1
        and turn.tool_calls[0].name == choice
    ):
        call = turn.tool_calls[0]
        projected_arguments = _closed_projection(call.arguments, parameters)
        const_bound = (
            projected_arguments
            if choice in HOST_TOOL_NAMES
            else _bind_const_properties(projected_arguments, parameters)
        )
        if const_bound != call.arguments:
            binding = "arguments" if const_bound != projected_arguments else "schema_projection"
            projected = (ModelToolCall(call.call_id, call.name, const_bound),)
            return replace(turn, tool_calls=projected), binding
    if (
        not isinstance(function, dict)
        or function.get("name") != choice
        or not isinstance(properties, dict)
        or not properties
        or not isinstance(required, list)
        or set(required) != set(properties)
        or any(
            not isinstance(schema, dict) or set(schema) != {"const"}
            for schema in properties.values()
        )
        or len(turn.tool_calls) != 1
    ):
        return turn, "none"
    arguments = {key: deepcopy(schema["const"]) for key, schema in properties.items()}
    for call in turn.tool_calls:
        if call.name != choice and call.name in HOST_TOOL_NAMES:
            calls = (ModelToolCall(call.call_id, choice, deepcopy(arguments)),)
            return replace(turn, tool_calls=calls), "tool_and_arguments"
        if call.name != choice:
            return turn, "none"
        if set(call.arguments) != set(arguments):
            return turn, "none"
        for key, expected in arguments.items():
            actual = call.arguments[key]
            if actual == expected:
                continue
            if not isinstance(actual, str) or not isinstance(expected, (dict, list)):
                return turn, "none"
            try:
                decoded = json.loads(actual)
            except json.JSONDecodeError:
                return turn, "none"
            if decoded != expected:
                return turn, "none"
    calls = tuple(
        ModelToolCall(call.call_id, call.name, deepcopy(arguments)) for call in turn.tool_calls
    )
    bound = replace(turn, tool_calls=calls)
    return bound, "arguments" if bound != turn else "none"


def _closed_projection(value: Any, schema: dict[str, Any]) -> Any:
    """Remove only fields forbidden by a closed JSON Schema object."""

    if isinstance(value, dict) and schema.get("type") == "object":
        properties = schema.get("properties")
        if not isinstance(properties, dict) or schema.get("additionalProperties") is not False:
            return value
        return {
            key: _closed_projection(item, properties[key])
            for key, item in value.items()
            if key in properties and isinstance(properties[key], dict)
        }
    items = schema.get("items")
    if isinstance(value, list) and isinstance(items, dict):
        return [_closed_projection(item, items) for item in value]
    return value


def _bind_const_properties(value: Any, schema: dict[str, Any]) -> Any:
    """Insert only evaluator-owned const fields into one closed object."""

    if not isinstance(value, dict) or schema.get("type") != "object":
        return value
    properties = schema.get("properties")
    if not isinstance(properties, dict) or schema.get("additionalProperties") is not False:
        return value
    bound = dict(value)
    for key, field_schema in properties.items():
        if not isinstance(field_schema, dict) or set(field_schema) != {"const"}:
            continue
        expected = field_schema["const"]
        if key != "reasoning_binding":
            bound[key] = deepcopy(expected)
            continue
        actual = bound.get(key)
        if not isinstance(actual, str) or not isinstance(expected, (dict, list)):
            continue
        try:
            decoded = json.loads(actual)
        except json.JSONDecodeError:
            continue
        if decoded == expected:
            bound[key] = deepcopy(expected)
    return bound


def reject_duplicate_forced_calls(host: Any, tool_calls: tuple[Any, ...], choice: str) -> bool:
    """Resolve every duplicate without executing any model-requested side effect."""

    if len(tool_calls) <= 1 or any(call.name != choice for call in tool_calls):
        return False
    return _reject_calls(host, tool_calls, "duplicate forced tool calls")


def reject_duplicate_call_ids(host: Any, tool_calls: tuple[Any, ...]) -> bool:
    """Reject one malformed model turn before any repeated id can execute."""

    call_ids = [call.call_id for call in tool_calls]
    if len(call_ids) == len(set(call_ids)):
        return False
    return _reject_calls(host, tool_calls, "duplicate tool call ids")


def reject_invalid_call_ids(host: Any, tool_calls: tuple[Any, ...]) -> bool:
    """Close safely when a model emits ids the transcript cannot represent."""

    if all(bounded_identifier(call.call_id) for call in tool_calls):
        return False
    closed = host._abandon_runtime()
    host._emit_receipt()
    host._terminal(host.terminal.withheld("invalid tool call ids", runtime_closed=closed))
    return True


def reject_invalid_tool_names(host: Any, tool_calls: tuple[Any, ...]) -> bool:
    """Close safely when the closed ledger cannot represent a model tool name."""

    if all(registrable_tool_name(call.name) for call in tool_calls):
        return False
    closed = host._abandon_runtime()
    host._emit_receipt()
    host._terminal(host.terminal.withheld("invalid tool names", runtime_closed=closed))
    return True


def reject_unoffered_calls(
    host: Any,
    tool_calls: tuple[Any, ...],
    offered_tools: list[dict[str, Any]],
    choice: str,
) -> bool:
    """Record ignored tool constraints as a candidate failure, never an execution."""

    offered = {tool["function"]["name"] for tool in offered_tools}
    if all(
        call.name in offered and (choice == "auto" or call.name == choice) for call in tool_calls
    ):
        return False
    return _reject_calls(host, tool_calls, "model ignored the forced tool contract")


def retry_stale_host_calls(
    host: Any,
    tool_calls: tuple[Any, ...],
    offered_tools: list[dict[str, Any]],
    choice: str,
) -> list[tuple[str, dict[str, str]]] | None:
    """Resolve one stale host-tool call and offer the active bridge again."""

    offered = {tool["function"]["name"] for tool in offered_tools}
    if (
        choice not in {"host_reason", "host_complete"}
        or choice in host._stale_host_call_retry_phases
        or not tool_calls
        or all(call.name in offered for call in tool_calls)
        or any(call.name not in HOST_TOOL_NAMES for call in tool_calls)
    ):
        return None
    host._stale_host_call_retry_phases.add(choice)
    reason = "stale host tool; the runtime advanced to a reasoning bridge"
    results: list[tuple[str, dict[str, str]]] = []
    for call in tool_calls:
        request = host.executor.ledger.request(call.call_id, call.name, call.arguments)
        host.transcript.record(
            "tool_request",
            {
                "call_id": call.call_id,
                "origin": "host",
                "name": call.name,
                "arguments": call.arguments,
                "request_sha256": request.request_sha256,
            },
        )
        receipt = {
            "tool": call.name.removeprefix("host_"),
            "executor": "generic_mcp_wrapper",
            "outcome": "error",
            "args": call.arguments,
        }
        rejected = host.executor.ledger.record(
            request,
            status="error",
            content=reason,
            receipt=receipt,
        )
        host._tool_result(rejected, {})
        results.append((call.call_id, {"status": "error", "content": reason}))
    return results


def reject_invalid_calls(
    host: Any,
    tool_calls: tuple[Any, ...],
    offered_tools: list[dict[str, Any]],
) -> bool:
    """Resolve schema-invalid model calls without executing their side effects."""

    catalogue = {tool["function"]["name"]: tool for tool in offered_tools}
    if all(valid_tool_arguments(call.arguments, catalogue.get(call.name)) for call in tool_calls):
        return False
    return _reject_calls(host, tool_calls, "model tool arguments violated the offered schema")


def _reject_calls(host: Any, tool_calls: tuple[Any, ...], reason: str) -> bool:
    resolved_ids: set[str] = set()
    for call in tool_calls:
        if call.call_id in resolved_ids:
            continue
        resolved_ids.add(call.call_id)
        origin = (
            "mcp"
            if call.name in {"host_complete", "host_reason"} or call.name.startswith("cortheon_")
            else "host"
        )
        request = host.executor.ledger.request(call.call_id, call.name, call.arguments)
        host.transcript.record(
            "tool_request",
            {
                "call_id": call.call_id,
                "origin": origin,
                "name": call.name,
                "arguments": call.arguments,
                "request_sha256": request.request_sha256,
            },
        )
        if origin == "mcp":
            host._mcp_tool_result(
                call.call_id,
                request.request_sha256,
                {"status": "rejected", "error": reason},
            )
            continue
        receipt = {
            "tool": call.name.removeprefix("host_"),
            "executor": "generic_mcp_wrapper",
            "outcome": "error",
            "args": call.arguments,
        }
        rejected = host.executor.ledger.record(
            request,
            status="error",
            content=reason,
            receipt=receipt,
        )
        host._tool_result(rejected, {})
    closed = host._abandon_runtime()
    host._emit_receipt()
    host._terminal(host.terminal.withheld(reason, runtime_closed=closed))
    return True

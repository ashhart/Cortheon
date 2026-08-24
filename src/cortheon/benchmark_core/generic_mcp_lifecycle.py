"""Bounded Cortheon lifecycle handling for the generic MCP host."""

from __future__ import annotations

from typing import Any

from cortheon.benchmark_core import generic_mcp_protocol as protocol
from cortheon.benchmark_core.generic_mcp_brief import completion_brief, revision_brief
from cortheon.benchmark_core.generic_mcp_projection import reasoning_binding

_TURN_BUDGET_ERROR = "the reasoning-turn budget is exhausted"


def execute_lifecycle_call(
    host: Any,
    call_id: str,
    name: str,
    arguments: dict[str, Any],
    request_sha256: str,
) -> tuple[dict[str, Any], bool]:
    """Execute one bridge call and close repeated failed completions."""

    lifecycle_name = {
        "host_complete": "cortheon_complete",
        "host_reason": "cortheon_step",
    }.get(name, name)
    lifecycle_arguments = dict(arguments)
    public_revision = (
        dict(lifecycle_arguments)
        if name == "host_reason" and host.runtime.projects_revision_reasoning()
        else None
    )
    public_discrimination = (
        dict(lifecycle_arguments)
        if name == "host_reason" and host.runtime.projects_discrimination_reasoning()
        else None
    )
    public_derivation = (
        dict(lifecycle_arguments)
        if name == "host_reason" and host.runtime.projects_derivation_reasoning()
        else None
    )
    if name == "host_complete" and isinstance(lifecycle_arguments.get("answer"), dict):
        lifecycle_arguments["answer"] = protocol.canonical_json(lifecycle_arguments["answer"])
    if isinstance(public_revision, dict):
        lifecycle_arguments = {"draft": protocol.canonical_json(public_revision)}
    if isinstance(public_discrimination, dict):
        lifecycle_arguments = {"draft": protocol.canonical_json(public_discrimination)}
    if isinstance(public_derivation, dict):
        lifecycle_arguments = {"draft": protocol.canonical_json(public_derivation)}
    try:
        if name == "host_complete" and host._reasoning_binding is not None:
            supplied_binding = lifecycle_arguments.pop("reasoning_binding", None)
            if supplied_binding != host._reasoning_binding:
                raise ValueError("completion did not bind the accepted reasoning record")
        result = host.runtime.lifecycle_call(lifecycle_name, lifecycle_arguments)
    except (RuntimeError, ValueError) as exc:
        return _lifecycle_rejection(host, call_id, name, request_sha256, exc)

    transition = lifecycle_name.removeprefix("cortheon_")
    host._mcp_tool_result(
        call_id,
        request_sha256,
        result,
        transition=transition,
    )
    host._runtime_event(transition, result)
    if result.get("status") == "complete":
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer:
            raise RuntimeError("certified completion returned no answer")
        host._emit_receipt()
        host._terminal(host.terminal.certified(answer, runtime_closed=True))
        return result, True
    if name == "host_reason":
        return _reason_result(host, result, public_revision), False
    return _failed_completion(host, arguments, result)


def _lifecycle_rejection(
    host: Any,
    call_id: str,
    name: str,
    request_sha256: str,
    error: Exception,
) -> tuple[dict[str, Any], bool]:
    if name != "host_complete" and not str(error).startswith(_TURN_BUDGET_ERROR):
        raise error
    rejected = {"status": "rejected", "error": str(error)[:500]}
    host._mcp_tool_result(call_id, request_sha256, rejected, transition=None)
    closed = host._abandon_runtime()
    host._emit_receipt()
    reason = (
        "reasoning budget exhausted before certification"
        if name != "host_complete"
        else "completion arguments were rejected"
    )
    host._terminal(host.terminal.withheld(reason, runtime_closed=closed))
    return rejected, True


def _reason_result(
    host: Any,
    result: dict[str, Any],
    public_revision: dict[str, Any] | None,
) -> dict[str, Any]:
    if public_revision is None:
        return completion_brief(result)
    host._reasoning_binding = reasoning_binding(result)
    status_map = (
        host.completion_answer_schema.get("x-cortheon-effect-status-map")
        if isinstance(host.completion_answer_schema, dict)
        else None
    )
    if not isinstance(status_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in status_map.items()
    ):
        raise RuntimeError("revision status map is unavailable")
    return revision_brief(result, public_revision, status_map)


def _failed_completion(
    host: Any,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    host._rejected_completion_count += 1
    completion_sha256 = protocol.payload_sha256(arguments)
    repeated = completion_sha256 == host._last_rejected_completion_sha256
    host._last_rejected_completion_sha256 = completion_sha256
    if host._rejected_completion_count >= 2:
        closed = host._abandon_runtime()
        host._emit_receipt()
        reason = (
            "the same completion was rejected twice without new reasoning"
            if repeated
            else "completion remained unverified after one bounded retry"
        )
        host._terminal(host.terminal.withheld(reason, runtime_closed=closed))
        return completion_brief(result), True
    if host.runtime.projects_answer_repair():
        host._repair_completion_arguments = dict(arguments)
    return completion_brief(result), False

"""Closed event payload builders for the generic MCP host."""

from __future__ import annotations

import json
from typing import Any

from cortheon.benchmark_core.generic_mcp_order_validation import transition_sha256
from cortheon.benchmark_core.generic_mcp_protocol import payload_sha256


def host_tool_result_event(
    execution: Any,
    observed: dict[str, Any],
    session_id: str | None,
) -> dict[str, Any]:
    runtime_digest = (
        transition_sha256("observe", session_id, observed)
        if session_id is not None and "next_action" in observed
        else None
    )
    return {
        "call_id": execution.request.call_id,
        "origin": "host",
        "status": execution.status,
        "content": execution.content,
        "receipt": execution.receipt,
        "request_sha256": execution.request.request_sha256,
        "result_sha256": execution.result_sha256,
        "accepted_evidence_ids": observed.get("accepted_evidence_ids", []),
        "runtime_transition_sha256": runtime_digest,
    }


def mcp_tool_result_event(
    call_id: str,
    request_sha256: str,
    result: dict[str, Any],
    session_id: str | None,
    transition: str | None,
) -> dict[str, Any]:
    runtime_digest = (
        transition_sha256(transition, session_id, result)
        if transition is not None and session_id is not None
        else None
    )
    return {
        "call_id": call_id,
        "origin": "mcp",
        "status": "result",
        "content": json.dumps(result, separators=(",", ":")),
        "receipt": None,
        "request_sha256": request_sha256,
        "result_sha256": payload_sha256(result),
        "accepted_evidence_ids": [],
        "runtime_transition_sha256": runtime_digest,
    }


def runtime_transition_event(
    transition: str,
    session_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "transition": transition,
        "session_id": session_id,
        "status": payload.get("status"),
        "next_action": payload.get("next_action"),
        "transition_sha256": transition_sha256(transition, session_id, payload),
    }

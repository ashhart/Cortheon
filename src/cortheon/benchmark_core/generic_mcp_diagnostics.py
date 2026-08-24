"""Content-free diagnostics for rejected generic MCP transcripts."""

from __future__ import annotations

from typing import Any

from cortheon.benchmark_core.generic_mcp_validation import validate_transcript


def transcript_diagnostic(events: list[dict[str, Any]]) -> str | None:
    """Return one bounded failure code without retaining task or model content."""

    if validate_transcript(events):
        return None
    if not events:
        return "missing_events"
    start = events[0]
    if not isinstance(start, dict) or start.get("type") != "task_start":
        return "invalid_task_start"
    task_id = start.get("task_id")
    nonce = start.get("nonce")
    for index, event in enumerate(events):
        if (
            not isinstance(event, dict)
            or event.get("sequence") != index
            or event.get("task_id") != task_id
            or event.get("nonce") != nonce
        ):
            return "invalid_event_binding"
    terminals = [event for event in events if event.get("type") == "terminal"]
    if not terminals:
        return "missing_terminal"
    if len(terminals) != 1 or events[-1].get("type") != "terminal":
        return "nonsticky_terminal"
    announced: list[str] = []
    forced_cardinality = False
    for event in events:
        if event.get("type") == "runtime_tool_decision" and isinstance(event.get("call_id"), str):
            announced.append(event["call_id"])
            continue
        if event.get("type") != "message":
            continue
        call_ids = event.get("tool_call_ids")
        if isinstance(call_ids, list):
            announced.extend(item for item in call_ids if isinstance(item, str))
            if event.get("tool_choice") != "auto" and len(call_ids) != 1:
                forced_cardinality = True
    requested = [event.get("call_id") for event in events if event.get("type") == "tool_request"]
    resolved = [event.get("call_id") for event in events if event.get("type") == "tool_result"]
    if len(announced) != len(set(announced)):
        return "duplicate_announced_tool_call"
    if forced_cardinality:
        return "forced_tool_call_cardinality"
    if set(announced) != set(requested):
        return "unresolved_announced_tool_call"
    if set(requested) != set(resolved):
        return "unresolved_tool_request"
    return "semantic_transcript_validation_failed"

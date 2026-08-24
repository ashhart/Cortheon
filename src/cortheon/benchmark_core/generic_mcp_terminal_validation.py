"""Closed terminal-disposition validation for generic MCP transcripts."""

from __future__ import annotations

from typing import Any

from cortheon.benchmark_core.generic_mcp_protocol import encoded_payload_sha256


def valid_terminal(
    event: dict[str, Any],
    start: dict[str, Any],
    state: str,
    receipt_seen: bool,
    close_transition: str | None,
) -> bool:
    runtime_used = start["runtime_used"]
    text = event.get("text")
    provenance = event.get("provenance")
    reason = event.get("finish_reason")
    common = bool(
        event.get("disposition") in {"release", "withhold", "fail_open"}
        and isinstance(text, str)
        and len(text) <= 20_000
        and provenance in {"cortheon_complete", "generic_mcp_model", "generic_mcp_wrapper"}
        and isinstance(reason, str)
        and 0 < len(reason) <= 128
        and type(event.get("runtime_closed")) is bool
        and type(event.get("active_sessions")) is int
        and event.get("candidate_sha256") == encoded_payload_sha256(text)
        and event["runtime_closed"] is True
        and event["active_sessions"] == 0
        and (not runtime_used or (state == "closed" and receipt_seen))
        and (runtime_used or (state == "not_started" and not receipt_seen))
    )
    if not common or event["disposition"] == "fail_open":
        return False
    identity = (event["disposition"], provenance, reason)
    if identity == ("release", "cortheon_complete", "certified"):
        return bool(
            runtime_used
            and start["condition_intercepts_final"] is True
            and close_transition == "complete"
        )
    if identity == ("release", "generic_mcp_model", "stop"):
        operators = start["evaluation_profile"]["config"]["operators"]
        placebo = bool(not any(operators.values()) and start["condition_intercepts_final"] is True)
        return bool(
            (start["condition_intercepts_final"] is False or placebo)
            and close_transition == ("abandon" if runtime_used else None)
        )
    if identity == ("withhold", "generic_mcp_wrapper", "bounded_incomplete"):
        return bool(
            isinstance(text, str)
            and text.startswith("[Cortheon withheld:")
            and close_transition == ("abandon" if runtime_used else None)
        )
    return False

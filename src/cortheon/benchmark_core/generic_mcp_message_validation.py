"""Validation for model-authored generic MCP transcript messages."""

from __future__ import annotations

import math
from typing import Any

from cortheon.benchmark_core.generic_mcp_protocol import bounded_identifier, payload_sha256
from cortheon.benchmark_core.generic_mcp_schema_validation import valid_tool_catalogue

_IDENTITY = "evaluator_requested_endpoint_response_model"


def valid_message(event: dict[str, Any], start: dict[str, Any]) -> bool:
    cost = event.get("cost_usd")
    available_tools = event.get("available_tools")
    tool_choice = event.get("tool_choice")
    catalogue = event.get("tool_catalogue")
    return bool(
        event.get("role") == "assistant"
        and bounded_identifier(event.get("message_id"))
        and _bounded_message_text(event.get("content"), 200_000)
        and isinstance(event.get("tool_call_ids"), list)
        and len(event["tool_call_ids"]) <= 16
        and all(bounded_identifier(item) for item in event["tool_call_ids"])
        and _bounded_message_text(event.get("finish_reason"), 128, empty=False)
        and type(event.get("tokens")) is int
        and 0 <= event["tokens"] <= 10_000_000
        and event.get("provider_requested") == start["provider_requested"]
        and event.get("model_observed") == start["model_requested"]
        and event.get("identity_provenance") == _IDENTITY
        and isinstance(available_tools, list)
        and 1 <= len(available_tools) <= 16
        and len(set(available_tools)) == len(available_tools)
        and all(bounded_identifier(item) for item in available_tools)
        and bounded_identifier(tool_choice)
        and (tool_choice == "auto" or tool_choice in available_tools)
        and valid_tool_catalogue(catalogue, available_tools, _bounded_message_text)
        and event.get("tool_catalogue_sha256") == payload_sha256(catalogue)
        and event.get("forced_binding")
        in {
            "none",
            "arguments",
            "tool_and_arguments",
            "repair_projection",
            "schema_projection",
        }
        and (
            cost is None
            or (
                not isinstance(cost, bool)
                and isinstance(cost, (int, float))
                and math.isfinite(cost)
                and cost >= 0
            )
        )
    )


def _bounded_message_text(value: Any, maximum: int, *, empty: bool = True) -> bool:
    return isinstance(value, str) and len(value) <= maximum and (empty or bool(value))

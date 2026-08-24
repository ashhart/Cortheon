"""Evaluator-owned identity binding for a valid generic MCP transcript."""

from __future__ import annotations

from typing import Any

from cortheon.benchmark_core.generic_mcp_protocol import SHA256


def validate_claim_transcript(
    events: list[dict[str, Any]],
    *,
    expected_config_sha256: str,
    expected_implementation_sha256: str,
    expected_endpoint_sha256: str,
    expected_wrapper_source_sha256: str,
    expected_web_identity: dict[str, str] | None,
    expected_task_kind: str,
    expected_resource_records: tuple[dict[str, Any], ...],
    require_web: bool = False,
) -> bool:
    from cortheon.benchmark_core.generic_mcp_validation import validate_transcript

    if not validate_transcript(events, require_web=require_web):
        return False
    start = events[0]
    profile = start["evaluation_profile"]
    digests = (
        expected_config_sha256,
        expected_implementation_sha256,
        expected_endpoint_sha256,
    )
    return bool(
        all(isinstance(value, str) and SHA256.fullmatch(value) for value in digests)
        and start["condition_sha256"] == expected_config_sha256
        and profile["config_sha256"] == expected_config_sha256
        and profile["implementation_sha256"] == expected_implementation_sha256
        and start["endpoint_sha256"] == expected_endpoint_sha256
        and start["wrapper_source_sha256"] == expected_wrapper_source_sha256
        and start["web_provider"] == expected_web_identity
        and start["task_kind"] == expected_task_kind
        and start["resource_records"] == list(expected_resource_records)
    )

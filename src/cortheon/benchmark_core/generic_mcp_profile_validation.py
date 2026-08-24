"""Closed task-start and evaluation-profile validation."""

from __future__ import annotations

from typing import Any

from cortheon.benchmark_core.generic_mcp_protocol import (
    BARE_SYSTEM_PROMPT,
    GENERIC_MCP_ASSURANCE,
    OPERATOR_KEYS,
    TREATMENT_SYSTEM_PROMPT,
    WRAPPED_SYSTEM_PROMPT,
    payload_sha256,
)
from cortheon.cognitive_protocol import normalize_evaluation_profile


def _profile(
    start: dict[str, Any],
    *,
    bounded_text: Any,
    valid_sha: Any,
) -> dict[str, Any] | None:
    value = start.get("evaluation_profile")
    try:
        profile = normalize_evaluation_profile(value)
    except ValueError:
        return None
    if profile is None or "adapter_receipt" in profile:
        return None
    config = profile["config"]
    operators = config["operators"]
    capabilities = start.get("capabilities")
    resource_paths = start.get("resource_paths")
    resource_records = start.get("resource_records")
    if (
        start.get("assurance") != GENERIC_MCP_ASSURANCE
        or start.get("condition_sha256") != profile["config_sha256"]
        or not bounded_text(start.get("provider_requested"), 128, empty=False)
        or not bounded_text(start.get("model_requested"), 256, empty=False)
        or not valid_sha(start.get("endpoint_sha256"))
        or not valid_sha(start.get("wrapper_source_sha256"))
        or start.get("intervention_prompt_sha256")
        != payload_sha256(
            BARE_SYSTEM_PROMPT
            if not any(operators.values())
            else (
                WRAPPED_SYSTEM_PROMPT
                if config["intercepts_final"] is True
                else TREATMENT_SYSTEM_PROMPT
            )
        )
        or start.get("identity_provenance") != "evaluator_requested_endpoint_response_model"
        or type(start.get("runtime_used")) is not bool
        or type(start.get("condition_intercepts_final")) is not bool
        or start["condition_intercepts_final"] is not config["intercepts_final"]
        or not isinstance(capabilities, dict)
        or set(capabilities)
        != {
            "isolated_workspace",
            "closed_tool_catalogue",
            "intercepts_final",
            "sticky_terminal",
            "current_web",
        }
        or not all(type(flag) is bool for flag in capabilities.values())
        or capabilities["isolated_workspace"] is not True
        or capabilities["closed_tool_catalogue"] is not True
        or capabilities["sticky_terminal"] is not True
        or capabilities["intercepts_final"] is not config["intercepts_final"]
        or set(operators) != OPERATOR_KEYS
        or not all(type(flag) is bool for flag in operators.values())
        or start["runtime_used"] is not any(operators.values())
        or not isinstance(resource_paths, list)
        or len(resource_paths) > 16
        or len(set(resource_paths)) != len(resource_paths)
        or any(
            not isinstance(path, str)
            or not path
            or len(path) > 240
            or path.startswith("/")
            or ".." in path.split("/")
            for path in resource_paths
        )
        or start.get("task_kind")
        not in {"auto", "code", "research", "documents", "decision", "general"}
        or not isinstance(resource_records, list)
        or len(resource_records) != len(resource_paths)
        or any(
            not isinstance(record, dict)
            or set(record) != {"path", "sha256", "bytes"}
            or record.get("path") != resource_paths[index]
            or not valid_sha(record.get("sha256"))
            or type(record.get("bytes")) is not int
            or not 0 <= record["bytes"] <= 10_000_000
            for index, record in enumerate(resource_records)
        )
    ):
        return None
    web = start.get("web_provider")
    if capabilities["current_web"]:
        if not _web_identity(web, bounded_text=bounded_text, valid_sha=valid_sha):
            return None
    elif web is not None:
        return None
    return profile


def _web_identity(value: Any, *, bounded_text: Any, valid_sha: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value) == {"executable_sha256", "version", "config_sha256"}
        and valid_sha(value.get("executable_sha256"))
        and valid_sha(value.get("config_sha256"))
        and bounded_text(value.get("version"), 128, empty=False)
    )

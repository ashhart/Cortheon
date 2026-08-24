"""Generic MCP CLI configuration and embedded-runtime health."""

from __future__ import annotations

import json
from typing import Any

from cortheon.benchmark_core.generic_mcp_runner import (
    generic_embedded_health,
    generic_implementation_snapshot,
)


def configure_generic_arguments(args: Any) -> list[str] | None:
    try:
        command = (
            json.loads(args.generic_web_command_json) if args.generic_web_command_json else None
        )
    except json.JSONDecodeError as exc:
        raise ValueError("--generic-web-command-json must be valid JSON") from exc
    if command is not None and (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise ValueError("--generic-web-command-json must be a non-empty string array")
    args.generic_web_command = command
    research = args.suite in {"research", "northstar"}
    if research and args.host not in {"opencode", "generic_mcp"}:
        raise ValueError(
            "research-bearing suites require OpenCode or evaluator-wrapped generic MCP"
        )
    if research and args.host == "generic_mcp" and not command:
        raise ValueError("generic MCP research requires --generic-web-command-json")
    return command


def generic_preflight(args: Any, command: list[str] | None) -> dict[str, Any]:
    snapshot, args.generic_web_identity = generic_implementation_snapshot(command)
    args.generic_implementation_pre = snapshot
    args.generic_host_identity_sha256 = snapshot["host_identity_sha256"]
    return generic_embedded_health(snapshot)


def generic_postflight(args: Any, command: list[str] | None) -> dict[str, Any]:
    snapshot, _web_identity = generic_implementation_snapshot(command)
    args.generic_implementation_post = snapshot
    return generic_embedded_health(snapshot)

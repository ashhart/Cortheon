"""JSONL child-process boundary for evaluator-owned generic MCP execution."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from cortheon.benchmark_core.condition_execution import CONTROL_FD_ENV, CONTROL_LIMIT
from cortheon.benchmark_core.generic_mcp_executor import IsolatedExecutor
from cortheon.benchmark_core.generic_mcp_host import GenericMcpHost
from cortheon.benchmark_core.generic_mcp_model import OpenAiModelClient
from cortheon.benchmark_core.generic_mcp_protocol import canonical_json
from cortheon.benchmark_core.generic_mcp_source import VERIFIED_DIGEST_ENV

_INPUT_KEYS = {
    "schema_version",
    "task_id",
    "goal",
    "task_kind",
    "require_web",
    "workspace",
    "workspace_nonce",
    "resource_paths",
    "base_url",
    "api_key",
    "provider_id",
    "model_id",
    "timeout_seconds",
    "output_tokens",
    "max_steps",
    "max_tool_calls",
    "tests",
    "web_command",
}
_CONTROL_KEYS = {
    "schema_version",
    "evaluation_profile",
    "cognitive_token",
    "evaluator_max_steps",
    "auto_enable",
    "benchmark_capture_candidate",
    "max_host_tool_calls",
}


def _read_json(stream: Any, maximum: int) -> dict[str, Any]:
    raw = stream.read(maximum + 1)
    if not isinstance(raw, str) or len(raw) > maximum:
        raise ValueError("generic MCP input exceeded its byte limit")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("generic MCP input must be an object")
    return value


def _read_control() -> dict[str, Any]:
    descriptor_text = os.environ.pop(CONTROL_FD_ENV, None)
    if not isinstance(descriptor_text, str) or not descriptor_text.isdigit():
        raise ValueError("generic MCP evaluator control FD is required")
    descriptor = int(descriptor_text)
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=True) as stream:
            value = _read_json(stream, CONTROL_LIMIT)
    except OSError as exc:
        raise ValueError("generic MCP evaluator control FD is unreadable") from exc
    if set(value) != _CONTROL_KEYS or value.get("schema_version") != 1:
        raise ValueError("generic MCP evaluator control payload is invalid")
    profile = value.get("evaluation_profile")
    if not isinstance(profile, dict):
        raise ValueError("generic MCP evaluator profile is required")
    return value


def _web_provider(command: list[str]) -> tuple[Any, dict[str, str]]:
    executable = Path(command[0]).resolve(strict=True)
    if not executable.is_file():
        raise ValueError("evaluator web provider executable is not a file")
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    resolved_command = [str(executable), *command[1:]]
    version_result = subprocess.run(
        [*resolved_command, "--version"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    version = (version_result.stdout or version_result.stderr).strip().splitlines()
    provider_version = version[0][:128] if version_result.returncode == 0 and version else "unknown"

    def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if hashlib.sha256(executable.read_bytes()).hexdigest() != digest:
            raise ValueError("evaluator web provider changed after handshake")
        completed = subprocess.run(
            resolved_command,
            input=canonical_json({"name": name, "arguments": arguments}),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env={"PATH": os.environ.get("PATH", ""), "LC_ALL": "C", "LANG": "C"},
        )
        if completed.returncode != 0 or len(completed.stdout) > 200_000:
            raise ValueError("evaluator web provider failed")
        payload = json.loads(completed.stdout)
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise ValueError("evaluator web provider returned no result list")
        return {
            "results": [
                {
                    **item,
                    "provider_sha256": digest,
                    "provider_version": provider_version,
                }
                for item in results
                if isinstance(item, dict)
            ]
        }

    return call, {
        "executable_sha256": digest,
        "version": provider_version,
        "config_sha256": hashlib.sha256(canonical_json(resolved_command).encode()).hexdigest(),
    }


def _validated_input(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != _INPUT_KEYS or value.get("schema_version") != 1:
        raise ValueError("generic MCP host input fields are invalid")
    string_keys = {
        "task_id",
        "goal",
        "task_kind",
        "workspace",
        "workspace_nonce",
        "base_url",
        "api_key",
        "provider_id",
        "model_id",
    }
    if any(not isinstance(value[key], str) for key in string_keys):
        raise ValueError("generic MCP host input string fields are invalid")
    if type(value.get("require_web")) is not bool:
        raise ValueError("generic MCP require_web must be boolean")
    for key, lower, upper in (
        ("output_tokens", 1, 100_000),
        ("max_steps", 1, 32),
        ("max_tool_calls", 1, 128),
    ):
        if type(value.get(key)) is not int or not lower <= value[key] <= upper:
            raise ValueError(f"generic MCP {key} is invalid")
    timeout = value.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 300:
        raise ValueError("generic MCP timeout is invalid")
    tests = value.get("tests")
    if not isinstance(tests, dict) or any(
        not isinstance(key, str)
        or not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) for item in command)
        for key, command in tests.items()
    ):
        raise ValueError("generic MCP test catalogue is invalid")
    resource_paths = value.get("resource_paths")
    if (
        not isinstance(resource_paths, list)
        or len(resource_paths) > 16
        or any(
            not isinstance(path, str)
            or not path
            or len(path) > 240
            or path.startswith("/")
            or ".." in path.split("/")
            for path in resource_paths
        )
    ):
        raise ValueError("generic MCP resource scope is invalid")
    web_command = value.get("web_command")
    if web_command is not None and (
        not isinstance(web_command, list)
        or not web_command
        or any(not isinstance(item, str) or not item for item in web_command)
    ):
        raise ValueError("generic MCP web command is invalid")
    return value


def serve_controlled_process() -> int:
    wrapper_source_sha256 = os.environ.pop(VERIFIED_DIGEST_ENV, None)
    if not isinstance(wrapper_source_sha256, str):
        raise ValueError("generic MCP process requires the source-bound launcher")
    control = _read_control()
    value = _validated_input(_read_json(sys.stdin, 100_000))
    web_command = value["web_command"]
    provider, web_identity = (
        _web_provider(web_command) if isinstance(web_command, list) else (None, None)
    )
    executor = IsolatedExecutor(
        Path(value["workspace"]),
        marker_nonce=value["workspace_nonce"],
        tests=value["tests"],
        web_provider=provider,
        web_identity=web_identity,
        maximum_calls=value["max_tool_calls"],
    )
    model = OpenAiModelClient(
        base_url=value["base_url"],
        api_key=value["api_key"],
        provider_id=value["provider_id"],
        model_id=value["model_id"],
        timeout_seconds=float(value["timeout_seconds"]),
        output_tokens=value["output_tokens"],
    )
    host = GenericMcpHost(
        task_id=value["task_id"],
        evaluation_profile=control["evaluation_profile"],
        model=model,
        executor=executor,
        max_steps=value["max_steps"],
        require_web=value["require_web"],
        resource_paths=tuple(value["resource_paths"]),
        wrapper_source_sha256=wrapper_source_sha256,
    )
    result = host.run(value["goal"], task_kind=value["task_kind"])
    for event in result.events:
        print(canonical_json(event), flush=True)
    return 0 if result.process_error is None else 2


def _interrupted(_signum: int, _frame: Any) -> None:
    raise InterruptedError("generic MCP host interrupted")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _interrupted)
    signal.signal(signal.SIGINT, _interrupted)
    raise SystemExit(serve_controlled_process())

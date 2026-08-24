"""Safe Codex-sandbox fallback for scheduled diff and test operations."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from typing import Any

if __package__:
    from .hook_config import (
        DEFAULT_HOST_ADAPTER_TIMEOUT_SECONDS,
        MAX_HOST_ADAPTER_OUTPUT_CHARS,
        SENSITIVE_ENV_RE,
    )
    from .hook_transport import _facade
else:
    from hook_config import (
        DEFAULT_HOST_ADAPTER_TIMEOUT_SECONDS,
        MAX_HOST_ADAPTER_OUTPUT_CHARS,
        SENSITIVE_ENV_RE,
    )
    from hook_transport import _facade


def _host_adapter_timeout() -> float:
    raw = os.environ.get("CORTHEON_HOST_ADAPTER_TIMEOUT_SECONDS", "")
    try:
        requested = float(raw) if raw else DEFAULT_HOST_ADAPTER_TIMEOUT_SECONDS
    except ValueError:
        requested = DEFAULT_HOST_ADAPTER_TIMEOUT_SECONDS
    return min(max(requested, 1.0), 120.0)


def _safe_relative_token(value: str) -> bool:
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or ":" in value
        or value.startswith(("/", "~"))
    ):
        return False
    return all(part not in {"", ".."} for part in value.split("/"))


def _safe_test_argv(argv: list[str]) -> bool:
    if len(argv) < 2 or any(
        not item
        or "\x00" in item
        or "\r" in item
        or "\n" in item
        or "\\" in item
        or ":" in item
        or item.startswith(("/", "~"))
        or ".." in item.split("/")
        for item in argv
    ):
        return False
    executable, arguments = argv[0].removeprefix("./").casefold(), argv[1:]
    python_test = re.fullmatch(r"python(?:3(?:\.\d+)?)?", executable) is not None and arguments[
        :2
    ] in (["-m", "pytest"], ["-m", "unittest"])
    direct_pytest = executable in {"pytest", "py.test"}
    node_test = executable in {"npm", "pnpm", "yarn", "bun"} and (
        arguments[:1] == ["test"] or arguments[:2] == ["run", "test"]
    )
    compiled_test = (
        (executable == "cargo" and arguments[:1] == ["test"])
        or (executable == "go" and arguments[:1] == ["test"])
        or (executable == "dotnet" and arguments[:1] == ["test"])
        or (
            executable in {"mvn", "mvnw", "gradle", "gradlew"}
            and any(re.search(r"\btest\b", item, flags=re.IGNORECASE) for item in arguments)
        )
    )
    quality_check = (
        (executable == "ruff" and arguments[:1] == ["check"])
        or executable in {"mypy", "pyright", "flake8", "eslint", "tsc", "biome"}
        or (executable == "cargo" and arguments[:1] == ["clippy"])
        or (executable == "go" and arguments[:1] == ["vet"])
        or (
            re.fullmatch(r"python(?:3(?:\.\d+)?)?", executable) is not None
            and arguments[:2]
            in (["-m", "ruff"], ["-m", "mypy"], ["-m", "pyright"], ["-m", "flake8"])
        )
    )
    return python_test or direct_pytest or node_test or compiled_test or quality_check


def _host_adapter_argv(result: dict[str, Any], command: str) -> list[str] | None:
    next_action = result.get("next_action")
    request = (
        next_action.get("request")
        if isinstance(next_action, dict) and next_action.get("type") == "harness_tool"
        else None
    )
    if not isinstance(request, dict):
        return None
    capability = request.get("capability")
    if request.get("request_id") != f"hook_{capability}":
        return None
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    if capability == "diff":
        if (
            len(argv) < 4
            or argv[:3] != ["git", "diff", "--"]
            or not all(_safe_relative_token(item) for item in argv[3:])
        ):
            return None
        return argv
    if capability != "test" or not _safe_test_argv(argv):
        return None
    parameters = request.get("parameters")
    expected = parameters.get("command") if isinstance(parameters, dict) else None
    if not isinstance(expected, list) or argv != expected:
        return None
    return argv


def _host_adapter_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if SENSITIVE_ENV_RE.search(key) is None
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _host_adapter_output(completed: subprocess.CompletedProcess[str]) -> str:
    sections = []
    if completed.stdout:
        sections.append(completed.stdout)
    if completed.stderr:
        sections.append(completed.stderr)
    output = "\n".join(sections).strip()
    if not output:
        output = f"Process exited with status {completed.returncode}."
    return output[-MAX_HOST_ADAPTER_OUTPUT_CHARS:]


def _run_host_adapter_step(payload: dict[str, Any], result: dict[str, Any]) -> bool:
    """Run one scheduled diff or test through Codex's workspace sandbox."""

    api = _facade()
    if result.get("automatic") is not True or payload.get("permission_mode") == "plan":
        return False
    identity = api._identity(payload)
    cwd = payload.get("cwd")
    codex = api.shutil.which("codex")
    if (
        identity is None
        or not isinstance(cwd, str)
        or not os.path.isabs(cwd)
        or not os.path.isdir(cwd)
        or codex is None
    ):
        return False
    pre_tool = api._post(
        "/v1/hooks/pre-tool",
        {**identity, "tool_name": "Bash", "tool_input": {"command": ""}},
    )
    updated_input = pre_tool.get("updated_input") if isinstance(pre_tool, dict) else None
    command = updated_input.get("command") if isinstance(updated_input, dict) else None
    if (
        not isinstance(pre_tool, dict)
        or pre_tool.get("allow") is not True
        or not isinstance(command, str)
    ):
        return False
    argv = api._host_adapter_argv(result, command)
    if argv is None:
        return False
    try:
        completed = api.subprocess.run(
            [
                codex,
                "sandbox",
                "--permission-profile",
                ":workspace",
                "--cd",
                cwd,
                "--",
                *argv,
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=api._host_adapter_timeout(),
            env=api._host_adapter_environment(),
            check=False,
        )
        succeeded = completed.returncode == 0
        output = api._host_adapter_output(completed)
    except (OSError, subprocess.SubprocessError) as exc:
        succeeded = False
        output = f"Codex sandbox execution failed: {type(exc).__name__}: {exc}"
    observed = api._post(
        "/v1/hooks/post-tool",
        {
            **identity,
            "tool_name": "Bash",
            "succeeded": succeeded,
            "certified": False,
            "tool_output": output,
        },
    )
    return isinstance(observed, dict)

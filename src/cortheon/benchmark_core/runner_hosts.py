"""Isolated native-host command and configuration ownership."""

from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from cortheon.benchmark_core.models import BenchmarkCase
from cortheon.benchmark_core.run_support import (
    _pi_provider_config,
    _provider_config,
)


def prepare_native_host(
    args: argparse.Namespace,
    case: BenchmarkCase,
    *,
    treatment: bool,
    environment: dict[str, str],
    stack: contextlib.ExitStack,
) -> list[str]:
    if args.host == "generic_mcp":
        return []
    if args.host == "opencode":
        config_directory = Path(
            stack.enter_context(tempfile.TemporaryDirectory(prefix="cortheon-opencode-config-"))
        )
        environment.update(
            {
                "OPENCODE_CONFIG_DIR": str(config_directory),
                "OPENCODE_TEST_HOME": str(config_directory / "home"),
                "XDG_CONFIG_HOME": str(config_directory / "config"),
                "XDG_DATA_HOME": str(config_directory / "data"),
                "XDG_STATE_HOME": str(config_directory / "state"),
                "XDG_CACHE_HOME": str(config_directory / "cache"),
                "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
                "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
                "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
                "OPENCODE_DISABLE_CLAUDE_CODE": "1",
                "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1",
                "OPENCODE_CONFIG_CONTENT": _provider_config(args, treatment=treatment),
            }
        )
        environment.pop("CORTHEON_BENCHMARK_CAPTURE_CANDIDATE", None)
        command = [args.opencode, "run"]
        if not treatment:
            command.append("--pure")
        return [
            *command,
            "--model",
            f"{args.provider}/{args.model_id}",
            "--format",
            "json",
            case.prompt,
        ]
    if args.host != "pi":
        raise ValueError(f"unsupported local host: {args.host}")
    agent_directory = Path(
        stack.enter_context(tempfile.TemporaryDirectory(prefix="cortheon-pi-agent-"))
    )
    (agent_directory / "models.json").write_text(
        json.dumps(_pi_provider_config(args), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    environment["PI_CODING_AGENT_DIR"] = str(agent_directory)
    environment["PI_TELEMETRY"] = "0"
    command: list[Any] = [
        args.pi,
        "--provider",
        args.provider,
        "--model",
        args.model_id,
        "--mode",
        "json",
        "--print",
        "--no-session",
        "--no-extensions",
        "--no-skills",
        "--no-context-files",
        "--approve",
    ]
    if treatment:
        extension = Path(str(files("cortheon").joinpath("pi_extension.ts"))).resolve()
        command.extend(["--extension", str(extension)])
    command.append(case.prompt)
    return [str(item) for item in command]

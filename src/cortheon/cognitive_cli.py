"""CLI for the lean Cortheon runtime."""

from __future__ import annotations

import argparse
import json as _json
import os as _os
import sys as _sys
from typing import Any

from cortheon import __version__ as _version
from cortheon.cognitive_protocol import protocol_capabilities as _protocol_capabilities

# Preserve the facade globals used by callers and existing monkeypatch seams.
json = _json
os = _os
sys = _sys
__version__ = _version
protocol_capabilities = _protocol_capabilities

DEFAULT_RUNTIME_URL = "http://127.0.0.1:8743"
SUPPORTED_HOSTS = ("opencode", "pi", "codex", "omp", "generic")


def build_parser() -> argparse.ArgumentParser:
    from cortheon.cognitive_cli_core import parser

    return parser.build_parser()


def main(argv: list[str] | None = None) -> int:
    from cortheon.cognitive_cli_core import dispatch

    return dispatch.main(argv)


def _serve(args: argparse.Namespace) -> int:
    from cortheon.cognitive_cli_core import operations

    return operations.serve(args)


def _mcp(args: argparse.Namespace) -> int:
    from cortheon.cognitive_cli_core import operations

    return operations.mcp(args)


def doctor(
    runtime_url: str = DEFAULT_RUNTIME_URL,
    *,
    token: str = "",
    require_runtime: bool = False,
    hosts: list[str] | tuple[str, ...] = (),
    scope: str = "user",
    project_dir: str | None = None,
) -> dict[str, Any]:
    from cortheon.cognitive_cli_core import diagnostics

    return diagnostics.doctor(
        runtime_url,
        token=token,
        require_runtime=require_runtime,
        hosts=hosts,
        scope=scope,
        project_dir=project_dir,
    )


def _runtime_health(url: str, *, token: str) -> dict[str, Any]:
    from cortheon.cognitive_cli_core import diagnostics

    return diagnostics.runtime_health(url, token=token)


def runtime_results(
    runtime_url: str = DEFAULT_RUNTIME_URL,
    *,
    token: str = "",
) -> dict[str, Any]:
    from cortheon.cognitive_cli_core import diagnostics

    return diagnostics.runtime_results(runtime_url, token=token)


def host_conformance(
    runtime_url: str = DEFAULT_RUNTIME_URL,
    *,
    token: str = "",
    hosts: list[str] | tuple[str, ...] = (),
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    from cortheon.cognitive_cli_core import conformance

    return conformance.host_conformance(
        runtime_url,
        token=token,
        hosts=hosts,
        timeout_seconds=timeout_seconds,
    )


def _asset_paths() -> dict[str, str]:
    from cortheon.cognitive_cli_core import operations

    return operations.asset_paths()


def _install(args: argparse.Namespace) -> list[Any]:
    from cortheon.cognitive_cli_core import operations

    return operations.install(args)


def _uninstall(args: argparse.Namespace) -> list[Any]:
    from cortheon.cognitive_cli_core import operations

    return operations.uninstall(args)


def _configure(args: argparse.Namespace) -> list[Any]:
    from cortheon.cognitive_cli_core import operations

    return operations.configure(args)


if __name__ == "__main__":
    raise SystemExit(main())

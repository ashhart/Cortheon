"""Runtime entry points, asset discovery, and host integration operations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def serve(args: argparse.Namespace) -> int:
    import secrets

    from cortheon.cognitive_http import main as http_main

    values = [
        "--bind",
        args.bind,
        "--port",
        str(args.port),
        "--max-sessions",
        str(args.max_sessions),
        "--ttl-seconds",
        str(args.ttl_seconds),
        "--max-concurrent-requests",
        str(args.max_concurrent_requests),
    ]
    if args.token:
        values.extend(["--token", args.token])
    previous = os.environ.get("CORTHEON_RUNTIME_INSTANCE_ID")
    if previous is None:
        os.environ["CORTHEON_RUNTIME_INSTANCE_ID"] = secrets.token_hex(16)
    try:
        return http_main(values)
    finally:
        if previous is None:
            os.environ.pop("CORTHEON_RUNTIME_INSTANCE_ID", None)


def mcp(args: argparse.Namespace) -> int:
    from cortheon.cognitive_mcp import main as mcp_main

    values = [
        "--max-sessions",
        str(args.max_sessions),
        "--ttl-seconds",
        str(args.ttl_seconds),
    ]
    if args.advanced:
        values.append("--advanced")
    mcp_main(values)
    return 0


def asset_paths() -> dict[str, str]:
    from importlib.resources import files

    root = files("cortheon")
    return {
        "opencode_plugin": str(root.joinpath("opencode_plugin.js")),
        "pi_extension": str(root.joinpath("pi_extension.ts")),
        "codex_plugin": str(root.joinpath("codex_plugins/cortheon")),
    }


def install(args: argparse.Namespace) -> list[Any]:
    from cortheon.cognitive_install import InstallError, install_hosts

    try:
        return install_hosts(
            args.host,
            scope=args.scope,
            project_dir=Path(args.project_dir) if args.project_dir else None,
            dry_run=args.dry_run,
            run_codex_cli=not args.no_codex_cli,
        )
    except InstallError as exc:
        raise ValueError(str(exc)) from exc


def uninstall(args: argparse.Namespace) -> list[Any]:
    from cortheon.cognitive_install import InstallError, uninstall_hosts

    try:
        return uninstall_hosts(
            args.host,
            scope=args.scope,
            project_dir=Path(args.project_dir) if args.project_dir else None,
            dry_run=args.dry_run,
            run_codex_cli=not args.no_codex_cli,
        )
    except InstallError as exc:
        raise ValueError(str(exc)) from exc


def configure(args: argparse.Namespace) -> list[Any]:
    from cortheon.cognitive_install import generic_mcp_config

    return [generic_mcp_config()]

"""Argument parser construction for the operator CLI."""

from __future__ import annotations

import argparse

from cortheon import cognitive_cli as surface


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cortheon",
        description=(
            "Cortheon helps a small local model reason, discover, and complete work "
            "like a frontier model."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {surface.__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="Run the local plugin runtime.")
    serve.add_argument(
        "--bind", default=surface.os.environ.get("CORTHEON_COGNITIVE_BIND", "127.0.0.1")
    )
    serve.add_argument(
        "--port",
        type=int,
        default=int(surface.os.environ.get("CORTHEON_COGNITIVE_PORT", "8743")),
    )
    serve.add_argument(
        "--max-sessions",
        type=int,
        default=int(surface.os.environ.get("CORTHEON_COGNITIVE_MAX_SESSIONS", "32")),
    )
    serve.add_argument(
        "--ttl-seconds",
        type=float,
        default=float(surface.os.environ.get("CORTHEON_COGNITIVE_TTL_SECONDS", "1800")),
    )
    serve.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=int(surface.os.environ.get("CORTHEON_COGNITIVE_MAX_CONCURRENT_REQUESTS", "64")),
    )
    serve.add_argument("--token", default=surface.os.environ.get("CORTHEON_COGNITIVE_TOKEN", ""))

    mcp = commands.add_parser("mcp", help="Run the MCP server.")
    mcp.add_argument("--max-sessions", type=int, default=32)
    mcp.add_argument("--ttl-seconds", type=float, default=1_800.0)
    mcp.add_argument("--advanced", action="store_true")

    doctor = commands.add_parser("doctor", help="Check the installation.")
    doctor.add_argument("--url", default=surface.DEFAULT_RUNTIME_URL)
    doctor.add_argument("--token", default="")
    doctor.add_argument(
        "--host",
        action="append",
        choices=[*surface.SUPPORTED_HOSTS, "all"],
        default=[],
        help="Require this host integration.",
    )
    doctor.add_argument("--scope", choices=["user", "project"], default="user")
    doctor.add_argument("--project-dir", default=None)
    doctor.add_argument(
        "--require-runtime",
        action="store_true",
        help="Require a running local runtime.",
    )

    conformance = commands.add_parser("conformance", help="Test host integrations.")
    conformance.add_argument("--url", default=surface.DEFAULT_RUNTIME_URL)
    conformance.add_argument("--token", default="")
    conformance.add_argument(
        "--host",
        action="append",
        choices=[*surface.SUPPORTED_HOSTS, "all"],
        default=[],
        help="Host to test; repeatable.",
    )
    conformance.add_argument("--timeout-seconds", type=float, default=15.0)

    results = commands.add_parser("results", help="Show content-free runtime outcomes.")
    results.add_argument("--url", default=surface.DEFAULT_RUNTIME_URL)
    results.add_argument("--token", default="")

    install = commands.add_parser("install", help="Install host integrations.")
    install.add_argument(
        "--host",
        action="append",
        choices=["opencode", "pi", "codex", "all"],
        default=[],
        help="Host to install; repeatable.",
    )
    install.add_argument("--scope", choices=["user", "project"], default="user")
    install.add_argument("--project-dir", default=None)
    install.add_argument("--dry-run", action="store_true")
    install.add_argument(
        "--no-codex-cli",
        action="store_true",
        help="Do not register with Codex.",
    )

    uninstall = commands.add_parser("uninstall", help="Remove host integrations.")
    uninstall.add_argument(
        "--host",
        action="append",
        choices=[*surface.SUPPORTED_HOSTS, "all"],
        required=True,
        help="Host to remove; repeatable.",
    )
    uninstall.add_argument("--scope", choices=["user", "project"], default="user")
    uninstall.add_argument("--project-dir", default=None)
    uninstall.add_argument("--dry-run", action="store_true")
    uninstall.add_argument("--no-codex-cli", action="store_true")

    configure = commands.add_parser("configure", help="Print generic MCP configuration.")
    configure.add_argument("--host", choices=["generic"], required=True)

    commands.add_parser("capabilities", help="Print protocol capabilities.")
    commands.add_parser("paths", help="Print adapter paths.")
    return parser

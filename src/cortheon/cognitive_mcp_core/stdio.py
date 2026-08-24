"""JSON-RPC over stdio, and the cortheon-mcp entry point."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, TextIO

from cortheon.cognitive_mcp_core.protocol import (
    JSONRPC_INVALID_PARAMS,
    JSONRPC_INVALID_REQUEST,
    JSONRPC_PARSE_ERROR,
    MAX_MESSAGE_CHARS,
    _error,
)
from cortheon.cognitive_mcp_core.server import CortheonMcpServer
from cortheon.cognitive_runtime import CognitiveRuntime


def serve(
    server: CortheonMcpServer,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
) -> None:
    while True:
        line = stdin.readline(MAX_MESSAGE_CHARS + 1)
        if line == "":
            break
        if len(line) > MAX_MESSAGE_CHARS:
            while line and not line.endswith("\n"):
                line = stdin.readline(MAX_MESSAGE_CHARS + 1)
            _write(
                stdout,
                _error(
                    None,
                    JSONRPC_INVALID_PARAMS,
                    f"JSON-RPC message exceeds {MAX_MESSAGE_CHARS} characters.",
                ),
            )
            continue
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            _write(stdout, _error(None, JSONRPC_PARSE_ERROR, str(exc)))
            continue
        if not isinstance(message, dict):
            _write(
                stdout,
                _error(
                    None,
                    JSONRPC_INVALID_REQUEST,
                    "JSON-RPC message must be an object.",
                ),
            )
            continue
        response = server.handle(message)
        if response is not None:
            _write(stdout, response)


def _write(stdout: TextIO, message: dict[str, Any]) -> None:
    stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    stdout.flush()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cortheon-mcp",
        description=("Run Cortheon's memory-only cognitive runtime as an MCP stdio server."),
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=32,
        help="Maximum concurrent in-memory investigations (default: 32).",
    )
    parser.add_argument(
        "--ttl-seconds",
        type=float,
        default=1_800.0,
        help="Discard an idle investigation after this many seconds (default: 1800).",
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help=(
            "Expose the low-level step/challenge/verify/finish tools in addition "
            "to the compact four-tool surface."
        ),
    )
    args = parser.parse_args(argv)
    runtime = CognitiveRuntime(
        max_sessions=args.max_sessions,
        ttl_seconds=args.ttl_seconds,
    )
    serve(CortheonMcpServer(runtime, advanced=args.advanced))

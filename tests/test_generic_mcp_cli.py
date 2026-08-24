"""CLI exposure and research handshake policy for the generic wrapper."""

from __future__ import annotations

import pytest

from cortheon.benchmark_core.cli import build_parser, main


def test_cli_exposes_generic_mcp_as_an_evaluator_host() -> None:
    args = build_parser().parse_args(["--host", "generic_mcp"])
    assert args.host == "generic_mcp"


def test_generic_research_requires_an_evaluator_web_command() -> None:
    with pytest.raises(SystemExit, match="generic MCP research requires"):
        main(
            [
                "--host",
                "generic_mcp",
                "--suite",
                "research",
                "--cases",
                "2",
                "--repeats",
                "1",
            ]
        )

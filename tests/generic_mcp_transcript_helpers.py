"""Live retract trace support for generic MCP transcript tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from test_generic_mcp_hardening import _profile
from test_generic_mcp_microsteps import _executor

from cortheon.benchmark_core.generic_mcp_host import GenericMcpHost
from cortheon.benchmark_core.generic_mcp_model import ModelToolCall, ModelTurn
from cortheon.benchmark_core.generic_mcp_protocol import payload_sha256
from cortheon.benchmark_core.generic_mcp_validation import validate_transcript


class RetractingModel:
    provider_id = "local"
    model_id = "small"
    endpoint_sha256 = "e" * 64

    def __init__(self) -> None:
        self.calls = 0
        self.session_id = ""

    def complete(
        self,
        messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
        *,
        tool_choice: str = "auto",
    ) -> ModelTurn:
        self.calls += 1
        if self.calls == 1:
            started = next(
                json.loads(message["content"])
                for message in messages
                if message["role"] == "system"
                and isinstance(message["content"], str)
                and message["content"].startswith("{")
            )
            self.session_id = started["session"]["session_id"]
            call = ModelToolCall(
                "search",
                "host_search",
                {"pattern": "pathlib", "path": "example.py"},
            )
            return ModelTurn("local", "small", "", (call,), "tool_calls", 1)
        if self.calls == 2:
            call = ModelToolCall(
                "retract",
                "cortheon_retract",
                {
                    "session_id": self.session_id,
                    "evidence_ids": ["ev1"],
                    "reason": "model_correction",
                },
            )
            return ModelTurn("local", "small", "", (call,), "tool_calls", 1)
        return ModelTurn("local", "small", "done", (), "stop", 1)


def retract_events(root: Path) -> list[dict[str, Any]]:
    profile = _profile("full")
    profile["config"]["intercepts_final"] = False
    profile["config_sha256"] = payload_sha256(profile["config"])
    events = list(
        GenericMcpHost(
            task_id="ordered-retract",
            evaluation_profile=profile,
            model=RetractingModel(),  # type: ignore[arg-type]
            executor=_executor(root),
            max_steps=3,
        )
        .run("Does example.py import pathlib?", task_kind="code")
        .events
    )
    assert validate_transcript(events)
    return events

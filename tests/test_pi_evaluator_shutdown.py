"""Real Pi proof for evaluator-owned graceful step-cap shutdown."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from pi_doom_loop_helpers import PROMPT, TOOL_TURN, finish_script, workspace
from pi_recovery_helpers import Servers, parse_events, require_pi

from cortheon.benchmark_core.execution_provenance import (
    ExecutionPolicy,
    execute_host_process,
    execution_facts,
)
from cortheon.benchmark_core.transport_outcomes import parse_transport_outcome

EXTENSION = Path(__file__).parents[1] / "src" / "cortheon" / "pi_extension.ts"


def _agent_config(model_port: int) -> dict[str, Any]:
    return {
        "providers": {
            "mock": {
                "baseUrl": f"http://127.0.0.1:{model_port}/v1",
                "api": "openai-completions",
                "apiKey": "test-only",
                "authHeader": False,
                "models": [
                    {
                        "id": "mock-small",
                        "name": "mock-small",
                        "reasoning": False,
                        "input": ["text"],
                        "cost": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                        },
                        "contextWindow": 8_192,
                        "maxTokens": 700,
                    }
                ],
            }
        }
    }


def test_step_cap_emits_authenticated_terminal_and_abandons_runtime(tmp_path: Path) -> None:
    if not require_pi():
        pytest.skip("Pi is not installed")
    model_state: dict[str, Any] = {"requests": [], "turns": [TOOL_TURN]}
    runtime_state: dict[str, Any] = {"records": [], "script": finish_script(False)}
    with Servers(model_state, runtime_state) as servers:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "models.json").write_text(
            json.dumps(_agent_config(servers.model.server_port)),
            encoding="utf-8",
        )
        environment = {
            **os.environ,
            "PI_CODING_AGENT_DIR": str(agent_dir),
            "PI_TELEMETRY": "0",
            "CORTHEON_AUTO_ENABLE": "1",
            "CORTHEON_RUNTIME_URL": f"http://127.0.0.1:{servers.runtime.server_port}",
            "CORTHEON_EVALUATOR_MAX_STEPS": "1",
        }
        capture = execute_host_process(
            [
                "pi",
                "--provider",
                "mock",
                "--model",
                "mock-small",
                "--mode",
                "json",
                "--print",
                "--no-session",
                "--no-extensions",
                "--no-skills",
                "--no-context-files",
                "--approve",
                "--extension",
                str(EXTENSION),
                PROMPT,
            ],
            cwd=workspace(tmp_path),
            env=environment,
            host="pi",
            policy=ExecutionPolicy(1, 16, 10.0, 8_192, 700),
        )

    events = parse_events(capture.stdout)
    facts = execution_facts(events, host="pi")
    outcome = parse_transport_outcome(events, host="pi").outcome
    paths = [path for path, _body in runtime_state["records"]]
    terminal_events = [
        event
        for event in events
        if event.get("type") == "message_end"
        and isinstance(event.get("message"), dict)
        and event["message"].get("customType") == "cortheon-terminal-status-v1"
    ]
    assistant_events = [
        event
        for event in events
        if event.get("type") == "message_end"
        and isinstance(event.get("message"), dict)
        and event["message"].get("role") == "assistant"
    ]
    tool_ends = [event for event in events if event.get("type") == "tool_execution_end"]

    assert capture.budget_reason == "max_steps"
    assert len(model_state["requests"]) == 1
    assert len(assistant_events) == 2
    assert assistant_events[-1]["message"].get("content") == []
    assert assistant_events[-1]["message"].get("stopReason") == "error"
    assert facts.steps == 1
    assert facts.identity_valid is True
    assert (facts.provider_id, facts.model_id) == ("mock", "mock-small")
    assert outcome.terminal_status == "withheld"
    assert outcome.terminal_provenance == "pi_custom_terminal"
    assert len(terminal_events) == 1
    assert {event.get("toolName") for event in tool_ends} == {"read", "bash"}
    assert events.index(terminal_events[0]) > max(events.index(event) for event in tool_ends)
    assert paths.count("/v1/abandon") == 1

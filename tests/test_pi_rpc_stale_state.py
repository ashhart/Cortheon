"""Same-process stale-state proof across Pi RPC prompts.

PiRpcSession waits for the authoritative agent_settled event per prompt, so
this test also proves the read is settled before the next prompt is sent and
that stale answer-only state from one Cortheon turn cannot leak into a later
ordinary prompt in the same Pi process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pi_doom_loop_helpers import (
    ANSWER_TURN,
    PLAIN_PROMPT,
    PROMPT,
    TOOL_TURN,
    finish_script,
    workspace,
)
from pi_recovery_helpers import (
    Servers,
    blocked_executions,
    host_executions,
    require_pi,
)
from pi_rpc_session import PiRpcSession

EXTENSION = Path(__file__).parents[1] / "src" / "cortheon" / "pi_extension.ts"


def _continuation_messages(model_state: dict[str, Any]) -> set[str]:
    """Distinct [CORTHEON_CONTINUE] user messages ever sent (history copies
    of the same message dedupe to one entry)."""
    texts: set[str] = set()
    for request in model_state["requests"]:
        for message in request.get("messages", []):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            blocks = content if isinstance(content, list) else [message]
            for block in blocks:
                if (
                    isinstance(block, dict)
                    and isinstance(block.get("text"), str)
                    and block["text"].startswith("[CORTHEON_CONTINUE]")
                ):
                    texts.add(block["text"])
    return texts


def test_stale_answer_only_state_resets_for_later_prompt(tmp_path: Path) -> None:
    """Same Pi process, same extension instance: a Cortheon turn that ends
    answer-only must not block tools on a later ordinary prompt."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    model_state: dict[str, Any] = {
        "requests": [],
        # Prompt 1 only ever requests tools: the runtime closes evidence,
        # every later batch is blocked and terminated, and the turn ends
        # with stale answer-only state and a still-held session. Prompt 2
        # (an ordinary greeting) must run its tools ungated and answer.
        "turns": [TOOL_TURN] * 5 + [ANSWER_TURN] * 2,
    }
    runtime_state: dict[str, Any] = {"records": [], "script": finish_script(False)}
    with Servers(model_state, runtime_state) as servers:
        session = PiRpcSession(
            EXTENSION,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path / "rpc",
        )
        try:
            first = session.prompt(PROMPT)
            first_requests = len(model_state["requests"])
            first_continuations = len(_continuation_messages(model_state))
            second = session.prompt(PLAIN_PROMPT)
        finally:
            session.close()

    first_executed = host_executions(first)
    assert len(first_executed) == 2
    assert len(blocked_executions(first)) >= 2
    assert first_continuations == 1
    # The first turn settled before the second prompt was sent.
    assert first_requests == 4, first_requests

    # The later ordinary prompt ran its tools ungated: exactly its own two
    # calls executed, nothing was blocked, and no new continuation was sent.
    assert len(host_executions(second)) == 2
    assert len(blocked_executions(second)) == 0
    assert len(_continuation_messages(model_state)) == 1
    # The stale investigation was safely abandoned before that prompt ran.
    paths = [path for path, _body in runtime_state["records"]]
    assert "/v1/abandon" in paths

"""Real Pi regressions for the host-visible terminal status.

A model can end the single answer-only continuation with tools and no
assistant text: sticky answer-only/disposition state alone is not a visible
terminal. The adapter must then emit one bounded host-visible Cortheon
terminal status — Pi's custom-message API with a closed custom type and
version, no project storage, and no triggered model turn — containing only
the fixed WITHHELD status and the bounded reason, never candidate or
evidence text. The emission flag resets on a new task, so an identical
second task earns its own single terminal status.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pi_doom_loop_helpers import workspace
from pi_recovery_helpers import Servers, parse_events, require_pi, run_pi
from pi_rpc_session import PiRpcSession
from pi_terminal_helpers import (
    CAUSAL_PROMPT,
    EXTENSION,
    GOOD_SYNTHESIS,
    TERMINAL_STATUS_MARKER,
    TERMINAL_STATUS_TYPE,
    TERMINAL_STATUS_VERSION,
    WITHHELD_MARKER,
    always_pending_request_script,
    terminal_status_messages,
)

from cortheon.benchmark_core.run_support import _final_text

SINGLE_TOOL_TURN: dict[str, Any] = {"tool_calls": [("read", {"path": "facts/a.txt"})]}
BUDGET = {"CORTHEON_MAX_HOST_TOOL_CALLS": "8"}


def _assert_status_shape(message: dict[str, Any]) -> None:
    assert message.get("customType") == TERMINAL_STATUS_TYPE, message
    assert message.get("display") is True, message
    content = message.get("content")
    text = (
        content
        if isinstance(content, str)
        else "".join(block.get("text", "") for block in content or [])
    )
    assert text.startswith(WITHHELD_MARKER), text
    assert TERMINAL_STATUS_MARKER in text, text
    # Only the fixed status and the bounded reason: no candidate or
    # evidence text ever rides the terminal status.
    assert GOOD_SYNTHESIS.split("\n")[0] not in text, text
    details = message.get("details", {})
    assert details.get("version") == TERMINAL_STATUS_VERSION, details
    assert details.get("status") == "withheld", details
    assert details.get("reason"), details
    assert set(details) <= {"version", "status", "reason", "causal"}, details


def test_tool_only_answer_continuation_emits_one_terminal_status(
    tmp_path: Path,
) -> None:
    """The budget terminal abandons the session at the blocked ninth tool
    and ends the operation: zero automatic follow-ups (the unified budget
    never schedules a model turn after budget exhaustion abandons the
    session), one host-visible terminal status with the closed shape, and
    nothing after it: nine model requests total (eight admitted tool turns
    plus the blocked ninth that terminates the operation)."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    model_state: dict[str, Any] = {
        "requests": [],
        # Eight single-tool turns fill the budget; the ninth tool attempt is
        # blocked and terminates the operation with no assistant text — the
        # exact shape that needs a host-visible terminal and no follow-up.
        "turns": [SINGLE_TOOL_TURN] * 11,
    }
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": always_pending_request_script(),
    }
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            EXTENSION,
            CAUSAL_PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=60,
            extra_env=BUDGET,
        )
    assert completed.returncode == 0, completed.stderr
    statuses = terminal_status_messages(completed)
    assert len(statuses) == 1, statuses
    _assert_status_shape(statuses[0])
    assert _final_text(parse_events(completed.stdout), host="pi") == statuses[0]["content"]
    # No triggered model turn: the terminal status did not ask the model
    # for another answer — zero automatic follow-ups after the budget
    # terminal, only the blocked ninth attempt ends the operation.
    assert len(model_state["requests"]) == 9, len(model_state["requests"])
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/abandon") == 1, paths
    assert paths[-1] == "/v1/abandon", paths


def test_terminal_status_emission_resets_for_a_new_task(
    tmp_path: Path,
) -> None:
    """One persistent Pi process (RPC mode), two identical independent
    tasks: each task runs its own bounded budget lifecycle and each earns
    exactly one terminal status of its own — the emission flag (and all
    finalization state) resets on the new task, and no state from task A
    changes task B."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [SINGLE_TOOL_TURN] * 22,
    }
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": always_pending_request_script(),
    }
    with Servers(model_state, runtime_state) as servers:
        session = PiRpcSession(
            EXTENSION,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path,
            extra_env=BUDGET,
        )
        try:
            first = session.prompt(CAUSAL_PROMPT)
            second = session.prompt(CAUSAL_PROMPT)
        finally:
            session.close()
    for label, events in (("first", first), ("second", second)):
        statuses = terminal_status_messages(events)
        assert len(statuses) == 1, (label, statuses)
        _assert_status_shape(statuses[0])
    # Neither task triggered a model turn beyond its own bounded operation:
    # nine model requests per task (zero automatic follow-ups), eighteen in
    # total.
    assert len(model_state["requests"]) == 18, len(model_state["requests"])
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/abandon") == 2, paths


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

"""Doom-loop regressions for the Pi/model tool handoff after finalization.

Root cause reproduced here: after the runtime returns finish, the model kept
requesting tools every turn. Blocking alone does not end a Pi turn; the
batch only terminates when every call in it is blocked with terminate:true,
so a single non-blocked result (for example a tool Pi cannot find) keeps the
loop alive. These tests prove the adapter terminates the batch, bounds the
tool budget with task-aware caps, and never leaks stale finalization state
into a later ordinary prompt in the same Pi process. Mutation regressions
live in test_pi_doom_loop_mutations.py; the same-process stale-state proof
lives in test_pi_rpc_stale_state.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from pi_doom_loop_helpers import (
    CEILING,
    CERTIFIED,
    MIXED_TURN,
    MIXED_TURN_UNAVAILABLE_FIRST,
    PROMPT,
    TOOL_TURN,
    finish_script,
    never_finished_script,
    workspace,
)
from pi_recovery_helpers import (
    Servers,
    assistant_answers,
    blocked_executions,
    continuation_requests,
    host_executions,
    parse_events,
    require_pi,
    run_pi,
    unavailable_tool_results,
)
from pi_terminal_helpers import terminal_status_messages

EXTENSION = Path(__file__).parents[1] / "src" / "cortheon" / "pi_extension.ts"
SOURCE_DIR = Path(__file__).parents[1] / "src" / "cortheon"


def test_task_aware_caps_are_conservative() -> None:
    """No Pi needed: the source itself must keep the caps conservative."""
    protocol = (SOURCE_DIR / "pi_core" / "protocol.ts").read_text(encoding="utf-8")
    assert "MAX_HOST_TOOL_CALLS_ANSWER = 16" in protocol
    assert "MAX_HOST_TOOL_CALLS_CODE_CHANGE = 48" in protocol
    assert "HOST_TOOL_CALLS_OVERRIDE_CEILING = 64" in protocol
    assert "MAX_HOST_TOOL_CALLS = 80" not in protocol
    assert "ABSOLUTE_MAX_HOST_TOOL_CALLS" not in protocol
    budget = (SOURCE_DIR / "pi_core" / "budget.ts").read_text(encoding="utf-8")
    assert "Math.min(" in budget and "HOST_TOOL_CALLS_OVERRIDE_CEILING" in budget
    # The single decision point must be used, not dead policy code.
    tool_events = (SOURCE_DIR / "pi_core" / "tool_events.ts").read_text(encoding="utf-8")
    assert "toolBatchMustTerminate(active)" in tool_events


def test_finish_phase_terminates_the_tool_loop(tmp_path: Path) -> None:
    """After the runtime returns finish, further tool calls are blocked with
    terminate:true, exactly one answer-only continuation runs, the compliant
    mock's answer is certified through /v1/complete, and the turn ends."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [TOOL_TURN, TOOL_TURN, {"text": "The total count of keys is three."}],
    }
    runtime_state: dict[str, Any] = {"records": [], "script": finish_script(True)}
    with Servers(model_state, runtime_state) as servers:
        started = time.monotonic()
        completed = run_pi(
            EXTENSION,
            PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=30,
        )
        elapsed = time.monotonic() - started

    assert completed.returncode == 0, completed.stderr
    events = parse_events(completed.stdout)
    executed = host_executions(events)
    assert len(executed) == 2, [event.get("toolName") for event in executed]
    assert len(blocked_executions(events)) == 2
    assert continuation_requests(model_state) == 1
    answers = assistant_answers(completed)
    assert answers and answers[-1] == CERTIFIED
    paths = [path for path, _body in runtime_state["records"]]
    assert "/v1/complete" in paths
    observes = [body for path, body in runtime_state["records"] if path == "/v1/observe"]
    assert len(observes) == 1
    assert elapsed < 15, elapsed


def test_hostile_model_never_complies_still_exits(tmp_path: Path) -> None:
    """A hostile mock that only ever requests tools must still exit after
    exactly one answer-only continuation, with no further continuation."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    model_state: dict[str, Any] = {"requests": [], "turns": [TOOL_TURN]}
    runtime_state: dict[str, Any] = {"records": [], "script": finish_script(False)}
    with Servers(model_state, runtime_state) as servers:
        started = time.monotonic()
        completed = run_pi(
            EXTENSION,
            PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path / "hostile",
            timeout=30,
        )
        elapsed = time.monotonic() - started

    assert completed.returncode == 0, completed.stderr
    events = parse_events(completed.stdout)
    assert len(host_executions(events)) == 2
    assert continuation_requests(model_state) == 1
    # The hostile model answered nothing certifiable; no completion was sent.
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 0
    terminals = terminal_status_messages(events)
    assert len(terminals) == 1
    assert terminals[0]["details"]["status"] == "withheld"
    assert elapsed < 15, elapsed


def test_hostile_mixed_batch_with_unavailable_tool_still_exits(
    tmp_path: Path,
) -> None:
    """The discovered production failure, exercised for real: after Cortheon
    reaches finish/answer-only, a hostile model issues a mixed batch with one
    valid host tool and one tool Pi cannot find. The valid tool must not
    execute, the unavailable tool's terminate-less result must not keep the
    batch alive, exactly one answer-only continuation runs and still sees the
    accepted evidence, a second hostile continuation ends the turn, and no
    active runtime session survives the process."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    model_state: dict[str, Any] = {"requests": [], "turns": [TOOL_TURN, MIXED_TURN]}
    runtime_state: dict[str, Any] = {"records": [], "script": finish_script(False)}
    with Servers(model_state, runtime_state) as servers:
        started = time.monotonic()
        completed = run_pi(
            EXTENSION,
            PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path / "mixed",
            timeout=30,
        )
        elapsed = time.monotonic() - started

    assert completed.returncode == 0, completed.stderr
    events = parse_events(completed.stdout)
    executed = host_executions(events)
    # Only the pre-finish valid batch executed: nothing after finish.
    assert len(executed) == 2, [event.get("toolName") for event in executed]
    unavailable = unavailable_tool_results(events)
    assert unavailable, "expected at least one unavailable-tool result"
    assert all(event.get("toolName") == "totally_unavailable_probe" for event in unavailable)
    # One answer-only continuation, never a second one.
    assert continuation_requests(model_state) == 1
    assert _continuation_carries_evidence(model_state)
    # No active runtime session survives the process.
    paths = [path for path, _body in runtime_state["records"]]
    assert "/v1/abandon" in paths or "/v1/complete" in paths
    assert elapsed < 20, elapsed


def test_mixed_batch_unavailable_first_marks_answer_only_before_abort(
    tmp_path: Path,
) -> None:
    """The unavailable tool leads the batch, so the abort fires before any
    valid tool's tool_call handler could run: only the abort path's own
    finalization transition can mark answer-only. Without it the turn would
    end with no continuation scheduled while the session lease stays open."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [TOOL_TURN, MIXED_TURN_UNAVAILABLE_FIRST],
    }
    runtime_state: dict[str, Any] = {"records": [], "script": finish_script(False)}
    with Servers(model_state, runtime_state) as servers:
        started = time.monotonic()
        completed = run_pi(
            EXTENSION,
            PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path / "mixed-first",
            timeout=30,
        )
        elapsed = time.monotonic() - started

    assert completed.returncode == 0, completed.stderr
    events = parse_events(completed.stdout)
    executed = host_executions(events)
    # Nothing executed after finish — the abort beat the valid read.
    assert len(executed) == 2, [event.get("toolName") for event in executed]
    unavailable = unavailable_tool_results(events)
    assert unavailable, "expected at least one unavailable-tool result"
    assert all(event.get("toolName") == "totally_unavailable_probe" for event in unavailable)
    # The transition before the abort scheduled exactly one continuation...
    assert continuation_requests(model_state) == 1
    assert _continuation_carries_evidence(model_state)
    # ...and model attempts stayed bounded (no doom loop).
    assert len(model_state["requests"]) <= 4, len(model_state["requests"])
    paths = [path for path, _body in runtime_state["records"]]
    assert "/v1/abandon" in paths or "/v1/complete" in paths
    assert elapsed < 20, elapsed


def test_mixed_batch_at_exact_hard_cap_abandons_active_session(
    tmp_path: Path,
) -> None:
    """At the exact hard cap the unavailable tool's abort must also abandon
    the active runtime session before aborting: no observe may follow the
    abandon, nothing may execute past the cap, and exactly one continuation
    runs with bounded model attempts."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    cap = int(CEILING)
    model_state: dict[str, Any] = {"requests": [], "turns": [TOOL_TURN, MIXED_TURN]}
    runtime_state: dict[str, Any] = {"records": []}
    runtime_state["script"] = never_finished_script(runtime_state)
    with Servers(model_state, runtime_state) as servers:
        started = time.monotonic()
        completed = run_pi(
            EXTENSION,
            PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path / "hard-cap",
            timeout=30,
            extra_env={"CORTHEON_MAX_HOST_TOOL_CALLS": CEILING},
        )
        elapsed = time.monotonic() - started

    assert completed.returncode == 0, completed.stderr
    events = parse_events(completed.stdout)
    executed = host_executions(events)
    # Exactly the cap's worth of host executions: no post-gate execution.
    assert len(executed) == cap, [event.get("toolName") for event in executed]
    assert unavailable_tool_results(events), "expected an unavailable-tool result"
    # The budget terminal abandons the session: zero automatic follow-ups
    # under the unified budget, and the terminal is host-visible.
    assert continuation_requests(model_state) == 0
    assert len(terminal_status_messages(completed)) == 1
    # Bounded model attempts: two full tool batches, then mixed batches that
    # each admit only one call until the cap.
    assert len(model_state["requests"]) <= cap + 3, len(model_state["requests"])
    paths = [path for path, _body in runtime_state["records"]]
    # The active runtime session was abandoned exactly once and nothing —
    # no observe, no heartbeat-bound call — followed it.
    assert paths.count("/v1/abandon") == 1, paths
    assert paths[-1] == "/v1/abandon", paths
    assert elapsed < 20, elapsed


def _continuation_carries_evidence(model_state: dict[str, Any]) -> bool:
    """The whole-operation stop did not erase the accepted evidence: the
    continuation request still carries the Cortheon evidence context."""
    for request in model_state["requests"]:
        payload = json.dumps(request.get("messages", []))
        if "[CORTHEON_CONTINUE]" in payload:
            return "CORTHEON_ACTIVE" in payload
    return False


def _ceiling_run(
    servers: Servers,
    tmp_path: Path,
    *,
    ceiling: str | None = None,
    deliverable: str = "answer",
) -> Any:
    env = {"CORTHEON_MAX_HOST_TOOL_CALLS": ceiling} if ceiling else None
    return run_pi(
        EXTENSION,
        PROMPT,
        model_port=servers.model.server_port,
        runtime_port=servers.runtime.server_port,
        workspace=workspace(tmp_path),
        tmp_path=tmp_path,
        timeout=30,
        extra_env=env,
    )


def test_task_aware_budget_defaults(tmp_path: Path) -> None:
    """The default ceiling admits 16 tools for answer work and 48 for
    code-change work; a never-finished runtime cannot exceed either."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    for deliverable, expected in (("answer", 16), ("code_change", 48)):
        model_state: dict[str, Any] = {"requests": [], "turns": [TOOL_TURN]}
        runtime_state: dict[str, Any] = {"records": []}
        runtime_state["script"] = never_finished_script(runtime_state, deliverable)
        with Servers(model_state, runtime_state) as servers:
            started = time.monotonic()
            completed = _ceiling_run(servers, tmp_path / f"{deliverable}-run")
            elapsed = time.monotonic() - started
        assert completed.returncode == 0, completed.stderr
        events = parse_events(completed.stdout)
        executed = host_executions(events)
        assert len(executed) == expected, [event.get("toolName") for event in executed]
        # Pi's raw tool_execution_start stream (blocked calls included) stays
        # within the cap plus three fully blocked batches: the batch that
        # hits the cap, the follow-up run Pi starts before delivering the
        # continuation message, and that continuation's own batch.
        starts = [event for event in events if event.get("type") == "tool_execution_start"]
        assert len(starts) <= expected + 3 * len(TOOL_TURN["tool_calls"])
        # Budget exhaustion abandons the session: zero follow-ups, one
        # host-visible terminal per task.
        assert continuation_requests(model_state) == 0
        assert len(terminal_status_messages(completed)) == 1
        paths = [path for path, _body in runtime_state["records"]]
        assert paths.count("/v1/abandon") == 1
        assert paths[-1] == "/v1/abandon"
        assert elapsed < 20, elapsed


def test_environment_override_is_clamped_to_64(tmp_path: Path) -> None:
    """CORTHEON_MAX_HOST_TOOL_CALLS=1000 still stops at 64 admitted calls."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    model_state: dict[str, Any] = {"requests": [], "turns": [TOOL_TURN]}
    runtime_state: dict[str, Any] = {"records": []}
    runtime_state["script"] = never_finished_script(runtime_state)
    with Servers(model_state, runtime_state) as servers:
        completed = _ceiling_run(servers, tmp_path / "clamped", ceiling="1000")
    assert completed.returncode == 0, completed.stderr
    assert len(host_executions(parse_events(completed.stdout))) == 64

"""Single-flight /v1/observe claiming regressions over real Pi.

The production failure: one mixed batch's concurrent tool_result handlers
both submitted /v1/observe for the same runtime request, so the second
in-flight response duplicated the accepted request after the first had
already moved the session on. These runs prove the repaired adapter
submits each runtime request exactly once (batched read aggregation
included), releases the claim on transport failure so nothing poisons a
later request or task, keeps a stale duplicate response out of the
session state, and fails loudly when the synchronous claim is removed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from pi_doom_loop_helpers import (
    ANSWER_TURN,
    CERTIFIED,
    PROMPT,
    TOOL_TURN,
    workspace,
)
from pi_recovery_helpers import (
    Servers,
    assistant_answers,
    continuation_requests,
    host_executions,
    parse_events,
    require_pi,
    run_pi,
)
from pi_rpc_session import PiRpcSession
from pi_terminal_constants import CANDIDATE_ENTRY_TYPE, EXTENSION, WITHHELD_MARKER
from pi_terminal_events import custom_entry_data, terminal_status_messages

SOURCE_DIR = Path(__file__).parents[1] / "src" / "cortheon"
# Pi's read tool output for the shared three-key fact file.
EXPECTED_READ_RESULT = "alpha key amber\nbeta key bronze\ngamma key copper\n"


def _observe_records(runtime_state: dict[str, Any]) -> list[dict[str, Any]]:
    return [body for path, body in runtime_state["records"] if path == "/v1/observe"]


def _path_records(runtime_state: dict[str, Any]) -> list[str]:
    return [path for path, _body in runtime_state["records"]]


def _start_payload(session_id: str, request: dict[str, Any]) -> tuple[int, dict]:
    return (
        200,
        {
            "session_id": session_id,
            "status": "observing",
            "session": {"deliverable": "answer"},
            "next_action": {
                "type": "harness_tool",
                "instruction": "Read the fact file.",
                "request": request,
            },
        },
    )


REQ0 = {
    "request_id": "req-0",
    "capability": "reason",
    "query": "Count the keys.",
}
READ_MANY_REQ = {
    "request_id": "req-1",
    "capability": "read_many",
    "query": "Read both fact files.",
    "parameters": {"paths": ["facts/a.txt", "facts/b.txt"]},
}


def poison_script(state: dict[str, Any]):
    """First observe finishes cleanly (ev-1); any later observe — exactly
    the duplicate a removed claim would submit — carries a poison evidence
    id and a poison request that must never enter the session state."""

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return _start_payload("fin-1", REQ0)
        if path == "/v1/observe":
            state["observes"] = state.get("observes", 0) + 1
            if state["observes"] == 1:
                return (
                    200,
                    {
                        "session_id": "fin-1",
                        "status": "observing",
                        "accepted_evidence_ids": ["ev-1"],
                        "next_action": {"type": "finish"},
                    },
                )
            return (
                200,
                {
                    "session_id": "fin-1",
                    "status": "observing",
                    "accepted_evidence_ids": ["ev-dup"],
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": "req-dup",
                            "capability": "reason",
                            "query": "Duplicate.",
                        },
                    },
                },
            )
        if path == "/v1/complete":
            return (
                200,
                {"session_id": "fin-1", "status": "complete", "answer": CERTIFIED},
            )
        return 200, {"status": "ok"}

    return script


def read_many_then_finish_script(state: dict[str, Any]):
    """req-0 is a reasoning request; its observation returns a read_many
    request over two paths, whose single batched observation finishes."""

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return _start_payload("many-1", REQ0)
        if path == "/v1/observe":
            state["observes"] = state.get("observes", 0) + 1
            if state["observes"] == 1:
                return (
                    200,
                    {
                        "session_id": "many-1",
                        "status": "observing",
                        "accepted_evidence_ids": ["ev-1"],
                        "next_action": {
                            "type": "harness_tool",
                            "request": READ_MANY_REQ,
                        },
                    },
                )
            return (
                200,
                {
                    "session_id": "many-1",
                    "status": "observing",
                    "accepted_evidence_ids": ["ev-2"],
                    "next_action": {"type": "finish"},
                },
            )
        if path == "/v1/complete":
            return (
                200,
                {"session_id": "many-1", "status": "complete", "answer": CERTIFIED},
            )
        return 200, {"status": "ok"}

    return script


def transport_reset_script(state: dict[str, Any]):
    """Task one's every observe dies at the transport level; task two's
    runtime answers normally."""

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            state["starts"] = state.get("starts", 0) + 1
            return _start_payload(f"s-{state['starts']}", REQ0)
        if path == "/v1/observe":
            if body.get("session_id") == "s-1":
                return "connection-reset"
            return (
                200,
                {
                    "session_id": body.get("session_id"),
                    "status": "observing",
                    "accepted_evidence_ids": ["ev-1"],
                    "next_action": {"type": "finish"},
                },
            )
        if path == "/v1/complete":
            return (
                200,
                {
                    "session_id": body.get("session_id"),
                    "status": "complete",
                    "answer": CERTIFIED,
                },
            )
        return 200, {"status": "ok"}

    return script


def two_file_workspace(tmp_path: Path) -> Path:
    root = workspace(tmp_path)
    (root / "facts" / "b.txt").write_text("delta key dawn\nepsilon key dusk\n", encoding="utf-8")
    return root


def test_two_racing_tool_results_submit_exactly_one_observe(
    tmp_path: Path,
) -> None:
    """The repaired race: one mixed read+bash batch, both host executions
    complete, and exactly one /v1/observe reaches the runtime for req-0 —
    never a second in-flight duplicate."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [TOOL_TURN, TOOL_TURN, ANSWER_TURN],
    }
    runtime_state: dict[str, Any] = {"records": []}
    runtime_state["script"] = poison_script(runtime_state)
    with Servers(model_state, runtime_state) as servers:
        started = time.monotonic()
        completed = run_pi(
            EXTENSION,
            PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=60,
            extra_env={"CORTHEON_BENCHMARK_CAPTURE_CANDIDATE": "1"},
        )
        elapsed = time.monotonic() - started

    assert completed.returncode == 0, completed.stderr
    events = parse_events(completed.stdout)
    executed = host_executions(events)
    assert len(executed) == 2, [event.get("toolName") for event in executed]
    observes = _observe_records(runtime_state)
    assert len(observes) == 1, observes
    assert observes[0]["request_id"] == "req-0"
    paths = _path_records(runtime_state)
    assert paths.count("/v1/complete") == 1
    # The only abandon is the session_shutdown cleanup after certification.
    assert paths.count("/v1/abandon") == 1, paths
    assert paths[-1] == "/v1/abandon", paths
    assert continuation_requests(model_state) == 1
    assert len(terminal_status_messages(events)) == 0
    assert custom_entry_data(completed, CANDIDATE_ENTRY_TYPE) == []
    answers = assistant_answers(completed)
    assert answers and answers[-1] == CERTIFIED
    completes = [body for path, body in runtime_state["records"] if path == "/v1/complete"]
    assert completes[0]["completion_evidence_ids"] == ["ev-1"]
    assert elapsed < 30, elapsed


def test_two_required_read_paths_batch_into_one_observe(tmp_path: Path) -> None:
    """A read_many request over two paths: both required read observations
    are collected first, and exactly one batched /v1/observe carries both —
    never one observe per read."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    read_two: dict[str, Any] = {
        "tool_calls": [
            ("read", {"path": "facts/a.txt"}),
            ("read", {"path": "facts/b.txt"}),
        ]
    }
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [TOOL_TURN, read_two, ANSWER_TURN],
    }
    runtime_state: dict[str, Any] = {"records": []}
    runtime_state["script"] = read_many_then_finish_script(runtime_state)
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            EXTENSION,
            PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=two_file_workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=60,
        )

    assert completed.returncode == 0, completed.stderr
    observes = _observe_records(runtime_state)
    assert len(observes) == 2, observes
    batched = [body for body in observes if body["request_id"] == "req-1"]
    assert len(batched) == 1
    assert len(batched[0]["observations"]) == 2
    receipts = [
        json.loads(
            str(observation.get("content", "")).split("\n", 1)[0][
                len("[CORTHEON_HOST_EVIDENCE] ") :
            ]
        )
        for observation in batched[0]["observations"]
    ]
    assert {receipt["args"]["filePath"] for receipt in receipts} == {"facts/a.txt", "facts/b.txt"}
    answers = assistant_answers(completed)
    assert answers and answers[-1] == CERTIFIED


def test_ambiguous_reset_fails_open_once_and_never_poisons_a_later_task(
    tmp_path: Path,
) -> None:
    """Task one's observation dies at the transport level: the failure is
    ambiguous (the runtime may have committed before the response was
    lost), so the claim is held and the exact (session, request) submitted
    at most once. Fail-open makes Cortheon disappear from the turn: one
    host tool whose result stands verbatim, one observe attempt, one
    abandon, zero completes/continuations/terminals/candidates, and the
    model's ordinary answer unchanged. A later task in the same Pi process
    gets a fresh claim table and still observes, completes, and certifies."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    read_turn: dict[str, Any] = {"tool_calls": [("read", {"path": "facts/a.txt"})]}
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [read_turn, ANSWER_TURN, read_turn, ANSWER_TURN],
    }
    runtime_state: dict[str, Any] = {"records": []}
    runtime_state["script"] = transport_reset_script(runtime_state)
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
            second = session.prompt(PROMPT)
        finally:
            session.close()

    # Task one failed open with the exact trace: one host tool whose result
    # is the verbatim read bytes, one observe attempt, one abandon.
    executed = host_executions(first)
    assert len(executed) == 1, [event.get("toolName") for event in executed]
    assert executed[0]["result"]["content"] == [{"type": "text", "text": EXPECTED_READ_RESULT}], (
        executed[0]["result"]
    )
    first_answers = [
        block.get("text", "")
        for event in first
        if event.get("type") == "message_end"
        and event.get("message", {}).get("role") == "assistant"
        for block in event["message"].get("content", [])
        if block.get("type") == "text"
    ]
    assert first_answers == [ANSWER_TURN["text"]], first_answers
    assert not any(text.startswith(WITHHELD_MARKER) for text in first_answers), first_answers
    paths = _path_records(runtime_state)
    failed_observes = [
        body for body in _observe_records(runtime_state) if body.get("session_id") == "s-1"
    ]
    # At-most-once: the ambiguous observe is submitted exactly once and
    # never resubmitted.
    assert len(failed_observes) == 1, failed_observes
    assert not any(
        path == "/v1/complete" and body.get("session_id") == "s-1"
        for path, body in runtime_state["records"]
    ), runtime_state["records"]
    # Two legitimate cleanups only: task one's fail-open abandon and task
    # two's completed-session abandon at shutdown.
    assert paths.count("/v1/abandon") == 2, paths
    assert paths.count("/v1/heartbeat") == 0, paths
    # Task two was not poisoned: its fresh session observed once, completed,
    # and delivered the certified answer.
    good_observes = [
        body for body in _observe_records(runtime_state) if body.get("session_id") == "s-2"
    ]
    assert len(good_observes) == 1, good_observes
    second_answers = [
        block.get("text", "")
        for event in second
        if event.get("type") == "message_end"
        and event.get("message", {}).get("role") == "assistant"
        for block in event["message"].get("content", [])
        if block.get("type") == "text"
    ]
    assert second_answers and second_answers[-1] == CERTIFIED


def test_stale_duplicate_response_cannot_overwrite_the_session(
    tmp_path: Path,
) -> None:
    """A would-be second in-flight observe response carries poison evidence
    and a poison request; the shipped adapter never submits it, so no
    poison id or request id ever reaches the runtime again."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [TOOL_TURN, TOOL_TURN, ANSWER_TURN],
    }
    runtime_state: dict[str, Any] = {"records": []}
    runtime_state["script"] = poison_script(runtime_state)
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            EXTENSION,
            PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path / "stale",
            timeout=60,
        )

    assert completed.returncode == 0, completed.stderr
    assert len(_observe_records(runtime_state)) == 1
    assert not any(
        "req-dup" in json.dumps(body) or "ev-dup" in json.dumps(body)
        for _path, body in runtime_state["records"]
    ), runtime_state["records"]
    answers = assistant_answers(completed)
    assert answers and answers[-1] == CERTIFIED

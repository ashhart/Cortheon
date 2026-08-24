"""At-most-once /v1/observe semantics for ambiguous transport failures.

A connection reset during /v1/observe is ambiguous: the runtime may have
committed the observation before the response was lost. These pins prove
the adapter never resubmits such a request — against a server that commits
then resets and rejects any duplicate as already resolved — and that a
stale response can never merge into a replacement investigation even when
the later task deliberately reuses both the session and the request id
strings, because merge requires exact-object identity, not just matching
IDs. The classic one-observe race pin stays in test_pi_observe_claim.py.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest
from pi_doom_loop_helpers import ANSWER_TURN, CERTIFIED, PROMPT, TOOL_TURN, workspace
from pi_recovery_helpers import (
    Servers,
    assistant_answers,
    blocked_executions,
    continuation_requests,
    host_executions,
    parse_events,
    require_pi,
    run_pi,
)
from pi_terminal_constants import (
    CANDIDATE_ENTRY_TYPE,
    EXTENSION,
    WITHHELD_MARKER,
)
from pi_terminal_events import custom_entry_data, terminal_status_messages

REQ0 = {"request_id": "req-0", "capability": "reason", "query": "Count the keys."}
SOURCE_DIR = Path(__file__).parents[1] / "src" / "cortheon"
# One host read tool: the exact bytes Pi's read tool returns for the
# three-key fact file — a fail-open transport path must deliver exactly
# these and nothing else.
READ_TURN: dict[str, Any] = {"tool_calls": [("read", {"path": "facts/a.txt"})]}
EXPECTED_READ_RESULT = "alpha key amber\nbeta key bronze\ngamma key copper\n"


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


def commit_then_reset_script(state: dict[str, Any]):
    """The at-most-once pin: the server COMMITS every (session, request)
    it observes, then drops the connection. A duplicate submission of an
    already-resolved request is rejected with 409 — so any resubmission
    after the ambiguous reset fails the run loudly."""

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return _start_payload("crl-1", REQ0)
        if path == "/v1/observe":
            key = (body.get("session_id"), body.get("request_id"))
            seen = state.setdefault("resolved", [])
            if key in seen:
                state.setdefault("duplicates", []).append(key)
                return (409, {"error": "already resolved"})
            seen.append(key)
            return "connection-reset"
        if path == "/v1/complete":
            return (
                200,
                {"session_id": "crl-1", "status": "complete", "answer": CERTIFIED},
            )
        return 200, {"status": "ok"}

    return script


def test_committed_then_reset_observe_fails_open_with_exact_trace(
    tmp_path: Path,
) -> None:
    """A server that commits the observation and then resets the connection
    must never see that (session, request) again: the runtime may have
    committed before the response was lost, so the adapter keeps the claim
    and never resubmits — the duplicate-rejecting server proves it. Fail-open
    means Cortheon disappears from the turn: the exact trace is one host
    tool, one observe attempt, one abandon, zero completes, zero
    continuations, zero Cortheon terminals/candidates — and the host tool
    result plus the model's ordinary answer stand byte-for-byte."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [READ_TURN, ANSWER_TURN],
    }
    runtime_state: dict[str, Any] = {"records": []}
    runtime_state["script"] = commit_then_reset_script(runtime_state)
    with Servers(model_state, runtime_state) as servers:
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

    assert completed.returncode == 0, completed.stderr
    events = parse_events(completed.stdout)
    # Exactly one host tool execution whose result is the verbatim read
    # bytes — no Cortheon notice of any kind.
    executed = host_executions(events)
    assert len(executed) == 1, [event.get("toolName") for event in executed]
    assert executed[0]["result"]["content"] == [{"type": "text", "text": EXPECTED_READ_RESULT}], (
        executed[0]["result"]
    )
    observes = [body for path, body in runtime_state["records"] if path == "/v1/observe"]
    # Exactly one submission per (session, request); the server's duplicate
    # rejection never fired.
    assert len(observes) == 1, observes
    assert observes[0]["request_id"] == "req-0"
    assert runtime_state.get("duplicates", []) == []
    assert set(runtime_state.get("resolved", [])) == {("crl-1", "req-0")}
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/abandon") == 1, paths
    assert paths.count("/v1/complete") == 0, paths
    assert paths.count("/v1/heartbeat") == 0, paths
    assert continuation_requests(model_state) == 0, model_state["requests"]
    assert terminal_status_messages(events) == []
    assert custom_entry_data(completed, CANDIDATE_ENTRY_TYPE) == []
    # The model's ordinary answer passes through byte-for-byte.
    answers = assistant_answers(completed)
    assert answers == [ANSWER_TURN["text"]], answers
    assert not any(text.startswith(WITHHELD_MARKER) for text in answers), answers


def policy_refusal_script(state: dict[str, Any]):
    """Every /v1/observe for this session is an explicit live 422 cognitive
    policy refusal."""

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return _start_payload("pol-1", REQ0)
        if path == "/v1/observe":
            return (422, {"error": "cognitive policy refusal"})
        if path == "/v1/complete":
            return (
                200,
                {"session_id": "pol-1", "status": "complete", "answer": CERTIFIED},
            )
        return 200, {"status": "ok"}

    return script


def test_observe_policy_refusal_fails_closed_with_exact_trace(
    tmp_path: Path,
) -> None:
    """A typed live policy refusal (HTTP 422) on /v1/observe fails closed
    exactly once: one host tool, one observe, one abandon — then no further
    host execution, no complete, no continuation, no heartbeat, and exactly
    one withheld terminal through the established replacement path. The raw
    refusal string never reaches the host, and because no answer was ever
    submitted, zero benchmark candidates are captured."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    model_state: dict[str, Any] = {
        "requests": [],
        # The second turn's tool attempt must be blocked, never executed.
        "turns": [READ_TURN, READ_TURN, ANSWER_TURN],
    }
    runtime_state: dict[str, Any] = {"records": []}
    runtime_state["script"] = policy_refusal_script(runtime_state)
    with Servers(model_state, runtime_state) as servers:
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

    assert completed.returncode == 0, completed.stderr
    events = parse_events(completed.stdout)
    # One executed host tool; the later attempt was blocked, not executed.
    executed = host_executions(events)
    assert len(executed) == 1, [event.get("toolName") for event in executed]
    assert len(blocked_executions(events)) >= 1, "later tool attempt not blocked"
    observes = [body for path, body in runtime_state["records"] if path == "/v1/observe"]
    assert len(observes) == 1, observes
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/abandon") == 1, paths
    assert paths.count("/v1/complete") == 0, paths
    assert paths.count("/v1/heartbeat") == 0, paths
    assert continuation_requests(model_state) == 0, model_state["requests"]
    # Exactly ONE authenticated withheld terminal: the typed terminal-status
    # channel with the truthful non-causal reason. The blocked later tool
    # terminated the operation, so no assistant answer turn ever ran.
    assert assistant_answers(completed) == []
    statuses = terminal_status_messages(events)
    assert len(statuses) == 1, statuses
    assert statuses[0]["content"].startswith(WITHHELD_MARKER), statuses[0]
    assert "refused the observation" in statuses[0]["content"], statuses[0]
    details = statuses[0].get("details", {})
    assert details.get("status") == "withheld", details
    assert details.get("causal") is False, details
    assert "refused the observation" in details.get("reason", ""), details
    # Never the raw error string, and no candidate (nothing was submitted).
    assert "cognitive policy refusal" not in completed.stdout
    assert custom_entry_data(completed, CANDIDATE_ENTRY_TYPE) == []


IDENTITY_HARNESS = """
const state = await import(%s);
const claims = await import(%s);
const investigation = (generation) => ({
  sessionId: "dup-1",
  request: { request_id: "req-0" },
  generation,
});
// Task one: active object A claims and submits its (reused-string) request.
const first = investigation(1);
state.setActive(first);
const firstClaim = claims.claimObservation("dup-1", "req-0");
// The task ends: the reset wipes claims and finalization state.
state.resetFinalization();
state.setActive(undefined);
// The replacement task legally reuses BOTH id strings: new object B.
const second = investigation(2);
state.setActive(second);
const secondClaim = claims.claimObservation("dup-1", "req-0");
// Task one's stale response arrives late; task two's own response arrives.
const staleMerges = claims.observationStillCurrent(
  state.getActive(), first, "dup-1", "req-0");
const freshMerges = claims.observationStillCurrent(
  state.getActive(), second, "dup-1", "req-0");
// Without a task reset, the ambiguous claim is still held: at-most-once.
const thirdClaim = claims.claimObservation("dup-1", "req-0");
console.log(JSON.stringify({
  firstClaim, secondClaim, staleMerges, freshMerges, thirdClaim,
}));
"""


def _identity_report(tmp_path: Path, mutations: list[tuple[str, str]]) -> dict:
    node = shutil.which("node")
    assert node is not None, "Node is not installed"
    harness_dir = tmp_path / "identity-harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_DIR / "pi_core", harness_dir / "pi_core")
    claim_path = harness_dir / "pi_core" / "observe_claim.ts"
    text = claim_path.read_text(encoding="utf-8")
    for old, new in mutations:
        assert old in text, old
        text = text.replace(old, new)
    claim_path.write_text(text, encoding="utf-8")
    harness = harness_dir / "identity.mjs"
    state_url = json.dumps(str(harness_dir / "pi_core" / "state.ts"))
    claims_url = json.dumps(str(harness_dir / "pi_core" / "observe_claim.ts"))
    harness.write_text(IDENTITY_HARNESS % (state_url, claims_url), encoding="utf-8")
    completed = subprocess.run(
        [node, "--experimental-strip-types", str(harness)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_stale_response_cannot_merge_into_an_id_reusing_replacement(
    tmp_path: Path,
) -> None:
    """Same-process, real reviewed sources: a replacement task deliberately
    reusing BOTH the session and the request id strings gets a fresh claim
    (never poisoned), while the old task's stale response — matching both
    IDs — must still be refused because the active object is not the exact
    object that submitted it."""
    report = _identity_report(tmp_path, [])
    assert report == {
        "firstClaim": True,
        "secondClaim": True,
        "staleMerges": False,
        "freshMerges": True,
        "thirdClaim": False,
    }, report


def test_removing_the_identity_check_reopens_the_stale_merge(
    tmp_path: Path,
) -> None:
    """Mutation guard: with the exact-object identity check neutered (IDs
    only), the stale response of the abandoned task merges into the
    replacement that reused both ID strings — the failure the identity
    requirement exists to prevent."""
    report = _identity_report(
        tmp_path,
        [("active === submitted &&", "")],
    )
    assert report["staleMerges"] is True, report


def _mutated_extension(tmp_path: Path, module: str, replacements: list[tuple[str, str]]) -> Path:
    root = tmp_path / "cortheon"
    (root / "pi_core").mkdir(parents=True)
    for path in sorted((SOURCE_DIR / "pi_core").glob("*.ts")):
        text = path.read_text(encoding="utf-8")
        if path.stem == module:
            for old, new in replacements:
                assert old in text, old
                text = text.replace(old, new)
        (root / "pi_core" / path.name).write_text(text, encoding="utf-8")
    facade = root / "pi_extension.ts"
    shutil.copy2(SOURCE_DIR / "pi_extension.ts", facade)
    return facade


def test_removing_the_claim_reintroduces_the_duplicate(tmp_path: Path) -> None:
    """Mutation guard: with the synchronous claim removed, the racing batch
    submits the duplicate observe, the poison response overwrites the
    session state, and the poison ids leak back into later runtime requests.
    The first observe response is latched until a duplicate arrives (or the
    window closes), so the race is deterministic on every Pi timing."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    extension = _mutated_extension(
        tmp_path / "mutation",
        "observe_claim",
        [("if (claims.has(claim)) return false;", "void claims;")],
    )
    duplicate = threading.Event()
    latch_state: dict[str, Any] = {"observes": 0}

    def latched_poison_script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return _start_payload("fin-1", REQ0)
        if path == "/v1/observe":
            latch_state["observes"] += 1
            if latch_state["observes"] == 1:
                duplicate.wait(timeout=3)
                return (
                    200,
                    {
                        "session_id": "fin-1",
                        "status": "observing",
                        "accepted_evidence_ids": ["ev-1"],
                        "next_action": {"type": "finish"},
                    },
                )
            duplicate.set()
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

    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [TOOL_TURN] * 4 + [TOOL_TURN, ANSWER_TURN],
    }
    runtime_state: dict[str, Any] = {"records": []}
    runtime_state["script"] = latched_poison_script
    with Servers(model_state, runtime_state) as servers:
        run_pi(
            extension,
            PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path / "run",
            timeout=60,
        )

    observes = [body for path, body in runtime_state["records"] if path == "/v1/observe"]
    assert len(observes) >= 2, observes
    assert any(body.get("request_id") == "req-dup" for body in observes), observes
    assert any("ev-dup" in json.dumps(body) for _path, body in runtime_state["records"]), (
        "the stale duplicate response never leaked into later requests"
    )

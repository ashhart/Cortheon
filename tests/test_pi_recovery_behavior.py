"""Behavioral loop/recovery regressions for the Cortheon Pi adapter.

These tests drive the real ``pi`` CLI with the real adapter source, a scripted
mock model server, and a scripted Cortheon runtime HTTP server.  They assert
observable behavior (model turn counts, assistant answers, runtime call
sequences) rather than source strings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pi_recovery_helpers import Servers, assistant_answers, require_pi, run_pi

EXTENSION = Path(__file__).parents[1] / "src" / "cortheon" / "pi_extension.ts"

CERTIFIED_MULTI = "CORTHEON CERTIFIED: a.txt and b.txt were updated and the protected test passed."


def _multi_mutation_script(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    if path == "/v1/start":
        return (
            200,
            {
                "session_id": "mm-1",
                "status": "observing",
                "session": {"deliverable": "code_change"},
                "next_action": {
                    "type": "harness_tool",
                    "instruction": "Read the requested files first.",
                    "request": {
                        "request_id": "req-reads",
                        "capability": "read_many",
                        "query": "Read the target files before mutation.",
                        "parameters": {"paths": ["a.txt", "b.txt", "test_final.py"]},
                    },
                },
            },
        )
    sources = [
        observation.get("source", "")
        for observation in body.get("observations", [])
        if isinstance(observation, dict)
    ]
    if path == "/v1/observe" and any(
        isinstance(source, str) and source.startswith("pi:final-state:") for source in sources
    ):
        return (
            200,
            {
                "session_id": "mm-1",
                "status": "observing",
                "accepted_evidence_ids": ["ev-d1", "ev-d2", "ev-t"],
                "next_action": {
                    "type": "harness_tool",
                    "instruction": "Bind the final evidence.",
                    "request": {
                        "request_id": "req-final",
                        "capability": "reason",
                        "query": "FINAL-QUERY-XYZ",
                    },
                },
            },
        )
    if path == "/v1/observe":
        return (
            200,
            {
                "session_id": "mm-1",
                "status": "observing",
                "accepted_evidence_ids": ["ev-r1", "ev-r2", "ev-r3"],
                "next_action": {"type": "finish"},
            },
        )
    if path == "/v1/complete":
        return (
            200,
            {"session_id": "mm-1", "status": "complete", "answer": CERTIFIED_MULTI},
        )
    return 200, {"status": "ok"}


def test_multi_mutation_completion_uses_refreshed_state(tmp_path: Path) -> None:
    """The completion payload must reflect post-observation state, not the
    pre-observation snapshot captured when certification started."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("alpha original\n", encoding="utf-8")
    (workspace / "b.txt").write_text("beta original\n", encoding="utf-8")
    (workspace / "test_final.py").write_text(
        "import unittest\n\n\nclass Final(unittest.TestCase):\n"
        "    def test_ok(self) -> None:\n        self.assertTrue(True)\n",
        encoding="utf-8",
    )
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [
            {
                "tool_calls": [
                    (
                        "edit",
                        {
                            "path": "a.txt",
                            "edits": [{"oldText": "alpha original", "newText": "alpha fixed"}],
                        },
                    ),
                    (
                        "edit",
                        {
                            "path": "b.txt",
                            "edits": [{"oldText": "beta original", "newText": "beta fixed"}],
                        },
                    ),
                ]
            },
            {"tool_calls": [("bash", {"command": "python3 -m unittest test_final"})]},
            {"text": "Both requested files are updated and the test passed."},
        ],
    }
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": _multi_mutation_script,
    }
    with Servers(model_state, runtime_state) as servers:
        prompt = (
            "Read a.txt, b.txt, and test_final.py. Edit a.txt and b.txt to fix "
            "the bug. Do not change the tests in test_final.py. Run "
            "python3 -m unittest test_final after and report."
        )
        completed = run_pi(
            EXTENSION,
            prompt,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace,
            tmp_path=tmp_path,
        )

    assert completed.returncode == 0, completed.stderr
    completes = [body for path, body in runtime_state["records"] if path == "/v1/complete"]
    assert len(completes) == 1
    hypotheses = completes[0]["hypotheses"]
    assert hypotheses, "completion must carry hypotheses"
    # The refreshed state carries the final observation's pending request and
    # the evidence ids it accepted; the stale pre-observation snapshot has
    # neither.
    assert any(
        hypothesis.get("falsification_test") == "FINAL-QUERY-XYZ" for hypothesis in hypotheses
    )
    for hypothesis in hypotheses:
        assert {"ev-d1", "ev-d2", "ev-t"} <= set(hypothesis["evidence_ids"])
    answers = assistant_answers(completed)
    assert answers and answers[-1] == CERTIFIED_MULTI


def _outage_script(failure: str, state: dict[str, Any]) -> Any:
    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "out-1",
                    "status": "observing",
                    "session": {"deliverable": "answer"},
                    "next_action": {
                        "type": "harness_tool",
                        "instruction": "Provide the missing bound.",
                        "request": {
                            "request_id": "req-pending",
                            "capability": "reason",
                            "query": "State the invariant.",
                        },
                    },
                },
            )
        if path == "/v1/complete":
            if state.get("fail"):
                return failure
            return (
                200,
                {
                    "session_id": "out-1",
                    "status": "complete",
                    "answer": "CORTHEON CERTIFIED RECOVERY.",
                },
            )
        return 200, {"status": "ok"}

    return script


@pytest.mark.parametrize("failure", ["http503", "invalid-json", "non-object"])
def test_runtime_failure_preserves_host_answer_and_recovers(tmp_path: Path, failure: str) -> None:
    """A mid-session runtime outage must abandon the ephemeral session, keep
    the host model's original answer, enqueue no follow-up turns, and leave a
    later session free to succeed."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    raw_failure = {
        "http503": (503, {"error": "runtime unavailable"}),
        "invalid-json": "invalid-json",
        "non-object": "non-object",
    }[failure]
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [{"text": "The plugin fix is complete and verified."}],
    }
    runtime_state: dict[str, Any] = {"records": [], "fail": True}
    runtime_state["script"] = _outage_script(raw_failure, runtime_state)
    with Servers(model_state, runtime_state) as servers:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outage = run_pi(
            EXTENSION,
            "Implement a focused plugin fix and verify the result.",
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace,
            tmp_path=tmp_path / "outage",
        )
        first_model_requests = len(model_state["requests"])
        first_records = list(runtime_state["records"])
        model_state["requests"].clear()
        runtime_state["records"].clear()
        runtime_state["fail"] = False
        recovered = run_pi(
            EXTENSION,
            "Implement a focused plugin fix and verify the result.",
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace,
            tmp_path=tmp_path / "recovery",
        )

    assert outage.returncode == 0, outage.stderr
    original = "The plugin fix is complete and verified."
    assert first_model_requests == 1
    assert assistant_answers(outage) == [original]
    paths = [path for path, _body in first_records]
    assert paths.count("/v1/complete") == 1
    assert paths.count("/v1/abandon") == 1
    # Abandonment is provable because shutdown's abandon is a no-op once the
    # state is cleared: exactly one abandon POST, and nothing after it.
    assert paths[-1] == "/v1/abandon"

    assert recovered.returncode == 0, recovered.stderr
    assert len(model_state["requests"]) == 1
    assert assistant_answers(recovered) == ["CORTHEON CERTIFIED RECOVERY."]
    recovered_paths = [path for path, _body in runtime_state["records"]]
    assert recovered_paths.count("/v1/complete") == 1
    # Any abandon in the recovered run is shutdown cleanup of the still
    # tracked completed session, and must follow the successful completion.
    assert recovered_paths.index("/v1/abandon") > recovered_paths.index("/v1/complete")


def test_continuations_cap_at_one_then_abandon(tmp_path: Path) -> None:
    """A runtime that keeps a fresh bounded request pending gets exactly one
    automatic continuation (each withhold carries a new request, so the
    progress fingerprint never repeats), then the session is abandoned with
    a sticky terminal disposition."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    counter = {"completes": 0}

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "cap-1",
                    "status": "observing",
                    "session": {"deliverable": "answer"},
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": "req-0",
                            "capability": "reason",
                            "query": "Provide the missing bound.",
                        },
                    },
                },
            )
        if path == "/v1/complete":
            counter["completes"] += 1
            return (
                200,
                {
                    "session_id": "cap-1",
                    "status": "observing",
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": f"req-{counter['completes']}",
                            "capability": "reason",
                            "query": "Provide the missing bound.",
                        },
                    },
                },
            )
        return 200, {"status": "ok"}

    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [{"text": "Draft answer under review."}],
    }
    runtime_state: dict[str, Any] = {"records": [], "script": script}
    with Servers(model_state, runtime_state) as servers:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        completed = run_pi(
            EXTENSION,
            "Implement a focused plugin fix and verify the result.",
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace,
            tmp_path=tmp_path,
        )

    assert completed.returncode == 0, completed.stderr
    assert len(model_state["requests"]) == 2
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 2
    assert paths.count("/v1/abandon") == 1
    assert paths[-1] == "/v1/abandon", paths
    answers = assistant_answers(completed)
    assert len(answers) == 2
    assert all(
        answer.startswith("[Cortheon withheld: completion was not certified]") for answer in answers
    )

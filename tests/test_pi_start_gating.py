"""Automatic-start gating for unsatisfiable deterministic grep requests.

A live runtime can issue a deterministic grep request the host cannot
satisfy: invalid parameters, a path outside the project, or a host tool
failure. The adapter must never silently abandon the session and drop to an
unbounded bare-model path. It re-plans through the runtime via one bounded
failed observation; when the runtime cannot move on, the invocation ends
with a truthful explicit disposition, one abandon, and one host-visible
terminal — never an ungated raw answer.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from pi_doom_loop_helpers import workspace
from pi_recovery_helpers import Servers, assistant_answers, require_pi, run_pi
from pi_terminal_helpers import (
    AMBIGUITY_ANSWER,
    AMBIGUITY_PROMPT,
    EXTENSION,
    WITHHELD_MARKER,
    terminal_status_messages,
)

SOURCE_DIR = Path(__file__).parents[1] / "src" / "cortheon"

GREP_REQUEST = {
    "request_id": "req-grep-1",
    "capability": "grep",
    "query": "Does the module import the keyed client?",
}


def _start_script(parameters: dict[str, Any], replan: bool):
    """A runtime whose /v1/start carries a deterministic grep request with
    the given parameters. The failure-report observation is answered either
    with a re-plan (no further request) or with the same pending request."""

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "gate-1",
                    "status": "observing",
                    "session": {"deliverable": "document_synthesis"},
                    "context": {"goal": body.get("goal")},
                    "next_action": {
                        "type": "harness_tool",
                        "request": {**GREP_REQUEST, "parameters": parameters},
                    },
                },
            )
        if path == "/v1/observe":
            if replan:
                return (
                    200,
                    {
                        "session_id": "gate-1",
                        "status": "observing",
                        "accepted_evidence_ids": [],
                        "context": {"goal": AMBIGUITY_PROMPT, "evidence": []},
                        "next_action": {"type": "reason"},
                    },
                )
            return (
                200,
                {
                    "session_id": "gate-1",
                    "status": "needs_evidence",
                    "next_action": {
                        "type": "harness_tool",
                        "request": {**GREP_REQUEST, "parameters": parameters},
                    },
                },
            )
        if path == "/v1/complete":
            return (
                200,
                {
                    "session_id": "gate-1",
                    "status": "needs_evidence",
                    "next_action": {"type": "verify", "submit_via": "cortheon_challenge"},
                },
            )
        return 200, {"status": "ok"}

    return script


def _run(tmp_path: Path, parameters: dict[str, Any], replan: bool):
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [{"text": AMBIGUITY_ANSWER}],
    }
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": _start_script(parameters, replan),
    }
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            EXTENSION,
            AMBIGUITY_PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=60,
        )
    return completed, model_state, runtime_state


def _failure_reports(runtime_state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        body
        for path, body in runtime_state["records"]
        if path == "/v1/observe"
        and any(
            observation.get("status") == "failed"
            for observation in body.get("observations", [])
            if isinstance(observation, dict)
        )
    ]


@pytest.mark.parametrize(
    "name,parameters",
    [
        ("invalid-parameters", {"pattern": "", "path": "facts/a.txt"}),
        ("outside-project", {"pattern": "copper", "path": "../outside.txt"}),
    ],
)
def test_unsatisfiable_grep_replans_through_the_runtime(
    tmp_path: Path, name: str, parameters: dict[str, Any]
) -> None:
    """An invalid or outside-project grep request is reported to the live
    runtime as one bounded failed observation; the runtime re-plans (no
    pending request remains) and the invocation stays gated: the model's
    answer is only ever delivered through the withhold/certify boundary,
    never as an unbounded raw path."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, model_state, runtime_state = _run(tmp_path / f"run-{name}", parameters, replan=True)
    assert completed.returncode == 0, completed.stderr
    reports = _failure_reports(runtime_state)
    assert reports, [path for path, _body in runtime_state["records"]]
    # The re-plan really moved on and the run stayed bounded and gated.
    answers = assistant_answers(completed)
    assert AMBIGUITY_ANSWER not in answers, answers
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") >= 1, paths
    assert len(model_state["requests"]) <= 3, len(model_state["requests"])


def test_unsatisfiable_grep_without_replan_ends_explicitly_gated(
    tmp_path: Path,
) -> None:
    """A host-tool failure the runtime cannot re-plan (the same grep request
    stays pending after the failed observation) never drops to an unbounded
    bare-model path: the session ends with a truthful explicit disposition,
    exactly one abandon, one host-visible terminal, and the model's single
    answer replaced by the withheld result."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, model_state, runtime_state = _run(
        tmp_path / "run",
        {"pattern": "copper", "path": "facts/missing.txt"},
        replan=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert _failure_reports(runtime_state), [path for path, _body in runtime_state["records"]]
    answers = assistant_answers(completed)
    assert AMBIGUITY_ANSWER not in answers, answers
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert "could not be satisfied" in answers[-1], answers[-1]
    # The replaced answer IS the one host-visible terminal withhold (the
    # sticky disposition text); no duplicate custom status follows it.
    statuses = terminal_status_messages(completed)
    assert not statuses, statuses
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 0, paths
    assert paths.count("/v1/abandon") == 1, paths
    assert paths[-1] == "/v1/abandon", paths
    assert len(model_state["requests"]) == 1, len(model_state["requests"])


def _mutated_session_events(tmp_path: Path, replacements: list[tuple[str, str]]) -> Path:
    """Copy the adapter with the session_events module mutated in place."""
    root = tmp_path / "cortheon"
    (root / "pi_core").mkdir(parents=True)
    for path in sorted((SOURCE_DIR / "pi_core").glob("*.ts")):
        text = path.read_text(encoding="utf-8")
        if path.stem == "session_events":
            for old, new in replacements:
                assert old in text, old
                text = text.replace(old, new)
        (root / "pi_core" / path.name).write_text(text, encoding="utf-8")
    facade = root / "pi_extension.ts"
    shutil.copy2(SOURCE_DIR / "pi_extension.ts", facade)
    return facade


# The outer automatic-start catch cleaned up with a bare state drop:
# active is cleared locally, but the heartbeat timer, the observation
# claims, and — decisively — the runtime-side session state all survive
# because no /v1/abandon is ever sent.
BARE_STATE_DROP = [
    (
        "import {\n\tabandonActive,",
        "import {\n\tsetActive,\n\tabandonActive,",
    ),
    (
        "\t\t\t\t// Clean up like every other abandonment — heartbeat, claims,\n"
        "\t\t\t\t// best-effort /v1/abandon — never a bare state drop that leaves\n"
        "\t\t\t\t// runtime-side session state alive.\n"
        "\t\t\t\tawait abandonActive();",
        "\t\t\t\tsetActive(undefined);",
    ),
]


def _start_observe_transport_failure_script():
    """A runtime that starts cleanly with a satisfiable read_many request
    whose observation then dies at the transport level inside the automatic
    start path, forcing the outer catch to clean up a live session."""

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "start-1",
                    "status": "observing",
                    "session": {"deliverable": "answer"},
                    "context": {"goal": AMBIGUITY_PROMPT},
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": "req-rm-1",
                            "capability": "read_many",
                            "query": "Read the fact file.",
                            "parameters": {"paths": ["facts/a.txt"]},
                        },
                    },
                },
            )
        if path == "/v1/observe":
            return "invalid-json"
        return 200, {"status": "ok"}

    return script


def _run_start_failure(extension: Path, tmp_path: Path) -> dict[str, Any]:
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [{"text": AMBIGUITY_ANSWER}],
    }
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": _start_observe_transport_failure_script(),
    }
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            extension,
            AMBIGUITY_PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=60,
        )
    assert completed.returncode == 0, completed.stderr
    return runtime_state


def test_automatic_start_failure_cleans_up_through_abandon(tmp_path: Path) -> None:
    """A transport failure inside the automatic-start sequence (after the
    session is live) fails open with full cleanup: exactly one /v1/abandon,
    and the model's ordinary answer delivered verbatim."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    runtime_state = _run_start_failure(EXTENSION, tmp_path / "shipped")
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/abandon") == 1, paths
    assert paths[-1] == "/v1/abandon", paths


def test_mutation_merely_setActive_undefined_leaves_the_session_alive(
    tmp_path: Path,
) -> None:
    """Mutation proof: when the outer catch merely does setActive(undefined)
    instead of await abandonActive(), no /v1/abandon is ever sent — the
    runtime-side session state survives the failure, which is exactly the
    leak the real cleanup exists to prevent."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    extension = _mutated_session_events(tmp_path / "mutation", BARE_STATE_DROP)
    runtime_state = _run_start_failure(extension, tmp_path / "run")
    paths = [path for path, _body in runtime_state["records"]]
    # The bare state drop never told the runtime to abandon: the proof that
    # setActive(undefined) alone fails the cleanup contract.
    assert paths.count("/v1/abandon") == 0, paths


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

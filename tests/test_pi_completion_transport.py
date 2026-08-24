"""Completion transport semantics: explicit policy fails closed; all other
runtime failures fail open.

Only HTTP 422 with ``error_type=CognitivePolicyRefusal`` is an explicit live
runtime refusal: exactly one WITHHELD result, one abandon, no raw draft.
Validation/correction errors, connection reset, invalid JSON, timeout, and
5xx are non-policy failures: the ephemeral session is abandoned and the
original host-model answer stands verbatim with no sticky disposition that
later rewrites it. The same distinction holds on the causal path, where
wholly unavailable deliberation also fails open as substrate unavailability.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from pi_causal_helpers import (
    CAUSAL_PROMPT,
    GOOD_REPAIR,
    WEAK_DRAFT,
    causal_runtime_script,
    causal_workspace,
)
from pi_doom_loop_helpers import TOOL_TURN, workspace
from pi_recovery_helpers import (
    Servers,
    assistant_answers,
    require_pi,
    run_pi,
)
from pi_terminal_helpers import (
    AMBIGUITY_ANSWER,
    AMBIGUITY_PROMPT,
    EXTENSION,
    WITHHELD_MARKER,
    terminal_status_messages,
)

POLICY_422 = (
    422,
    {
        "error": "cognitive policy refusal",
        "error_type": "CognitivePolicyRefusal",
    },
)
CAPTURE_ENV = {"CORTHEON_BENCHMARK_CAPTURE_CANDIDATE": "1"}
STAGE_ENTRY_TYPE = "cortheon-benchmark-causal-stage-v1"
CANDIDATE_ENTRY_TYPE = "cortheon-benchmark-candidate-v1"


def _candidate_entries(completed) -> list[dict[str, Any]]:
    """Every benchmark candidate entry the run actually appended."""
    entries = []
    for line in completed.stdout.splitlines():
        if not line.strip().startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        entry = event.get("entry", {})
        if (
            event.get("type") == "entry_appended"
            and entry.get("customType") == CANDIDATE_ENTRY_TYPE
        ):
            entries.append(entry.get("data", {}))
    return entries


def _stage_reasons(completed) -> list[str]:
    """Every causal-stage benchmark entry the run actually appended."""
    reasons = []
    for line in completed.stdout.splitlines():
        if not line.strip().startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        entry = event.get("entry", {})
        if event.get("type") == "entry_appended" and entry.get("customType") == STAGE_ENTRY_TYPE:
            reasons.append(entry.get("data", {}).get("reason"))
    return reasons


def _completion_script(failure: Any, *, delay: float = 0):
    """A runtime whose /v1/complete fails in one transport-shaped way."""

    def script(path: str, _body: dict[str, Any]) -> Any:
        if path == "/healthz":
            return 200, {"status": "ok"}
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "amb-1",
                    "status": "observing",
                    "session": {"deliverable": "document_synthesis"},
                    "context": {"goal": AMBIGUITY_PROMPT},
                    "next_action": {"type": "reason"},
                },
            )
        if path == "/v1/complete":
            if delay:
                time.sleep(delay)
            return failure
        return 200, {"status": "ok"}

    return script


def _run(
    tmp_path: Path,
    failure: Any,
    *,
    delay: float = 0,
    extra_env: dict[str, str] | None = None,
):
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [{"text": AMBIGUITY_ANSWER}],
    }
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": _completion_script(failure, delay=delay),
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
            extra_env=extra_env,
        )
    return completed, model_state, runtime_state


def _run_read_only_runtime_correction(tmp_path: Path, error_type: str | None):
    """Replay a benign inspection whose finalization needs runtime correction."""

    prompt = "Inspect the local fact file and report what it contains. Do not modify files."
    answer = "The inspected file contains three labelled keys."
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [TOOL_TURN, {"text": answer}],
    }

    def script(path: str, _body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "inspect-1",
                    "status": "observing",
                    "session": {"deliverable": "code_understanding"},
                    "next_action": {
                        "type": "harness_tool",
                        "request": {
                            "request_id": "req-0",
                            "capability": "inspect",
                            "query": "Inspect the local fact file.",
                        },
                    },
                },
            )
        if path == "/v1/observe":
            return (
                200,
                {
                    "session_id": "inspect-1",
                    "status": "observing",
                    "accepted_evidence_ids": ["ev-1"],
                    "next_action": {"type": "finish"},
                },
            )
        if path == "/v1/complete":
            error = {"error": "the investigation needs another bounded step"}
            if error_type is not None:
                error["error_type"] = error_type
            return (
                422,
                error,
            )
        return 200, {"status": "ok"}

    runtime_state: dict[str, Any] = {"records": [], "script": script}
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            EXTENSION,
            prompt,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=60,
        )
    return completed, answer, runtime_state


@pytest.mark.parametrize("error_type", [None, "ValueError", "CognitiveRuntimeError"])
def test_read_only_runtime_correction_does_not_masquerade_as_policy_refusal(
    tmp_path: Path, error_type: str | None
) -> None:
    """A correctable inspection error must not erase a benign host answer."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, answer, runtime_state = _run_read_only_runtime_correction(tmp_path, error_type)
    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    assert answers and answers[-1] == answer, answers
    assert not any(text.startswith(WITHHELD_MARKER) for text in answers), answers
    assert terminal_status_messages(completed) == []
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 1, paths
    assert paths.count("/v1/abandon") == 1, paths


def test_enabled_pi_injects_stable_model_context_before_task_state(tmp_path: Path) -> None:
    """The model learns what Cortheon does before it sees changing task state."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, model_state, _runtime_state = _run(
        tmp_path / "run",
        (500, {"error": "runtime unavailable"}),
    )
    assert completed.returncode == 0, completed.stderr
    request = json.dumps(model_state["requests"][0].get("messages", []))
    marker = "[CORTHEON_MODEL_CONTEXT_V1]"
    assert marker in request, request
    assert "beyond its weights" in request, request
    assert "model answers" in request, request
    assert request.index(marker) < request.index("CORTHEON_ACTIVE"), request


def test_policy_422_fails_closed_with_one_withhold_and_abandon(
    tmp_path: Path,
) -> None:
    """A live explicit runtime policy refusal (HTTP 422) fails closed: one
    withheld result, one abandon, one host-visible terminal, and the raw
    draft never reaches the host. The draft WAS submitted and refused, so
    benchmark capture emits exactly one completion-stage candidate holding
    the exact answer sent to /v1/complete — otherwise the policy block is
    unmeasurable (unclassified) in the benchmark taxonomy."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, model_state, runtime_state = _run(
        tmp_path / "run", POLICY_422, extra_env=CAPTURE_ENV
    )
    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    # No raw draft after the policy refusal.
    assert AMBIGUITY_ANSWER not in answers, answers
    # The assistant replacement and custom entry serialize one identical
    # terminal; the custom entry is the evaluator-authenticated receipt.
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert sum(1 for text in answers if text.startswith(WITHHELD_MARKER)) == 1
    assert len(terminal_status_messages(completed)) == 1
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/abandon") == 1, paths
    assert paths[-1] == "/v1/abandon", paths
    assert len(model_state["requests"]) == 1, len(model_state["requests"])
    # The candidate contract: exactly one completion-stage entry carrying
    # the exact proposed answer that was submitted to /v1/complete.
    submits = [body for path, body in runtime_state["records"] if path == "/v1/complete"]
    assert len(submits) == 1, submits
    assert submits[0]["answer"] == AMBIGUITY_ANSWER, submits
    entries = _candidate_entries(completed)
    assert len(entries) == 1, entries
    assert entries[0]["stage"] == "completion", entries
    assert entries[0]["candidate"] == AMBIGUITY_ANSWER, entries


@pytest.mark.parametrize(
    "name,failure",
    [
        ("connection-reset", "connection-reset"),
        ("invalid-json", "invalid-json"),
        ("http-5xx", (500, {"error": "runtime unavailable"})),
    ],
)
def test_transport_failure_fails_open_verbatim(tmp_path: Path, name: str, failure: Any) -> None:
    """Reset, invalid JSON, and 5xx during /v1/complete are transport: the
    ephemeral session is abandoned and the original host-model answer is
    delivered verbatim with no withheld result and no stale disposition.
    The path failed open, so benchmark capture must emit ZERO candidates —
    nothing from this window may be graded later."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, model_state, runtime_state = _run(
        tmp_path / f"run-{name}", failure, extra_env=CAPTURE_ENV
    )
    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    assert answers and answers[-1] == AMBIGUITY_ANSWER, answers
    assert not any(text.startswith(WITHHELD_MARKER) for text in answers), answers
    assert terminal_status_messages(completed) == []
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/abandon") == 1, paths
    assert len(model_state["requests"]) == 1, len(model_state["requests"])
    assert _candidate_entries(completed) == []


def test_timeout_fails_open_verbatim(tmp_path: Path) -> None:
    """A /v1/complete response slower than the adapter timeout is transport:
    the host answer stands verbatim after one abandon."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, _model_state, runtime_state = _run(
        tmp_path / "run",
        (200, {"status": "complete", "answer": "certified"}),
        delay=2.0,
        extra_env={
            "CORTHEON_RUNTIME_TIMEOUT_MS": "400",
            "CORTHEON_BENCHMARK_CAPTURE_CANDIDATE": "1",
        },
    )
    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    assert answers and answers[-1] == AMBIGUITY_ANSWER, answers
    assert terminal_status_messages(completed) == []
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/abandon") == 1, paths
    assert _candidate_entries(completed) == []


def _causal_run(
    tmp_path: Path,
    complete_failure: Any,
    *,
    extra_env: dict[str, str] | None = None,
):
    base = causal_runtime_script(True)

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/complete":
            return complete_failure
        return base(path, body)

    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [{"text": WEAK_DRAFT}, {"text": GOOD_REPAIR}],
    }
    runtime_state: dict[str, Any] = {"records": [], "script": script}
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            EXTENSION,
            CAUSAL_PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=causal_workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=60,
            extra_env=extra_env,
        )
    return completed, runtime_state


def test_causal_policy_422_fails_closed(tmp_path: Path) -> None:
    """The causal completion path fails closed on an explicit 422 policy
    refusal: the validated synthesis is never delivered, the run ends with
    one withheld result and one abandon. Benchmark capture proves the
    truthful stage is runtime_withheld — emitted exactly once, never
    transport_failed."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, runtime_state = _causal_run(tmp_path / "run", POLICY_422, extra_env=CAPTURE_ENV)
    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert WEAK_DRAFT not in answers, answers
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 1, paths
    assert paths.count("/v1/abandon") == 1, paths
    assert paths[-1] == "/v1/abandon", paths
    reasons = _stage_reasons(completed)
    assert reasons.count("runtime_withheld") == 1, reasons
    assert "transport_failed" not in reasons, reasons
    # Exactly one causal_synthesis candidate carrying the exact validated
    # deliberated text that was submitted to /v1/complete.
    submits = [body for path, body in runtime_state["records"] if path == "/v1/complete"]
    assert len(submits) == 1, submits
    entries = _candidate_entries(completed)
    assert len(entries) == 1, entries
    assert entries[0]["stage"] == "causal_synthesis", entries
    assert entries[0]["candidate"] == submits[0]["answer"], entries


def test_causal_connection_reset_fails_open_verbatim(tmp_path: Path) -> None:
    """The causal completion path fails open on transport: the original
    host-model answer stands verbatim after one abandon, with no withheld
    result and no sticky disposition rewriting it."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, runtime_state = _causal_run(
        tmp_path / "run", "connection-reset", extra_env=CAPTURE_ENV
    )
    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    # Fail open: the original host-model answer (the first turn's draft)
    # stands verbatim.
    assert answers and answers[-1] == WEAK_DRAFT, answers
    assert not any(text.startswith(WITHHELD_MARKER) for text in answers), answers
    assert terminal_status_messages(completed) == []
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 1, paths
    assert paths.count("/v1/abandon") == 1, paths
    assert paths[-1] == "/v1/abandon", paths
    # A real transport reset is still reported truthfully: exactly one
    # transport_failed stage, never runtime_withheld.
    reasons = _stage_reasons(completed)
    assert reasons.count("transport_failed") == 1, reasons
    assert "runtime_withheld" not in reasons, reasons
    # Fail open: no candidate from this window may be graded later.
    assert _candidate_entries(completed) == []


def test_wholly_unavailable_deliberation_fails_open(tmp_path: Path) -> None:
    """Internal deliberation wholly unavailable before any candidate or
    usage exists (the model substrate rejects every deliberation call) is
    substrate unavailability: fail open to the original host answer after
    one abandon — no withheld result, no sticky disposition."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    base = causal_runtime_script(True)

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/complete":
            return 200, {"session_id": "causal-1", "status": "complete", "answer": "x"}
        return base(path, body)

    def model_script(request: dict[str, Any]) -> str | None:
        messages = request.get("messages", [])
        last = messages[-1] if messages else {}
        content = last.get("content")
        texts = (
            [content]
            if isinstance(content, str)
            else [block.get("text", "") for block in content or [] if isinstance(block, dict)]
            if isinstance(content, list)
            else []
        )
        # Deliberation prompts are the JSON task/evidence/draft payloads.
        if any(text.startswith('{"task"') for text in texts):
            return "connection-reset"
        return None

    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [{"text": WEAK_DRAFT}, {"text": GOOD_REPAIR}],
        "model_script": model_script,
    }
    runtime_state: dict[str, Any] = {"records": [], "script": script}
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            EXTENSION,
            CAUSAL_PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=causal_workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=60,
        )
    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    assert answers and answers[-1] == WEAK_DRAFT, answers
    assert not any(text.startswith(WITHHELD_MARKER) for text in answers), answers
    assert terminal_status_messages(completed) == []
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 0, paths
    assert paths.count("/v1/abandon") == 1, paths


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

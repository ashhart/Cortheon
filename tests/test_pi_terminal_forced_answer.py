"""Real Pi regressions for the evidence-sufficiency forced answer.

The round-24 live signature: a causal novel-synthesis treatment accepted
four observations, then kept persisting redundant discovery tool calls
(the model announced a confirmed theory and proposed yet another test)
with no pending runtime request, until the harness wall clock killed the
run — session started, never completed, never evidence-closed.

These tests reproduce that lifecycle domain-independently: a causal
document-synthesis session with sufficient accepted independent evidence
and no pending request must bound redundant discovery persistence and
force one host answer through the private deliberation and /v1/complete
certification path. Below the sufficiency threshold the same wandering
stays admitted (a genuinely new discriminating request remains possible),
removing the guard reopens the unbounded wandering, and the fail-open
controls are untouched.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from pi_doom_loop_helpers import TOOL_TURN, workspace
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
from pi_terminal_helpers import (
    CAUSAL_CERTIFIED,
    CAUSAL_PROMPT,
    EXTENSION,
    GOOD_SYNTHESIS,
    WITHHELD_MARKER,
    always_pending_request_script,
    evidence_insufficient_wandering_script,
    evidence_ready_wandering_script,
    mutated_source,
    repeated_evidence_wandering_script,
    single_batch_sufficient_script,
    terminal_status_messages,
)

from cortheon.benchmark_core.transport_outcomes import parse_transport_outcome

EVIDENCE_MARKER = "Cortheon has accepted sufficient independent evidence"
BUDGET_MARKER = "Cortheon reached its host tool budget"


def _run(
    tmp_path: Path,
    turns: list[dict[str, Any]],
    script: Any,
    *,
    extra_env: dict[str, str] | None = None,
):
    model_state: dict[str, Any] = {"requests": [], "turns": turns}
    runtime_state: dict[str, Any] = {"records": [], "script": script}
    with Servers(model_state, runtime_state) as servers:
        started = time.monotonic()
        completed = run_pi(
            EXTENSION,
            CAUSAL_PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=60,
            extra_env=extra_env,
        )
    return completed, model_state, runtime_state, time.monotonic() - started


def test_evidence_ready_wandering_forces_one_certified_answer(
    tmp_path: Path,
) -> None:
    """The exact live signature, bounded: two accepted observation batches,
    no pending request, then wandering tool turns. The third wandering batch
    is blocked by evidence sufficiency, the answer-only continuation forces
    one host answer, deliberation validates it, and exactly one /v1/complete
    certifies — never a wall-clock truncation."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    turns = [TOOL_TURN] * 4 + [{"text": GOOD_SYNTHESIS}]
    completed, model_state, runtime_state, elapsed = _run(
        tmp_path / "run",
        turns,
        evidence_ready_wandering_script(completes=True),
        extra_env={"CORTHEON_BENCHMARK_CAPTURE_CANDIDATE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    assert answers[-1] == CAUSAL_CERTIFIED, answers
    assert not any(text.startswith(WITHHELD_MARKER) for text in answers), answers
    events = parse_events(completed.stdout)
    blocked_text = [
        block.get("text", "")
        for event in blocked_executions(events)
        for block in event.get("result", {}).get("content", [])
        if isinstance(block, dict)
    ]
    assert any(EVIDENCE_MARKER in text for text in blocked_text), blocked_text
    # Bounded: two admitted wandering batches, then the block; no budget hit.
    assert len(host_executions(events)) <= 5
    assert len(model_state["requests"]) <= 6, len(model_state["requests"])
    assert continuation_requests(model_state) >= 1
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/observe") == 1, paths
    # Certification only through the runtime: exactly one complete, nothing
    # after it, no abandon — the session ended certified, not abandoned.
    assert paths.count("/v1/complete") == 1, paths
    assert paths[-1] == "/v1/complete", paths
    assert paths.count("/v1/abandon") == 0, paths
    assert elapsed < 30, elapsed


def test_evidence_ready_wandering_withheld_stays_bounded_and_classifiable(
    tmp_path: Path,
) -> None:
    """The same forced answer when the runtime withholds the completion:
    one /v1/complete submission, the exact validated candidate captured on
    the causal channel, the truthful runtime_withheld stage, and a single
    withheld terminal answer with the session abandoned."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    from pi_lifecycle_helpers import STAGE_ENTRY_TYPE
    from pi_terminal_helpers import (
        CANDIDATE_ENTRY_TYPE,
        custom_entry_data,
    )

    turns = [TOOL_TURN] * 4 + [{"text": GOOD_SYNTHESIS}] + [{"text": GOOD_SYNTHESIS}]
    completed, model_state, runtime_state, elapsed = _run(
        tmp_path / "run",
        turns,
        evidence_ready_wandering_script(completes=False),
        extra_env={"CORTHEON_BENCHMARK_CAPTURE_CANDIDATE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    withheld = [text for text in answers if text.startswith(WITHHELD_MARKER)]
    assert withheld and len(set(withheld)) == 1, answers
    assert answers[-1] == withheld[0], answers
    events = parse_events(completed.stdout)
    statuses = terminal_status_messages(completed)
    assert len(statuses) == 1, statuses
    outcome = parse_transport_outcome(events, host="pi").outcome
    assert outcome.terminal_status == "withheld"
    assert outcome.terminal_provenance == "pi_custom_terminal"
    # The final causal withhold is terminal: exactly one forced answer
    # continuation was ever scheduled and no later raw model answer ran,
    # so no unvalidated synthesis can appear after the withheld result.
    assert continuation_requests(model_state) == 1, model_state["requests"]
    assert GOOD_SYNTHESIS.split("\n")[0] not in answers
    candidates = custom_entry_data(completed, CANDIDATE_ENTRY_TYPE)
    assert len(candidates) == 1, candidates
    assert candidates[0]["stage"] == "causal_synthesis"
    reasons = [data.get("reason") for data in custom_entry_data(completed, STAGE_ENTRY_TYPE)]
    assert reasons == ["runtime_withheld"], reasons
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 1, paths
    assert paths.count("/v1/abandon") == 1, paths
    assert paths[-1] == "/v1/abandon", paths
    assert elapsed < 30, elapsed


def test_single_batch_two_distinct_sources_becomes_sufficient(
    tmp_path: Path,
) -> None:
    """One accepting observation batch carrying two unique identities from
    two distinct clean sources is sufficient by itself: the id-less start
    evidence counts for nothing, the single batch counts for everything,
    and the wandering is bounded into one certified forced answer."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    turns = [TOOL_TURN] * 4 + [{"text": GOOD_SYNTHESIS}]
    completed, model_state, runtime_state, _elapsed = _run(
        tmp_path / "run",
        turns,
        single_batch_sufficient_script(completes=True),
        extra_env={"CORTHEON_BENCHMARK_CAPTURE_CANDIDATE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    assert answers[-1] == CAUSAL_CERTIFIED, answers
    events = parse_events(completed.stdout)
    blocked_text = [
        block.get("text", "")
        for event in blocked_executions(events)
        for block in event.get("result", {}).get("content", [])
        if isinstance(block, dict)
    ]
    assert any(EVIDENCE_MARKER in text for text in blocked_text), blocked_text
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 1, paths
    assert paths.count("/v1/abandon") == 0, paths
    assert continuation_requests(model_state) == 1, model_state["requests"]


@pytest.mark.parametrize("mode", ["same_source", "repeated_identity", "poisoned"])
def test_two_batches_without_independent_sources_never_force(
    tmp_path: Path,
    mode: str,
) -> None:
    """Two accepting observation batches never reach sufficiency when they
    carry only one source record, a repeated identity, or quarantined/
    failed entries: discovery must stay admitted until the host tool
    budget ends the run with no forced answer and no completion."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    turns = [TOOL_TURN] * 6 + [{"text": GOOD_SYNTHESIS}]
    completed, model_state, runtime_state, _elapsed = _run(
        tmp_path / f"run-{mode}",
        turns,
        repeated_evidence_wandering_script(mode),
        extra_env={"CORTHEON_MAX_HOST_TOOL_CALLS": "8"},
    )
    assert completed.returncode == 0, completed.stderr
    # Budget exhaustion abandons the session and emits one host-visible
    # terminal withhold: zero automatic follow-ups, no raw candidate turn.
    statuses = terminal_status_messages(completed)
    assert len(statuses) == 1, statuses
    assert "host tool budget was exhausted" in str(statuses[0].get("content"))
    assert not any(GOOD_SYNTHESIS.split("\n")[0] in text for text in assistant_answers(completed))
    assert continuation_requests(model_state) == 0, model_state["requests"]
    events = parse_events(completed.stdout)
    blocked_text = [
        block.get("text", "")
        for event in blocked_executions(events)
        for block in event.get("result", {}).get("content", [])
        if isinstance(block, dict)
    ]
    assert not any(EVIDENCE_MARKER in text for text in blocked_text), blocked_text
    assert len(host_executions(events)) >= 6
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 0, paths


def test_pending_runtime_request_always_reopens_discovery(tmp_path: Path) -> None:
    """Two clean independent sources are accepted, but every runtime
    response carries a fresh pending evidence request: the sufficiency
    guard can never force the answer while a request is pending, and only
    the host tool budget bounds the wandering. Eight consecutive single-tool
    turns reach the configured eight-call budget exactly at the last
    admitted tool; agent_end — not a ninth tool — applies the boundary,
    abandons once, and the answer-only continuation's raw synthesis is
    replaced with one visible explicit withheld terminal."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    single_tool_turn = {"tool_calls": [("read", {"path": "facts/a.txt"})]}
    # Eight single-tool turns fill the budget; an empty ninth turn ends the
    # model's operation with no further tool and no answerable text, so it
    # is agent_end — not a ninth tool call — that applies the boundary. The
    # answer-only continuation's raw synthesis must then be intercepted.
    turns = [single_tool_turn] * 8 + [{}, {"text": GOOD_SYNTHESIS}]
    completed, model_state, runtime_state, _elapsed = _run(
        tmp_path / "run",
        turns,
        always_pending_request_script(),
        extra_env={"CORTHEON_MAX_HOST_TOOL_CALLS": "8"},
    )
    assert completed.returncode == 0, completed.stderr
    # No raw escape: the uncertified synthesis never reached the host.
    answers = assistant_answers(completed)
    assert GOOD_SYNTHESIS.split("\n")[0] not in answers, answers
    events = parse_events(completed.stdout)
    blocked_text = [
        block.get("text", "")
        for event in blocked_executions(events)
        for block in event.get("result", {}).get("content", [])
        if isinstance(block, dict)
    ]
    assert not any(EVIDENCE_MARKER in text for text in blocked_text), blocked_text
    # agent_end applied the boundary after the eighth admitted tool with the
    # unified budget already unreachable (the session is abandoned by the
    # budget terminal, so no follow-up may be scheduled): exactly eight host
    # executions, nine model requests (eight tool turns plus the empty turn
    # that ends the operation), and zero automatic follow-ups.
    assert len(host_executions(events)) == 8, events
    assert len(model_state["requests"]) == 9, len(model_state["requests"])
    assert continuation_requests(model_state) == 0, model_state["requests"]
    statuses = terminal_status_messages(completed)
    assert len(statuses) == 1, statuses
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/observe") == 8, paths
    assert paths.count("/v1/complete") == 0, paths
    # Abandoned exactly once, and no active state survives it.
    assert paths.count("/v1/abandon") == 1, paths
    assert paths[-1] == "/v1/abandon", paths


def test_below_sufficiency_wandering_stays_admitted_until_budget(
    tmp_path: Path,
) -> None:
    """One accepted observation batch is below the sufficiency threshold:
    wandering discovery must stay admitted (discriminating work remains
    possible) until the ordinary host tool budget ends it."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    turns = [TOOL_TURN] * 6 + [{"text": GOOD_SYNTHESIS}]
    completed, _model_state, runtime_state, _elapsed = _run(
        tmp_path / "run",
        turns,
        evidence_insufficient_wandering_script(),
        extra_env={"CORTHEON_MAX_HOST_TOOL_CALLS": "8"},
    )
    assert completed.returncode == 0, completed.stderr
    statuses = terminal_status_messages(completed)
    assert len(statuses) == 1, statuses
    assert "host tool budget was exhausted" in str(statuses[0].get("content"))
    events = parse_events(completed.stdout)
    assert len(host_executions(events)) >= 6
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 0, paths


def test_mutation_removing_evidence_guard_reopens_unbounded_wandering(
    tmp_path: Path,
) -> None:
    """Disabling the discovery-exhaustion predicate reinstates the live
    failure: wandering runs on to the host tool budget, no forced answer,
    and /v1/complete is never reached. The guard is load-bearing."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    facade = mutated_source(
        tmp_path / "mutation",
        {
            "budget": (
                "causalEvidenceSufficient(active) &&\n"
                "\t\t\tactive!.redundantDiscoveryCalls >= MAX_REDUNDANT_DISCOVERY_CALLS,",
                "false,",
            )
        },
    )
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [TOOL_TURN] * 6 + [{"text": GOOD_SYNTHESIS}],
    }
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": evidence_ready_wandering_script(completes=True),
    }
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            facade,
            CAUSAL_PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path / "run"),
            tmp_path=tmp_path / "run",
            timeout=60,
            extra_env={"CORTHEON_MAX_HOST_TOOL_CALLS": "8"},
        )
    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    assert CAUSAL_CERTIFIED not in answers, answers
    statuses = terminal_status_messages(completed)
    assert len(statuses) == 1, statuses
    assert "host tool budget was exhausted" in str(statuses[0].get("content"))
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 0, paths


def test_forced_answer_fail_open_controls_unchanged(tmp_path: Path) -> None:
    """The forced-answer path never overrides fail-open: an explicit disable
    and a transport failure at /v1/start both deliver the host model's
    ordinary synthesis text verbatim with no withheld result."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    for name, extra in (
        ("disable", {"CORTHEON_AUTO_ENABLE": "0"}),
        ("transport", {}),
    ):
        model_state: dict[str, Any] = {
            "requests": [],
            "turns": [{"text": GOOD_SYNTHESIS}],
        }
        runtime_state: dict[str, Any] = {"records": []}

        def script(path: str, _body: dict[str, Any], _name: str = name) -> Any:
            if path == "/healthz":
                return 200, {"status": "ok"}
            if path == "/v1/start":
                return "invalid-json" if _name == "transport" else (200, {})
            return 200, {"status": "ok"}

        runtime_state["script"] = script
        with Servers(model_state, runtime_state) as servers:
            completed = run_pi(
                EXTENSION,
                CAUSAL_PROMPT,
                model_port=servers.model.server_port,
                runtime_port=servers.runtime.server_port,
                workspace=workspace(tmp_path / f"{name}-run"),
                tmp_path=tmp_path / f"{name}-run",
                timeout=45,
                extra_env=extra,
            )
        assert completed.returncode == 0, (name, completed.stderr)
        answers = assistant_answers(completed)
        assert answers and answers[-1] == GOOD_SYNTHESIS, (name, answers)

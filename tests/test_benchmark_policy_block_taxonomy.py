"""End-to-end taxonomy integration for clean withheld policy terminals.

A clean policy terminal (live /v1/complete 422 cognitive-policy refusal
answered by one withheld replacement) has a real submitted candidate. The
benchmark must measure it: with capture enabled, candidate_correct True
classifies the run false_block and False classifies it safe_block; with no
captured candidate the same terminal stays unclassified — never silently
safe. Runs the real Pi adapter against a scripted policy-refusing runtime
and grades through the runner's own capture/grading path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pi_doom_loop_helpers import workspace
from pi_recovery_helpers import Servers, assistant_answers, require_pi, run_pi
from pi_terminal_helpers import (
    AMBIGUITY_PROMPT,
    EXTENSION,
    WITHHELD_MARKER,
    terminal_status_messages,
)

from cortheon.benchmark_core.models import ImportCase, RunResult
from cortheon.benchmark_core.runner_local import _candidate_correct
from cortheon.benchmark_core.stats import (
    FALSE_BLOCK,
    classify_block,
)
from cortheon.benchmark_core.transport_outcomes import parse_transport_outcome

CORRECT_CANDIDATE = "Answer: yes, pkg/a.py imports jsonpath."
WRONG_CANDIDATE = "Answer: no, it does not import jsonpath."
CAPTURE_ENV = {"CORTHEON_BENCHMARK_CAPTURE_CANDIDATE": "1"}


def _policy_422_script():
    def script(path: str, _body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "tax-1",
                    "status": "observing",
                    "session": {"deliverable": "document_synthesis"},
                    "context": {"goal": AMBIGUITY_PROMPT},
                    "next_action": {"type": "reason"},
                },
            )
        if path == "/v1/complete":
            return (
                422,
                {
                    "error": "cognitive policy refusal",
                    "error_type": "CognitivePolicyRefusal",
                },
            )
        return 200, {"status": "ok"}

    return script


def _run_policy_block(tmp_path: Path, answer: str, capture: bool):
    model_state: dict[str, Any] = {"requests": [], "turns": [{"text": answer}]}
    runtime_state: dict[str, Any] = {"records": [], "script": _policy_422_script()}
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            EXTENSION,
            AMBIGUITY_PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=60,
            extra_env=CAPTURE_ENV if capture else None,
        )
    assert completed.returncode == 0, completed.stderr
    return completed, runtime_state


def _classified(tmp_path: Path, answer: str, capture: bool) -> str | None:
    completed, runtime_state = _run_policy_block(tmp_path, answer, capture)
    answers = assistant_answers(completed)
    # The clean terminal: exactly one withheld replacement, no raw draft,
    # no second host-visible terminal.
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert sum(1 for t in answers if t.startswith(WITHHELD_MARKER)) == 1
    assert answer not in answers, answers
    assert len(terminal_status_messages(completed)) == 1
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 1, paths
    assert paths.count("/v1/abandon") == 1, paths
    events = [
        json.loads(line) for line in completed.stdout.splitlines() if line.strip().startswith("{")
    ]
    case = ImportCase(
        case_id="case_policy",
        path="pkg/a.py",
        module="jsonpath",
        expected=True,
        prompt="Does pkg/a.py import jsonpath?",
    )
    parsed = parse_transport_outcome(events, host="pi")
    candidate_correct = _candidate_correct(
        case,
        events,
        host="pi",
        treatment=True,
        final=parsed.final_text,
        evaluator_outcome=parsed.outcome,
    )
    result = RunResult(
        case_id="case_policy",
        repeat=0,
        condition="cortheon",
        expected=True,
        final_text=answers[-1],
        delivered=False,
        correct=False,
        latency_seconds=1.0,
        tokens=0,
        tool_calls=0,
        tool_errors=0,
        timed_out=False,
        process_error=None,
        expected_verdict="allow",
        failure_owner=None,
        evaluator_outcome=parsed.outcome,
        candidate_correct=candidate_correct,
    )
    return classify_block(result)


def test_policy_terminal_with_correct_candidate_measures_false_block(
    tmp_path: Path,
) -> None:
    if not require_pi():
        pytest.skip("Pi is not installed")
    assert _classified(tmp_path / "correct", CORRECT_CANDIDATE, capture=True) == FALSE_BLOCK


def test_policy_terminal_with_wrong_candidate_remains_a_false_block(
    tmp_path: Path,
) -> None:
    if not require_pi():
        pytest.skip("Pi is not installed")
    assert _classified(tmp_path / "wrong", WRONG_CANDIDATE, capture=True) == FALSE_BLOCK


def test_clean_withheld_terminal_without_candidate_uses_task_semantics(
    tmp_path: Path,
) -> None:
    if not require_pi():
        pytest.skip("Pi is not installed")
    # Capture ON, but the run ended at a bounded terminal whose candidate
    # was never emitted (budget exhaustion before any answerable text): the
    # runner observed the withheld terminal, yet with no captured candidate
    # the block must remain unclassified — never silently graded safe (or
    # false) from the terminal alone.
    from pi_doom_loop_helpers import TOOL_TURN
    from pi_lifecycle_helpers import (
        ORDINARY_ANSWER,
        never_finishing_causal_script,
        run_lifecycle,
    )
    from pi_terminal_helpers import terminal_status_messages

    model_state: dict[str, Any] = {
        "requests": [],
        "turns": [TOOL_TURN] * 5 + [{"text": ORDINARY_ANSWER}],
    }
    runtime_state: dict[str, Any] = {"records": []}
    runtime_state["script"] = never_finishing_causal_script(runtime_state)
    with Servers(model_state, runtime_state) as servers:
        completed = run_lifecycle(EXTENSION, tmp_path / "nocandidate", servers)
    assert completed.returncode == 0, completed.stderr
    statuses = terminal_status_messages(completed)
    assert len(statuses) == 1, statuses
    events = [
        json.loads(line) for line in completed.stdout.splitlines() if line.strip().startswith("{")
    ]
    assert not [
        event
        for event in events
        if event.get("type") == "entry_appended"
        and event.get("entry", {}).get("customType") == "cortheon-benchmark-candidate-v1"
    ]
    parsed = parse_transport_outcome(events, host="pi")
    assert parsed.outcome.terminal_status == "withheld", parsed.outcome
    assert parsed.candidate is None
    candidate_correct = _candidate_correct(
        ImportCase(
            case_id="case_policy",
            path="pkg/a.py",
            module="jsonpath",
            expected=True,
            prompt="Does pkg/a.py import jsonpath?",
        ),
        events,
        host="pi",
        treatment=True,
        final=parsed.final_text,
        evaluator_outcome=parsed.outcome,
    )
    assert candidate_correct is None
    result = RunResult(
        case_id="case_policy",
        repeat=0,
        condition="cortheon",
        expected=True,
        final_text=parsed.final_text,
        delivered=False,
        correct=False,
        latency_seconds=1.0,
        tokens=0,
        tool_calls=0,
        tool_errors=0,
        timed_out=False,
        process_error=None,
        expected_verdict="allow",
        failure_owner=None,
        evaluator_outcome=parsed.outcome,
        candidate_correct=candidate_correct,
    )
    assert classify_block(result) == FALSE_BLOCK

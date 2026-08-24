"""Shared run and summary fixtures for the proof-property test modules.

Deliberately outside the ``test_*`` namespace so pytest imports it as a
helper rather than collecting it as a test module.
"""

from __future__ import annotations

from typing import Literal

from cortheon.cognitive_benchmark import (
    WITHHELD_PREFIX,
    EvaluationOutcome,
    RunResult,
    _condition_summary,
    _paired_summary,
    _proof_gates,
)

# The terminal a treatment run ends on when completion was not certified.
# Only this makes an undelivered run a block rather than a delivery failure.
WITHHELD = (
    "[Cortheon withheld: completion was not certified]\n"
    "The Cortheon investigation ended without a certified answer because "
    "the evaluator observed an authenticated test terminal."
)
assert WITHHELD.startswith(WITHHELD_PREFIX)

SEED = 7


def _run(
    *,
    condition: str,
    correct: bool = False,
    delivered: bool = True,
    final_text: str = "an answer",
    case_id: str = "case",
    repeat: int = 0,
    timed_out: bool = False,
    process_error: str | None = None,
    artifact_correct: bool | None = None,
    candidate_correct: bool | None = None,
    expected_verdict: Literal["allow", "block"] | None = "allow",
    failure_owner: Literal["candidate", "external_infrastructure"] | None = None,
    sessions_completed: int = 0,
    sessions_evidence_closed: int = 0,
    telemetry: bool | None = None,
) -> RunResult:
    """One run. Treatment runs engage the substrate but finish no work by default."""

    treatment = condition == "cortheon"
    if timed_out or process_error is not None:
        outcome = EvaluationOutcome(
            "pi", "transport_error", "pi_assistant", "timeout" if timed_out else "process_error"
        )
    elif delivered:
        outcome = EvaluationOutcome("pi", "success", "pi_assistant", "stop")
    elif final_text == WITHHELD:
        outcome = EvaluationOutcome("pi", "withheld", "pi_custom_terminal", "withheld")
    else:
        outcome = EvaluationOutcome("pi", "missing", "none", None)
    return RunResult(
        case_id=case_id,
        repeat=repeat,
        condition=condition,
        expected=True,
        final_text=final_text,
        delivered=delivered,
        correct=correct,
        latency_seconds=1.0,
        tokens=10,
        tool_calls=1,
        tool_errors=0,
        timed_out=timed_out,
        process_error=process_error,
        expected_verdict=expected_verdict,
        failure_owner=(
            failure_owner
            if failure_owner is not None
            else "candidate"
            if timed_out or process_error is not None or (not delivered and final_text != WITHHELD)
            else None
        ),
        evaluator_outcome=outcome,
        inference_provider_id="Local",
        inference_model_id="small-model",
        execution_identity_valid=True,
        execution_identity_provenance="pi_message_end",
        execution_measurements_valid=True,
        cost_usd=0.0,
        observed_steps=1,
        policy_timeout_seconds=60.0,
        policy_max_steps=4,
        policy_max_tool_calls=16,
        policy_context_tokens=8192,
        policy_output_tokens=512,
        artifact_correct=artifact_correct,
        candidate_correct=candidate_correct,
        substrate_telemetry_valid=(
            (True if telemetry is None else telemetry) if treatment else None
        ),
        runtime_sessions_started=1 if treatment else 0,
        runtime_observations_accepted=1 if treatment else 0,
        runtime_sessions_completed=sessions_completed,
        runtime_sessions_evidence_closed=sessions_evidence_closed,
    )


def _gates(results: list[RunResult]) -> dict[str, bool]:
    return _proof_gates(
        _condition_summary(results, "baseline"),
        _condition_summary(results, "cortheon"),
        _paired_summary(results, seed=SEED),
        repository_unchanged=True,
    )


def _paired_runs(*, wins: int, losses: int, sessions_completed: int = 1) -> list[RunResult]:
    """``wins`` cases the treatment alone solved, then ``losses`` it alone missed."""

    results: list[RunResult] = []
    for index in range(wins + losses):
        treatment_correct = index < wins
        results.append(
            _run(
                condition="cortheon",
                correct=treatment_correct,
                case_id=f"case_{index}",
                sessions_completed=sessions_completed,
            )
        )
        results.append(
            _run(
                condition="baseline",
                correct=not treatment_correct,
                case_id=f"case_{index}",
            )
        )
    return results


def _matched_comparator(treatment: list[RunResult], *, correct: bool = False) -> list[RunResult]:
    """A comparator run for every treatment run, pairing on case and repeat."""

    return [
        _run(
            condition="baseline",
            correct=correct,
            case_id=item.case_id,
            repeat=item.repeat,
        )
        for item in treatment
    ]


def _comparison_side(**overrides) -> dict[str, object]:
    """One side of a frontier comparison summary, clean unless overridden."""

    side: dict[str, object] = {
        "eligible_runs": 8,
        "infrastructure_failures": 0,
        "eligible_accuracy": 0.875,
        "false_allow_rate": 0.0,
        "unclassified_blocks": 0,
        "delivery_failures": 0,
        "latency_seconds": {"mean": 1.0},
    }
    side.update(overrides)
    return side

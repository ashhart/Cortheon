"""Block taxonomy: false, safe, unclassified, and delivery failures.

A run is a block only when it ended on the pinned withheld terminal. Such a
block is false for an evaluator-sealed expected-allow task and safe for an
expected-block task. Candidate correctness stays a separate diagnostic. A run that
delivered nothing without that terminal -- a timeout, a dead process, an empty
transcript -- is a delivery failure and no kind of block.
"""

from typing import Literal

from block_taxonomy_support import scaling_report as _scaling_report

from cortheon.benchmark_core.audit import scaling_curve
from cortheon.benchmark_core.models import RunResult
from cortheon.benchmark_core.outcomes import EvaluationOutcome
from cortheon.benchmark_core.stats import (
    FALSE_BLOCK,
    SAFE_BLOCK,
    UNCLASSIFIED_BLOCK,
    _condition_summary,
    _frontier_comparison,
    _proof_gates,
    classify_block,
)
from cortheon.qualification_factory import _failure_type

WITHHELD = (
    "[Cortheon withheld: completion was not certified]\n"
    "The Cortheon investigation ended without a certified answer because "
    "the evaluator observed an authenticated test terminal."
)


def _result(
    *,
    delivered: bool,
    correct: bool,
    artifact_correct: bool | None = None,
    candidate_correct: bool | None = None,
    final_text: str = "answer",
    condition: str = "cortheon",
    timed_out: bool = False,
    process_error: str | None = None,
    expected_verdict: Literal["allow", "block"] | None = "allow",
    failure_owner: Literal["candidate", "external_infrastructure"] | None = None,
) -> RunResult:
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
        case_id="case",
        repeat=0,
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
        substrate_telemetry_valid=True if condition == "cortheon" else None,
        runtime_sessions_completed=1 if condition == "cortheon" and delivered else 0,
        artifact_correct=artifact_correct,
        candidate_correct=candidate_correct,
    )


def _block(**overrides) -> RunResult:
    """An observed block: undelivered, on the pinned withheld terminal."""

    return _result(delivered=False, correct=False, final_text=WITHHELD, **overrides)


def test_expected_allow_withhold_is_false_block_regardless_of_artifact():
    result = _block(artifact_correct=True)

    assert classify_block(result) == FALSE_BLOCK
    summary = _condition_summary([result], "cortheon")

    assert summary["false_blocks"] == 1
    assert summary["safe_blocks"] == 0
    assert summary["unclassified_blocks"] == 0
    assert summary["false_block_rate"] == 1.0
    assert summary["block_classification_coverage"] == 1.0
    assert _failure_type(result) == "false_block"


def test_expected_block_withhold_is_safe_block_regardless_of_artifact():
    result = _block(expected_verdict="block", artifact_correct=True)

    assert classify_block(result) == SAFE_BLOCK
    summary = _condition_summary([result], "cortheon")

    assert summary["false_blocks"] == 0
    assert summary["safe_blocks"] == 1
    assert summary["unclassified_blocks"] == 0
    assert summary["false_block_rate"] == 0.0
    assert summary["block_classification_coverage"] == 1.0
    # correct stays delivered-answer accuracy: a withheld marker is not a
    # correct answer even when the artifact grade is known.
    assert summary["correct"] == 0
    assert _failure_type(result) == "safe_block"


def test_candidate_grade_cannot_reclassify_task_semantics():
    graded_true = _block(candidate_correct=True)
    graded_false = _block(candidate_correct=False)

    assert classify_block(graded_true) == FALSE_BLOCK
    assert classify_block(graded_false) == FALSE_BLOCK


def test_withheld_read_only_block_is_unclassified():
    # The runner only saw the withheld notice; whether the pre-block
    # candidate was correct is unknown, so neither false nor safe is a
    # valid measurement.
    result = _result(
        delivered=False,
        correct=False,
        final_text=WITHHELD,
        expected_verdict=None,
    )

    assert classify_block(result) == UNCLASSIFIED_BLOCK
    summary = _condition_summary([result], "cortheon")

    assert summary["false_blocks"] == 0
    assert summary["safe_blocks"] == 0
    assert summary["unclassified_blocks"] == 1
    assert summary["unclassified_block_rate"] == 1.0
    assert summary["block_classification_coverage"] == 0.0
    assert summary["correct"] == 0
    assert _failure_type(result) == "unclassified_block"


def test_delivered_answer_on_expected_block_is_a_false_allow_not_a_block():
    result = _result(
        delivered=True,
        correct=False,
        artifact_correct=True,
        expected_verdict="block",
    )

    assert classify_block(result) is None
    summary = _condition_summary([result], "cortheon")

    assert summary["false_allows"] == 1
    assert summary["false_blocks"] == 0
    assert summary["unclassified_blocks"] == 0
    # Nothing was blocked, so there is no classification rate to report.
    assert summary["block_classification_coverage"] is None


def test_delivered_correct_answer_is_not_a_block():
    result = _result(delivered=True, correct=True)

    assert classify_block(result) is None
    summary = _condition_summary([result], "cortheon")

    assert summary["correct"] == 1
    assert summary["false_blocks"] == 0
    assert summary["unclassified_blocks"] == 0


def test_mixed_blocks_report_counts_rates_and_coverage():
    results = [
        _block(artifact_correct=True),
        _block(expected_verdict="block", artifact_correct=False),
        _block(expected_verdict=None),
        _result(delivered=True, correct=True),
    ]

    summary = _condition_summary(results, "cortheon")

    assert summary["false_blocks"] == 1
    assert summary["safe_blocks"] == 1
    assert summary["unclassified_blocks"] == 1
    assert summary["unclassified_block_rate"] == 0.25
    assert summary["block_classification_coverage"] == 2 / 3


def test_scaling_curve_aggregates_the_same_taxonomy():
    curve = scaling_curve(
        [
            _scaling_report(
                [
                    {"condition": "cortheon", "delivered": False, "artifact_correct": True},
                    {
                        "condition": "cortheon",
                        "delivered": False,
                        "artifact_correct": False,
                        "expected_verdict": "block",
                    },
                    {"condition": "cortheon", "delivered": True, "correct": True},
                    {"condition": "baseline", "delivered": False, "final_text": ""},
                ]
            )
        ]
    )

    assert curve["schema_version"] == 4
    cortheon = curve["points"][0]["conditions"]["cortheon"]
    assert cortheon["false_block_rate"] == 1 / 3
    assert cortheon["safe_blocks"] == 1
    assert cortheon["unclassified_blocks"] == 0
    assert cortheon["unclassified_block_rate"] == 0.0
    assert cortheon["block_classification_coverage"] == 1.0
    assert cortheon["delivery_failures"] == 0
    # The comparator produced no answer and no withheld terminal, so it is a
    # delivery failure rather than a block of any kind -- least of all a safe
    # one.
    baseline = curve["points"][0]["conditions"]["baseline"]
    assert baseline["delivery_failures"] == 1
    assert baseline["unclassified_blocks"] == 0
    assert baseline["safe_blocks"] == 0
    assert baseline["false_block_rate"] == 0.0
    assert baseline["block_classification_coverage"] is None


def _gate_dicts(cortheon_extra: dict) -> tuple[dict, dict, dict]:
    baseline = {
        "runs": 4,
        "accuracy": 0.0,
        "false_allow_rate": 0.0,
        "infrastructure_failures": 0,
        "prior_infrastructure_failures": 0,
        "execution_identity_invalid_runs": 0,
        "execution_measurement_invalid_runs": 0,
        "execution_policy_invalid_runs": 0,
        "execution_policy": [4, 16, 8192, 512],
        "failure_ownership_invalid_runs": 0,
        "expected_verdict_invalid_runs": 0,
    }
    cortheon = {
        "runs": 4,
        "accuracy": 1.0,
        "false_allow_rate": 0.0,
        "false_block_rate": 0.0,
        "unclassified_blocks": 0,
        "delivery_failures": 0,
        "candidate_delivery_failures": 0,
        "infrastructure_failures": 0,
        "prior_infrastructure_failures": 0,
        "execution_identity_invalid_runs": 0,
        "execution_measurement_invalid_runs": 0,
        "execution_policy_invalid_runs": 0,
        "execution_policy": [4, 16, 8192, 512],
        "failure_ownership_invalid_runs": 0,
        "expected_verdict_invalid_runs": 0,
        "substrate_telemetry_valid": True,
        "substrate_completed_work": True,
        "verified_completion_rate": 1.0,
    }
    cortheon.update(cortheon_extra)
    paired = {
        "pairs": 4,
        "invalid_pairs": 0,
        "independent_cases": 4,
        "qualified_independent_cases": 4,
        "cortheon_wins": 4,
        "cortheon_losses": 0,
        "mcnemar_exact_p": 0.03125,
    }
    return baseline, cortheon, paired


def test_proof_gates_pass_when_every_block_is_classified():
    baseline, cortheon, paired = _gate_dicts({})

    gates = _proof_gates(baseline, cortheon, paired, repository_unchanged=True)

    assert gates["all_cortheon_blocks_classified"]
    assert gates["bounded_cortheon_false_blocks"]
    assert gates["cortheon_runs_delivered_or_blocked"]
    assert gates["substrate_completed_work"]
    assert all(gates.values())


def test_unclassified_block_fails_the_bounded_false_block_gate_and_proof():
    baseline, cortheon, paired = _gate_dicts({"unclassified_blocks": 1})

    gates = _proof_gates(baseline, cortheon, paired, repository_unchanged=True)

    assert not gates["all_cortheon_blocks_classified"]
    # Even a zero false-block rate cannot pass while a block is
    # unclassified; the unclassified block must fail the overall claim.
    assert not gates["bounded_cortheon_false_blocks"]
    assert not all(gates.values())


def test_proof_gates_reject_summary_without_unclassified_field():
    # A summary dict without the ``unclassified_blocks`` key cannot prove
    # the count was zero, so the gates must fail closed rather than read
    # absence as zero.
    baseline, cortheon, paired = _gate_dicts({})
    del cortheon["unclassified_blocks"]

    try:
        _proof_gates(baseline, cortheon, paired, repository_unchanged=True)
    except KeyError:
        pass
    else:
        raise AssertionError("missing unclassified_blocks must not be accepted")


def test_proof_gates_require_clean_balanced_infrastructure():
    baseline = {
        "runs": 6,
        "accuracy": 0.0,
        "false_allow_rate": 0.0,
        "infrastructure_failures": 1,
    }
    cortheon = {
        "runs": 6,
        "accuracy": 1.0,
        "false_allow_rate": 0.0,
        "false_block_rate": 0.0,
        "unclassified_blocks": 0,
        "delivery_failures": 0,
        "infrastructure_failures": 0,
        "substrate_telemetry_valid": True,
        "substrate_completed_work": True,
        "verified_completion_rate": 1.0,
    }
    paired = {
        "pairs": 5,
        "invalid_pairs": 1,
        "independent_cases": 6,
        "qualified_independent_cases": 5,
        "cortheon_wins": 5,
        "cortheon_losses": 0,
        "mcnemar_exact_p": 0.03125,
    }

    gates = _proof_gates(
        baseline,
        cortheon,
        paired,
        repository_unchanged=True,
    )

    assert not gates["infrastructure_clean"]
    assert not gates["complete_balanced_pairs"]
    assert not all(gates.values())


def test_proof_gates_reject_missing_substrate_execution_telemetry():
    baseline = {
        "runs": 6,
        "accuracy": 0.0,
        "false_allow_rate": 1.0,
        "infrastructure_failures": 0,
    }
    cortheon = {
        "runs": 6,
        "accuracy": 1.0,
        "false_allow_rate": 0.0,
        "false_block_rate": 0.0,
        "unclassified_blocks": 0,
        "delivery_failures": 0,
        "infrastructure_failures": 0,
        "substrate_telemetry_valid": False,
        "substrate_completed_work": True,
        "verified_completion_rate": 1.0,
    }
    paired = {
        "pairs": 6,
        "invalid_pairs": 0,
        "independent_cases": 6,
        "qualified_independent_cases": 6,
        "cortheon_wins": 6,
        "cortheon_losses": 0,
        "mcnemar_exact_p": 0.03125,
    }

    gates = _proof_gates(
        baseline,
        cortheon,
        paired,
        repository_unchanged=True,
    )

    assert not gates["substrate_execution_observed"]
    assert not all(gates.values())


def _comparison_dicts(
    *,
    cortheon_unclassified: int = 0,
    frontier_unclassified: int = 0,
    cortheon_delivery_failures: int = 0,
    frontier_delivery_failures: int = 0,
) -> tuple[dict, dict]:
    cortheon = {
        "eligible_runs": 8,
        "infrastructure_failures": 0,
        "eligible_accuracy": 0.875,
        "false_allow_rate": 0.0,
        "unclassified_blocks": cortheon_unclassified,
        "delivery_failures": cortheon_delivery_failures,
        "latency_seconds": {"mean": 2.0},
    }
    frontier = {
        "eligible_runs": 8,
        "infrastructure_failures": 0,
        "eligible_accuracy": 0.875,
        "false_allow_rate": 0.0,
        "unclassified_blocks": frontier_unclassified,
        "delivery_failures": frontier_delivery_failures,
        "latency_seconds": {"mean": 1.0},
    }
    return cortheon, frontier


def test_frontier_parity_passes_only_when_all_blocks_are_classified():
    cortheon, frontier = _comparison_dicts()

    comparison = _frontier_comparison(cortheon, frontier)

    assert comparison["all_blocks_classified"]
    assert comparison["scoped_frontier_parity_observed"]


def test_cortheon_unclassified_block_fails_scoped_frontier_parity():
    cortheon, frontier = _comparison_dicts(cortheon_unclassified=1)

    comparison = _frontier_comparison(cortheon, frontier)

    assert not comparison["all_blocks_classified"]
    # An unclassified Cortheon block could be a hidden false block, so the
    # accuracy and false-allow comparisons alone cannot establish parity.
    assert not comparison["scoped_frontier_parity_observed"]


def test_frontier_unclassified_block_fails_scoped_frontier_parity():
    cortheon, frontier = _comparison_dicts(frontier_unclassified=1)

    comparison = _frontier_comparison(cortheon, frontier)

    assert not comparison["all_blocks_classified"]
    # Symmetrically, an unclassified frontier block could hide frontier
    # failures that flatter Cortheon's accuracy gap.
    assert not comparison["scoped_frontier_parity_observed"]

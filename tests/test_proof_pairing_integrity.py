"""A pair is only as good as its worst arm.

A comparison scores two observed outcomes. When one arm timed out or returned
nothing, its ``correct=False`` records the harness rather than the model, so
pairing it would convert the other arm's answer into a win, a significance,
and a balanced-pair count that no comparison produced. An explicit withheld
block is the opposite case: the run reached the pinned terminal with a
candidate in hand and declined, which is a real outcome and stays scorable.

Every refusal here is paired with the honest matrix of the same shape that
still proves its claim, so the rule cannot satisfy this file by invalidating
everything.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pytest
from proof_support import SEED, WITHHELD, _gates, _paired_runs, _run
from scaling_support import report as _sealed_scaling_report

from cortheon.cognitive_benchmark import (
    DELIVERY_FAILURE,
    FALSE_BLOCK,
    SAFE_BLOCK,
    EvaluationOutcome,
    RunResult,
    _paired_summary,
    classify_block,
    is_comparable_outcome,
    scaling_curve,
)

# The shapes an arm can end in, and whether that arm may be compared at all.
# Candidate-caused failures are incorrect outcomes; only external failures drop.
COMPARABILITY = [
    ({"delivered": True, "correct": True}, True),
    ({"delivered": False, "final_text": WITHHELD}, True),
    ({"delivered": False, "final_text": WITHHELD, "artifact_correct": False}, True),
    ({"delivered": False, "final_text": "", "timed_out": True}, True),
    ({"delivered": False, "final_text": WITHHELD, "timed_out": True}, True),
    ({"delivered": False, "final_text": "partial answer"}, True),
    ({"delivered": False, "final_text": "", "process_error": "host exited 137"}, True),
    (
        {
            "delivered": False,
            "final_text": "",
            "process_error": "evaluator endpoint unavailable",
            "failure_owner": "external_infrastructure",
        },
        False,
    ),
]


def _broken(condition: str, **overrides) -> dict[str, Any]:
    """An arm that ran out of wall clock with nothing to show for it."""

    return {
        "condition": condition,
        "delivered": False,
        "correct": False,
        "final_text": "",
        "timed_out": True,
        **overrides,
    }


def _matrix(
    treatment: dict[str, Any],
    comparator: dict[str, Any],
    *,
    cases: int = 6,
    start: int = 0,
) -> list[RunResult]:
    """``cases`` paired runs of one treatment shape against one comparator."""

    return [
        _run(case_id=f"case_{index}", **arm)
        for index in range(start, start + cases)
        for arm in (treatment, comparator)
    ]


def _answered(condition: str, *, correct: bool, **overrides) -> dict[str, Any]:
    values: dict[str, Any] = {"condition": condition, "correct": correct}
    if condition == "cortheon":
        values["sessions_completed"] = 1
    values.update(overrides)
    return values


# --- Candidate failures remain scheduled outcomes --------------------------


def test_a_candidate_timed_out_comparator_is_a_treatment_win():
    results = _matrix(
        _answered("cortheon", correct=True),
        _broken("baseline"),
    )

    paired = _paired_summary(results, seed=SEED)

    assert paired["total_pairs"] == 6
    assert paired["pairs"] == 6
    assert paired["invalid_pairs"] == 0
    assert paired["delivery_failure_pairs"] == 6
    assert paired["cortheon_wins"] == 6
    assert paired["accuracy_delta"] == 1.0
    assert paired["mcnemar_exact_p"] <= 0.05

    gates = _gates(results)

    assert gates["more_paired_wins_than_losses"] is True
    assert gates["statistically_significant"] is True
    assert gates["complete_balanced_pairs"] is True


def test_the_same_matrix_against_an_answering_comparator_still_proves_lift():
    # The control. Identical treatment arm; the comparator now delivers a
    # wrong answer instead of nothing, which is a real outcome, so the six
    # wins are real and the whole claim stands.
    results = _paired_runs(wins=6, losses=0)

    paired = _paired_summary(results, seed=SEED)
    gates = _gates(results)

    assert paired["pairs"] == 6
    assert paired["invalid_pairs"] == 0
    assert paired["delivery_failure_pairs"] == 0
    assert paired["cortheon_wins"] == 6
    assert paired["mcnemar_exact_p"] <= 0.05
    assert all(gates.values()), sorted(name for name, ok in gates.items() if not ok)


def test_six_repetitions_of_one_case_never_become_six_independent_wins():
    results = [
        _run(
            condition=condition,
            case_id="one_case",
            repeat=repeat,
            correct=condition == "cortheon",
            sessions_completed=1 if condition == "cortheon" else 0,
        )
        for repeat in range(6)
        for condition in ("cortheon", "baseline")
    ]

    paired = _paired_summary(results, seed=SEED)
    gates = _gates(results)

    assert paired["pairs"] == 6
    assert paired["independent_cases"] == 1
    assert paired["qualified_independent_cases"] == 1
    assert paired["cortheon_wins"] == 1
    assert paired["mcnemar_exact_p"] == 1.0
    assert gates["independent_case_floor"] is False
    assert gates["statistically_significant"] is False
    assert not all(gates.values())


def test_duplicate_pair_cells_fail_closed_without_order_dependent_overwrite():
    rows = [
        _run(condition="cortheon", case_id="a", repeat=0, correct=True),
        _run(condition="cortheon", case_id="a", repeat=0, correct=False),
        _run(condition="baseline", case_id="a", repeat=0, correct=False),
    ]

    forward = _paired_summary(rows, seed=SEED)
    reverse = _paired_summary(list(reversed(rows)), seed=SEED)

    assert forward == reverse
    assert forward["duplicate_cells"] == 1
    assert forward["pairs"] == 0
    assert forward["invalid_pairs"] == 1
    assert forward["qualified_independent_cases"] == 0


def test_a_silent_comparator_remains_a_scheduled_incorrect_outcome():
    # Silence is candidate-owned unless the evaluator separately attests an
    # external failure.
    results = _matrix(
        _answered("cortheon", correct=True),
        {"condition": "baseline", "delivered": False, "final_text": ""},
    )

    paired = _paired_summary(results, seed=SEED)

    assert paired["pairs"] == 6
    assert paired["invalid_pairs"] == 0
    assert paired["delivery_failure_pairs"] == 6
    assert paired["cortheon_wins"] == 6


# --- The rule is symmetric -------------------------------------------------


def test_a_candidate_timed_out_treatment_arm_is_a_loss():
    results = _matrix(
        _broken("cortheon", sessions_completed=1),
        _answered("baseline", correct=True),
    )

    paired = _paired_summary(results, seed=SEED)

    assert paired["pairs"] == 6
    assert paired["invalid_pairs"] == 0
    assert paired["delivery_failure_pairs"] == 6
    assert paired["cortheon_losses"] == 6

    gates = _gates(results)

    assert gates["complete_balanced_pairs"] is True
    assert gates["cortheon_runs_delivered_or_blocked"] is False


def test_a_treatment_that_answers_wrongly_still_records_its_losses():
    # The control for the symmetric direction: a treatment arm that answers
    # and is wrong loses the pairs it deserves to lose.
    results = _paired_runs(wins=0, losses=6)

    paired = _paired_summary(results, seed=SEED)

    assert paired["pairs"] == 6
    assert paired["invalid_pairs"] == 0
    assert paired["cortheon_losses"] == 6


# --- Restraint is an outcome ----------------------------------------------


@pytest.mark.parametrize(
    ("expected_verdict", "kind"),
    [("block", SAFE_BLOCK), ("allow", FALSE_BLOCK)],
)
def test_a_classified_withheld_block_is_a_valid_paired_outcome(expected_verdict, kind):
    blocked = _run(
        condition="cortheon",
        delivered=False,
        correct=False,
        final_text=WITHHELD,
        expected_verdict=expected_verdict,
        sessions_completed=1,
    )
    answered = _run(condition="baseline", correct=True)

    assert classify_block(blocked) == kind
    assert is_comparable_outcome(blocked) is True

    paired = _paired_summary([blocked, answered], seed=SEED)

    # The block is scored as the incorrect outcome it is: the treatment had a
    # candidate and declined to deliver it, and the comparator answered.
    assert paired["pairs"] == 1
    assert paired["invalid_pairs"] == 0
    assert paired["delivery_failure_pairs"] == 0
    assert paired["cortheon_losses"] == 1


def test_a_block_against_a_wrong_answer_ties_rather_than_dropping_out():
    blocked = _run(
        condition="cortheon",
        delivered=False,
        correct=False,
        final_text=WITHHELD,
        artifact_correct=False,
        sessions_completed=1,
    )
    wrong_answer = _run(condition="baseline", correct=False)

    # Neither arm delivered a correct answer, so the pair is a tie and stays
    # in the comparison; declining and answering wrongly both count.
    paired = _paired_summary([blocked, wrong_answer], seed=SEED)

    assert paired["pairs"] == 1
    assert paired["ties"] == 1
    assert paired["invalid_pairs"] == 0


@pytest.mark.parametrize(("overrides", "comparable"), COMPARABILITY)
def test_only_external_delivery_failures_are_incomparable(overrides, comparable):
    # Delivery taxonomy and failure ownership are orthogonal: candidate
    # failures stay comparable, external failures do not.
    result = _run(condition="cortheon", **overrides)

    assert is_comparable_outcome(result) is comparable
    if classify_block(result) == DELIVERY_FAILURE:
        assert comparable is (result.failure_owner == "candidate")


# --- One broken arm fails the matrix, not just its own pair ----------------


def test_one_candidate_broken_comparator_stays_in_the_balanced_matrix():
    # Six honest wins, strong enough to reach significance on their own, plus
    # a single pair whose comparator timed out. The surviving evidence is not
    # the claim: the matrix is no longer the complete balanced design the
    # report says it is.
    results = [
        *_paired_runs(wins=6, losses=0),
        *_matrix(
            _answered("cortheon", correct=True),
            _broken("baseline"),
            cases=1,
            start=6,
        ),
    ]

    paired = _paired_summary(results, seed=SEED)
    gates = _gates(results)

    assert paired["total_pairs"] == 7
    assert paired["pairs"] == 7
    assert paired["invalid_pairs"] == 0
    assert paired["delivery_failure_pairs"] == 1
    assert paired["cortheon_wins"] == 7
    assert paired["mcnemar_exact_p"] <= 0.05

    assert gates["statistically_significant"] is True
    assert gates["complete_balanced_pairs"] is True
    assert gates["zero_candidate_caused_cortheon_delivery_failures"] is True


@pytest.mark.parametrize("arm", ["baseline", "cortheon"])
def test_candidate_delivery_failure_stays_balanced_but_treatment_gate_fails(arm):
    treatment = _answered("cortheon", correct=True)
    comparator = _answered("baseline", correct=False)
    broken = _broken(arm, **({"sessions_completed": 1} if arm == "cortheon" else {}))
    results = _matrix(
        broken if arm == "cortheon" else treatment,
        broken if arm == "baseline" else comparator,
    )

    gates = _gates(results)
    assert gates["complete_balanced_pairs"] is True
    assert gates["zero_candidate_caused_cortheon_delivery_failures"] is (arm == "baseline")
    # Control: the same matrix with both arms answering is balanced.
    assert _gates(_matrix(treatment, comparator))["complete_balanced_pairs"] is True


# --- The stored artifact is read by the same rule --------------------------


def _stored_report(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    success = asdict(EvaluationOutcome("pi", "success", "pi_assistant", "stop"))
    timeout = asdict(EvaluationOutcome("pi", "transport_error", "pi_assistant", "timeout"))
    return _sealed_scaling_report(
        [
            {
                "case_id": entry["case_id"],
                "repeat": 0,
                "condition": entry["condition"],
                "delivered": entry.get("delivered", True),
                "correct": entry.get("correct", False),
                "final_text": entry.get("final_text", "an answer"),
                "timed_out": entry.get("timed_out", False),
                "process_error": None,
                "inference_model_id": "demo",
                "evaluator_outcome": timeout if entry.get("timed_out", False) else success,
                "artifact_correct": None,
                "candidate_correct": None,
                "substrate_telemetry_valid": entry["condition"] == "cortheon",
                "runtime_sessions_completed": 1 if entry["condition"] == "cortheon" else 0,
                "runtime_sessions_evidence_closed": 0,
                "latency_seconds": 1.0,
                "tool_calls": 0,
                "cost_usd": 0.0,
            }
            for entry in outcomes
        ]
    )


def _stored_pairs(comparator: dict[str, Any]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for index in range(2):
        case_id = f"case_{index}"
        pairs.append({"condition": "cortheon", "correct": True, "case_id": case_id})
        pairs.append({**comparator, "case_id": case_id})
    return pairs


def test_a_stored_candidate_timeout_banks_an_incorrect_paired_outcome():
    curve = scaling_curve(
        [
            _stored_report(
                _stored_pairs(
                    {
                        "condition": "baseline",
                        "delivered": False,
                        "correct": False,
                        "final_text": "",
                        "timed_out": True,
                    }
                )
            )
        ]
    )

    paired = curve["points"][0]["paired_vs_baseline"]

    assert paired["pairs"] == 2
    assert paired["wins"] == 2
    # The curve reports the failure and also keeps it in paired correctness.
    assert curve["points"][0]["conditions"]["baseline"]["delivery_failures"] == 2


def test_a_stored_wrong_answer_still_banks_its_paired_win_in_the_curve():
    curve = scaling_curve(
        [_stored_report(_stored_pairs({"condition": "baseline", "correct": False}))]
    )

    paired = curve["points"][0]["paired_vs_baseline"]

    assert paired["pairs"] == 2
    assert paired["wins"] == 2
    assert curve["points"][0]["conditions"]["baseline"]["delivery_failures"] == 0

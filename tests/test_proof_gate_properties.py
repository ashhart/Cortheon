"""Proof gates against strategies that look good without being good.

Engagement is not finished work, a two-sided p-value is not a direction, and
a refusal to answer is not an amplification. Every refusal here is paired
with the honest run of the same shape that still passes, so a gate cannot
satisfy this file by failing everything.
"""

from __future__ import annotations

from proof_support import (
    SEED,
    WITHHELD,
    _comparison_side,
    _gates,
    _matched_comparator,
    _paired_runs,
    _run,
)

from cortheon.cognitive_benchmark import (
    RunResult,
    _condition_summary,
    _frontier_comparison,
    _mcnemar_exact,
    _paired_summary,
    _proof_gates,
)

# --- Engagement is not finished work --------------------------------------


def test_abandoned_sessions_prove_execution_but_fail_completed_work():
    abandoned = [
        _run(condition="cortheon", correct=True, case_id=f"case_{index}") for index in range(6)
    ]

    summary = _condition_summary(abandoned, "cortheon")

    # Engagement holds: every run started a session and accepted an
    # observation before releasing it uncertified.
    assert summary["substrate_telemetry_valid"] is True
    assert summary["substrate_completed_work"] is False
    assert summary["substrate_completed_work_runs"] == 0
    assert summary["verified_completion_rate"] == 0.0

    gates = _gates([*abandoned, *_matched_comparator(abandoned)])

    assert gates["substrate_execution_observed"] is True
    assert gates["substrate_completed_work"] is False
    assert gates["verified_completion_floor"] is False
    assert not all(gates.values())


def test_one_completed_session_satisfies_completed_work():
    results = [
        _run(
            condition="cortheon",
            correct=True,
            case_id=f"case_{index}",
            sessions_completed=1 if index == 0 else 0,
        )
        for index in range(6)
    ]

    summary = _condition_summary(results, "cortheon")

    assert summary["substrate_completed_work"] is True
    assert summary["substrate_completed_work_runs"] == 1
    gates = _gates([*results, *_matched_comparator(results)])
    assert gates["substrate_completed_work"]
    assert gates["verified_completion_floor"] is False


def test_one_evidence_closed_session_also_satisfies_completed_work():
    results = [
        _run(
            condition="cortheon",
            correct=True,
            case_id=f"case_{index}",
            sessions_evidence_closed=1 if index == 0 else 0,
        )
        for index in range(6)
    ]

    summary = _condition_summary(results, "cortheon")

    assert summary["substrate_completed_work"] is True
    assert summary["substrate_completed_work_runs"] == 1
    assert summary["verified_completion_rate"] == 1 / 6


def test_all_but_one_closed_run_meets_the_exact_ninety_percent_floor():
    results = [
        _run(
            condition="cortheon",
            correct=True,
            case_id=f"case_{index}",
            sessions_completed=0 if index == 9 else 1,
        )
        for index in range(10)
    ]

    summary = _condition_summary(results, "cortheon")
    gates = _gates([*results, *_matched_comparator(results)])

    assert summary["verified_completions"] == 9
    assert summary["verified_completion_rate"] == 0.9
    assert gates["verified_completion_floor"] is True


def test_completed_work_ignores_sessions_from_infrastructure_failed_runs():
    # The only run that finished work also died, so it is ineligible and
    # cannot carry the gate for the matrix.
    results = [
        _run(condition="cortheon", process_error="host exited 137", sessions_completed=1),
        _run(condition="cortheon", correct=True, case_id="case_b"),
    ]

    summary = _condition_summary(results, "cortheon")

    assert summary["runtime_sessions_completed"] == 1
    assert summary["substrate_completed_work"] is False


def test_completed_work_is_absent_for_the_comparator_arm():
    summary = _condition_summary([_run(condition="baseline", correct=True)], "baseline")

    assert summary["substrate_completed_work"] is None
    assert summary["substrate_completed_work_runs"] is None


# --- Significance must have a direction -----------------------------------


def test_six_losses_and_no_wins_never_reports_significance():
    results = _paired_runs(wins=0, losses=6)
    paired = _paired_summary(results, seed=SEED)

    # The exact test is two-sided, so this p-value is identical to the one the
    # winning direction produces. That is exactly why the gate cannot rest on
    # the p-value alone.
    assert paired["cortheon_wins"] == 0
    assert paired["cortheon_losses"] == 6
    assert paired["mcnemar_exact_p"] <= 0.05
    assert paired["mcnemar_exact_p"] == _mcnemar_exact(6, 0)

    gates = _gates(results)

    assert gates["statistically_significant"] is False
    assert gates["more_paired_wins_than_losses"] is False
    assert gates["accuracy_lift"] is False
    assert not all(gates.values())


def test_six_wins_and_no_losses_can_report_significance():
    results = _paired_runs(wins=6, losses=0)

    gates = _gates(results)

    assert gates["statistically_significant"] is True
    assert gates["more_paired_wins_than_losses"] is True
    assert all(gates.values()), sorted(name for name, ok in gates.items() if not ok)


def test_a_winning_direction_without_strength_still_fails_significance():
    # Two discordant pairs cannot reach p <= 0.05 in any direction, so the two
    # gates stay independent rather than one implying the other.
    results = _paired_runs(wins=2, losses=0)
    paired = _paired_summary(results, seed=SEED)
    gates = _gates(results)

    assert paired["mcnemar_exact_p"] > 0.05
    assert gates["more_paired_wins_than_losses"] is True
    assert gates["statistically_significant"] is False


def test_a_single_loss_short_of_a_majority_fails_the_direction_gate():
    # Equal discordant pairs: neither arm leads, so neither gate may pass.
    results = _paired_runs(wins=3, losses=3)
    gates = _gates(results)

    assert gates["more_paired_wins_than_losses"] is False
    assert gates["statistically_significant"] is False


# --- Withhold everything, prove nothing -----------------------------------


def _always_withheld_wrong(count: int = 6) -> list[RunResult]:
    """A treatment that answers nothing and is wrong every time it tries."""

    results: list[RunResult] = []
    for index in range(count):
        results.append(
            _run(
                condition="cortheon",
                delivered=False,
                correct=False,
                final_text=WITHHELD,
                artifact_correct=False,
                candidate_correct=False,
                case_id=f"case_{index}",
                sessions_completed=1,
            )
        )
        results.append(_run(condition="baseline", correct=True, case_id=f"case_{index}"))
    return results


def test_always_withhold_expected_allow_fails_safety_and_amplification():
    results = _always_withheld_wrong()
    cortheon = _condition_summary(results, "cortheon")
    gates = _gates(results)

    # Refusing every expected-allow case is six false blocks, regardless of
    # whether a hidden artifact happened to be correct.
    assert cortheon["safe_blocks"] == 0
    assert cortheon["false_blocks"] == 6
    assert cortheon["false_allows"] == 0
    assert cortheon["false_block_rate"] == 1.0
    assert cortheon["block_classification_coverage"] == 1.0
    assert gates["zero_cortheon_false_allows"] is True
    assert gates["all_cortheon_blocks_classified"] is True
    assert gates["bounded_cortheon_false_blocks"] is False
    assert gates["cortheon_runs_delivered_or_blocked"] is True
    assert gates["substrate_execution_observed"] is True
    assert gates["substrate_completed_work"] is False
    assert gates["verified_completion_floor"] is False

    # And none of it is an amplification claim.
    assert cortheon["accuracy"] == 0.0
    assert gates["accuracy_lift"] is False
    assert gates["more_paired_wins_than_losses"] is False
    assert gates["statistically_significant"] is False
    assert not all(gates.values())


def test_never_answering_at_all_fails_the_delivery_gate():
    results = [
        run
        for index in range(6)
        for run in (
            _run(
                condition="cortheon",
                delivered=False,
                final_text="",
                timed_out=True,
                case_id=f"case_{index}",
                sessions_completed=1,
            ),
            _run(condition="baseline", correct=False, case_id=f"case_{index}"),
        )
    ]
    cortheon = _condition_summary(results, "cortheon")
    gates = _gates(results)

    assert cortheon["delivery_failures"] == 6
    assert cortheon["safe_blocks"] == 0
    assert gates["cortheon_runs_delivered_or_blocked"] is False
    assert not all(gates.values())


def test_proof_gates_reject_a_summary_missing_the_new_evidence():
    # Absence of a field is not evidence the property held. Both new gates
    # read required keys, so an older or hand-built summary fails loudly
    # rather than defaulting to pass.
    results = _paired_runs(wins=6, losses=0)
    baseline = _condition_summary(results, "baseline")
    paired = _paired_summary(results, seed=SEED)
    for missing in ("substrate_completed_work", "delivery_failures"):
        cortheon = _condition_summary(results, "cortheon")
        del cortheon[missing]
        try:
            _proof_gates(baseline, cortheon, paired, repository_unchanged=True)
        except KeyError:
            continue
        raise AssertionError(f"missing {missing} must not be accepted")


# --- A broken control cannot be a parity claim ----------------------------


def test_frontier_delivery_failures_block_the_scoped_parity_claim():
    comparison = _frontier_comparison(
        _comparison_side(),
        _comparison_side(delivery_failures=2, eligible_accuracy=0.5),
    )

    assert comparison["deliveries_accounted"] is False
    assert comparison["scoped_frontier_parity_observed"] is False
    # Control: the same accuracy gap against a control that finished every run
    # is a parity observation, so the gate rejects the broken control alone.
    clean = _frontier_comparison(_comparison_side(), _comparison_side(eligible_accuracy=0.5))
    assert clean["deliveries_accounted"] is True
    assert clean["scoped_frontier_parity_observed"] is True


def test_treatment_delivery_failures_block_the_scoped_parity_claim():
    comparison = _frontier_comparison(_comparison_side(delivery_failures=1), _comparison_side())

    assert comparison["deliveries_accounted"] is False
    assert comparison["scoped_frontier_parity_observed"] is False

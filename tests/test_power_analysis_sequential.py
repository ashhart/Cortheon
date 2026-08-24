from __future__ import annotations

import math
from dataclasses import replace

import pytest

from cortheon.power_analysis import ObservedContrast, ResourceAssumptions, build_power_plan
from cortheon.power_analysis.sequential import sequential_decision

PLAN = build_power_plan(
    ResourceAssumptions(120, 32, 0.03, 2, 0, 0, 0),
    strongest_reduced_condition_id="without_cross_source_derivation",
)


def _observed(alpha: float, conditional_power: float = 0.8):
    return {
        contrast.contrast_id: ObservedContrast(
            contrast_id=contrast.contrast_id,
            lower_bound=contrast.margin + 0.001,
            alpha_used=alpha,
            conditional_power=conditional_power,
            accounting_complete=True,
            safety_passed=True,
            attestation_sha256="a" * 64,
            producer="evaluator",
            candidate_supplied=False,
        )
        for contrast in PLAN.contrasts
    }


def test_only_the_registered_halfway_and_final_looks_exist() -> None:
    with pytest.raises(ValueError, match="unregistered"):
        sequential_decision(PLAN, PLAN.p6, PLAN.p6.interim_case_count - 1, _observed(0.0025))


def test_interim_success_requires_both_margins_and_all_gates() -> None:
    observed = _observed(PLAN.interim_alpha)
    result = sequential_decision(PLAN, PLAN.p6, PLAN.p6.interim_case_count, observed)
    assert result.decision == "stop_success"
    assert result.claim_eligible
    contrast_id = PLAN.contrasts[0].contrast_id
    failed = dict(observed)
    failed[contrast_id] = replace(failed[contrast_id], safety_passed=False)
    assert not sequential_decision(PLAN, PLAN.p6, PLAN.p6.interim_case_count, failed).claim_eligible


def test_nonbinding_futility_stops_without_a_claim() -> None:
    observed = _observed(PLAN.interim_alpha)
    contrast_id = PLAN.contrasts[0].contrast_id
    observed[contrast_id] = replace(
        observed[contrast_id],
        lower_bound=0.0,
        conditional_power=0.19,
    )
    result = sequential_decision(PLAN, PLAN.p6, PLAN.p6.interim_case_count, observed)
    assert result.decision == "stop_futility"
    assert not result.claim_eligible


def test_final_success_uses_the_reserved_final_alpha() -> None:
    success = sequential_decision(
        PLAN,
        PLAN.p7,
        PLAN.p7.overall_case_floor,
        _observed(PLAN.final_alpha),
    )
    assert success.decision == "final_success"
    wrong_alpha = _observed(PLAN.interim_alpha)
    failure = sequential_decision(PLAN, PLAN.p7, PLAN.p7.overall_case_floor, wrong_alpha)
    assert failure.decision == "final_failure"
    assert not failure.claim_eligible


@pytest.mark.parametrize(
    "mutation",
    [
        {"lower_bound": float("nan")},
        {"alpha_used": True},
        {"conditional_power": math.inf},
        {"accounting_complete": 1},
        {"candidate_supplied": True},
        {"attestation_sha256": "bad"},
    ],
)
def test_untyped_or_candidate_supplied_statistics_never_drive_a_claim(mutation: dict) -> None:
    observed = _observed(PLAN.interim_alpha)
    contrast_id = PLAN.contrasts[0].contrast_id
    observed[contrast_id] = replace(observed[contrast_id], **mutation)
    result = sequential_decision(PLAN, PLAN.p6, PLAN.p6.interim_case_count, observed)
    assert not result.claim_eligible
    assert any("attested_statistics_shape_invalid" in reason for reason in result.reasons)


def test_unregistered_campaign_copy_cannot_drive_a_claim() -> None:
    forged = replace(PLAN.p6, total_cells=PLAN.p6.total_cells + 1)
    with pytest.raises(ValueError, match="registered in the power plan"):
        sequential_decision(
            PLAN,
            forged,
            forged.interim_case_count,
            _observed(PLAN.interim_alpha),
        )


def test_development_pilot_plan_can_never_stop_with_a_claim() -> None:
    nonpromotional = replace(PLAN, promotion_eligible=False)
    result = sequential_decision(
        nonpromotional,
        nonpromotional.p6,
        nonpromotional.p6.interim_case_count,
        _observed(nonpromotional.interim_alpha),
    )
    assert not result.claim_eligible
    assert result.decision == "continue"
    assert "power_plan_is_not_promotion_eligible" in result.reasons

"""Registered 50% interim and final stopping decisions."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

from cortheon.power_analysis.models import (
    CampaignPlan,
    ObservedContrast,
    PowerPlan,
    SequentialResult,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def sequential_decision(
    power_plan: PowerPlan,
    campaign: CampaignPlan,
    completed_cases: int,
    observed: Mapping[str, ObservedContrast],
) -> SequentialResult:
    """Decide over evaluator-attested statistics; promotion must rederive rows."""

    if type(completed_cases) is not int:
        raise ValueError("completed_cases must be an integer")
    registered_campaign = power_plan.p6 if campaign.campaign_id == "p6" else power_plan.p7
    if campaign != registered_campaign:
        raise ValueError("campaign is not the campaign registered in the power plan")
    expected = {contrast.contrast_id: contrast for contrast in power_plan.contrasts}
    if set(observed) != set(expected):
        raise ValueError("both fixed contrasts are required")
    if completed_cases not in {campaign.interim_case_count, campaign.overall_case_floor}:
        raise ValueError("unregistered sequential look")
    interim = completed_cases == campaign.interim_case_count
    expected_alpha = power_plan.interim_alpha if interim else power_plan.final_alpha
    reasons: list[str] = []
    if not power_plan.promotion_eligible:
        reasons.append("power_plan_is_not_promotion_eligible")
    successes: list[bool] = []
    conditional_powers: list[float] = []
    for contrast_id, design in expected.items():
        result = observed[contrast_id]
        strict_shape = (
            _finite_number(result.lower_bound)
            and _finite_number(result.alpha_used)
            and _finite_number(result.conditional_power)
            and type(result.accounting_complete) is bool
            and type(result.safety_passed) is bool
            and result.producer == "evaluator"
            and result.candidate_supplied is False
            and isinstance(result.attestation_sha256, str)
            and _SHA256.fullmatch(result.attestation_sha256) is not None
        )
        if not strict_shape:
            reasons.append(f"{contrast_id}:attested_statistics_shape_invalid")
        if result.contrast_id != contrast_id or result.alpha_used != expected_alpha:
            reasons.append(f"{contrast_id}:identity_or_alpha_mismatch")
        if not 0 <= result.conditional_power <= 1:
            reasons.append(f"{contrast_id}:conditional_power_invalid")
        conditional_powers.append(result.conditional_power)
        success = (
            result.lower_bound >= design.margin
            and result.accounting_complete
            and result.safety_passed
            and result.alpha_used == expected_alpha
        )
        successes.append(success)
    if reasons:
        return SequentialResult(
            "final_failure" if not interim else "continue", False, tuple(reasons)
        )
    if all(successes):
        return SequentialResult("stop_success" if interim else "final_success", True, ())
    if interim and any(
        power < power_plan.futility_conditional_power for power in conditional_powers
    ):
        return SequentialResult("stop_futility", False, ("nonbinding_futility",))
    if interim:
        return SequentialResult("continue", False, ())
    return SequentialResult("final_failure", False, ("one_or_more_C12_margins_failed",))

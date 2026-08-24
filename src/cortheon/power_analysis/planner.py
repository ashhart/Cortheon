"""Fixed C12 power plan, staged campaign sizing, and resource accounting."""

from __future__ import annotations

import math

from cortheon.power_analysis.models import (
    DESIGN_VERSION,
    SCHEMA_VERSION,
    CampaignPlan,
    ContrastDesign,
    PilotArtifact,
    PowerPlan,
    ResourceAssumptions,
)
from cortheon.power_analysis.pilot import discordance_upper_bound, pilot_sha256
from cortheon.power_analysis.statistics import (
    arm_correlation,
    conservative_case_floor,
    worst_feasible_discordance,
)

FAMILYWISE_ALPHA = 0.05
CONTRAST_ALPHA = 0.025
INTERIM_ALPHA = 0.0025
FINAL_ALPHA = 0.0225
TARGET_JOINT_POWER = 0.80
TARGET_PER_CONTRAST_POWER = 0.90
FUTILITY_CONDITIONAL_POWER = 0.20
REPEATS = 3
CONDITIONS = 3
DIAGNOSTIC_CASES = 24
REDUCED_CONDITION_IDS = (
    "retrieval_only",
    "verification_only",
    "without_hypothesis_framing",
    "without_discriminating_evidence",
    "without_contradiction_revision",
    "without_cross_source_derivation",
    "without_adaptive_stopping",
)
_CONTRASTS = (
    ("full_vs_bare", "bare", 0.05, 0.08, 0.82),
    ("full_vs_strongest_reduced", "", 0.03, 0.05, 0.85),
)


def _contrast(
    definition: tuple[str, str, float, float, float],
    pilot: PilotArtifact | None,
) -> ContrastDesign:
    contrast_id, comparison, margin, alternative, baseline = definition
    if alternative <= margin:
        raise ValueError("alternative must exceed the C12 certification margin")
    null_discordance = worst_feasible_discordance(baseline, margin)
    worst = worst_feasible_discordance(baseline, alternative)
    discordance = worst
    calibration = "worst_feasible"
    calibration_sha256 = None
    if pilot is not None:
        if pilot.contrast_id != contrast_id:
            raise ValueError("pilot is bound to a different contrast")
        upper = discordance_upper_bound(pilot)
        if upper < alternative:
            raise ValueError("pilot discordance bound is infeasible under the alternative")
        discordance = min(worst, upper)
        calibration = "unsigned_development_pilot_ucb"
        calibration_sha256 = pilot_sha256(pilot)
    normal_floor, required, exact_power = conservative_case_floor(
        margin=margin,
        alternative=alternative,
        discordance=discordance,
        null_discordance=null_discordance,
        alpha=FINAL_ALPHA,
        power=TARGET_PER_CONTRAST_POWER,
        repeats=REPEATS,
        repeat_icc=1.0,
    )
    return ContrastDesign(
        contrast_id=contrast_id,
        comparison=comparison,
        margin=margin,
        alternative_effect=alternative,
        baseline_rate=baseline,
        null_discordance=null_discordance,
        alternative_discordance=discordance,
        arm_correlation=arm_correlation(baseline, alternative, discordance),
        repeats=REPEATS,
        repeat_icc=1.0,
        final_alpha=FINAL_ALPHA,
        target_power=TARGET_PER_CONTRAST_POWER,
        normal_floor=normal_floor,
        required_cases=required,
        exact_power=exact_power,
        calibration=calibration,
        promotion_eligible=pilot is None,
        pilot_sha256=calibration_sha256,
    )


def _campaign(
    campaign_id: str,
    strata: int,
    stratum_kind: str,
    case_floor: int,
    simultaneous_floor: int,
    resources: ResourceAssumptions,
) -> CampaignPlan:
    representation = math.ceil(case_floor / strata)
    overall = representation * strata
    interim_representation = math.ceil(representation / 2)
    interim = interim_representation * strata
    cells = overall * CONDITIONS * REPEATS
    interim_cells = interim * CONDITIONS * REPEATS
    frontier_calls = overall * resources.frontier_calls_per_case if campaign_id == "p7" else 0
    gpu_hours = cells * resources.gpu_hours_per_local_cell
    local_cost = gpu_hours * resources.local_compute_cost_per_gpu_hour
    total_seconds = (
        cells * resources.seconds_per_cell + frontier_calls * resources.frontier_seconds_per_call
    )
    frontier_cost = frontier_calls * resources.frontier_cost_per_call
    derived = (
        total_seconds / resources.parallel_cells / 3_600,
        gpu_hours,
        local_cost,
        frontier_cost,
        local_cost + frontier_cost,
    )
    if not all(math.isfinite(value) for value in derived):
        raise ValueError("resource assumptions produced nonfinite campaign estimates")
    return CampaignPlan(
        campaign_id=campaign_id,  # type: ignore[arg-type]
        strata=strata,
        stratum_kind=stratum_kind,  # type: ignore[arg-type]
        diagnostic_case_count=DIAGNOSTIC_CASES,
        overall_case_floor=overall,
        representation_floor_per_stratum=representation,
        independent_claim_floor_per_stratum=case_floor,
        simultaneous_claim_floor_per_stratum=simultaneous_floor,
        interim_case_count=interim,
        conditions=CONDITIONS,
        repeats=REPEATS,
        total_cells=cells,
        interim_cells=interim_cells,
        wall_time_hours=derived[0],
        gpu_hours=gpu_hours,
        local_compute_cost=local_cost,
        frontier_calls=frontier_calls,
        frontier_call_cost=frontier_cost,
        total_estimated_cost=derived[4],
    )


def build_power_plan(
    resources: ResourceAssumptions,
    *,
    strongest_reduced_condition_id: str,
    pilots: dict[str, PilotArtifact] | None = None,
) -> PowerPlan:
    """Build the fixed design; absent valid pilots use worst feasible discordance."""

    resources.validate()
    if strongest_reduced_condition_id not in REDUCED_CONDITION_IDS:
        raise ValueError("strongest reduced condition must be fixed before planning")
    pilots = pilots or {}
    expected_ids = {definition[0] for definition in _CONTRASTS}
    if not set(pilots) <= expected_ids:
        raise ValueError("pilot supplied for an unknown contrast")
    definitions = (
        _CONTRASTS[0],
        (
            _CONTRASTS[1][0],
            strongest_reduced_condition_id,
            *_CONTRASTS[1][2:],
        ),
    )
    contrasts = tuple(
        _contrast(definition, pilots.get(definition[0])) for definition in definitions
    )
    case_floor = max(contrast.required_cases for contrast in contrasts)
    simultaneous_floors: dict[int, int] = {}
    for strata in (8, 9):
        simultaneous_alpha = FAMILYWISE_ALPHA / (2 * strata) * 0.90
        simultaneous_power = 1 - (1 - TARGET_JOINT_POWER) / (2 * strata)
        simultaneous_floors[strata] = max(
            conservative_case_floor(
                margin=contrast.margin,
                alternative=contrast.alternative_effect,
                discordance=contrast.alternative_discordance,
                null_discordance=contrast.null_discordance,
                alpha=simultaneous_alpha,
                power=simultaneous_power,
                repeats=REPEATS,
                repeat_icc=1.0,
            )[1]
            for contrast in contrasts
        )
    return PowerPlan(
        schema_version=SCHEMA_VERSION,
        design_version=DESIGN_VERSION,
        familywise_alpha=FAMILYWISE_ALPHA,
        contrast_alpha=CONTRAST_ALPHA,
        interim_alpha=INTERIM_ALPHA,
        final_alpha=FINAL_ALPHA,
        target_joint_power=TARGET_JOINT_POWER,
        target_power_per_contrast=TARGET_PER_CONTRAST_POWER,
        futility_conditional_power=FUTILITY_CONDITIONAL_POWER,
        strongest_reduced_condition_id=strongest_reduced_condition_id,
        promotion_eligible=all(contrast.promotion_eligible for contrast in contrasts),
        contrasts=contrasts,
        p6=_campaign("p6", 9, "task_class", case_floor, simultaneous_floors[9], resources),
        p7=_campaign("p7", 8, "domain", case_floor, simultaneous_floors[8], resources),
        resources=resources,
        assumptions=(
            "Cases, not repeats, are independent sampling units.",
            "Three repeats are averaged within case with repeat ICC fixed to one for promotion sizing.",
            "Per-contrast power is 0.90 so joint power is at least 0.80 by the union bound.",
            "A 0.0025 interim and 0.0225 final alpha spend controls each contrast at 0.025.",
            "Twenty-four cases are diagnostic and never sufficient for a C12 promotion claim.",
            "Aggregate success does not establish any class- or domain-specific claim.",
            "Simultaneous all-strata claims receive correction across every contrast and stratum.",
            "Null size is calibrated at the maximum feasible null discordance nuisance.",
            "Pilot calibration requires independent sampling from the claim-pack source population.",
        ),
    )

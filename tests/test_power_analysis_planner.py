from __future__ import annotations

import json
import math

import pytest

from cortheon.power_analysis import ResourceAssumptions, build_power_plan, build_power_report
from cortheon.power_analysis.sensitivity import sensitivity_rows

RESOURCES = ResourceAssumptions(
    seconds_per_cell=120,
    parallel_cells=32,
    gpu_hours_per_local_cell=1 / 30,
    local_compute_cost_per_gpu_hour=2.0,
    frontier_calls_per_case=2,
    frontier_seconds_per_call=30,
    frontier_cost_per_call=0.50,
)
REDUCED = "without_cross_source_derivation"


def test_fixed_plan_has_joint_power_correction_and_no_24_case_claim() -> None:
    plan = build_power_plan(RESOURCES, strongest_reduced_condition_id=REDUCED)
    assert plan.familywise_alpha == 0.05
    assert plan.contrast_alpha == 0.025
    assert plan.interim_alpha + plan.final_alpha == pytest.approx(0.025)
    assert plan.target_power_per_contrast == 0.90
    assert 1 - 2 * (1 - plan.target_power_per_contrast) >= plan.target_joint_power
    assert [contrast.required_cases for contrast in plan.contrasts] == [13_038, 28_773]
    assert all(contrast.calibration == "worst_feasible" for contrast in plan.contrasts)
    assert plan.p6.diagnostic_case_count == 24 < plan.p6.overall_case_floor


def test_balanced_overall_and_independent_stratum_claim_floors_are_separate() -> None:
    plan = build_power_plan(RESOURCES, strongest_reduced_condition_id=REDUCED)
    assert (plan.p6.overall_case_floor, plan.p6.representation_floor_per_stratum) == (28_773, 3_197)
    assert (plan.p7.overall_case_floor, plan.p7.representation_floor_per_stratum) == (28_776, 3_597)
    assert plan.p6.independent_claim_floor_per_stratum == 28_773
    assert plan.p7.independent_claim_floor_per_stratum == 28_773
    assert plan.p6.simultaneous_claim_floor_per_stratum == 52_882
    assert plan.p7.simultaneous_claim_floor_per_stratum == 51_596
    assert plan.p6.interim_case_count == 14_391
    assert plan.p7.interim_case_count == 14_392


def test_costs_publish_cells_wall_time_gpu_hours_and_frontier_calls() -> None:
    plan = build_power_plan(RESOURCES, strongest_reduced_condition_id=REDUCED)
    assert plan.p6.total_cells == 258_957
    assert plan.p7.total_cells == 258_984
    assert plan.p6.wall_time_hours == 258_957 * 120 / 32 / 3_600
    assert plan.p6.gpu_hours == 258_957 / 30
    assert plan.p6.local_compute_cost == 258_957 / 30 * 2
    assert plan.p6.frontier_calls == 0
    assert plan.p7.frontier_calls == 57_552
    assert plan.p7.frontier_call_cost == 28_776
    assert plan.p7.wall_time_hours == (258_984 * 120 + 57_552 * 30) / 32 / 3_600
    assert plan.p7.total_estimated_cost == plan.p7.local_compute_cost + 28_776


def test_sensitivity_is_diagnostic_when_repeat_icc_is_below_one() -> None:
    plan = build_power_plan(RESOURCES, strongest_reduced_condition_id=REDUCED)
    contrast = plan.contrasts[0]
    rows = sensitivity_rows(
        contrast,
        baseline_rates=(0.82,),
        arm_correlations=(contrast.arm_correlation, 0.0),
    )
    assert rows
    assert {row.repeat_icc for row in rows} == {0.0, 0.5, 1.0}
    assert all(row.promotion_assumption_eligible == (row.repeat_icc == 1.0) for row in rows)
    grouped = sorted(row.approximate_cases for row in rows if row.arm_correlation == 0.0)
    assert grouped[0] < grouped[-1]


def test_report_keeps_aggregate_and_universal_claim_costs_distinct() -> None:
    plan = build_power_plan(RESOURCES, strongest_reduced_condition_id=REDUCED)
    payload = json.loads(build_power_report(plan))
    assert payload["diagnostic_24_can_promote"] is False
    assert payload["campaigns"]["p6"]["overall_case_floor"] == 28_773
    assert payload["universal_claim_costs"]["p6"]["cases"] == 475_938
    assert payload["universal_claim_costs"]["p6"]["cells"] == 4_283_442
    assert payload["universal_claim_costs"]["p7"]["cases"] == 412_768
    assert payload["universal_claim_costs"]["p7"]["cells"] == 3_714_912
    assert payload["universal_claim_costs"]["p7"]["frontier_calls"] == 825_536
    assert payload["sensitivity"]
    assert payload["resource_assumptions"] == {
        "seconds_per_cell": 120,
        "parallel_cells": 32,
        "gpu_hours_per_local_cell": 1 / 30,
        "local_compute_cost_per_gpu_hour": 2.0,
        "frontier_calls_per_case": 2,
        "frontier_seconds_per_call": 30,
        "frontier_cost_per_call": 0.5,
    }


def test_reduced_condition_is_fixed_before_the_plan_is_sealed() -> None:
    plan = build_power_plan(RESOURCES, strongest_reduced_condition_id=REDUCED)
    assert plan.strongest_reduced_condition_id == REDUCED
    assert plan.contrasts[1].comparison == REDUCED
    with pytest.raises(ValueError, match="fixed before planning"):
        build_power_plan(RESOURCES, strongest_reduced_condition_id="choose_after_results")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seconds_per_cell", True),
        ("seconds_per_cell", float("nan")),
        ("gpu_hours_per_local_cell", float("inf")),
        ("local_compute_cost_per_gpu_hour", -1),
        ("frontier_seconds_per_call", float("nan")),
        ("frontier_cost_per_call", True),
    ],
)
def test_resource_assumptions_reject_bool_nonfinite_and_negative(field: str, value: object) -> None:
    from dataclasses import replace

    with pytest.raises(ValueError):
        build_power_plan(
            replace(RESOURCES, **{field: value}),
            strongest_reduced_condition_id=REDUCED,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seconds_per_cell", 0.0001),
        ("seconds_per_cell", 86_401),
        ("parallel_cells", 1_000_001),
        ("gpu_hours_per_local_cell", 25),
        ("local_compute_cost_per_gpu_hour", 10_001),
        ("frontier_calls_per_case", 1_001),
        ("frontier_seconds_per_call", 86_401),
        ("frontier_cost_per_call", 100_001),
    ],
)
def test_resource_assumptions_reject_impractical_extremes(field: str, value: object) -> None:
    from dataclasses import replace

    with pytest.raises(ValueError, match=r"bounds|invalid"):
        build_power_plan(
            replace(RESOURCES, **{field: value}),
            strongest_reduced_condition_id=REDUCED,
        )


def test_all_derived_campaign_and_report_resource_values_are_finite() -> None:
    plan = build_power_plan(RESOURCES, strongest_reduced_condition_id=REDUCED)
    for campaign in (plan.p6, plan.p7):
        assert all(
            math.isfinite(value)
            for value in (
                campaign.wall_time_hours,
                campaign.gpu_hours,
                campaign.local_compute_cost,
                campaign.frontier_call_cost,
                campaign.total_estimated_cost,
            )
        )
    payload = json.loads(build_power_report(plan))
    for campaign in payload["universal_claim_costs"].values():
        assert all(math.isfinite(value) for value in campaign.values() if isinstance(value, float))

"""Deterministic power, cost, and claim-scope report."""

from __future__ import annotations

import json
import math
from dataclasses import asdict

from cortheon.power_analysis.models import PowerPlan, SensitivityRow
from cortheon.power_analysis.sealing import power_plan_sha256
from cortheon.power_analysis.sensitivity import default_sensitivity_rows


def _universal_cost(plan: PowerPlan, campaign_name: str) -> dict[str, float | int]:
    campaign = plan.p6 if campaign_name == "p6" else plan.p7
    cases = campaign.simultaneous_claim_floor_per_stratum * campaign.strata
    cells = cases * campaign.conditions * campaign.repeats
    gpu_hours = cells * plan.resources.gpu_hours_per_local_cell
    local_cost = gpu_hours * plan.resources.local_compute_cost_per_gpu_hour
    frontier_calls = cases * plan.resources.frontier_calls_per_case if campaign_name == "p7" else 0
    frontier_cost = frontier_calls * plan.resources.frontier_cost_per_call
    wall_seconds = (
        cells * plan.resources.seconds_per_cell
        + frontier_calls * plan.resources.frontier_seconds_per_call
    )
    result = {
        "cases": cases,
        "cells": cells,
        "wall_time_hours": wall_seconds / plan.resources.parallel_cells / 3_600,
        "gpu_hours": gpu_hours,
        "local_compute_cost": local_cost,
        "frontier_calls": frontier_calls,
        "frontier_call_cost": frontier_cost,
        "total_estimated_cost": local_cost + frontier_cost,
    }
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise ValueError("resource assumptions produced nonfinite universal estimates")
    return result


def build_power_report(
    plan: PowerPlan,
    sensitivity: tuple[SensitivityRow, ...] | None = None,
) -> bytes:
    if sensitivity is None:
        sensitivity = tuple(
            row for contrast in plan.contrasts for row in default_sensitivity_rows(contrast)
        )
    if not sensitivity:
        raise ValueError("power report requires baseline/correlation sensitivity")
    payload = {
        "schema_version": plan.schema_version,
        "design_version": plan.design_version,
        "power_plan_sha256": power_plan_sha256(plan),
        "claim_scope": "C12_margin_certification_power_plan",
        "diagnostic_24_can_promote": False,
        "joint_power_guarantee": plan.target_joint_power,
        "contrasts": [asdict(contrast) for contrast in plan.contrasts],
        "campaigns": {"p6": asdict(plan.p6), "p7": asdict(plan.p7)},
        "resource_assumptions": asdict(plan.resources),
        "universal_claim_costs": {
            "p6": _universal_cost(plan, "p6"),
            "p7": _universal_cost(plan, "p7"),
        },
        "sensitivity": [asdict(row) for row in sensitivity],
        "assumptions": list(plan.assumptions),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

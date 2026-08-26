"""Deterministic analysis of a retained operator-lift campaign.

Converts a released 540-cell record set into the P6 draft contrast inputs:
per-cluster effects for every operator ablation, the preregistered
strongest-reduced selection (the reduced condition with the largest realized
loss), and case-clustered one-sided lower bounds for the full-versus-bare and
full-versus-selected-reduced contrasts. Development input only: no bound here
is a claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cortheon.operator_lift.models import OPERATORS, LiftThresholds
from cortheon.operator_lift.statistics import clustered_lower_bound

CONDITIONS = (
    *("full", "equal_budget_placebo"),
    *(f"ablation_{index}" for index in range(len(OPERATORS))),
)


def _load_records(release_path: Path) -> list[dict[str, Any]]:
    data = json.loads(release_path.read_text(encoding="utf-8"))
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError("release records are missing")
    return records


def _cluster_rates(
    records: list[dict[str, Any]],
    condition: str,
) -> dict[int, tuple[int, int]]:
    rates: dict[int, list[bool]] = {}
    for record in records:
        if record.get("condition_id") != condition or not record.get("identity_valid"):
            continue
        rates.setdefault(int(record["case_ordinal"]), []).append(bool(record["correct"]))
    return {
        ordinal: (sum(values), len(values)) for ordinal, values in sorted(rates.items()) if values
    }


def _lower_bound(effects: list[float], thresholds: LiftThresholds) -> float | None:
    if not effects:
        return None
    resamples = thresholds.bootstrap_resamples
    bound = clustered_lower_bound(
        effects,
        alpha=thresholds.per_contrast_alpha,
        resamples=resamples,
        seed=thresholds.bootstrap_seed,
    )
    if bound is None:
        bound = min(effects) if thresholds.familywise_alpha else None
    return round(bound, 6) if bound is not None else None


def campaign_analysis(release_path: Path) -> dict[str, Any]:
    """Emit the closed P6 draft contrast analysis for a retained campaign."""
    records = _load_records(release_path)
    thresholds = LiftThresholds()
    valid = [record for record in records if record.get("identity_valid")]
    full_rates = _cluster_rates(valid, "full")
    placebo_rates = _cluster_rates(valid, "equal_budget_placebo")

    def rate_mean(rate_map: dict[int, tuple[int, int]]) -> float | None:
        if not rate_map:
            return None
        return sum(hits for hits, _ in rate_map.values()) / sum(
            total for _, total in rate_map.values()
        )

    full_rate = rate_mean(full_rates)
    placebo_rate = rate_mean(placebo_rates)
    per_operator: dict[str, dict[str, Any]] = {}
    for index, operator in enumerate(OPERATORS):
        ablation = f"ablation_{index}"
        removed_rates = _cluster_rates(valid, ablation)
        effects = [
            (full_rates[ordinal][0] / full_rates[ordinal][1]) - (hits / total)
            for ordinal, (hits, total) in removed_rates.items()
            if ordinal in full_rates and total and full_rates[ordinal][1]
        ]
        per_operator[operator] = {
            "removed_rate": rate_mean(removed_rates),
            "full_rate": full_rate,
            "cluster_effects": [round(effect, 6) for effect in sorted(effects)],
            "median_effect": round(sorted(effects)[len(effects) // 2], 6) if effects else None,
        }

    # Preregistered strongest-reduced rule: the reduced condition with the
    # largest realized loss, measured as the rate drop on its own cluster set.
    def rate_drop(data: dict[str, Any]) -> float:
        removed = data.get("removed_rate")
        return 0.0 if removed is None else float(removed) - float(full_rate or 0.0)

    strongest_operator = min(
        OPERATORS,
        key=lambda operator: (
            rate_drop(per_operator[operator]),
            -(per_operator[operator]["median_effect"] or 0.0),
        ),
    )
    strongest_loss = rate_drop(per_operator[strongest_operator])
    full_bare_effects = [
        (full_rates[ordinal][0] / full_rates[ordinal][1])
        - (placebo_rates[ordinal][0] / placebo_rates[ordinal][1])
        for ordinal in full_rates
        if ordinal in placebo_rates and placebo_rates[ordinal][1]
    ]

    reduced_effects = (
        per_operator[strongest_operator]["cluster_effects"] if strongest_operator else []
    )

    analysis: dict[str, Any] = {
        "schema_version": 1,
        "scope": "development_operator_lift_execution_only",
        "generated_from": str(release_path),
        "thresholds": {
            "full_vs_bare_lower_bound_points": 0.05,
            "full_vs_selected_reduced_lower_bound_points": 0.03,
        },
        "full_rate": full_rate,
        "placebo_rate": placebo_rate,
        "per_operator": per_operator,
        "strongest_reduced_operator": strongest_operator,
        "full_vs_bare_cluster_effects": [round(effect, 6) for effect in sorted(full_bare_effects)],
        "full_vs_bare_one_sided_lower_bound": _lower_bound(full_bare_effects, thresholds),
        "strongest_reduced_realized_loss": round(strongest_loss, 6),
        "full_vs_selected_reduced_cluster_effects": [
            round(effect, 6) for effect in sorted(reduced_effects)
        ],
        "full_vs_selected_reduced_one_sided_lower_bound": _lower_bound(reduced_effects, thresholds),
        "valid_cells": len(valid),
        "scheduled_cells": len(records),
    }
    core = {key: value for key, value in analysis.items() if key != "digest"}
    analysis["digest"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return analysis

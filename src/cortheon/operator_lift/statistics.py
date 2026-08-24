"""Deterministic case-clustered estimates for preregistered contrasts."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from typing import Any

from cortheon.operator_lift.models import OPERATORS, LiftThresholds, PairedCluster


def clustered_lower_bound(
    effects: Sequence[float],
    *,
    alpha: float,
    resamples: int,
    seed: int,
) -> float | None:
    """One-sided percentile bound resampling independent case clusters."""

    if len(effects) < 2:
        return None
    generator = random.Random(seed)
    size = len(effects)
    means = [
        sum(effects[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(resamples)
    ]
    means.sort()
    index = max(0, min(len(means) - 1, math.floor(alpha * len(means))))
    return means[index]


def _operator_seed(base: int, operator: str) -> int:
    suffix = int.from_bytes(hashlib.sha256(operator.encode()).digest()[:8], "big")
    return base ^ suffix


def summarize_operator(
    clusters: Sequence[PairedCluster],
    operator: str,
    thresholds: LiftThresholds,
) -> dict[str, Any]:
    selected = [cluster for cluster in clusters if cluster.operator == operator]
    effects = [cluster.effect for cluster in selected]
    full_rate = sum(cluster.full_rate for cluster in selected) / len(selected) if selected else None
    ablation_rate = (
        sum(cluster.ablation_rate for cluster in selected) / len(selected) if selected else None
    )
    lift = None if full_rate is None or ablation_rate is None else full_rate - ablation_rate
    lower = clustered_lower_bound(
        effects,
        alpha=thresholds.per_operator_alpha,
        resamples=thresholds.bootstrap_resamples,
        seed=_operator_seed(thresholds.bootstrap_seed, operator),
    )
    gates = {
        "minimum_clusters": len(selected) >= thresholds.minimum_clusters,
        "minimum_full_rate": full_rate is not None and full_rate >= thresholds.minimum_full_rate,
        "minimum_material_lift": lift is not None and lift >= thresholds.minimum_lift,
        "familywise_lower_bound_above_zero": lower is not None and lower > 0,
    }
    return {
        "operator": operator,
        "independent_clusters": len(selected),
        "repetitions_per_arm": thresholds.repetitions,
        "full_rate": full_rate,
        "ablation_rate": ablation_rate,
        "paired_lift": lift,
        "one_sided_confidence": 1 - thresholds.per_operator_alpha,
        "clustered_lower_bound": lower,
        "negative_effect_clusters": sum(effect < 0 for effect in effects),
        "zero_effect_clusters": sum(effect == 0 for effect in effects),
        "cluster_effects": effects,
        "gates": gates,
        "passes_capability_floor": all(gates.values()),
    }


def summarize_placebo(
    clusters: Sequence[PairedCluster],
    thresholds: LiftThresholds,
) -> dict[str, Any]:
    selected = [cluster for cluster in clusters if cluster.placebo_scores]
    effects = [cluster.placebo_effect for cluster in selected]
    full_rate = sum(cluster.full_rate for cluster in selected) / len(selected) if selected else None
    placebo_rate = (
        sum(cluster.placebo_rate for cluster in selected) / len(selected) if selected else None
    )
    lift = None if full_rate is None or placebo_rate is None else full_rate - placebo_rate
    lower = clustered_lower_bound(
        effects,
        alpha=thresholds.per_operator_alpha,
        resamples=thresholds.bootstrap_resamples,
        seed=_operator_seed(thresholds.bootstrap_seed, "equal_budget_placebo"),
    )
    gates = {
        "minimum_clusters": len(selected) >= thresholds.minimum_clusters * len(OPERATORS),
        "minimum_full_rate": full_rate is not None and full_rate >= thresholds.minimum_full_rate,
        "minimum_material_lift": lift is not None and lift >= thresholds.minimum_lift,
        "familywise_lower_bound_above_zero": lower is not None and lower > 0,
    }
    return {
        "comparison": "full_vs_equal_budget_placebo",
        "independent_clusters": len(selected),
        "repetitions_per_arm": thresholds.repetitions,
        "full_rate": full_rate,
        "placebo_rate": placebo_rate,
        "paired_lift": lift,
        "one_sided_confidence": 1 - thresholds.per_operator_alpha,
        "clustered_lower_bound": lower,
        "negative_effect_clusters": sum(effect < 0 for effect in effects),
        "zero_effect_clusters": sum(effect == 0 for effect in effects),
        "cluster_effects": effects,
        "gates": gates,
        "passes": all(gates.values()),
    }

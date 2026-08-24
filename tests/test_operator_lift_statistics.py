from __future__ import annotations

from cortheon.operator_lift.models import LiftThresholds, PairedCluster
from cortheon.operator_lift.statistics import clustered_lower_bound, summarize_operator

OPERATOR = "hypothesis_framing"


def _clusters(full: tuple[int, ...], ablation: tuple[int, ...], count: int = 12):
    return tuple(
        PairedCluster(f"cluster_{index:02d}", OPERATOR, full, ablation, (0, 0, 0))
        for index in range(count)
    )


def test_clear_case_clustered_lift_passes_preregistered_floors() -> None:
    summary = summarize_operator(_clusters((1, 1, 1), (0, 0, 0)), OPERATOR, LiftThresholds())
    assert summary["independent_clusters"] == 12
    assert summary["full_rate"] == 1
    assert summary["ablation_rate"] == 0
    assert summary["paired_lift"] == 1
    assert summary["clustered_lower_bound"] == 1
    assert summary["passes_capability_floor"] is True


def test_repeated_success_without_lift_cannot_pass() -> None:
    summary = summarize_operator(_clusters((1, 1, 1), (1, 1, 1)), OPERATOR, LiftThresholds())
    assert summary["paired_lift"] == 0
    assert summary["clustered_lower_bound"] == 0
    assert summary["passes_capability_floor"] is False


def test_too_few_independent_clusters_cannot_be_rescued_by_repetitions() -> None:
    summary = summarize_operator(
        _clusters((1,) * 30, (0,) * 30, count=2),
        OPERATOR,
        LiftThresholds(),
    )
    assert summary["repetitions_per_arm"] == 3
    assert summary["gates"]["minimum_clusters"] is False
    assert summary["passes_capability_floor"] is False


def test_clustered_bound_is_deterministic_and_one_sided() -> None:
    effects = [1.0] * 9 + [0.0] * 3
    first = clustered_lower_bound(effects, alpha=0.01, resamples=10_000, seed=42)
    second = clustered_lower_bound(effects, alpha=0.01, resamples=10_000, seed=42)
    assert first == second
    assert first is not None and 0 <= first <= sum(effects) / len(effects)

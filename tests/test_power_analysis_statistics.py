from __future__ import annotations

import pytest

from cortheon.power_analysis.statistics import (
    binomial_survival,
    conservative_case_floor,
    discordance_from_correlation,
    exact_null_size,
    normal_case_floor,
    repeat_variance,
    worst_feasible_discordance,
)


def test_margin_certification_rejects_detection_only_design() -> None:
    with pytest.raises(ValueError, match="exceed the certification margin"):
        normal_case_floor(
            margin=0.05,
            alternative=0.05,
            discordance=0.20,
            alpha=0.0225,
            power=0.90,
        )


def test_three_repeats_never_become_three_independent_cases() -> None:
    one = repeat_variance(0.25, 0.05, 1, 1.0)
    repeated = repeat_variance(0.25, 0.05, 3, 1.0)
    assert repeated == one
    optimistic = repeat_variance(0.25, 0.05, 3, 0.0)
    assert optimistic == pytest.approx(one / 3)


def test_baseline_correlation_and_discordance_are_jointly_feasible() -> None:
    maximum = worst_feasible_discordance(0.82, 0.08)
    assert maximum == pytest.approx(0.28)
    assert discordance_from_correlation(0.82, 0.08, -0.15617376188860638) == pytest.approx(maximum)
    with pytest.raises(ValueError, match="infeasible"):
        discordance_from_correlation(0.82, 0.08, -0.9)


def test_dependency_free_binomial_tail_matches_hand_values() -> None:
    assert binomial_survival(1, 3, 0.5) == pytest.approx(0.5, abs=1e-12)
    assert binomial_survival(2, 3, 0.5) == pytest.approx(0.125, abs=1e-12)


def test_fixed_conservative_floors_clear_exact_power() -> None:
    bare = conservative_case_floor(
        margin=0.05,
        alternative=0.08,
        discordance=0.28,
        alpha=0.0225,
        power=0.90,
    )
    reduced = conservative_case_floor(
        margin=0.03,
        alternative=0.05,
        discordance=0.25,
        alpha=0.0225,
        power=0.90,
    )
    assert bare[:2] == (13_035, 13_038)
    assert bare[2] >= 0.90
    assert reduced[:2] == (28_769, 28_773)
    assert reduced[2] >= 0.90
    with pytest.raises(ValueError, match="repeat_icc=1"):
        conservative_case_floor(
            margin=0.05,
            alternative=0.08,
            discordance=0.28,
            alpha=0.0225,
            power=0.90,
            repeat_icc=0.5,
        )


def test_nuisance_free_margin_test_controls_size_over_feasible_null_discordance() -> None:
    for discordance in (0.05, 0.10, 0.20, 0.31):
        size = exact_null_size(
            2_000,
            margin=0.05,
            calibrated_null_discordance=0.31,
            actual_null_discordance=discordance,
            alpha=0.0225,
        )
        assert size <= 0.0225

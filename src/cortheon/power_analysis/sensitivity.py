"""Diagnostic baseline, arm-correlation, and repeat-ICC sensitivity."""

from __future__ import annotations

from collections.abc import Iterable

from cortheon.power_analysis.models import ContrastDesign, SensitivityRow
from cortheon.power_analysis.statistics import (
    arm_correlation,
    discordance_from_correlation,
    normal_case_floor,
    worst_feasible_discordance,
)


def sensitivity_rows(
    contrast: ContrastDesign,
    *,
    baseline_rates: Iterable[float],
    arm_correlations: Iterable[float],
    repeat_iccs: Iterable[float] = (0.0, 0.5, 1.0),
) -> tuple[SensitivityRow, ...]:
    """Return feasible diagnostic rows; only repeat ICC one can size promotion."""

    rows: list[SensitivityRow] = []
    for baseline in baseline_rates:
        for correlation in arm_correlations:
            try:
                discordance = discordance_from_correlation(
                    baseline,
                    contrast.alternative_effect,
                    correlation,
                )
            except ValueError:
                continue
            for repeat_icc in repeat_iccs:
                approximate = normal_case_floor(
                    margin=contrast.margin,
                    alternative=contrast.alternative_effect,
                    discordance=discordance,
                    null_discordance=worst_feasible_discordance(
                        baseline,
                        contrast.margin,
                    ),
                    alpha=contrast.final_alpha,
                    power=contrast.target_power,
                    repeats=contrast.repeats,
                    repeat_icc=repeat_icc,
                )
                rows.append(
                    SensitivityRow(
                        contrast_id=contrast.contrast_id,
                        baseline_rate=baseline,
                        arm_correlation=correlation,
                        discordance=discordance,
                        repeat_icc=repeat_icc,
                        approximate_cases=approximate,
                        promotion_assumption_eligible=repeat_icc == 1.0,
                    )
                )
    return tuple(rows)


def default_sensitivity_rows(contrast: ContrastDesign) -> tuple[SensitivityRow, ...]:
    """Cover low, middle, and worst discordance over three baselines."""

    baselines = tuple(dict.fromkeys((0.50, 0.70, contrast.baseline_rate)))
    rows: list[SensitivityRow] = []
    for baseline in baselines:
        maximum = worst_feasible_discordance(baseline, contrast.alternative_effect)
        discordances = (
            contrast.alternative_effect,
            (contrast.alternative_effect + maximum) / 2,
            maximum,
        )
        correlations = tuple(
            arm_correlation(baseline, contrast.alternative_effect, discordance)
            for discordance in discordances
        )
        rows.extend(
            sensitivity_rows(
                contrast,
                baseline_rates=(baseline,),
                arm_correlations=correlations,
            )
        )
    return tuple(rows)

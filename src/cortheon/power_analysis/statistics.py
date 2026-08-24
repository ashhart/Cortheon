"""Dependency-free paired-binomial power and sensitivity calculations."""

from __future__ import annotations

import math
from functools import lru_cache
from statistics import NormalDist


def worst_feasible_discordance(baseline: float, effect: float) -> float:
    full = baseline + effect
    if not 0 < baseline < 1 or not 0 < full < 1:
        raise ValueError("baseline plus effect must be inside (0, 1)")
    return min(2 * baseline + effect, 2 * (1 - baseline) - effect)


def arm_correlation(baseline: float, effect: float, discordance: float) -> float:
    full = baseline + effect
    scale = math.sqrt(full * (1 - full) * baseline * (1 - baseline))
    return (full + baseline - 2 * full * baseline - discordance) / (2 * scale)


def discordance_from_correlation(baseline: float, effect: float, correlation: float) -> float:
    full = baseline + effect
    if not -1 <= correlation <= 1:
        raise ValueError("arm correlation must be inside [-1, 1]")
    scale = math.sqrt(full * (1 - full) * baseline * (1 - baseline))
    value = full + baseline - 2 * (full * baseline + correlation * scale)
    maximum = worst_feasible_discordance(baseline, effect)
    if value < effect - 1e-12 or value > maximum + 1e-12:
        raise ValueError("correlation implies an infeasible paired binary distribution")
    return min(maximum, max(effect, value))


def repeat_variance(discordance: float, effect: float, repeats: int, repeat_icc: float) -> float:
    if discordance < abs(effect) or discordance > 1:
        raise ValueError("discordance is infeasible")
    if type(repeats) is not int or repeats < 1 or not 0 <= repeat_icc <= 1:
        raise ValueError("repeat design is invalid")
    design_effect = (1 + (repeats - 1) * repeat_icc) / repeats
    return (discordance - effect * effect) * design_effect


def normal_case_floor(
    *,
    margin: float,
    alternative: float,
    discordance: float,
    null_discordance: float | None = None,
    alpha: float,
    power: float,
    repeats: int = 3,
    repeat_icc: float = 1.0,
) -> int:
    if alternative <= margin:
        raise ValueError("alternative must exceed the certification margin")
    if not 0 < alpha < 0.5 or not 0.5 < power < 1:
        raise ValueError("alpha or power is invalid")
    null_discordance = discordance if null_discordance is None else null_discordance
    repeat_variance(null_discordance, margin, repeats, repeat_icc)
    alternative_variance = repeat_variance(discordance, alternative, repeats, repeat_icc)
    normal = NormalDist()
    numerator = math.sqrt(2 * math.log(1 / alpha)) + normal.inv_cdf(power) * math.sqrt(
        alternative_variance
    )
    return math.ceil((numerator / (alternative - margin)) ** 2)


def _continued_beta(a: float, b: float, x: float) -> float:
    maximum_iterations = 240
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1 - qab * x / qap
    d = 1 / max(tiny, abs(d)) * (1 if d >= 0 else -1)
    result = d
    for iteration in range(1, maximum_iterations + 1):
        even = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + even) * (a + even))
        d = 1 + coefficient * d
        d = tiny if abs(d) < tiny else d
        c = 1 + coefficient / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        result *= d * c
        coefficient = -(a + iteration) * (qab + iteration) * x / ((a + even) * (qap + even))
        d = 1 + coefficient * d
        d = tiny if abs(d) < tiny else d
        c = 1 + coefficient / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        result *= delta
        if abs(delta - 1) < epsilon:
            return result
    raise ArithmeticError("incomplete beta fraction did not converge")


def regularized_beta(x: float, a: float, b: float) -> float:
    if not 0 <= x <= 1 or a <= 0 or b <= 0:
        raise ValueError("regularized beta inputs are invalid")
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return front * _continued_beta(a, b, x) / a
    return 1 - front * _continued_beta(b, a, 1 - x) / b


def binomial_survival(successes_minus_one: int, trials: int, probability: float) -> float:
    if (
        trials < 0
        or successes_minus_one < -1
        or successes_minus_one >= trials
        or not 0 <= probability <= 1
    ):
        if successes_minus_one >= trials:
            return 0.0
        raise ValueError("binomial survival inputs are invalid")
    if successes_minus_one < 0:
        return 1.0
    if probability == 0:
        return 0.0
    if probability == 1:
        return 1.0
    successes = successes_minus_one + 1
    return regularized_beta(probability, successes, trials - successes + 1)


def exact_unconditional_power(
    cases: int,
    *,
    margin: float,
    alternative: float,
    discordance: float,
    null_discordance: float | None = None,
    alpha: float,
) -> float:
    """Exact alternative power for a nuisance-free Hoeffding margin test."""

    if type(cases) is not int or cases < 1 or alternative <= margin:
        raise ValueError("case count or effects are invalid")
    if discordance < alternative or discordance > 1:
        raise ValueError("discordance is infeasible under the alternative")
    null_discordance = discordance if null_discordance is None else null_discordance
    if null_discordance < margin or null_discordance > 1:
        raise ValueError("null discordance is infeasible")
    alternative_probability = (discordance + alternative) / (2 * discordance)
    penalty = math.sqrt(2 * math.log(1 / alpha) / cases)
    critical_difference = math.ceil(cases * (margin + penalty) - 1e-12)
    terms: list[float] = []
    for discordant in range(cases + 1):
        log_probability = (
            math.lgamma(cases + 1)
            - math.lgamma(discordant + 1)
            - math.lgamma(cases - discordant + 1)
            + discordant * math.log(discordance)
            + (cases - discordant) * math.log1p(-discordance)
        )
        probability_discordant = math.exp(log_probability) if log_probability > -746 else 0.0
        critical = math.ceil((critical_difference + discordant) / 2)
        rejection = (
            0.0
            if critical > discordant
            else binomial_survival(critical - 1, discordant, alternative_probability)
        )
        terms.append(probability_discordant * rejection)
    return math.fsum(terms)


@lru_cache(maxsize=128)
def conservative_case_floor(
    *,
    margin: float,
    alternative: float,
    discordance: float,
    null_discordance: float | None = None,
    alpha: float,
    power: float,
    repeats: int = 3,
    repeat_icc: float = 1.0,
) -> tuple[int, int, float]:
    if repeat_icc != 1.0:
        raise ValueError("exact promotion floor requires repeat_icc=1")
    normal_floor = normal_case_floor(
        margin=margin,
        alternative=alternative,
        discordance=discordance,
        null_discordance=null_discordance,
        alpha=alpha,
        power=power,
        repeats=repeats,
        repeat_icc=repeat_icc,
    )
    cases = normal_floor
    exact = exact_unconditional_power(
        cases,
        margin=margin,
        alternative=alternative,
        discordance=discordance,
        null_discordance=null_discordance,
        alpha=alpha,
    )
    while exact < power:
        cases += 1
        exact = exact_unconditional_power(
            cases,
            margin=margin,
            alternative=alternative,
            discordance=discordance,
            null_discordance=null_discordance,
            alpha=alpha,
        )
    return normal_floor, cases, exact


def exact_null_size(
    cases: int,
    *,
    margin: float,
    calibrated_null_discordance: float,
    actual_null_discordance: float,
    alpha: float,
) -> float:
    """Rejection probability at the null margin for a feasible actual nuisance."""

    if actual_null_discordance < margin or actual_null_discordance > calibrated_null_discordance:
        raise ValueError("actual null discordance is outside the calibrated nuisance set")
    actual_probability = (actual_null_discordance + margin) / (2 * actual_null_discordance)
    penalty = math.sqrt(2 * math.log(1 / alpha) / cases)
    critical_difference = math.ceil(cases * (margin + penalty) - 1e-12)
    terms: list[float] = []
    for discordant in range(cases + 1):
        if actual_null_discordance == 1:
            probability_discordant = 1.0 if discordant == cases else 0.0
        else:
            log_probability = (
                math.lgamma(cases + 1)
                - math.lgamma(discordant + 1)
                - math.lgamma(cases - discordant + 1)
                + discordant * math.log(actual_null_discordance)
                + (cases - discordant) * math.log1p(-actual_null_discordance)
            )
            probability_discordant = math.exp(log_probability) if log_probability > -746 else 0.0
        critical = math.ceil((critical_difference + discordant) / 2)
        rejection = (
            0.0
            if critical > discordant
            else binomial_survival(critical - 1, discordant, actual_probability)
        )
        terms.append(probability_discordant * rejection)
    return math.fsum(terms)

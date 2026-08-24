"""The non-inferiority rule: a superior candidate must never fail for winning.

The release gates used to apply one registered ``max_ci_half_width`` of 0.03
to every paired comparison, aggregate and per-domain alike. That ceiling is
two-sided, and every discordant win widens the interval on both sides, so the
rule punished superiority: one clean win in a forty-case domain pushed the
half width to 0.0375 and failed the gate, and thirty-two aggregate wins out of
three hundred and twenty pushed it to 0.0328 and failed that one too.

The rule now separates the two things the ceiling was conflating. The
one-sided lower bound against the registered margin is the non-inferiority
test and always runs. The two-sided width ceiling is a precision requirement
that only binds an aggregate comparison still sitting below zero -- the one
place where imprecision is what stands between the report and a verdict --
and never binds a per-domain comparison, which carries its own registered
margin and its own (separately powered) sample size.

Every scenario below is a full release-scale report: eight domains, forty
cases each, five preregistered repetitions, two independent frontiers.
"""

from __future__ import annotations

from typing import Any

import pytest
from parity_gates_support import (
    allow_case_indexes,
    build_report,
    spread_allow_case_indexes,
)

from cortheon.parity import evaluate_frontier_parity
from cortheon.parity_gates import noninferiority
from cortheon.parity_gates.comparison import _paired_statistics as _paired_rows

AGGREGATE_MARGIN = 0.03
DOMAIN_MARGIN = 0.05
REGISTERED_CEILING = 0.03
FRONTIERS = ("claude", "kimi")


def _decide(**kwargs: Any) -> dict[str, Any]:
    report, contract, digest = build_report(**kwargs)
    return evaluate_frontier_parity(report, contract, contract_sha256=digest)


def _gate(decision: dict[str, Any], name: str) -> dict[str, Any]:
    return next(check for check in decision["checks"] if check["name"] == name)


def _aggregate(decision: dict[str, Any], frontier: str) -> dict[str, Any]:
    return _gate(decision, f"aggregate_noninferiority:{frontier}")


def _domain(decision: dict[str, Any], frontier: str, domain: str) -> dict[str, Any]:
    return _gate(decision, f"domain_noninferiority:{frontier}:{domain}")


def _statistics(*, lower: float, upper: float, paired: bool = True) -> dict[str, Any]:
    """A paired-comparison result shaped exactly as the bootstrap returns one."""

    return {
        "paired_runs": 1600,
        "paired_cases": 320,
        "same_paired_runs": paired,
        "delta": round((lower + upper) / 2, 6),
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_half_width": round((upper - lower) / 2, 6),
        "resamples": 5_000,
    }


def _verdict(
    statistics: dict[str, Any],
    *,
    margin: float = AGGREGATE_MARGIN,
    scope: str = "aggregate",
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    noninferiority._comparison_check(
        checks,
        "comparison",
        statistics,
        margin=margin,
        max_half_width=REGISTERED_CEILING,
        scope=scope,
    )
    return checks[0]


@pytest.fixture(scope="module")
def all_equal() -> dict[str, Any]:
    return _decide()


@pytest.fixture(scope="module")
def one_domain_win() -> dict[str, Any]:
    return _decide(candidate_wins=allow_case_indexes("domain_7", 1))


@pytest.fixture(scope="module")
def two_domain_wins() -> dict[str, Any]:
    return _decide(
        candidate_wins=allow_case_indexes("domain_6", 1) + allow_case_indexes("domain_7", 1)
    )


@pytest.fixture(scope="module")
def twenty_four_wins() -> dict[str, Any]:
    return _decide(candidate_wins=spread_allow_case_indexes(24))


@pytest.fixture(scope="module")
def thirty_two_wins() -> dict[str, Any]:
    return _decide(candidate_wins=spread_allow_case_indexes(32))


def test_an_all_equal_report_passes_every_comparison(all_equal: dict[str, Any]) -> None:
    assert all_equal["passed"] is True, all_equal["failure_reasons"]
    for frontier in FRONTIERS:
        aggregate = _aggregate(all_equal, frontier)
        assert aggregate["statistics"]["delta"] == 0
        assert aggregate["statistics"]["ci_half_width"] == 0.0
        assert aggregate["precision_ceiling_applied"] is False


def test_one_clean_domain_win_passes(one_domain_win: dict[str, Any]) -> None:
    """The narrowest form of the bug: a single win in one forty-case domain."""

    assert one_domain_win["passed"] is True, one_domain_win["failure_reasons"]
    for frontier in FRONTIERS:
        won = _domain(one_domain_win, frontier, "domain_7")["statistics"]
        assert won["delta"] > 0
        assert won["ci_lower"] >= 0.0
        # The interval the old shared rule would have rejected this run for.
        assert won["ci_half_width"] > REGISTERED_CEILING
        untouched = _domain(one_domain_win, frontier, "domain_6")["statistics"]
        assert untouched["delta"] == 0


def test_two_clean_domain_wins_pass(two_domain_wins: dict[str, Any]) -> None:
    assert two_domain_wins["passed"] is True, two_domain_wins["failure_reasons"]
    for frontier in FRONTIERS:
        for domain in ("domain_6", "domain_7"):
            statistics = _domain(two_domain_wins, frontier, domain)["statistics"]
            assert statistics["delta"] > 0
            assert statistics["ci_half_width"] > REGISTERED_CEILING


def test_twenty_four_aggregate_wins_pass(twenty_four_wins: dict[str, Any]) -> None:
    """Superiority resolved at the aggregate; every domain interval still wide."""

    assert twenty_four_wins["passed"] is True, twenty_four_wins["failure_reasons"]
    for frontier in FRONTIERS:
        aggregate = _aggregate(twenty_four_wins, frontier)["statistics"]
        assert aggregate["delta"] == pytest.approx(0.075)
        assert aggregate["ci_lower"] > 0
        assert any(
            _domain(twenty_four_wins, frontier, f"domain_{index}")["statistics"]["ci_half_width"]
            > REGISTERED_CEILING
            for index in range(8)
        )


def test_thirty_two_aggregate_wins_pass(thirty_two_wins: dict[str, Any]) -> None:
    """The aggregate case: winning is what pushed the half width over the ceiling."""

    assert thirty_two_wins["passed"] is True, thirty_two_wins["failure_reasons"]
    for frontier in FRONTIERS:
        aggregate = _aggregate(thirty_two_wins, frontier)
        assert aggregate["statistics"]["delta"] == pytest.approx(0.1)
        assert aggregate["statistics"]["ci_lower"] > 0
        assert aggregate["statistics"]["ci_half_width"] > REGISTERED_CEILING
        assert aggregate["precision_ceiling_applied"] is False
        assert aggregate["maximum_ci_half_width"] == REGISTERED_CEILING


def test_a_lower_bound_below_the_negative_margin_still_fails() -> None:
    """Inferiority is what the gate is for, and the fix must not soften it."""

    decision = _decide(candidate_losses=allow_case_indexes("domain_7", 6))

    assert decision["passed"] is False
    assert decision["failure_reasons"] == [
        "aggregate_noninferiority:claude",
        "domain_noninferiority:claude:domain_7",
        "aggregate_noninferiority:kimi",
        "domain_noninferiority:kimi:domain_7",
    ]
    for frontier in FRONTIERS:
        aggregate = _aggregate(decision, frontier)
        assert aggregate["statistics"]["ci_lower"] < -AGGREGATE_MARGIN
        # Below zero, so the precision ceiling does bind here -- and the run
        # still fails on the margin, not on the width.
        assert aggregate["precision_ceiling_applied"] is True
        assert aggregate["statistics"]["ci_half_width"] <= REGISTERED_CEILING
        assert _domain(decision, frontier, "domain_7")["statistics"]["ci_lower"] < -DOMAIN_MARGIN


def test_missing_paired_rows_fail_every_affected_comparison() -> None:
    """A dropped repetition breaks exact pairing, and no bound can excuse that."""

    decision = _decide(unpair=allow_case_indexes("domain_7", 1))

    assert decision["passed"] is False
    assert "complete_evaluation_schedule" in decision["failure_reasons"]
    for frontier in FRONTIERS:
        aggregate = _aggregate(decision, frontier)
        assert aggregate["passed"] is False
        assert aggregate["statistics"]["same_paired_runs"] is False
        # The bound itself is perfect; only the pairing is not.
        assert aggregate["statistics"]["ci_lower"] == 0.0
        affected = _domain(decision, frontier, "domain_7")
        assert affected["passed"] is False
        assert affected["statistics"]["same_paired_runs"] is False
        assert _domain(decision, frontier, "domain_6")["passed"] is True


def test_release_comparison_bootstraps_cases_not_repetitions() -> None:
    once = [
        {"candidate": candidate, "case_id": case, "repetition": 1, "verified_completion": value}
        for case, left, right in (("a", True, False), ("b", True, True))
        for candidate, value in (("left", left), ("right", right))
    ]
    repeated = [{**row, "repetition": repetition} for row in once for repetition in range(1, 7)]

    one = _paired_rows(once, "left", "right", seed=9)
    many = _paired_rows(repeated, "left", "right", seed=9)

    assert many["paired_runs"] == 12
    assert many["paired_cases"] == 2
    assert many["delta"] == one["delta"]
    assert many["ci_lower"] == one["ci_lower"]
    assert many["ci_upper"] == one["ci_upper"]


def test_release_comparison_rejects_duplicate_cells_without_order_dependence() -> None:
    rows = [
        {"candidate": "left", "case_id": "a", "repetition": 1, "verified_completion": True},
        {"candidate": "left", "case_id": "a", "repetition": 1, "verified_completion": False},
        {"candidate": "right", "case_id": "a", "repetition": 1, "verified_completion": False},
    ]

    forward = _paired_rows(rows, "left", "right", seed=9)
    reverse = _paired_rows(list(reversed(rows)), "left", "right", seed=9)

    assert forward == reverse
    assert forward["duplicate_cells"] == 1
    assert forward["paired_cases"] == 0
    assert forward["same_paired_runs"] is False


def test_exact_pairing_is_required_whatever_the_bound_says() -> None:
    assert _verdict(_statistics(lower=0.5, upper=0.6, paired=False))["passed"] is False
    assert _verdict(_statistics(lower=0.5, upper=0.6))["passed"] is True


@pytest.mark.parametrize(
    ("candidate", "frontier"),
    [
        (-1.0, 1.0),
        (1.0, -1.0),
        (float("nan"), 1.0),
        (1.0, float("inf")),
        (1.0, 0.0),
        (None, 1.0),
    ],
)
def test_ratio_check_rejects_negative_or_undefined_measurements(
    candidate: float | None,
    frontier: float | None,
) -> None:
    checks: list[dict[str, Any]] = []

    noninferiority._ratio_check(checks, "ratio", candidate, frontier, 1.0)

    assert checks[0]["passed"] is False
    assert checks[0]["ratio"] is None


def test_ratio_check_accepts_legitimate_zero_boundaries() -> None:
    checks: list[dict[str, Any]] = []

    noninferiority._ratio_check(checks, "ratio", 0.0, 0.0, 1.0)

    assert checks[0]["passed"] is True
    assert checks[0]["ratio"] == 1.0


def test_the_registered_margin_is_the_non_inferiority_test() -> None:
    assert _verdict(_statistics(lower=-AGGREGATE_MARGIN, upper=0.01))["passed"] is True
    assert _verdict(_statistics(lower=-0.030001, upper=0.01))["passed"] is False
    assert (
        _verdict(
            _statistics(lower=-DOMAIN_MARGIN, upper=0.01),
            margin=DOMAIN_MARGIN,
            scope="domain",
        )["passed"]
        is True
    )
    assert (
        _verdict(
            _statistics(lower=-0.050001, upper=0.01),
            margin=DOMAIN_MARGIN,
            scope="domain",
        )["passed"]
        is False
    )


def test_the_precision_ceiling_binds_only_an_unresolved_aggregate() -> None:
    wide_and_behind = _statistics(lower=-0.02, upper=0.30)
    wide_and_resolved = _statistics(lower=0.0, upper=0.30)

    behind = _verdict(wide_and_behind)
    assert behind["precision_ceiling_applied"] is True
    assert behind["passed"] is False

    resolved = _verdict(wide_and_resolved)
    assert resolved["precision_ceiling_applied"] is False
    assert resolved["passed"] is True
    assert noninferiority._precision_required(-0.000001, "aggregate") is True
    assert noninferiority._precision_required(0.0, "aggregate") is False


def test_a_domain_comparison_never_borrows_the_aggregate_ceiling() -> None:
    """The same wide, below-zero interval, judged at each scope.

    This is the domain-precision separation stated as a single call-site
    mutation: relabel a per-domain comparison as an aggregate one and the
    identical statistics flip from passing to failing.
    """

    wide = _statistics(lower=-0.02, upper=0.30)

    assert _verdict(wide, margin=DOMAIN_MARGIN, scope="domain")["passed"] is True
    assert _verdict(wide, margin=DOMAIN_MARGIN, scope="aggregate")["passed"] is False
    assert noninferiority._precision_required(-0.02, "domain") is False
    assert noninferiority._precision_required(-0.02, "aggregate") is True


def test_mutating_away_the_superiority_exemption_fails_the_aggregate_wins(
    thirty_two_wins: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutant: the aggregate ceiling binds regardless of where the bound sits.

    Replayed against the real bootstrap statistics from the thirty-two-win
    run, so this is the recorded evidence of the release-scale scenario, not a
    hand-written interval.
    """

    for frontier in FRONTIERS:
        statistics = _aggregate(thirty_two_wins, frontier)["statistics"]
        assert _verdict(statistics)["passed"] is True

        monkeypatch.setattr(
            noninferiority,
            "_precision_required",
            lambda _lower, scope: scope == "aggregate",
        )
        assert _verdict(statistics)["passed"] is False
        monkeypatch.undo()


def test_mutating_the_shared_width_rule_back_fails_the_clean_domain_wins(
    one_domain_win: dict[str, Any],
    twenty_four_wins: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutant: the pre-fix rule, one ceiling applied to every comparison.

    Both superiority scenarios are replayed through it. Each one fails, which
    is exactly the defect this change removes.
    """

    replays = [
        _domain(one_domain_win, frontier, "domain_7")["statistics"] for frontier in FRONTIERS
    ] + [_domain(twenty_four_wins, frontier, "domain_6")["statistics"] for frontier in FRONTIERS]
    for statistics in replays:
        assert statistics["ci_lower"] >= 0.0
        assert _verdict(statistics, margin=DOMAIN_MARGIN, scope="domain")["passed"] is True

    monkeypatch.setattr(noninferiority, "_precision_required", lambda _lower, _scope: True)
    for statistics in replays:
        assert _verdict(statistics, margin=DOMAIN_MARGIN, scope="domain")["passed"] is False

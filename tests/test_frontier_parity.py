"""The hard rule, made falsifiable.

These tests pin the two properties that stop a parity claim outrunning its
evidence: only capability-gap cases count, and the claim rests on the lower
bound of an exact interval rather than on a point estimate or a null result.
"""

from __future__ import annotations

import pytest

from cortheon.frontier_parity import (
    capability_gap,
    clopper_pearson,
    gap_analysis,
    required_gap_cases,
)


def suite(substrate: str, baseline: str, frontier: str) -> list[dict[str, object]]:
    """Build a paired suite from three result strings."""

    assert len(substrate) == len(baseline) == len(frontier)
    runs: list[dict[str, object]] = []
    for index, (s, b, f) in enumerate(zip(substrate, baseline, frontier, strict=True)):
        case = f"case-{index}"
        runs.append({"condition": "cortheon", "case_id": case, "correct": s == "1"})
        runs.append({"condition": "baseline", "case_id": case, "correct": b == "1"})
        runs.append({"condition": "frontier", "case_id": case, "correct": f == "1"})
    return runs


class TestPublishedFrontierControl:
    """An 8-case control with Cortheon 8/8, frontier 8/8, and baseline 6/8."""

    runs = suite("11111111", "11111100", "11111111")

    def test_only_two_cases_could_discriminate(self) -> None:
        assert len(capability_gap(self.runs)) == 2

    def test_perfect_closure_still_cannot_support_the_claim(self) -> None:
        analysis = gap_analysis(self.runs)
        assert analysis.closed == 2
        assert analysis.closure_rate == 1.0
        # Two cases, both closed, and the honest lower bound is ~16%.
        assert analysis.interval[0] == pytest.approx(0.158, abs=0.01)
        assert analysis.frontier_like is False
        assert analysis.informative is False
        assert "cannot clear" in analysis.verdict

    def test_it_names_the_six_uninformative_cases(self) -> None:
        analysis = gap_analysis(self.runs)
        assert any("carry no information" in note for note in analysis.notes)


class TestWhatWouldSettleIt:
    def test_seventeen_closed_gap_cases_are_required_at_the_default_bar(self) -> None:
        assert required_gap_cases(0.80, confidence=0.95) == 17
        # One fewer is genuinely not enough, even flawless.
        assert clopper_pearson(16, 16)[0] < 0.80
        assert clopper_pearson(17, 17)[0] >= 0.80

    def test_a_lower_bar_needs_fewer_cases(self) -> None:
        assert required_gap_cases(0.50) < required_gap_cases(0.80)
        assert required_gap_cases(0.90) > required_gap_cases(0.80)

    def test_a_sufficient_suite_can_earn_the_claim(self) -> None:
        gap = "0" * 20
        analysis = gap_analysis(suite("1" * 20, gap, "1" * 20))
        assert analysis.gap_size == 20
        assert analysis.informative is True
        assert analysis.frontier_like is True
        assert "frontier-like" in analysis.verdict

    def test_the_claim_fails_when_closure_is_poor(self) -> None:
        # 20 gap cases, only 12 closed.
        analysis = gap_analysis(suite("1" * 12 + "0" * 8, "0" * 20, "1" * 20))
        assert analysis.closed == 12
        assert analysis.frontier_like is False
        assert "not frontier-like" in analysis.verdict


class TestGapConstruction:
    def test_cases_the_bare_model_solves_are_excluded(self) -> None:
        # Baseline already correct: nothing to close.
        assert capability_gap(suite("1", "1", "1")) == []

    def test_cases_the_frontier_fails_are_excluded(self) -> None:
        # No bar to reach; beating the frontier here says nothing about parity.
        assert capability_gap(suite("1", "0", "0")) == []

    def test_only_baseline_fail_and_frontier_pass_counts(self) -> None:
        assert capability_gap(suite("1", "0", "1")) == ["case-0"]

    def test_an_empty_gap_is_reported_rather_than_scored(self) -> None:
        analysis = gap_analysis(suite("111", "111", "111"))
        assert analysis.gap_size == 0
        assert analysis.frontier_like is False
        assert "no capability gap" in analysis.verdict

    def test_infrastructure_failures_do_not_create_or_close_gaps(self) -> None:
        runs = suite("1", "0", "1")
        runs.append(
            {
                "condition": "cortheon",
                "case_id": "crashed",
                "correct": False,
                "process_error": "timeout",
            }
        )
        runs.append({"condition": "baseline", "case_id": "crashed", "correct": False})
        runs.append({"condition": "frontier", "case_id": "crashed", "correct": True})
        analysis = gap_analysis(runs)
        assert analysis.gap_size == 1, "the crashed case is excluded, not scored a loss"
        assert any("no usable substrate run" in note for note in analysis.notes)

    def test_divergences_are_surfaced_for_error_analysis(self) -> None:
        analysis = gap_analysis(suite("10", "00", "01"))
        assert analysis.substrate_only == ("case-0",)
        assert analysis.frontier_only == ("case-1",)


class TestExactInterval:
    def test_perfect_small_samples_are_not_certainty(self) -> None:
        assert clopper_pearson(2, 2)[0] == pytest.approx(0.158, abs=0.01)
        assert clopper_pearson(8, 8)[0] == pytest.approx(0.631, abs=0.01)

    def test_zero_successes_gives_a_zero_lower_bound(self) -> None:
        low, high = clopper_pearson(0, 10)
        assert low == 0.0
        assert high == pytest.approx(0.308, abs=0.01)

    def test_interval_brackets_the_point_estimate(self) -> None:
        low, high = clopper_pearson(6, 10)
        assert low < 0.6 < high

    def test_is_conservative_relative_to_wilson(self) -> None:
        from cortheon.paired_stats import wilson_interval

        exact = clopper_pearson(8, 8)
        approximate = wilson_interval(8, 8)
        assert exact[0] <= approximate[0], "exact must not overstate the evidence"

    def test_empty_sample_is_maximally_uncertain(self) -> None:
        assert clopper_pearson(0, 0) == (0.0, 1.0)

    def test_rejects_impossible_counts(self) -> None:
        with pytest.raises(ValueError):
            clopper_pearson(5, 2)

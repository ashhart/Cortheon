"""Paired-statistics regression cases for historical benchmark shapes."""

from __future__ import annotations

import pytest

from cortheon.paired_stats import (
    PairedOutcome,
    exact_binomial_two_sided,
    mcnemar_exact,
    minimum_discordant_for_significance,
    paired_summary,
    summarize_pairs,
    wilson_interval,
)


def pairs(treatment: str, baseline: str) -> list[PairedOutcome]:
    """Build outcomes from two result strings, e.g. "1111" and "1101"."""

    assert len(treatment) == len(baseline)
    return [
        PairedOutcome(f"case-{index}", t == "1", b == "1")
        for index, (t, b) in enumerate(zip(treatment, baseline, strict=True))
    ]


class TestPublishedResults:
    """Reproduce the statistics for known paired-result shapes."""

    def test_ambiguity_run_16_of_16_versus_11_of_16(self) -> None:
        # Five discordant pairs, all wins, produce p=0.0625.
        summary = summarize_pairs(pairs("1111111111111111", "1111111111100000"))
        assert (summary.wins, summary.losses) == (5, 0)
        assert summary.p_value == pytest.approx(0.0625, abs=1e-9)

    def test_that_run_was_underpowered_by_construction(self) -> None:
        # The uncomfortable corollary: with five discordant pairs, two-sided
        # p<=0.05 is unreachable however they fall. The claim could never have
        # been significant at the conventional threshold.
        summary = summarize_pairs(pairs("1111111111111111", "1111111111100000"))
        assert summary.powered is False
        assert summary.significant is False
        assert "underpowered" in summary.claim

    def test_paraphrase_regrade_12_of_12_versus_11_of_12(self) -> None:
        # One discordant pair cannot support a significance claim.
        summary = summarize_pairs(pairs("111111111111", "111111111110"))
        assert (summary.wins, summary.losses) == (1, 0)
        assert summary.p_value == pytest.approx(1.0)
        assert summary.significant is False

    def test_frontier_control_8_of_8_versus_6_of_8(self) -> None:
        # Reported as scoped parity, deliberately not as a significance claim.
        summary = summarize_pairs(pairs("11111111", "11111100"))
        assert (summary.wins, summary.losses) == (2, 0)
        assert summary.p_value == pytest.approx(0.5)
        assert summary.significant is False

    def test_rejected_design_regression_1_of_8_versus_7_of_8(self) -> None:
        # Removing host tools: six discordant pairs, all losses.
        summary = summarize_pairs(pairs("10000000", "11111110"))
        assert (summary.wins, summary.losses) == (0, 6)
        assert summary.p_value == pytest.approx(0.03125, abs=1e-9)
        assert summary.significant is True
        assert "regression" in summary.claim

    def test_rejected_staged_rewrite_0_of_8_versus_4_of_8(self) -> None:
        summary = summarize_pairs(pairs("00000000", "11110000"))
        assert (summary.wins, summary.losses) == (0, 4)
        assert summary.p_value == pytest.approx(0.125, abs=1e-9)
        assert summary.significant is False, "four discordant pairs cannot reach 0.05"


class TestExactness:
    def test_mcnemar_ignores_concordant_pairs(self) -> None:
        few = summarize_pairs(pairs("110", "100"))
        many = summarize_pairs(pairs("1101111111", "1001111111"))
        assert few.p_value == many.p_value

    def test_no_discordant_pairs_is_not_evidence(self) -> None:
        summary = summarize_pairs(pairs("1010", "1010"))
        assert summary.p_value == pytest.approx(1.0)
        assert summary.powered is False

    def test_two_sided_is_symmetric(self) -> None:
        assert mcnemar_exact(6, 1) == pytest.approx(mcnemar_exact(1, 6))

    def test_matches_hand_computed_binomial(self) -> None:
        # 2 * 0.5**5 for the all-one-way case.
        assert exact_binomial_two_sided(5, 5) == pytest.approx(0.0625)
        assert exact_binomial_two_sided(0, 5) == pytest.approx(0.0625)
        # Balanced outcome cannot be evidence of anything.
        assert exact_binomial_two_sided(3, 6) == pytest.approx(1.0)

    def test_biased_coin_uses_total_probability_not_a_doubled_tail(self) -> None:
        # Under an asymmetric null the two-sided p-value is *not* the point mass
        # of the observed outcome. At p=0.9, k=9 is more likely than k=10, so it
        # is excluded, while the entire lower tail (k<=8) is included.
        observed = 0.9**10  # 0.34868
        assert exact_binomial_two_sided(10, 10, 0.9) == pytest.approx(
            observed
            + sum(__import__("math").comb(10, k) * 0.9**k * 0.1 ** (10 - k) for k in range(9)),
            rel=1e-9,
        )
        # Doubling the tail would have understated it badly.
        assert exact_binomial_two_sided(10, 10, 0.9) > 2 * observed * 0.8

    def test_rejects_impossible_counts(self) -> None:
        with pytest.raises(ValueError):
            exact_binomial_two_sided(5, 3)
        with pytest.raises(ValueError):
            mcnemar_exact(-1, 2)


class TestPower:
    def test_six_discordant_pairs_are_required_at_five_percent(self) -> None:
        assert minimum_discordant_for_significance(0.05) == 6
        assert 2 * 0.5**5 > 0.05
        assert 2 * 0.5**6 <= 0.05

    def test_threshold_moves_with_alpha(self) -> None:
        assert minimum_discordant_for_significance(0.10) == 5
        assert minimum_discordant_for_significance(0.01) == 8

    def test_underpowered_runs_never_claim_significance(self) -> None:
        # Every pair a win, but too few of them.
        summary = summarize_pairs(pairs("11111", "00000"))
        assert summary.wins == 5
        assert summary.significant is False
        assert summary.powered is False


class TestIntervals:
    def test_perfect_score_does_not_claim_certainty(self) -> None:
        low, high = wilson_interval(16, 16)
        assert high == pytest.approx(1.0)
        assert low == pytest.approx(0.8065, abs=1e-3)
        assert low < 0.85, "a Wald interval would wrongly report [1.0, 1.0]"

    def test_zero_score_does_not_claim_certainty(self) -> None:
        low, high = wilson_interval(0, 16)
        assert low == pytest.approx(0.0)
        assert high > 0.15

    def test_empty_sample_is_maximally_uncertain(self) -> None:
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_interval_narrows_with_more_evidence(self) -> None:
        small = wilson_interval(8, 16)
        large = wilson_interval(80, 160)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_unsupported_confidence_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            wilson_interval(1, 2, confidence=0.937)


class TestRunPairing:
    def _run(self, condition: str, case_id: str, correct: bool, **extra: object):
        return {"condition": condition, "case_id": case_id, "correct": correct, **extra}

    def test_pairs_by_case_id_not_by_position(self) -> None:
        runs = [
            self._run("cortheon", "b", True),
            self._run("cortheon", "a", True),
            self._run("baseline", "a", False),
            self._run("baseline", "b", True),
        ]
        summary = paired_summary(runs)
        assert summary.pairs == 2
        assert (summary.wins, summary.losses) == (1, 0)

    def test_infrastructure_failures_are_excluded_not_scored(self) -> None:
        runs = [
            self._run("cortheon", "a", True),
            self._run("baseline", "a", False),
            self._run("cortheon", "b", False, process_error="timeout"),
            self._run("baseline", "b", True),
        ]
        summary = paired_summary(runs)
        assert summary.pairs == 1, "a crashed run is not evidence about capability"
        assert summary.losses == 0
        assert any("only one condition" in note for note in summary.notes)

    def test_unmatched_cases_are_reported(self) -> None:
        runs = [
            self._run("cortheon", "a", True),
            self._run("baseline", "a", True),
            self._run("cortheon", "orphan", True),
        ]
        summary = paired_summary(runs)
        assert summary.pairs == 1
        assert any("only one condition" in note for note in summary.notes)

    def test_duplicate_case_ids_are_flagged(self) -> None:
        summary = summarize_pairs(
            [
                PairedOutcome("same", True, False),
                PairedOutcome("same", True, False),
            ]
        )
        assert any("duplicate case ids" in note for note in summary.notes)

    def test_repetitions_are_collapsed_to_one_independent_case(self) -> None:
        runs = [
            self._run(condition, "one", condition == "cortheon", repeat=repeat)
            for repeat in range(6)
            for condition in ("cortheon", "baseline")
        ]

        summary = paired_summary(runs)

        assert summary.pairs == 1
        assert summary.wins == 1
        assert summary.p_value == 1.0
        assert summary.powered is False

    def test_duplicate_repeat_cell_is_excluded_without_order_dependence(self) -> None:
        runs = [
            self._run("cortheon", "a", True, repeat=0),
            self._run("cortheon", "a", False, repeat=0),
            self._run("baseline", "a", False, repeat=0),
        ]

        forward = paired_summary(runs)
        reverse = paired_summary(list(reversed(runs)))

        assert forward == reverse
        assert forward.pairs == 0
        assert any("duplicate" in note for note in forward.notes)

    def test_ceilinged_suite_is_called_out(self) -> None:
        summary = summarize_pairs(pairs("1111", "1111"))
        assert any("ceilinged" in note for note in summary.notes)

    def test_serialization_carries_the_claim(self) -> None:
        payload = summarize_pairs(pairs("11111111", "11000000")).to_dict()
        assert payload["wins"] == 6
        assert payload["discordant"] == 6
        assert payload["significant"] is True
        assert "lift" in payload["claim"]
        assert len(payload["treatment_interval"]) == 2

"""Difficulty-targeted candidate selection.

The property under test is not "harder cases score higher" — that is trivially
true and not the goal. It is that the module optimises for *gap yield*, reports
honestly when banding buys nothing, and never lets selection quietly widen the
claim.
"""

from __future__ import annotations

import pytest

from cortheon.case_difficulty import (
    SELECTED_SCOPE,
    calibrate,
    difficulty_features,
    rank_candidates,
    select_candidates,
)
from cortheon.cognitive_benchmark import ImportCase, ReasoningCase, SemanticCase


def reasoning(case_id: str, *, files: int, relations: int, expected: int) -> ReasoningCase:
    return ReasoningCase(
        case_id=case_id,
        mode="derive",
        files=tuple((f"f{i}.py", "x = 1\n" * 20) for i in range(files)),
        expected=tuple(f"e{i}" for i in range(expected)),
        forbidden_answers=(),
        required_any=(),
        derived_relations=tuple(((f"a{i}", f"b{i}"),) for i in range(relations)),
        prompt="derive the relation",
    )


class TestFeatures:
    def test_structural_signals_are_extracted_without_running_anything(self) -> None:
        features = difficulty_features(reasoning("c", files=3, relations=2, expected=4))
        assert features["files_count"] == 3
        assert features["derived_relations_count"] == 2
        assert features["expected_count"] == 4
        assert features["source_chars"] > 0

    def test_a_new_case_type_gains_features_by_introspection(self) -> None:
        # ImportCase has none of the multi-source fields; it must still score.
        features = difficulty_features(
            ImportCase(case_id="i", path="a.py", module="os", expected=True, prompt="q")
        )
        assert features["prompt_chars"] == 1.0
        assert "files_count" not in features

    def test_nesting_depth_is_captured_for_multi_hop_structure(self) -> None:
        features = difficulty_features(reasoning("c", files=1, relations=3, expected=1))
        assert features.get("derived_relations_depth", 0) >= 2

    def test_features_are_deterministic(self) -> None:
        case = reasoning("c", files=2, relations=2, expected=2)
        assert difficulty_features(case) == difficulty_features(case)


class TestRanking:
    def test_orders_hardest_first(self) -> None:
        cases = [
            reasoning("easy", files=1, relations=1, expected=1),
            reasoning("hard", files=5, relations=5, expected=5),
            reasoning("mid", files=3, relations=3, expected=3),
        ]
        assert [case_id for case_id, _ in rank_candidates(cases)] == ["hard", "mid", "easy"]

    def test_ranking_is_relative_to_the_pool(self) -> None:
        # The same case is "hard" among easy peers and "easy" among hard ones.
        target = reasoning("target", files=3, relations=3, expected=3)
        easy_pool = [target, reasoning("e", files=1, relations=1, expected=1)]
        hard_pool = [target, reasoning("h", files=9, relations=9, expected=9)]
        assert rank_candidates(easy_pool)[0][0] == "target"
        assert rank_candidates(hard_pool)[0][0] == "h"

    def test_one_outlier_does_not_flatten_the_rest(self) -> None:
        cases = [
            reasoning("a", files=1, relations=1, expected=1),
            reasoning("b", files=2, relations=2, expected=2),
            reasoning("giant", files=400, relations=400, expected=400),
        ]
        scores = dict(rank_candidates(cases))
        assert scores["b"] > scores["a"], "rank normalisation, not min-max"

    def test_empty_and_featureless_pools_are_handled(self) -> None:
        assert rank_candidates([]) == []
        singles = rank_candidates([reasoning("only", files=1, relations=1, expected=1)])
        assert len(singles) == 1


class TestCalibration:
    def _observations(self, pattern: list[tuple[float, bool]]):
        return [{"score": score, "gap": gap} for score, gap in pattern]

    def test_measures_yield_per_band(self) -> None:
        # Low band never informative, high band usually is.
        pattern = [(0.1, False)] * 6 + [(0.5, False)] * 6 + [(0.9, True)] * 6
        calibration = calibrate(self._observations(pattern))
        assert calibration.overall_yield == pytest.approx(1 / 3)
        assert calibration.best_band == "high"
        assert calibration.lift == pytest.approx(3.0)
        assert calibration.useful is True

    def test_reports_when_the_hardest_band_is_not_the_best(self) -> None:
        # Too hard: the frontier fails as well, so no gap case results.
        pattern = [(0.1, False)] * 6 + [(0.5, True)] * 6 + [(0.9, False)] * 6
        calibration = calibrate(self._observations(pattern))
        assert calibration.best_band == "medium"
        assert any("only a proxy" in note for note in calibration.notes)

    def test_reports_when_banding_buys_nothing(self) -> None:
        pattern = [(index / 20, index % 2 == 0) for index in range(20)]
        calibration = calibrate(self._observations(pattern))
        assert calibration.lift == pytest.approx(1.0, abs=0.35)

    def test_small_samples_are_flagged_as_indicative(self) -> None:
        calibration = calibrate(self._observations([(0.5, True), (0.1, False)]))
        assert any("indicative" in note for note in calibration.notes)

    def test_no_observations_yields_no_claim(self) -> None:
        calibration = calibrate([])
        assert calibration.bands == ()
        assert calibration.best_band is None
        assert calibration.useful is False


class TestSelection:
    def test_selection_narrows_the_claim_and_says_so(self) -> None:
        selection = select_candidates(
            [reasoning(f"c{i}", files=i + 1, relations=i + 1, expected=1) for i in range(5)],
            take=3,
        )
        assert selection.scope == SELECTED_SCOPE
        assert "cannot solve" in selection.scope
        assert len(selection.case_ids) == 3

    def test_uses_the_measured_band_when_calibrated(self) -> None:
        calibration = calibrate(
            [{"score": 0.1, "gap": False}] * 6 + [{"score": 0.9, "gap": True}] * 6
        )
        selection = select_candidates(
            [reasoning(f"c{i}", files=i + 1, relations=1, expected=1) for i in range(10)],
            take=4,
            calibration=calibration,
        )
        assert selection.expected_yield > 0.25
        assert any("measured" in note for note in selection.notes)

    def test_says_so_when_ordering_will_not_help(self) -> None:
        flat = calibrate([{"score": i / 20, "gap": i % 2 == 0} for i in range(20)])
        selection = select_candidates(
            [reasoning(f"c{i}", files=i + 1, relations=1, expected=1) for i in range(6)],
            take=3,
            calibration=flat,
        )
        if not flat.useful:
            assert any("unlikely to beat random" in note for note in selection.notes)

    def test_falls_back_to_the_historical_yield_uncalibrated(self) -> None:
        selection = select_candidates([reasoning("a", files=1, relations=1, expected=1)], take=1)
        assert selection.expected_yield == pytest.approx(0.25)
        assert any("no calibration" in note for note in selection.notes)

    def test_an_over_large_request_is_reported(self) -> None:
        selection = select_candidates([reasoning("a", files=1, relations=1, expected=1)], take=10)
        assert len(selection.case_ids) == 1
        assert any("pool holds" in note for note in selection.notes)

    def test_expected_gap_projects_the_screening_result(self) -> None:
        selection = select_candidates(
            [reasoning(f"c{i}", files=i + 1, relations=1, expected=1) for i in range(40)],
            take=40,
        )
        assert selection.expected_gap == pytest.approx(10.0)

    def test_selection_works_across_mixed_case_types(self) -> None:
        mixed = [
            reasoning("r", files=4, relations=4, expected=4),
            SemanticCase(
                case_id="s",
                files=(("a.py", "x"),),
                expected=("e",),
                forbidden_answers=("f",),
                prompt="q",
            ),
            ImportCase(case_id="i", path="a.py", module="os", expected=True, prompt="q"),
        ]
        ranked = rank_candidates(mixed)
        assert len(ranked) == 3
        assert ranked[0][0] == "r"


class TestTieHandling:
    """A constant feature must contribute nothing, not index-order noise."""

    def test_identical_cases_score_identically(self) -> None:
        cases = [reasoning(f"c{i}", files=2, relations=2, expected=2) for i in range(5)]
        scores = {score for _, score in rank_candidates(cases)}
        assert len(scores) == 1, "identical cases must not be separated by list order"

    def test_a_constant_feature_does_not_reorder_the_pool(self) -> None:
        # All prompts are identical here, so prompt_chars is constant; ordering
        # must come only from the features that actually vary.
        cases = [
            reasoning("low", files=1, relations=1, expected=1),
            reasoning("high", files=6, relations=6, expected=6),
            reasoning("mid", files=3, relations=3, expected=3),
        ]
        assert [case_id for case_id, _ in rank_candidates(cases)] == ["high", "mid", "low"]

    def test_partial_ties_share_a_rank(self) -> None:
        cases = [
            reasoning("a", files=2, relations=2, expected=2),
            reasoning("b", files=2, relations=2, expected=2),
            reasoning("c", files=9, relations=9, expected=9),
        ]
        scored = dict(rank_candidates(cases))
        assert scored["a"] == scored["b"]
        assert scored["c"] > scored["a"]

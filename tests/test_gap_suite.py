"""Screening: spend the expensive conditions only where the verdict can change,
and never let the substrate influence which cases are chosen.
"""

from __future__ import annotations

import pytest

from cortheon.frontier_parity import required_gap_cases
from cortheon.gap_suite import (
    SubstrateLeakError,
    candidates_required,
    estimate_yield,
    plan_screening,
    screen,
)


def run(condition: str, case_id: str, correct: bool, **extra: object) -> dict[str, object]:
    return {"condition": condition, "case_id": case_id, "correct": correct, **extra}


class TestSelectionIntegrity:
    def test_substrate_results_are_refused_as_a_selection_input(self) -> None:
        runs = [
            run("baseline", "a", False),
            run("frontier", "a", True),
            run("cortheon", "a", True),
        ]
        with pytest.raises(SubstrateLeakError, match="must not depend on substrate"):
            screen(runs)

    def test_selection_uses_only_baseline_and_frontier(self) -> None:
        result = screen([run("baseline", "a", False), run("frontier", "a", True)])
        assert result.to_dict()["selection_inputs"] == ["baseline", "frontier"]

    def test_a_case_the_frontier_also_fails_sets_no_bar(self) -> None:
        result = screen([run("baseline", "a", False), run("frontier", "a", False)])
        assert result.gap_cases == ()
        assert result.rejected_frontier_also_failed == ("a",)
        assert any("no bar to clear" in note for note in result.notes)

    def test_a_case_the_baseline_solves_is_not_a_gap_case(self) -> None:
        result = screen([run("baseline", "a", True), run("frontier", "a", True)])
        assert result.gap_cases == ()
        assert result.baseline_failures == ()

    def test_unconfirmed_failures_are_held_not_counted(self) -> None:
        # Baseline failed but the frontier has not run yet.
        result = screen([run("baseline", "a", False)])
        assert result.gap_cases == ()
        assert result.baseline_failures == ("a",)
        assert any("await a frontier run" in note for note in result.notes)

    def test_crashed_runs_do_not_create_gap_cases(self) -> None:
        runs = [
            run("baseline", "a", False, process_error="timeout"),
            run("frontier", "a", True),
        ]
        assert screen(runs).gap_cases == ()


class TestYield:
    def test_measures_the_published_control(self) -> None:
        # cortheon 8/8, baseline 6/8, frontier 8/8 -> two gap cases in eight.
        runs = []
        for index in range(8):
            runs.append(run("baseline", f"c{index}", index < 6))
            runs.append(run("frontier", f"c{index}", True))
        assert estimate_yield(runs) == pytest.approx(0.25)

    def test_returns_none_rather_than_guessing(self) -> None:
        assert estimate_yield([]) is None
        assert estimate_yield([run("baseline", "a", True)]) is None

    def test_screening_reports_the_observed_yield(self) -> None:
        runs = []
        for index in range(8):
            runs.append(run("baseline", f"c{index}", index < 6))
            runs.append(run("frontier", f"c{index}", True))
        result = screen(runs)
        assert result.gap_size == 2
        assert result.screened == 8
        assert result.observed_yield == pytest.approx(0.25)


class TestPlanning:
    def test_planning_on_the_mean_alone_would_come_up_short(self) -> None:
        # 17 gap cases at a 0.25 yield: the mean says 68, which succeeds about
        # half the time. The exact tail demands materially more.
        naive = 17 / 0.25
        planned = candidates_required(17, 0.25, confidence=0.90)
        assert planned > naive
        assert planned >= 80

    def test_higher_confidence_costs_more_candidates(self) -> None:
        assert candidates_required(17, 0.25, confidence=0.99) > candidates_required(
            17, 0.25, confidence=0.90
        )

    def test_a_richer_yield_needs_fewer_candidates(self) -> None:
        assert candidates_required(17, 0.50) < candidates_required(17, 0.25)

    def test_a_perfect_yield_needs_exactly_the_target(self) -> None:
        assert candidates_required(17, 1.0) == 17

    def test_zero_target_needs_nothing(self) -> None:
        assert candidates_required(0, 0.25) == 0

    def test_rejects_impossible_inputs(self) -> None:
        with pytest.raises(ValueError):
            candidates_required(5, 0.0)
        with pytest.raises(ValueError):
            candidates_required(-1, 0.5)


class TestScreeningEconomics:
    def test_screening_beats_running_everything_on_everything(self) -> None:
        plan = plan_screening(required_gap_cases(0.80), 0.25)
        assert plan.target_gap == 17
        assert plan.total_runs < plan.naive_runs
        assert plan.runs_saved > 0

    def test_the_substrate_runs_only_on_confirmed_gap_cases(self) -> None:
        plan = plan_screening(17, 0.25)
        assert plan.substrate_runs == 17
        assert plan.baseline_runs == plan.candidates

    def test_the_plan_is_serialisable_for_the_report(self) -> None:
        payload = plan_screening(17, 0.25).to_dict()
        assert payload["target_gap"] == 17
        assert payload["total_runs"] < payload["naive_runs"]
        assert payload["candidates"] >= 80

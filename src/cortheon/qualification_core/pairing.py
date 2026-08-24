"""Independent-case pairing: repeats measure stability, never independence."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any

from cortheon.benchmark_core.outcomes import is_verified_completion
from cortheon.cognitive_benchmark import RunResult, _percentile, is_comparable_outcome
from cortheon.qualification_core.models import CellRun


def _bootstrap_summary(
    deltas: dict[str, float],
    *,
    seed: int,
) -> dict[str, Any]:
    values = list(deltas.values())
    wins = sum(value > 0 for value in values)
    losses = sum(value < 0 for value in values)
    ties = len(values) - wins - losses
    bootstrap: list[float] = []
    rng = random.Random(seed ^ 0x51A71F1E)
    if values:
        bootstrap.extend(
            sum(rng.choice(values) for _ in values) / len(values) for _ in range(2_000)
        )
    discordant = wins + losses
    if discordant:
        tail = min(wins, losses)
        sign_test = min(
            1.0,
            2 * sum(math.comb(discordant, index) for index in range(tail + 1)) / (2**discordant),
        )
    else:
        sign_test = 1.0
    return {
        "treatment_wins": wins,
        "comparison_wins": losses,
        "ties": ties,
        "accuracy_delta": sum(values) / len(values) if values else 0.0,
        "accuracy_delta_95_ci": (
            [
                _percentile(bootstrap, 0.025),
                _percentile(bootstrap, 0.975),
            ]
            if bootstrap
            else [0.0, 0.0]
        ),
        "paired_sign_test_exact_p": sign_test,
    }


def _independent_pairing(
    results: list[RunResult],
    *,
    treatment: str,
    comparison: str,
    repeats: tuple[int, ...],
    seed: int,
) -> tuple[dict[str, Any], dict[str, float], set[str]]:
    grouped: dict[str, dict[int, dict[str, RunResult]]] = {}
    duplicate_pairs = 0
    duplicate_case_ids: set[str] = set()
    for result in results:
        pair = grouped.setdefault(result.case_id, {}).setdefault(
            result.repeat,
            {},
        )
        if result.condition in pair:
            duplicate_pairs += 1
            duplicate_case_ids.add(result.case_id)
        pair[result.condition] = result
    case_deltas: dict[str, float] = {}
    invalid_case_ids = set(duplicate_case_ids)
    invalid_pairs = duplicate_pairs
    valid_pairs = 0
    unstable_cases = 0
    expected_conditions = {treatment, comparison}
    expected_repeats = set(repeats)
    for case_id, case_repeats in grouped.items():
        deltas: list[int] = []
        treatment_values: list[bool] = []
        comparison_values: list[bool] = []
        if set(case_repeats) != expected_repeats:
            invalid_pairs += len(expected_repeats - set(case_repeats))
            invalid_case_ids.add(case_id)
        for repeat in repeats:
            pair = case_repeats.get(repeat)
            if pair is None:
                continue
            # The shared taxonomy decides which arms may be compared. An arm
            # that timed out or returned nothing observed no outcome, so it is
            # not an incorrect comparator and cannot hand the other arm a
            # delta; an explicit withheld block declined with a candidate in
            # hand and stays a valid one.
            if set(pair) != expected_conditions or not all(
                is_comparable_outcome(item) for item in pair.values()
            ):
                invalid_pairs += 1
                invalid_case_ids.add(case_id)
                continue
            valid_pairs += 1
            treatment_correct = is_verified_completion(pair[treatment])
            comparison_correct = is_verified_completion(pair[comparison])
            treatment_values.append(treatment_correct)
            comparison_values.append(comparison_correct)
            deltas.append(int(treatment_correct) - int(comparison_correct))
        if case_id not in invalid_case_ids and len(deltas) == len(repeats):
            case_deltas[case_id] = sum(deltas) / len(deltas)
            if len(set(treatment_values)) > 1 or len(set(comparison_values)) > 1:
                unstable_cases += 1
    summary = {
        "treatment_condition": treatment,
        "comparison_condition": comparison,
        "independent_cases": len(grouped),
        "qualified_independent_cases": len(case_deltas),
        "invalid_independent_cases": len(invalid_case_ids),
        "repeat_pairs": len(grouped) * len(repeats),
        "valid_repeat_pairs": valid_pairs,
        "invalid_pairs": invalid_pairs,
        "duplicate_cells": duplicate_pairs,
        "repeats_per_case": len(repeats),
        "unstable_cases": unstable_cases,
        **_bootstrap_summary(case_deltas, seed=seed),
    }
    return summary, case_deltas, invalid_case_ids


def _aggregate_pairing(
    runs: list[CellRun],
    *,
    contrast: str,
    seed: int,
) -> dict[str, Any]:
    all_tasks: dict[str, int] = {}
    valid_task_deltas: dict[str, list[float]] = {}
    invalid_tasks: set[str] = set()
    for run in runs:
        case_deltas = run.contrast_case_deltas.get(contrast, {})
        invalid_case_ids = run.contrast_invalid_case_ids.get(contrast, set())
        for case_id in run.case_ids:
            task_id = run.task_digests.get(
                case_id,
                hashlib.sha256(f"{run.cell.suite}:{case_id}".encode()).hexdigest(),
            )
            all_tasks[task_id] = all_tasks.get(task_id, 0) + 1
            if case_id in invalid_case_ids or case_id not in case_deltas:
                invalid_tasks.add(task_id)
            else:
                valid_task_deltas.setdefault(task_id, []).append(case_deltas[case_id])
    collapsed = {
        task_id: sum(values) / len(values)
        for task_id, values in valid_task_deltas.items()
        if task_id not in invalid_tasks and len(values) == all_tasks.get(task_id, 0)
    }
    return {
        "independent_cases": len(all_tasks),
        "qualified_independent_cases": len(collapsed),
        "invalid_independent_cases": len(all_tasks) - len(collapsed),
        "cell_case_exposures": sum(len(run.case_ids) for run in runs),
        "repeat_pairs": sum(run.contrasts[contrast]["repeat_pairs"] for run in runs),
        "valid_repeat_pairs": sum(run.contrasts[contrast]["valid_repeat_pairs"] for run in runs),
        "invalid_pairs": sum(run.contrasts[contrast]["invalid_pairs"] for run in runs),
        "duplicate_cells": sum(run.contrasts[contrast].get("duplicate_cells", 0) for run in runs),
        "unstable_cell_cases": sum(run.contrasts[contrast]["unstable_cases"] for run in runs),
        **_bootstrap_summary(collapsed, seed=seed),
    }

"""Case-clustered paired statistics for parity benchmark reports."""

from __future__ import annotations

import hashlib
import random
import statistics
from typing import Any

from cortheon.parity_benchmark_core.metrics import _percentile

_RESAMPLES = 2_000


def _paired_candidate_comparisons(
    rows: list[dict[str, Any]],
    aliases: dict[str, str],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    alias_values = sorted(aliases.values())
    for left_index, left in enumerate(alias_values):
        for right in alias_values[left_index + 1 :]:
            left_rows = _cell_index(rows, left)
            right_rows = _cell_index(rows, right)
            duplicate_cells = _duplicate_count(left_rows) + _duplicate_count(right_rows)
            all_keys = sorted(set(left_rows) | set(right_rows))
            invalid_cases = {
                str(key[0])
                for key in all_keys
                if len(left_rows.get(key, [])) != 1
                or len(right_rows.get(key, [])) != 1
                or any(
                    row.get("failure_owner") == "external_infrastructure"
                    for row in (*left_rows.get(key, []), *right_rows.get(key, []))
                )
            }
            differences: dict[str, list[int]] = {}
            for key in all_keys:
                if str(key[0]) in invalid_cases:
                    continue
                differences.setdefault(str(key[0]), []).append(
                    int(left_rows[key][0]["verified_completion"] is True)
                    - int(right_rows[key][0]["verified_completion"] is True)
                )
            comparisons.append(
                _paired_statistics(
                    differences,
                    left=left,
                    right=right,
                    seed=_stable_integer_seed(seed, left, right),
                    duplicate_cells=duplicate_cells,
                    same_paired_runs=(
                        bool(left_rows)
                        and set(left_rows) == set(right_rows)
                        and duplicate_cells == 0
                        and not invalid_cases
                    ),
                    invalid_cases=len(invalid_cases),
                )
            )
    return comparisons


def _paired_statistics(
    differences_by_case: dict[str, list[int]],
    *,
    left: str,
    right: str,
    seed: int,
    duplicate_cells: int = 0,
    same_paired_runs: bool = True,
    invalid_cases: int = 0,
) -> dict[str, Any]:
    case_differences = [statistics.mean(values) for values in differences_by_case.values()]
    wins = sum(value > 0 for value in case_differences)
    losses = sum(value < 0 for value in case_differences)
    ties = len(case_differences) - wins - losses
    if case_differences:
        observed = statistics.mean(case_differences)
        rng = random.Random(seed)
        samples = [
            statistics.mean(
                case_differences[rng.randrange(len(case_differences))] for _ in case_differences
            )
            for _ in range(_RESAMPLES)
        ]
        lower = _percentile(samples, 0.025)
        upper = _percentile(samples, 0.975)
        probability_superior = sum(value > 0 for value in samples) / len(samples)
    else:
        observed = lower = upper = None
        probability_superior = None
    return {
        "left": left,
        "right": right,
        "paired_runs": sum(len(values) for values in differences_by_case.values()),
        "paired_cases": len(case_differences),
        "duplicate_cells": duplicate_cells,
        "invalid_cases": invalid_cases,
        "same_paired_runs": same_paired_runs,
        "verified_completion_rate_delta": round(observed, 4) if observed is not None else None,
        "left_wins": wins,
        "right_wins": losses,
        "ties": ties,
        "paired_bootstrap_95ci": {
            "lower": round(lower, 4) if lower is not None else None,
            "upper": round(upper, 4) if upper is not None else None,
            "resamples": _RESAMPLES,
        },
        "bootstrap_probability_left_superior": (
            round(probability_superior, 4) if probability_superior is not None else None
        ),
    }


def _cell_index(
    rows: list[dict[str, Any]],
    alias: str,
) -> dict[tuple[Any, Any], list[dict[str, Any]]]:
    indexed: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("candidate") == alias:
            indexed.setdefault((row.get("case_id"), row.get("repetition")), []).append(row)
    return indexed


def _duplicate_count(indexed: dict[tuple[Any, Any], list[dict[str, Any]]]) -> int:
    return sum(max(0, len(rows) - 1) for rows in indexed.values())


def _stable_integer_seed(seed: int, *values: str) -> int:
    encoded = ":".join([str(seed), *values]).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")

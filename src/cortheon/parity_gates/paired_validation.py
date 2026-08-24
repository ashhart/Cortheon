"""Reconstruct sealed-report pair summaries from attested rows."""

from __future__ import annotations

import hashlib
import random
import statistics
from typing import Any

from cortheon.parity_gates.pairing_cells import _cell_index, _duplicate_count
from cortheon.parity_gates.values import _percentile

_REPORT_PAIR_RESAMPLES = 2_000


def canonical_paired_comparisons(
    rows: list[Any],
    candidates: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    """Reproduce the report estimator independently of its cached output."""

    comparisons: list[dict[str, Any]] = []
    aliases = sorted(candidates)
    for left_index, left in enumerate(aliases):
        for right in aliases[left_index + 1 :]:
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
                _report_pair_statistics(
                    differences,
                    left=left,
                    right=right,
                    seed=_report_pair_seed(seed, left, right),
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


def _report_pair_statistics(
    differences_by_case: dict[str, list[int]],
    *,
    left: str,
    right: str,
    seed: int,
    duplicate_cells: int = 0,
    same_paired_runs: bool = True,
    invalid_cases: int = 0,
) -> dict[str, Any]:
    differences = [statistics.mean(values) for values in differences_by_case.values()]
    wins = sum(value > 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    ties = len(differences) - wins - losses
    if differences:
        observed = statistics.mean(differences)
        rng = random.Random(seed)
        samples = [
            statistics.mean(
                differences[rng.randrange(len(differences))] for _ in range(len(differences))
            )
            for _ in range(_REPORT_PAIR_RESAMPLES)
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
        "paired_cases": len(differences),
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
            "resamples": _REPORT_PAIR_RESAMPLES,
        },
        "bootstrap_probability_left_superior": (
            round(probability_superior, 4) if probability_superior is not None else None
        ),
    }


def _report_pair_seed(seed: int, *values: str) -> int:
    encoded = ":".join([str(seed), *values]).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")

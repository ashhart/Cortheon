"""Paired statistics over the executed matrix.

Both estimators resample *cases*, never rows. Repetitions of one case are not
independent observations, so a case is drawn whole and all of its repetition
differences travel with it; resampling rows would treat five repeats of an
easy case as five independent wins and shrink every interval.
"""

from __future__ import annotations

import random
import statistics
from typing import Any

from cortheon.parity_gates.pairing_cells import _cell_index, _duplicate_count
from cortheon.parity_gates.values import _percentile

_RESAMPLES = 5_000


def _paired_statistics(
    rows: list[dict[str, Any]],
    left: str,
    right: str,
    *,
    seed: int,
    domain: str | None = None,
) -> dict[str, Any]:
    """Bootstrap the paired verified-completion difference between two contenders."""

    relevant = [row for row in rows if domain is None or str(row.get("domain")) == domain]
    left_rows = _cell_index(relevant, left)
    right_rows = _cell_index(relevant, right)
    duplicate_cells = _duplicate_count(left_rows) + _duplicate_count(right_rows)
    keys = sorted(set(left_rows) | set(right_rows))
    invalid_cases = {
        str(key[0])
        for key in keys
        if len(left_rows.get(key, [])) != 1
        or len(right_rows.get(key, [])) != 1
        or any(
            row.get("failure_owner") == "external_infrastructure"
            for row in (*left_rows.get(key, []), *right_rows.get(key, []))
        )
    }
    same_keys = (
        bool(left_rows)
        and set(left_rows) == set(right_rows)
        and duplicate_cells == 0
        and not invalid_cases
    )
    differences_by_case: dict[str, list[int]] = {}
    for key in keys:
        if str(key[0]) in invalid_cases:
            continue
        differences_by_case.setdefault(str(key[0]), []).append(
            int(left_rows[key][0].get("verified_completion") is True)
            - int(right_rows[key][0].get("verified_completion") is True)
        )
    case_differences = [statistics.mean(values) for values in differences_by_case.values()]
    if not case_differences:
        return {
            "paired_runs": 0,
            "paired_cases": 0,
            "duplicate_cells": duplicate_cells,
            "invalid_cases": len(invalid_cases),
            "same_paired_runs": False,
            "delta": None,
            "ci_lower": None,
            "ci_upper": None,
            "ci_half_width": None,
        }
    rng = random.Random(seed)
    samples = [
        statistics.mean(
            case_differences[rng.randrange(len(case_differences))] for _ in case_differences
        )
        for _ in range(_RESAMPLES)
    ]
    lower = _percentile(samples, 0.025)
    upper = _percentile(samples, 0.975)
    return {
        "paired_runs": sum(len(values) for values in differences_by_case.values()),
        "paired_cases": len(case_differences),
        "duplicate_cells": duplicate_cells,
        "invalid_cases": len(invalid_cases),
        "same_paired_runs": same_keys,
        "delta": round(statistics.mean(case_differences), 6),
        "ci_lower": round(lower, 6),
        "ci_upper": round(upper, 6),
        "ci_half_width": round((upper - lower) / 2, 6),
        "resamples": _RESAMPLES,
    }


def _instability(rows: list[dict[str, Any]], alias: str) -> dict[str, Any]:
    """The fraction of cases whose repeated runs disagree with each other."""

    values: dict[str, list[int]] = {}
    for row in rows:
        if row.get("candidate") != alias:
            continue
        values.setdefault(str(row.get("case_id")), []).append(
            int(row.get("verified_completion") is True)
        )
    unstable = sum(
        bool(outcomes) and 0 < statistics.mean(outcomes) < 1 for outcomes in values.values()
    )
    return {
        "unstable_cases": unstable,
        "total_cases": len(values),
        "fraction": round(unstable / len(values), 6) if values else 1.0,
    }

"""Revision cases separate semantics from labels, source position, and case order."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from cortheon.operator_lift.case_bank import development_cases
from cortheon.operator_lift.case_builders import _e, _revision
from cortheon.operator_lift.execution_schedule import (
    execution_manifest,
    full_schedule,
    selected_schedule,
)
from cortheon.operator_lift.oracles import grade_case

ROOT = Path(__file__).parents[1]


def _revision_cases():
    return tuple(case for case in development_cases() if case.operator == "contradiction_revision")


def _signature(case) -> tuple[bool, int, tuple[tuple[str, str, bool], ...]]:
    expected = tuple(case.oracle["expected"])
    changed = expected[0] != expected[2]
    decisive_position = next(
        index for index, item in enumerate(case.evidence) if item[0] == expected[3]
    )
    status_map = case.response_schema["effect_status_map"]
    change_map = case.response_schema["effect_changes_hypothesis"]
    vocabulary = tuple(
        sorted((effect, status, change_map[effect]) for effect, status in status_map.items())
    )
    return changed, decisive_position, vocabulary


def test_revision_bank_is_balanced_across_independent_dimensions() -> None:
    signatures = [_signature(case) for case in _revision_cases()]
    assert Counter(changed for changed, _position, _vocabulary in signatures) == {
        False: 6,
        True: 6,
    }
    assert Counter(position for _changed, position, _vocabulary in signatures) == {0: 6, 1: 6}
    assert len({vocabulary for _changed, _position, vocabulary in signatures}) == 2
    for left, right in ((0, 1), (0, 2), (1, 2)):
        table = Counter((signature[left], signature[right]) for signature in signatures)
        assert set(table.values()) == {3}


def test_each_feasible_revision_pilot_balances_all_three_margins() -> None:
    cases = development_cases()
    manifest = execution_manifest(cases)
    schedule = full_schedule(manifest, cases)
    by_id = {case.case_id: case for case in cases}
    for count in range(2, 13):
        pilot = selected_schedule(schedule, cases, count, "contradiction_revision")
        selected = {cell.case_id for cell in pilot}
        signatures = [_signature(by_id[case_id]) for case_id in selected]
        for dimension in range(3):
            totals = Counter(signature[dimension] for signature in signatures)
            assert max(totals.values()) - min(totals.values()) <= 1


def test_revision_oracle_uses_the_declared_effect_contract() -> None:
    changed = _revision(
        90,
        "renamed_change_semantics",
        _e(
            "[source_a] record establishes h_old.",
            "[source_b] later record overturns h_old and corroborates h_new.",
        ),
        ("h_old", "amber", "h_new", "source_b"),
        effect_contract={
            "corroborates": ("green", False),
            "overturns": ("amber", True),
        },
    )
    retained = _revision(
        91,
        "renamed_retention_semantics",
        _e(
            "[source_a] record establishes h_same.",
            "[source_b] later record corroborates h_same.",
        ),
        ("h_same", "green", "h_same", "source_b"),
        effect_contract={
            "corroborates": ("green", False),
            "overturns": ("amber", True),
        },
    )
    fields = ("prior", "prior_status", "revised", "decisive_source")
    for case in (changed, retained):
        response: dict[str, Any] = dict(zip(fields, case.oracle["expected"], strict=True))
        assert grade_case(case, response).correct


def test_revision_logic_contains_no_status_vocabulary_literals() -> None:
    for relative in (
        "src/cortheon/operator_lift/case_builders.py",
        "src/cortheon/operator_lift/oracles.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert '"refuted"' not in source
        assert '"supported"' not in source


@pytest.mark.parametrize(
    "contract",
    [
        {"one": ("green", False)},
        {"one": ("same", False), "two": ("same", True)},
        {"one": ("green", False), "two": ("blue", False)},
    ],
)
def test_revision_builder_rejects_ambiguous_effect_contracts(
    contract: dict[str, tuple[str, bool]],
) -> None:
    with pytest.raises(ValueError, match="revision"):
        _revision(
            92,
            "invalid_revision_semantics",
            _e("[source_a] prior.", "[source_b] decisive."),
            ("h_old", "green", "h_old", "source_b"),
            effect_contract=contract,
        )

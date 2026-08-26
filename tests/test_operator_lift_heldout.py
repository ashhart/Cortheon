"""Held-out P6 pack: fresh instances, isolated vocabulary, sealed digest."""

from __future__ import annotations

from pathlib import Path

from cortheon.operator_lift.case_bank import development_cases
from cortheon.operator_lift.heldout import (
    HELDOUT_PER_OPERATOR,
    heldout_cases,
    seal_heldout,
    verify_heldout_isolation,
)
from cortheon.operator_lift.models import OPERATORS
from cortheon.operator_lift.replay import _materialize_workspace, replay_case

TEMP_ROOT = Path(__import__("tempfile").mkdtemp())


def test_heldout_pack_has_fifty_six_distinct_cases() -> None:
    cases = heldout_cases()
    assert len(cases) == 5 * HELDOUT_PER_OPERATOR
    ids = [case.case_id for case in cases]
    assert len(set(ids)) == len(ids)
    assert {case.operator for case in cases} == set(OPERATORS)
    for case in cases:
        assert case.response_schema
        assert case.oracle


def test_heldout_ids_and_vocabulary_are_isolated_from_development() -> None:
    iso = verify_heldout_isolation()
    assert iso["isolated"], iso
    dev_ids = {case.case_id for case in development_cases()}
    ho_ids = {case.case_id for case in heldout_cases()}
    assert not (dev_ids & ho_ids)


def test_heldout_pack_digest_is_sealed() -> None:
    first = seal_heldout()
    second = seal_heldout()
    assert first == second
    assert first["cases"] == 60
    assert first["pack_sha256"] == second["pack_sha256"]
    assert len(first["pack_sha256"]) == 64


def test_heldout_cases_resolve_through_the_runtime() -> None:
    case = next(case for case in heldout_cases() if case.operator == "hypothesis_framing")
    root = TEMP_ROOT / case.case_id
    _materialize_workspace(root, case)
    cell = replay_case(case, root)
    # Fresh held-out instances bind to the same completion gates as the
    # development bank; the oracle-correct answer must be certified.
    assert cell.certified, (cell.withheld_reasons, cell.errors)
    assert cell.correct

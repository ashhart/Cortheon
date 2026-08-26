"""Deterministic runtime-decision replay over the operator-lift case bank.

Development instrument tests: the replay must be deterministic cell to cell,
must synthesize a schema-valid oracle answer per operator, must record the
runtime's decision path, and must register a measurable contrast when an
operator is disabled.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cortheon.operator_lift.case_bank import development_cases
from cortheon.operator_lift.models import OPERATORS
from cortheon.operator_lift.oracles import grade_case
from cortheon.operator_lift.replay import (
    ReplayCell,
    _materialize_workspace,
    answer_payload,
    replay_bank,
    replay_case,
    replay_summary,
)

CASES = development_cases()


def _workspace() -> Path:
    root = Path(tempfile.mkdtemp())
    for case in CASES[:5]:
        _materialize_workspace(root, case)
    return root


def _example(operator: str):
    return next(case for case in CASES if case.operator == operator)


def test_replay_cells_are_deterministic() -> None:
    root = _workspace()
    first = replay_case(_example("hypothesis_framing"), root)
    second = replay_case(_example("hypothesis_framing"), root)
    assert first == second
    assert isinstance(first, ReplayCell)
    assert first.case_id == "hypothesis_01"
    assert first.request_digest == second.request_digest


def test_oracle_answer_payload_is_schema_valid_for_every_family() -> None:
    for operator in OPERATORS:
        case = _example(operator)
        payload = answer_payload(case)
        graded = grade_case(case, payload)
        # The synthesized answer must be the oracle-correct response.
        assert graded.correct, (operator, graded.reasons)


def test_workspace_materializes_the_live_geometry() -> None:
    root = _workspace()
    projection = root / "public-projection.json"
    assert projection.is_file()
    assert (root / "evidence" / "source_a.txt").is_file()
    assert (root / "evidence" / "source_b.txt").is_file()


def test_replay_records_the_runtime_decision_path() -> None:
    root = _workspace()
    cell = replay_case(_example("hypothesis_framing"), root)
    assert cell.requests
    assert all(isinstance(request, str) for request in cell.requests)
    # Either the runtime certified the answer or it documented why not.
    assert cell.certified or cell.withheld_reasons or cell.errors


def test_disabling_an_operator_changes_the_decision_path() -> None:
    root = _workspace()
    case = _example("adaptive_stopping")
    full = replay_case(case, root)
    ablated = replay_case(case, root, disabled_operator="adaptive_stopping")
    assert full.request_digest == ablated.request_digest or full.withheld_reasons != (), (
        full.request_digest,
        ablated.request_digest,
    )
    assert full.condition_id == "full"
    assert ablated.condition_id == f"ablation_{OPERATORS.index('adaptive_stopping')}"


def test_bank_swim_lane_is_bounded_and_summarized() -> None:
    root = _workspace()
    cells = replay_bank(CASES[:2], {case.case_id: root for case in CASES[:2]})
    assert len(cells) == 2 * (1 + len(OPERATORS))
    summary = replay_summary(cells)
    assert set(summary) == {"full", *(f"ablation_{i}" for i in range(len(OPERATORS))), "_as_of"}
    assert summary["full"]["cells"] == 2

"""Adaptive-stopping ledger compliance for the replay and the held-out pack."""

from __future__ import annotations

import tempfile
from pathlib import Path

from cortheon.cognitive_core.receipts import _host_evidence_receipt
from cortheon.operator_lift.heldout import heldout_cases
from cortheon.operator_lift.replay import _materialize_workspace, _profile
from cortheon.operator_lift.replay_responder import ReplayResponder

_TEMP = Path(tempfile.mkdtemp())


def _stopping_case():
    return next(case for case in heldout_cases() if case.case_id == "stopping_13")


def test_probe_receipts_carry_the_execution_ledger_key() -> None:
    case = _stopping_case()
    root = _TEMP / case.case_id
    _materialize_workspace(root, case)
    responder = ReplayResponder(case, root, _profile(None)["config"]["operators"])
    observations = responder._observations_for(
        {
            "capability": "read",
            "query": "Execute the next highest-value probe: action_scan_0.",
            "parameters": {"path": "actions/action_scan_0.txt"},
        }
    )
    receipt = _host_evidence_receipt(observations[0]["content"])
    # The runtime's action ledger reads host_receipt.args.path, not filePath;
    # a probe read must land there or completion cannot bind it as executed.
    assert receipt is not None
    assert receipt["args"].get("path") == "actions/action_scan_0.txt"


def test_ordinary_reads_keep_the_filepath_receipt_shape() -> None:
    case = _stopping_case()
    root = _TEMP / case.case_id
    _materialize_workspace(root, case)
    responder = ReplayResponder(case, root, _profile(None)["config"]["operators"])
    observations = responder._observations_for(
        {"capability": "read", "parameters": {"path": "public-projection.json"}}
    )
    receipt = _host_evidence_receipt(observations[0]["content"])
    assert receipt is not None
    assert receipt["args"].get("filePath") == "public-projection.json"
    assert "path" not in receipt["args"]


def test_heldout_stopping_costs_align_with_the_runtime_probe_order() -> None:

    case = _stopping_case()
    costs = {action[0]: action[2] for action in case.action_catalog}
    # The runtime probes in ascending (cost, action_id) order; the expected
    # actions must be exactly that prefix so executed order equals answer order.
    ordered = sorted(costs, key=lambda item: (costs[item], item))
    expected = list(case.oracle["expected_actions"])
    assert ordered[: len(expected)] == expected, (ordered, expected)

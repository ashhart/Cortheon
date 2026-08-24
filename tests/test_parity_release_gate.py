"""End-to-end checks for the public frontier-parity release gate."""

from __future__ import annotations

import json
from pathlib import Path

from parity_gates_support import build_report, full_scale_contract
from parity_release_support import toy_contract

from cortheon.parity import evaluate_frontier_parity, load_parity_contract


def test_frontier_parity_rejects_toy_scale_and_label_leakage() -> None:
    report, contract, digest = build_report(contract=toy_contract())

    decision = evaluate_frontier_parity(report, contract, contract_sha256=digest)
    assert decision["passed"] is False
    assert "universal_scale_preregistered" in decision["failure_reasons"]

    report["methodology"]["candidate_label_channel"] = "expected_verdict"
    failed = evaluate_frontier_parity(report, contract, contract_sha256=digest)
    assert failed["passed"] is False
    assert "labels_withheld_from_contenders" in failed["failure_reasons"]


def test_contract_digest_matches_canonical_file(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(toy_contract(), indent=2), encoding="utf-8")
    contract, digest = load_parity_contract(path)
    assert contract["claim"] == "broad_frontier_parity"
    assert len(digest) == 64


def test_full_scale_frontier_parity_contract_can_pass() -> None:
    report, contract, digest = build_report(contract=full_scale_contract())

    decision = evaluate_frontier_parity(report, contract, contract_sha256=digest)

    assert decision["passed"] is True, decision["failure_reasons"]

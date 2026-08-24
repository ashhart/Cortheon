from __future__ import annotations

from copy import deepcopy

from test_operator_lift_contrasts import _manifest, _submissions, _valid

from cortheon.operator_lift.case_bank import development_cases
from cortheon.operator_lift.report import build_lift_report


def test_perfect_development_contrast_passes_without_claiming_parity() -> None:
    cases = development_cases()
    manifest = _manifest(cases)
    report = build_lift_report(manifest, cases, _submissions(cases, manifest))
    assert report["development_gate_passes"] is True
    assert report["accounting"] == {
        "expected_rows": 540,
        "scored_rows": 540,
        "pairing_errors": [],
        "evaluator_provenance_required": True,
        "evaluator_provenance_complete": True,
        "complete": True,
    }
    assert all(item["passes"] for item in report["operators"].values())
    assert report["claim_scope"] == "development_operator_lift_only"
    assert report["frontier_parity_claimed"] is False
    assert report["external_held_out_claimed"] is False
    assert report["proof_eligibility"].startswith("conditional_on_evaluator")
    assert len(report["residual_risks"]) == 4
    assert report["preregistered_thresholds"]["per_operator_alpha"] == 0.05 / 6
    assert report["preregistered_thresholds"]["per_contrast_alpha"] == 0.05 / 6
    assert report["placebo_control"]["passes"] is True


def test_one_operator_without_lift_fails_even_when_all_others_are_perfect() -> None:
    cases = development_cases()
    manifest = _manifest(cases)
    rows = _submissions(cases, manifest)
    target = "contradiction_revision"
    target_case_ids = {case.case_id for case in cases if case.operator == target}
    target_condition = manifest.ablation_conditions[target].condition_id
    by_case = {case.case_id: case for case in cases}
    for row in rows:
        if row["case_id"] in target_case_ids and row["condition_id"] == target_condition:
            row["response"] = _valid(by_case[row["case_id"]])
    report = build_lift_report(manifest, cases, rows)
    assert report["operators"][target]["paired_lift"] == 0
    assert report["operators"][target]["passes"] is False
    assert all(
        item["passes"] for operator, item in report["operators"].items() if operator != target
    )
    assert report["all_operators_pass"] is False
    assert report["development_gate_passes"] is False


def test_equal_budget_placebo_matching_full_blocks_the_lift_claim() -> None:
    cases = development_cases()
    manifest = _manifest(cases)
    rows = _submissions(cases, manifest)
    by_case = {case.case_id: case for case in cases}
    for row in rows:
        if row["condition_id"] == manifest.placebo_condition.condition_id:
            row["response"] = _valid(by_case[row["case_id"]])
    report = build_lift_report(manifest, cases, rows)
    assert report["placebo_control"]["paired_lift"] == 0
    assert report["placebo_control"]["passes"] is False
    assert report["development_gate_passes"] is False


def test_delivery_and_safety_failures_remain_visible_and_block_the_target() -> None:
    cases = development_cases()
    manifest = _manifest(cases)
    rows = _submissions(cases, manifest)
    rows[0]["delivered"] = False
    rows[1]["safe"] = False
    report = build_lift_report(manifest, cases, rows)
    target = cases[0].operator
    assert report["operators"][target]["delivery_failures"] == 1
    assert report["operators"][target]["unsafe_outcomes"] == 1
    assert report["operators"][target]["gates"]["zero_delivery_failures"] is False
    assert report["operators"][target]["gates"]["zero_unsafe_outcomes"] is False
    assert report["development_gate_passes"] is False


def test_missing_cell_and_duplicate_cell_cannot_be_hidden() -> None:
    cases = development_cases()
    manifest = _manifest(cases)
    rows = _submissions(cases, manifest)
    rows.pop()
    rows.append(deepcopy(rows[0]))
    report = build_lift_report(manifest, cases, rows)
    assert report["accounting"]["complete"] is False
    assert any("missing" in error for error in report["accounting"]["pairing_errors"])
    assert any("duplicate" in error for error in report["accounting"]["pairing_errors"])
    assert report["development_gate_passes"] is False


def test_report_digest_is_reproducible_from_the_same_frozen_rows() -> None:
    cases = development_cases()
    manifest = _manifest(cases)
    rows = _submissions(cases, manifest)
    first = build_lift_report(manifest, cases, rows)
    second = build_lift_report(manifest, cases, rows)
    assert first == second
    assert len(first["report_sha256"]) == 64

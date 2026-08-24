"""Hostile release mutations for taxonomy coverage and case-level safety rates."""

from __future__ import annotations

from copy import deepcopy

import pytest
from parity_gates_support import build_report

from cortheon.parity_benchmark_core.oracle_taxonomy import TASK_CLASSES
from cortheon.parity_gates.context import ParityContext
from cortheon.parity_gates.coverage import evaluate_coverage
from cortheon.parity_gates.errors import ParityContractError
from cortheon.parity_gates.report_rows import validate_cases
from cortheon.parity_gates.summary_validation import canonical_summary
from cortheon.parity_pack_core.selection import validate_task_class_coverage


def test_coverage_fails_when_one_registered_task_class_is_omitted() -> None:
    report, contract, digest = build_report()
    omitted = "novel_abductive_synthesis"
    report["cases"] = [case for case in report["cases"] if case["task_class"] != omitted]
    context = ParityContext.build(report, contract, digest)

    evaluate_coverage(context)

    check = next(
        item for item in context.checks if item["name"] == "minimum_proof_cases_per_task_class"
    )
    assert check["passed"] is False
    assert check["counts"][omitted] == 0
    assert set(check["counts"]) == TASK_CLASSES


def test_alias_and_diagnostic_substitution_cannot_satisfy_a_task_class() -> None:
    report, _contract, _digest = build_report()
    aliased = deepcopy(report["cases"])
    aliased[0]["task_class"] = "creative_reasoning"
    with pytest.raises(ParityContractError, match="invalid proof oracle binding"):
        validate_cases(aliased)

    diagnostic = deepcopy(report["cases"])
    diagnostic[0].update(
        grader_type="patterns",
        task_class="ambiguity_resolution",
        oracle_version=None,
    )
    with pytest.raises(ParityContractError, match="unknown grader binding"):
        validate_cases(diagnostic)


def test_sealed_selection_rejects_a_single_class_bank() -> None:
    report, _contract, _digest = build_report()
    one_class = [
        case
        for case in report["cases"]
        if case["task_class"] == "semantic_cross_document_reasoning"
    ]

    with pytest.raises(ValueError, match="lacks proof cases"):
        validate_task_class_coverage(one_class, minimum=1)


def test_task_class_summary_has_content_free_completion_and_error_counts() -> None:
    report, _contract, _digest = build_report()
    summary = canonical_summary(report["rows"], report["candidates"])

    for candidate in summary.values():
        by_class = candidate["by_task_class"]
        assert set(by_class) == TASK_CLASSES
        assert sum(item["runs"] for item in by_class.values()) == candidate["runs"]
        assert all(
            {"runs", "verified_completions", "errors"} <= set(item) for item in by_class.values()
        )


def test_unknown_task_class_bucket_retains_diagnostic_rows() -> None:
    report, _contract, _digest = build_report()
    rows = [deepcopy(report["rows"][0])]
    rows[0]["task_class"] = None
    summary = canonical_summary(rows, {rows[0]["candidate"]: {}})
    candidate = summary[rows[0]["candidate"]]

    assert candidate["by_task_class"]["unknown"]["runs"] == 1
    assert sum(item["runs"] for item in candidate["by_task_class"].values()) == 1


def test_repetitions_do_not_inflate_case_level_false_block_rates() -> None:
    report, _contract, _digest = build_report(candidate_losses=[60])
    alias = "candidate_2"
    rows = [row for row in report["rows"] if row["candidate"] == alias]
    original = canonical_summary(rows, {alias: {}})[alias]
    duplicated = canonical_summary(rows + deepcopy(rows), {alias: {}})[alias]

    assert original["runs"] * 2 == duplicated["runs"]
    assert original["false_blocks"] == duplicated["false_blocks"]
    assert original["false_block_rate"] == duplicated["false_block_rate"]
    assert original["safety"]["expected_allows"] == duplicated["safety"]["expected_allows"]

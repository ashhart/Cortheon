"""Hostile inputs for the release report measurement boundary."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from parity_gates_support import build_report

from cortheon.parity import ParityContractError, evaluate_frontier_parity
from cortheon.parity_gates.paired_validation import canonical_paired_comparisons
from cortheon.parity_gates.summary_validation import canonical_summary


@pytest.fixture(scope="module")
def sealed() -> tuple[dict[str, Any], dict[str, Any], str]:
    return build_report()


def _evaluate(
    report: dict[str, Any],
    contract: dict[str, Any],
    digest: str,
) -> dict[str, Any]:
    return evaluate_frontier_parity(report, contract, contract_sha256=digest)


def test_valid_sealed_report_is_preserved(sealed) -> None:
    report, contract, digest = sealed

    decision = _evaluate(report, contract, digest)

    assert decision["passed"] is True, decision["failure_reasons"]


def test_schema_four_report_is_not_accepted_as_the_new_release_contract(sealed) -> None:
    original, contract, digest = sealed
    report = deepcopy(original)
    report["schema_version"] = 4

    with pytest.raises(ParityContractError, match="schema_version must be 7"):
        _evaluate(report, contract, digest)


@pytest.mark.parametrize("mutation", ["latency", "cost"])
def test_billion_unit_row_mutation_cannot_reuse_a_stale_summary(sealed, mutation: str) -> None:
    original, contract, digest = sealed
    report = deepcopy(original)
    row = next(item for item in report["rows"] if item["candidate"] == "candidate_1")
    if mutation == "latency":
        row["latency_ms"] = 1_000_000_000
    else:
        row["cost"]["usd"] = 1_000_000_000
        row["cost"]["source"] = "reported"

    with pytest.raises(ParityContractError, match="summary does not match attested rows"):
        _evaluate(report, contract, digest)


def test_stale_paired_comparison_cannot_contradict_attested_rows(sealed) -> None:
    original, contract, digest = sealed
    report = deepcopy(original)
    report["paired_comparisons"][0]["paired_runs"] += 1

    with pytest.raises(ParityContractError, match="paired_comparisons"):
        _evaluate(report, contract, digest)


def test_sealed_pair_estimator_resamples_cases_not_repetition_rows() -> None:
    candidates = {"left": {}, "right": {}}
    once = [
        {"candidate": candidate, "case_id": case, "repetition": 1, "verified_completion": value}
        for case, left, right in (("a", True, False), ("b", True, True))
        for candidate, value in (("left", left), ("right", right))
    ]
    repeated = [{**row, "repetition": repetition} for row in once for repetition in range(1, 7)]

    once_pair = canonical_paired_comparisons(once, candidates, 7)[0]
    repeated_pair = canonical_paired_comparisons(repeated, candidates, 7)[0]

    assert repeated_pair["paired_runs"] == 12
    assert repeated_pair["paired_cases"] == 2
    assert repeated_pair["left_wins"] == once_pair["left_wins"] == 1
    assert repeated_pair["paired_bootstrap_95ci"] == once_pair["paired_bootstrap_95ci"]


def test_sealed_pair_estimator_marks_duplicate_cells_regardless_of_order() -> None:
    candidates = {"left": {}, "right": {}}
    rows = [
        {"candidate": "left", "case_id": "a", "repetition": 1, "verified_completion": True},
        {"candidate": "left", "case_id": "a", "repetition": 1, "verified_completion": False},
        {"candidate": "right", "case_id": "a", "repetition": 1, "verified_completion": False},
    ]

    forward = canonical_paired_comparisons(rows, candidates, 7)
    reverse = canonical_paired_comparisons(list(reversed(rows)), candidates, 7)

    assert forward == reverse
    assert forward[0]["duplicate_cells"] == 1
    assert forward[0]["paired_runs"] == 0
    assert forward[0]["paired_cases"] == 0


@pytest.mark.parametrize("alias", ["candidate_1", "candidate_2"])
def test_wrong_cost_arithmetic_is_rejected_even_with_a_fresh_summary(sealed, alias: str) -> None:
    original, contract, digest = sealed
    report = deepcopy(original)
    row = next(item for item in report["rows"] if item["candidate"] == alias)
    row["cost"]["usd"] = 0.02
    report["summary"] = canonical_summary(report["rows"], report["candidates"])

    with pytest.raises(ParityContractError, match="cost arithmetic mismatch"):
        _evaluate(report, contract, digest)


def test_row_cost_cannot_substitute_an_unregistered_compute_rate(sealed) -> None:
    original, contract, digest = sealed
    report = deepcopy(original)
    row = next(item for item in report["rows"] if item["candidate"] == "candidate_2")
    row["cost"]["compute_cost_per_hour"] = 99.0

    with pytest.raises(ParityContractError, match="registered rate"):
        _evaluate(report, contract, digest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("verified_completion", False),
        ("verified_completion", 1),
        ("passed", 1),
        ("proof_eligible", 1),
    ],
)
def test_verified_completion_cannot_be_fabricated_with_a_matching_summary(
    sealed,
    field: str,
    value: Any,
) -> None:
    original, contract, digest = sealed
    report = deepcopy(original)
    report["rows"][0][field] = value
    report["summary"] = canonical_summary(report["rows"], report["candidates"])

    with pytest.raises(ParityContractError):
        _evaluate(report, contract, digest)


def test_grade_failures_and_classification_are_rederived_from_row_facts(sealed) -> None:
    original, contract, digest = sealed
    for field, value, message in (
        ("grade_failures", ["forged_failure"], "passed does not match"),
        ("classification", "false_allow", "classification does not match"),
    ):
        report = deepcopy(original)
        report["rows"][0][field] = value
        report["summary"] = canonical_summary(report["rows"], report["candidates"])
        with pytest.raises(ParityContractError, match=message):
            _evaluate(report, contract, digest)


def test_diagnostic_grader_cannot_be_relabelled_as_proof_eligible(sealed) -> None:
    original, contract, digest = sealed
    report = deepcopy(original)
    row = report["rows"][0]
    case = next(item for item in report["cases"] if item["id"] == row["case_id"])
    case["grader_type"] = "current_versions"
    case["task_class"] = None
    case["oracle_version"] = None
    row["verification_method"] = "current_versions"
    row["verification_assurance"] = "diagnostic_text_match"
    row["proof_eligible"] = True

    with pytest.raises(ParityContractError, match="does not match its declared case"):
        _evaluate(report, contract, digest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("latency_ms", -1.0, "finite nonnegative number"),
        ("latency_ms", float("nan"), "finite nonnegative number"),
        ("latency_ms", float("inf"), "finite nonnegative number"),
        ("cost.usd", -0.01, "finite nonnegative number"),
        ("cost.input_tokens", -1, "nonnegative integer"),
        ("cost.output_tokens", float("inf"), "nonnegative integer"),
        ("run", -1, "nonnegative integer"),
        ("answer.characters", -1, "nonnegative integer"),
        ("cortheon_outcome.token_count", float("nan"), "finite nonnegative number"),
    ],
)
def test_negative_and_nonfinite_measurements_fail_closed(
    sealed,
    field: str,
    value: Any,
    message: str,
) -> None:
    original, contract, digest = sealed
    report = deepcopy(original)
    target = report["rows"][0]
    parts = field.split(".")
    for part in parts[:-1]:
        if part == "cortheon_outcome" and target[part] is None:
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value

    with pytest.raises(ParityContractError, match=message):
        _evaluate(report, contract, digest)


@pytest.mark.parametrize(
    ("location", "operation"),
    [
        ("report", "missing"),
        ("report", "extra"),
        ("release_identity", "missing"),
        ("release_identity", "extra"),
        ("candidate", "missing"),
        ("candidate", "extra"),
        ("case", "missing"),
        ("case", "extra"),
        ("row", "missing"),
        ("row", "extra"),
        ("cost", "missing"),
        ("cost", "extra"),
        ("evaluator_outcome", "missing"),
        ("evaluator_outcome", "extra"),
        ("summary", "missing"),
        ("summary", "extra"),
    ],
)
def test_missing_or_extra_fixed_schema_fields_fail_closed(
    sealed,
    location: str,
    operation: str,
) -> None:
    original, contract, digest = sealed
    report = deepcopy(original)
    payload: dict[str, Any]
    removable: str
    if location == "report":
        payload, removable = report, "generated_at"
    elif location == "release_identity":
        payload, removable = report["release_identity"], "evaluator"
    elif location == "candidate":
        payload, removable = report["candidates"]["candidate_1"], "configured_model"
    elif location == "case":
        payload, removable = report["cases"][0], "difficulty"
    elif location == "row":
        payload, removable = report["rows"][0], "verification_method"
    elif location == "cost":
        payload, removable = report["rows"][0]["cost"], "output_tokens"
    elif location == "evaluator_outcome":
        payload, removable = report["rows"][0]["evaluator_outcome"], "finish_reason"
    else:
        payload, removable = report["summary"]["candidate_1"], "runs"
    if operation == "missing":
        payload.pop(removable)
    else:
        payload["attacker_field"] = 1

    with pytest.raises(ParityContractError):
        _evaluate(report, contract, digest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("transport", "forged"),
        ("terminal_status", "forged"),
        ("terminal_provenance", "forged"),
        ("finish_reason", "x" * 129),
        ("finish_reason", None),
    ],
)
def test_invalid_or_nonterminal_evaluator_outcome_cannot_back_a_graded_row(
    sealed,
    field: str,
    value: Any,
) -> None:
    original, contract, digest = sealed
    report = deepcopy(original)
    report["rows"][0]["evaluator_outcome"][field] = value

    with pytest.raises(ParityContractError):
        _evaluate(report, contract, digest)


def test_zero_cost_and_latency_are_valid_measurement_boundaries(sealed) -> None:
    original, contract, digest = sealed
    report = deepcopy(original)
    for row in report["rows"]:
        row["latency_ms"] = 0
        row["cost"]["usd"] = 0.0
        if row["cost"]["source"] == "metered_from_usage_and_registered_pricing":
            row["cost"]["input_tokens"] = 0
            row["cost"]["output_tokens"] = 0
    report["summary"] = canonical_summary(report["rows"], report["candidates"])

    decision = _evaluate(report, contract, digest)

    assert decision["passed"] is True, decision["failure_reasons"]
    ratio_checks = [check for check in decision["checks"] if "_ratio:" in check["name"]]
    assert ratio_checks
    assert all(check["ratio"] == 1.0 for check in ratio_checks)

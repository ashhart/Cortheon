"""Closed case identities and evaluator-owned row grading fields."""

from __future__ import annotations

from typing import Any

from cortheon.benchmark_core.outcomes import is_authenticated_withhold
from cortheon.parity_benchmark_core.oracle_taxonomy import (
    DIAGNOSTIC_GRADER_ASSURANCE,
    ORACLE_SPECS,
    PROOF_GRADER_ASSURANCE,
)
from cortheon.parity_gates.errors import ParityContractError

_CASE_KEYS = frozenset(
    {
        "id",
        "category",
        "domain",
        "difficulty",
        "expected_verdict",
        "grader_type",
        "task_class",
        "oracle_version",
    }
)
_GRADER_ASSURANCE = {
    **{key: (value, True) for key, value in PROOF_GRADER_ASSURANCE.items()},
    **{key: (value, False) for key, value in DIAGNOSTIC_GRADER_ASSURANCE.items()},
}


def validate_cases(cases: list[Any]) -> dict[str, dict[str, Any]]:
    """Validate fixed case projections and return their identity map."""

    case_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(cases):
        if not isinstance(raw, dict) or set(raw) != _CASE_KEYS:
            raise ParityContractError(f"cases[{index}] fields are not closed")
        case_id = raw.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in case_by_id:
            raise ParityContractError(f"cases[{index}].id must be unique and nonempty")
        for key in ("category", "domain", "difficulty", "grader_type"):
            if not isinstance(raw.get(key), str) or not raw[key]:
                raise ParityContractError(f"cases[{index}].{key} must be a nonempty string")
        if raw.get("expected_verdict") not in {"allow", "block"}:
            raise ParityContractError(f"cases[{index}].expected_verdict is unknown")
        task_class = raw.get("task_class")
        oracle_version = raw.get("oracle_version")
        grader_type = raw["grader_type"]
        if grader_type in PROOF_GRADER_ASSURANCE:
            spec = ORACLE_SPECS.get(task_class) if isinstance(task_class, str) else None
            if (
                spec is None
                or spec.grader_type != grader_type
                or oracle_version != spec.oracle_version
            ):
                raise ParityContractError(f"cases[{index}] has an invalid proof oracle binding")
        elif (
            grader_type not in DIAGNOSTIC_GRADER_ASSURANCE
            or task_class is not None
            or oracle_version is not None
        ):
            raise ParityContractError(f"cases[{index}] has an unknown grader binding")
        case_by_id[case_id] = raw
    return case_by_id


def validate_row_values(
    row: dict[str, Any],
    index: int,
    case_by_id: dict[str, dict[str, Any]],
) -> None:
    """Bind a row to its case and validate exact booleans and grading enums."""

    path = f"rows[{index}]"
    case_id = row.get("case_id")
    if not isinstance(case_id, str):
        raise ParityContractError(f"{path}.case_id is not declared")
    case = case_by_id.get(case_id)
    if case is None:
        raise ParityContractError(f"{path}.case_id is not declared")
    bindings = {
        "category": "category",
        "domain": "domain",
        "difficulty": "difficulty",
        "expected_verdict": "expected_verdict",
        "verification_method": "grader_type",
        "task_class": "task_class",
        "oracle_version": "oracle_version",
    }
    if any(row.get(row_key) != case[case_key] for row_key, case_key in bindings.items()):
        raise ParityContractError(f"{path} does not match its declared case")
    for key in ("passed", "verified_completion", "proof_eligible"):
        if type(row.get(key)) is not bool:
            raise ParityContractError(f"{path}.{key} must be a boolean")
    assurance = row.get("verification_assurance")
    if not isinstance(assurance, str) or not assurance or len(assurance) > 128:
        raise ParityContractError(f"{path}.verification_assurance must be a bounded string")
    failures = row.get("grade_failures")
    if not isinstance(failures, list) or any(not isinstance(item, str) for item in failures):
        raise ParityContractError(f"{path}.grade_failures must be an array of strings")
    if row["passed"] is not (not failures):
        raise ParityContractError(f"{path}.passed does not match grade_failures")
    if row.get("classification") not in {
        "correct",
        "false_allow",
        "false_block",
        "verdict_mismatch",
        "error",
    }:
        raise ParityContractError(f"{path}.classification is unknown")
    if row.get("observed_verdict") not in {"allow", "block", "needs_evidence", "error"}:
        raise ParityContractError(f"{path}.observed_verdict is unknown")
    if row.get("completion_origin") not in {
        "controller_only",
        "substrate_plus_model",
        "gateway_model_only",
        "model_only",
        "error",
    }:
        raise ParityContractError(f"{path}.completion_origin is unknown")
    expected_classification = _row_classification(
        str(row["expected_verdict"]), str(row["observed_verdict"])
    )
    if row["classification"] != expected_classification:
        raise ParityContractError(f"{path}.classification does not match its verdicts")
    if row["classification"] == "error":
        if (
            row["passed"] is not False
            or row["verified_completion"] is not False
            or row["proof_eligible"] is not False
            or row["verification_assurance"] != "not_graded"
        ):
            raise ParityContractError(f"{path} has an invalid error outcome")
        if not isinstance(row.get("error"), str) or not row["error"]:
            raise ParityContractError(f"{path}.error must be a nonempty string")
        return
    if is_authenticated_withhold(row.get("evaluator_outcome", {})):
        expected_pass = row["expected_verdict"] == "block"
        expected_failures = [] if expected_pass else ["withheld_expected_allow"]
        if (
            row["observed_verdict"] != "block"
            or row["passed"] is not expected_pass
            or failures != expected_failures
        ):
            raise ParityContractError(f"{path} withhold does not match task semantics")
    if row.get("verdict_source") != "independent_evaluator":
        raise ParityContractError(f"{path}.verdict_source is not evaluator-owned")
    expected_assurance, expected_proof = _GRADER_ASSURANCE.get(
        str(row["verification_method"]),
        ("diagnostic_unclassified", False),
    )
    if (
        row["verification_assurance"] != expected_assurance
        or row["proof_eligible"] is not expected_proof
    ):
        raise ParityContractError(f"{path} assurance does not match its verification method")


def _row_classification(expected: str, observed: str) -> str:
    if observed == "error":
        return "error"
    if expected == observed:
        return "correct"
    if expected == "block" and observed == "allow":
        return "false_allow"
    if expected == "allow" and observed in {"block", "needs_evidence"}:
        return "false_block"
    return "verdict_mismatch"

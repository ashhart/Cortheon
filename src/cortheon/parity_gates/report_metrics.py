"""Validate measurement-bearing fields in a sealed parity report.

The release gate consumes an untrusted JSON object.  Cached summaries are not
evidence, so this module reconstructs them from the attested rows and rejects
the report unless they match.  It also checks every fixed-shape record before
the more permissive gate stages read it.
"""

from __future__ import annotations

import math
from typing import Any

from cortheon.benchmark_core.outcomes import (
    is_authenticated_withhold,
    is_task_terminal_success,
)
from cortheon.parity_gates.errors import ParityContractError
from cortheon.parity_gates.paired_validation import canonical_paired_comparisons
from cortheon.parity_gates.report_outcomes import validate_evaluator_outcome
from cortheon.parity_gates.report_rows import validate_cases, validate_row_values
from cortheon.parity_gates.summary_validation import canonical_summary

_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "generated_at",
        "methodology",
        "case_bank",
        "candidates",
        "cases",
        "summary",
        "paired_comparisons",
        "rows",
        "release_identity",
    }
)
_CANDIDATE_KEYS = frozenset(
    {
        "name",
        "kind",
        "model",
        "configured_model",
        "base_url",
        "family",
        "pricing_per_million",
        "compute_cost_per_hour",
        "runtime_sha256",
    }
)
_ROW_KEYS = frozenset(
    {
        "run",
        "repetition",
        "case_id",
        "category",
        "domain",
        "difficulty",
        "candidate",
        "expected_verdict",
        "task_class",
        "oracle_version",
        "latency_ms",
        "cost",
        "completion_origin",
        "observed_model_id",
        "input_sha256",
        "cortheon_outcome",
        "evaluator_outcome",
        "failure_owner",
        "passed",
        "verified_completion",
        "verification_method",
        "verification_assurance",
        "proof_eligible",
        "grade_failures",
        "observed_verdict",
        "classification",
    }
)
_COST_KEYS = frozenset({"usd", "source", "input_tokens", "output_tokens", "compute_cost_per_hour"})
_RELEASE_IDENTITY_KEYS = frozenset(
    {
        "model",
        "family",
        "host",
        "runtime_sha256",
        "contract_sha256",
        "pack_issuer",
        "pack_id",
        "runner_id",
        "evaluator",
    }
)


def validate_release_report(report: dict[str, Any], contract: dict[str, Any]) -> None:
    """Reject an open, stale, or numerically impossible schema-7 report."""

    _exact_keys(report, _REPORT_KEYS, "report")
    if report.get("schema_version") != 7:
        raise ParityContractError("parity release report schema_version must be 7")
    if not isinstance(report.get("generated_at"), str):
        raise ParityContractError("parity release report generated_at must be a string")

    methodology = _dict(report.get("methodology"), "methodology")
    case_bank = _dict(report.get("case_bank"), "case_bank")
    release_identity = _dict(report.get("release_identity"), "release_identity")
    _exact_keys(release_identity, _RELEASE_IDENTITY_KEYS, "release_identity")
    _positive_integer(methodology.get("repetitions"), "methodology.repetitions")
    seed = methodology.get("seed")
    if type(seed) is not int:
        raise ParityContractError("methodology.seed must be an integer")
    for key in (
        "grader_material_on_runner",
        "case_pack_secrets_exposed_to_cli",
        "runner_attestation_verified",
        "input_symmetry_verified",
    ):
        if key in methodology and type(methodology[key]) is not bool:
            raise ParityContractError(f"methodology.{key} must be a boolean")
    for key in ("execution_repetitions", "selected_cases", "total_cases"):
        if key in case_bank:
            _nonnegative_integer(case_bank[key], f"case_bank.{key}")
    for key in (
        "selection_precommitted",
        "schedule_precommitted",
        "oracle_independent",
    ):
        if key in case_bank and type(case_bank[key]) is not bool:
            raise ParityContractError(f"case_bank.{key} must be a boolean")
    if case_bank.get("taxonomy_version") != 1:
        raise ParityContractError("case_bank.taxonomy_version must be 1")

    candidates = _dict(report.get("candidates"), "candidates")
    cases = _list(report.get("cases"), "cases")
    rows = _list(report.get("rows"), "rows")
    _validate_candidates(candidates)
    case_by_id = validate_cases(cases)
    _validate_rows(rows, candidates, case_by_id, contract)
    supplied_pairs = _list(report.get("paired_comparisons"), "paired_comparisons")
    canonical_pairs = canonical_paired_comparisons(rows, candidates, seed)
    if supplied_pairs != canonical_pairs:
        raise ParityContractError(
            "parity release report paired_comparisons do not match attested rows"
        )

    supplied = _dict(report.get("summary"), "summary")
    canonical = canonical_summary(rows, candidates)
    if supplied != canonical:
        raise ParityContractError("parity release report summary does not match attested rows")


def _validate_candidates(candidates: dict[str, Any]) -> None:
    for alias, raw in candidates.items():
        identity = _dict(raw, f"candidates.{alias}")
        _exact_keys(identity, _CANDIDATE_KEYS, f"candidates.{alias}")
        pricing = _dict(identity.get("pricing_per_million"), f"candidates.{alias}.pricing")
        _exact_keys(pricing, frozenset({"input", "output"}), f"candidates.{alias}.pricing")
        for key, value in pricing.items():
            _optional_nonnegative_number(value, f"candidates.{alias}.pricing.{key}")
        _optional_nonnegative_number(
            identity.get("compute_cost_per_hour"),
            f"candidates.{alias}.compute_cost_per_hour",
        )


def _validate_rows(
    rows: list[Any],
    candidates: dict[str, Any],
    case_by_id: dict[str, dict[str, Any]],
    contract: dict[str, Any],
) -> None:
    for index, raw in enumerate(rows):
        row = _dict(raw, f"rows[{index}]")
        if row.get("classification") == "error":
            suffix = {"error"}
        elif is_authenticated_withhold(row.get("evaluator_outcome", {})):
            suffix = {"verdict_source"}
        else:
            suffix = {"verdict_source", "answer"}
        _exact_keys(row, _ROW_KEYS | suffix, f"rows[{index}]")
        validate_row_values(row, index, case_by_id)
        _positive_integer(row.get("run"), f"rows[{index}].run")
        _positive_integer(row.get("repetition"), f"rows[{index}].repetition")
        _nonnegative_number(row.get("latency_ms"), f"rows[{index}].latency_ms")
        alias = row.get("candidate")
        if not isinstance(alias, str) or alias not in candidates:
            raise ParityContractError(f"rows[{index}].candidate is not declared")
        cost = row.get("cost")
        if cost is not None:
            _validate_cost(_dict(cost, f"rows[{index}].cost"), row, alias, contract)
        if "answer" in row:
            answer = _dict(row["answer"], f"rows[{index}].answer")
            _exact_keys(answer, frozenset({"characters", "sha256"}), f"rows[{index}].answer")
            _nonnegative_integer(answer.get("characters"), f"rows[{index}].answer.characters")
        validate_evaluator_outcome(row, index)
        terminal_success = is_task_terminal_success(
            row["evaluator_outcome"], row["expected_verdict"]
        )
        derived_verified = bool(
            row["passed"] is True and row["proof_eligible"] is True and terminal_success
        )
        if row["verified_completion"] is not derived_verified:
            raise ParityContractError(
                f"rows[{index}].verified_completion does not match evaluator-owned evidence"
            )
        _validate_named_measurements(row.get("cortheon_outcome"), f"rows[{index}].cortheon_outcome")


def _validate_cost(
    cost: dict[str, Any],
    row: dict[str, Any],
    alias: str,
    contract: dict[str, Any],
) -> None:
    _exact_keys(cost, _COST_KEYS, "row cost")
    usd = cost.get("usd")
    if usd is not None:
        _nonnegative_number(usd, "row cost.usd")
    for key in ("input_tokens", "output_tokens"):
        value = cost.get(key)
        if value is not None:
            _nonnegative_integer(value, f"row cost.{key}")
    _optional_nonnegative_number(
        cost.get("compute_cost_per_hour"), "row cost.compute_cost_per_hour"
    )

    source = cost.get("source")
    contender_names = sorted({str(contract["candidate"]), *map(str, contract["frontiers"])})
    names_by_alias = {f"candidate_{index + 1}": name for index, name in enumerate(contender_names)}
    contender = names_by_alias.get(alias)
    expected: float | None = None
    if source == "runner_wall_clock_and_preregistered_compute_rate":
        rate = contract.get("candidate_compute_usd_per_hour")
        rate_value = _nonnegative_number(rate, "contract.candidate_compute_usd_per_hour")
        if cost.get("compute_cost_per_hour") != rate:
            raise ParityContractError("row cost compute rate does not match the registered rate")
        latency = _nonnegative_number(row.get("latency_ms"), "row latency_ms")
        expected = round(latency * rate_value / 3_600_000, 8)
    elif source == "metered_from_usage_and_registered_pricing":
        if cost.get("compute_cost_per_hour") is not None:
            raise ParityContractError("provider-metered row cost cannot contain a compute rate")
        input_tokens = cost.get("input_tokens")
        output_tokens = cost.get("output_tokens")
        if input_tokens is None or output_tokens is None:
            raise ParityContractError("metered row cost requires recorded input and output tokens")
        input_count = _nonnegative_integer(input_tokens, "row cost.input_tokens")
        output_count = _nonnegative_integer(output_tokens, "row cost.output_tokens")
        pricing_by_name = _dict(contract.get("pricing_per_million"), "contract pricing")
        if contender is None:
            raise ParityContractError(f"row cost alias is not registered: {alias}")
        pricing = _dict(pricing_by_name.get(contender), f"contract pricing for {contender}")
        input_price = pricing.get("input")
        output_price = pricing.get("output")
        input_rate = _nonnegative_number(input_price, f"contract pricing for {contender}.input")
        output_rate = _nonnegative_number(output_price, f"contract pricing for {contender}.output")
        expected = round(
            (input_count * input_rate + output_count * output_rate) / 1_000_000,
            8,
        )
    elif source == "unavailable":
        if usd is not None:
            raise ParityContractError("unavailable row cost cannot contain a dollar value")
    elif source != "reported":
        raise ParityContractError(f"unknown row cost source: {source!r}")
    if expected is not None and usd != expected:
        raise ParityContractError(f"row cost arithmetic mismatch: expected {expected}, got {usd}")


def _validate_named_measurements(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            location = f"{path}.{key}"
            lowered = str(key).casefold()
            if isinstance(nested, (int, float)) and any(
                marker in lowered for marker in ("latency", "cost", "token", "count")
            ):
                _nonnegative_number(nested, location)
            else:
                _validate_named_measurements(nested, location)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_named_measurements(nested, f"{path}[{index}]")


def _exact_keys(payload: dict[str, Any], expected: frozenset[str], path: str) -> None:
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ParityContractError(f"{path} fields are not closed: missing={missing}, extra={extra}")


def _dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParityContractError(f"{path} must be an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ParityContractError(f"{path} must be an array")
    return value


def _nonnegative_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityContractError(f"{path} must be a finite nonnegative number")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ParityContractError(f"{path} must be a finite nonnegative number")
    return float(value)


def _optional_nonnegative_number(value: Any, path: str) -> None:
    if value is not None:
        _nonnegative_number(value, path)


def _nonnegative_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ParityContractError(f"{path} must be a nonnegative integer")
    return value


def _positive_integer(value: Any, path: str) -> int:
    integer = _nonnegative_integer(value, path)
    if integer == 0:
        raise ParityContractError(f"{path} must be positive")
    return integer

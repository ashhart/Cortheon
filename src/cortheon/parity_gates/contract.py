"""Validation of a pre-registered parity contract.

Nothing downstream re-checks these invariants, so this is where an ambiguous
or unregisterable contract must be rejected outright rather than evaluated
into a green decision. Validation raises; it never records a check.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any

from cortheon.parity_gates.errors import (
    _TRUSTED_FRONTIER_HOSTS,
    SUPPORTED_CANDIDATE_HOSTS,
    ParityContractError,
)
from cortheon.parity_gates.values import _is_sha256

_REQUIRED_THRESHOLDS = {
    "min_frontiers",
    "min_cases",
    "min_cases_per_domain",
    "min_cases_per_task_class",
    "min_repetitions",
    "min_safety_cases",
    "max_errors",
    "min_completion_rate",
    "min_domain_completion_rate",
    "min_substrate_plus_model_fraction",
    "noninferiority_margin",
    "domain_noninferiority_margin",
    "max_ci_half_width",
    "max_false_allow_rate",
    "max_false_block_rate",
    "max_unstable_case_fraction",
    "max_latency_ratio",
    "max_cost_ratio",
    "require_metered_cost",
}
_POSITIVE_INTEGER_THRESHOLDS = (
    "min_frontiers",
    "min_cases",
    "min_cases_per_domain",
    "min_cases_per_task_class",
    "min_repetitions",
    "min_safety_cases",
)
_UNIT_INTERVAL_THRESHOLDS = (
    "min_completion_rate",
    "min_domain_completion_rate",
    "min_substrate_plus_model_fraction",
    "noninferiority_margin",
    "domain_noninferiority_margin",
    "max_ci_half_width",
    "max_false_allow_rate",
    "max_false_block_rate",
    "max_unstable_case_fraction",
)


def load_parity_contract(path: Path) -> tuple[dict[str, Any], str]:
    """Load and validate a pre-registered parity contract and return its digest."""

    raw = path.expanduser().read_bytes()
    try:
        contract = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParityContractError(f"invalid parity contract JSON: {exc}") from exc
    if not isinstance(contract, dict):
        raise ParityContractError("parity contract must be a JSON object")
    _validate_contract(contract)
    canonical = json.dumps(
        contract,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return contract, hashlib.sha256(canonical).hexdigest()


def _validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema_version") != 2:
        raise ParityContractError("parity contract schema_version must be 2")
    if contract.get("claim") != "broad_frontier_parity":
        raise ParityContractError("parity contract claim must be broad_frontier_parity")
    if contract.get("candidate_scope") != "substrate_plus_model_system":
        raise ParityContractError("candidate_scope must be substrate_plus_model_system")
    for key in (
        "candidate",
        "last_tuning_at",
    ):
        if not isinstance(contract.get(key), str) or not contract[key].strip():
            raise ParityContractError(f"parity contract needs {key}")
    for key in ("frontiers", "required_domains", "substrate_maintainers"):
        values = contract.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != len(values)
        ):
            raise ParityContractError(f"parity contract needs unique non-empty {key}")
    _validate_families(contract)
    _validate_contenders(contract)
    _validate_thresholds(contract)
    floors = contract.get("domain_floors", {})
    if not isinstance(floors, dict) or any(
        key not in contract["required_domains"]
        or not isinstance(value, (int, float))
        or not 0 <= float(value) <= 1
        for key, value in floors.items()
    ):
        raise ParityContractError("domain_floors must map required domains to 0..1")
    try:
        parsed_tuning = datetime.fromisoformat(
            str(contract["last_tuning_at"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ParityContractError("last_tuning_at must be ISO-8601") from exc
    if parsed_tuning.utcoffset() is None:
        raise ParityContractError("last_tuning_at must include a timezone")


def _validate_families(contract: dict[str, Any]) -> None:
    """Each frontier is an independent family, and the candidate is none of them."""

    frontier_families = contract.get("frontier_families")
    if (
        not isinstance(frontier_families, dict)
        or set(frontier_families) != set(contract["frontiers"])
        or any(
            not isinstance(value, str) or not value.strip() for value in frontier_families.values()
        )
        or len(set(frontier_families.values())) != len(frontier_families)
    ):
        raise ParityContractError("frontier_families must map every frontier to a distinct family")
    candidate_family = contract.get("candidate_family")
    if not isinstance(candidate_family, str) or not candidate_family.strip():
        raise ParityContractError("parity contract needs candidate_family")
    if candidate_family in {str(value) for value in frontier_families.values()}:
        raise ParityContractError("candidate_family must be distinct from every frontier family")
    if contract.get("candidate_host") not in SUPPORTED_CANDIDATE_HOSTS:
        raise ParityContractError(
            "candidate_host must be one of: " + ", ".join(sorted(SUPPORTED_CANDIDATE_HOSTS))
        )


def _validate_contenders(contract: dict[str, Any]) -> None:
    """Models, endpoints, compute rate, runtime digest, and pricing are all bound."""

    frontier_families = contract["frontier_families"]
    contender_models = contract.get("contender_models")
    contender_names = {str(contract["candidate"]), *contract["frontiers"]}
    if (
        not isinstance(contender_models, dict)
        or set(contender_models) != contender_names
        or any(
            not isinstance(value, str) or not value.strip() for value in contender_models.values()
        )
    ):
        raise ParityContractError("contender_models must bind the candidate and every frontier")
    contender_endpoints = contract.get("contender_endpoints")
    if (
        not isinstance(contender_endpoints, dict)
        or set(contender_endpoints) != contender_names
        or any(
            not isinstance(value, str)
            or not value.strip()
            or urllib.parse.urlparse(value).scheme not in {"http", "https"}
            or not urllib.parse.urlparse(value).hostname
            for value in contender_endpoints.values()
        )
    ):
        raise ParityContractError("contender_endpoints must bind absolute HTTP(S) endpoints")
    for frontier_name in contract["frontiers"]:
        endpoint = urllib.parse.urlparse(contender_endpoints[frontier_name])
        family = frontier_families[frontier_name]
        hostname = (endpoint.hostname or "").casefold()
        if endpoint.scheme != "https" or hostname not in _TRUSTED_FRONTIER_HOSTS.get(family, set()):
            raise ParityContractError(
                f"frontier {frontier_name} must use an official HTTPS "
                f"endpoint registered for family {family}"
            )
    candidate_endpoint = urllib.parse.urlparse(contender_endpoints[str(contract["candidate"])])
    if candidate_endpoint.scheme != "http" or (
        candidate_endpoint.hostname or ""
    ).casefold() not in {"127.0.0.1", "::1", "localhost"}:
        raise ParityContractError("the candidate must run on the independent evaluator's loopback")
    candidate_compute_rate = contract.get("candidate_compute_usd_per_hour")
    if (
        not isinstance(candidate_compute_rate, (int, float))
        or not math.isfinite(float(candidate_compute_rate))
        or float(candidate_compute_rate) <= 0
    ):
        raise ParityContractError("candidate_compute_usd_per_hour must be finite and positive")
    if not _is_sha256(contract.get("candidate_runtime_sha256")):
        raise ParityContractError("candidate_runtime_sha256 must be 64 lowercase hex")
    pricing = contract.get("pricing_per_million")
    if (
        not isinstance(pricing, dict)
        or set(pricing) != contender_names
        or any(
            not isinstance(value, dict)
            or set(value) != {"input", "output"}
            or any(
                not isinstance(price, (int, float))
                or not math.isfinite(float(price))
                or float(price) < 0
                for price in value.values()
            )
            for value in pricing.values()
        )
    ):
        raise ParityContractError(
            "pricing_per_million must bind non-negative input/output prices for every contender"
        )


def _validate_thresholds(contract: dict[str, Any]) -> None:
    """Every registered threshold is present, typed, and in range."""

    thresholds = contract.get("thresholds")
    if not isinstance(thresholds, dict) or not set(thresholds) >= _REQUIRED_THRESHOLDS:
        missing = sorted(_REQUIRED_THRESHOLDS - set(thresholds or {}))
        raise ParityContractError("parity contract thresholds missing: " + ", ".join(missing))
    for key in _POSITIVE_INTEGER_THRESHOLDS:
        if not isinstance(thresholds[key], int) or thresholds[key] < 1:
            raise ParityContractError(f"threshold {key} must be a positive integer")
    if not isinstance(thresholds["max_errors"], int) or thresholds["max_errors"] < 0:
        raise ParityContractError("threshold max_errors must be non-negative")
    for key in _UNIT_INTERVAL_THRESHOLDS:
        value = thresholds[key]
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise ParityContractError(f"threshold {key} must be between zero and one")
    for key in ("max_latency_ratio", "max_cost_ratio"):
        value = thresholds[key]
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ParityContractError(f"threshold {key} must be positive")
    if not isinstance(thresholds["require_metered_cost"], bool):
        raise ParityContractError("threshold require_metered_cost must be boolean")

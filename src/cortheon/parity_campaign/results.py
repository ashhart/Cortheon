"""Validation of the post-run campaign results artifact.

The results file is written after execution and does nothing but map each
preregistered ``cell_id`` to its attested submission and stored report,
exactly once, with paths resolved against the results file itself.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

from cortheon.parity_campaign.artifacts import resolve_artifact
from cortheon.parity_campaign.errors import CampaignContractError
from cortheon.parity_campaign.schema import (
    RegisteredCell,
    _is_sha256,
    _require_object,
)

RESULTS_SCHEMA_VERSION = 1
RESULT_FIELDS = (
    "cell_id",
    "submission_path",
    "submission_sha256",
    "report_path",
    "report_sha256",
)


def results_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_results(
    payload: Any,
    *,
    cells: list[RegisteredCell],
    expected_campaign_id: str,
    expected_registration_digest: str,
    base_dir: Path,
) -> dict[str, dict[str, str]]:
    """Validate post-run results against the preregistered matrix."""

    _require_object(payload, "results")
    if set(payload) != {"schema_version", "campaign_id", "registration_sha256", "results"}:
        raise CampaignContractError(
            "results fields must be exactly: schema_version, campaign_id, "
            "registration_sha256, results"
        )
    if payload["schema_version"] != RESULTS_SCHEMA_VERSION:
        raise CampaignContractError(f"results schema_version must be {RESULTS_SCHEMA_VERSION}")
    if not isinstance(payload["campaign_id"], str) or not payload["campaign_id"].strip():
        raise CampaignContractError("results campaign_id must be a non-empty string")
    if not hmac.compare_digest(payload["campaign_id"], expected_campaign_id):
        raise CampaignContractError("results campaign_id does not match the registration")
    if not hmac.compare_digest(
        str(payload["registration_sha256"] or ""),
        expected_registration_digest,
    ):
        raise CampaignContractError(
            "results registration_sha256 does not match the registration digest"
        )
    raw_results = payload["results"]
    if not isinstance(raw_results, list) or not raw_results:
        raise CampaignContractError("results must be a non-empty array")
    declared_ids = {cell["cell_id"] for cell in cells}

    observed: dict[str, dict[str, str]] = {}
    resolved_paths: dict[Path, str] = {}
    for index, raw_result in enumerate(raw_results):
        _require_object(raw_result, f"result {index}")
        if set(raw_result) != set(RESULT_FIELDS):
            raise CampaignContractError(
                f"result {index} fields must be exactly: " + ", ".join(RESULT_FIELDS)
            )
        result = {}
        for field in RESULT_FIELDS:
            value = raw_result[field]
            if not isinstance(value, str) or not value.strip():
                raise CampaignContractError(f"result {index} field {field} must be a string")
            result[field] = value
        cell_id = result["cell_id"]
        if cell_id in observed:
            raise CampaignContractError(f"result cell {cell_id} is declared more than once")
        if cell_id not in declared_ids:
            raise CampaignContractError(
                f"result cell {cell_id} was never preregistered; extra results are forbidden"
            )
        observed[cell_id] = result
        for digest_field in ("submission_sha256", "report_sha256"):
            if not _is_sha256(result[digest_field]):
                raise CampaignContractError(
                    f"result {cell_id} field {digest_field} must be 64 lowercase hex"
                )
        for path_field in ("submission_path", "report_path"):
            resolved = resolve_artifact(base_dir, result[path_field])
            role = f"{path_field}:{cell_id}"
            if resolved in resolved_paths:
                raise CampaignContractError(
                    f"artifact path {resolved} is reused by {role} and {resolved_paths[resolved]}"
                )
            resolved_paths[resolved] = role
    missing = sorted(declared_ids - set(observed))
    if missing:
        raise CampaignContractError(
            "results are missing preregistered cells: " + ", ".join(missing)
        )
    for digest_field in ("submission_sha256", "report_sha256"):
        digests = [result[digest_field] for result in observed.values()]
        if len(set(digests)) != len(digests):
            raise CampaignContractError(f"results reuse a {digest_field}")
    _reject_cross_artifact_aliases(cells, set(resolved_paths))
    return observed


def _reject_cross_artifact_aliases(
    cells: list[RegisteredCell],
    result_paths: set[Path],
) -> None:
    registration_paths = {cell.contract_path for cell in cells} | {cell.pack_path for cell in cells}
    aliased = registration_paths & result_paths
    if aliased:
        raise CampaignContractError(
            "artifact paths must not be reused across kinds: " + str(sorted(aliased))
        )

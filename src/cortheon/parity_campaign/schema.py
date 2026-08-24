"""Strict validation and canonical digests for replication-campaign schemas.

Two artifacts govern a campaign. The *registration* is the immutable
pre-run contract: it fixes the exact cell matrix and every commitment that
exists before execution (inner parity contract, sealed private pack). The
*results* artifact is written after execution and only maps preregistered
cells to their submission and report files. Digests that cannot exist before
execution (submissions, reports) must never appear in the registration.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from cortheon.parity import SUPPORTED_CANDIDATE_HOSTS
from cortheon.parity_campaign.artifacts import resolve_artifact
from cortheon.parity_campaign.errors import CampaignContractError

CAMPAIGN_SCHEMA_VERSION = 2
DECISION_SCHEMA_VERSION = 2
CAMPAIGN_CLAIM = "replicated_broad_frontier_parity"
INNER_CLAIM = "broad_frontier_parity"
MIN_MODEL_FAMILIES = 3
MIN_EVALUATORS = 2
REPORT_SCHEMA_VERSION = 7

REGISTRATION_FIELDS = ("schema_version", "claim", "campaign_id", "declared_at", "cells")
CELL_FIELDS = (
    "cell_id",
    "family",
    "model",
    "host",
    "runtime_sha256",
    "evaluator",
    "evaluator_key_sha256",
    "pack_issuer",
    "pack_id",
    "runner_id",
    "contract_path",
    "contract_sha256",
    "pack_path",
    "pack_sha256",
    "pack_key_env",
    "runner_key_env",
)
CELL_DIGEST_FIELDS = ("runtime_sha256", "contract_sha256", "pack_sha256", "evaluator_key_sha256")
IDENTITY_FIELDS = (
    "model",
    "family",
    "host",
    "runtime_sha256",
    "contract_sha256",
    "evaluator",
    "pack_issuer",
    "pack_id",
    "runner_id",
)

_CELL_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,79}")
_ENV_NAME_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*")


class RegisteredCell:
    """One preregistered cell with its resolved artifact paths."""

    def __init__(self, fields: dict[str, str], base_dir: Path) -> None:
        self.fields = fields
        self.contract_path = resolve_artifact(base_dir, fields["contract_path"])
        self.pack_path = resolve_artifact(base_dir, fields["pack_path"])

    def __getitem__(self, key: str) -> str:
        return self.fields[key]


def registration_digest(payload: dict[str, Any]) -> str:
    """Canonical digest over every material preregistered field."""

    return _canonical_digest(payload)


def validate_registration(
    payload: Any,
    *,
    base_dir: Path,
) -> list[RegisteredCell]:
    """Validate a pre-run registration. Fail closed on any deviation."""

    _require_object(payload, "registration")
    if set(payload) != set(REGISTRATION_FIELDS):
        raise CampaignContractError(
            "registration fields must be exactly: " + ", ".join(REGISTRATION_FIELDS)
        )
    if payload["schema_version"] != CAMPAIGN_SCHEMA_VERSION:
        raise CampaignContractError(
            f"registration schema_version must be {CAMPAIGN_SCHEMA_VERSION}"
        )
    if payload["claim"] != CAMPAIGN_CLAIM:
        raise CampaignContractError(f"registration claim must be {CAMPAIGN_CLAIM}")
    campaign_id = payload["campaign_id"]
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise CampaignContractError("campaign_id must be a non-empty string")
    _require_utc_timestamp(payload["declared_at"], "declared_at")
    raw_cells = payload["cells"]
    if not isinstance(raw_cells, list) or not raw_cells:
        raise CampaignContractError("registration must declare a non-empty cells array")

    cells: list[RegisteredCell] = []
    for index, raw_cell in enumerate(raw_cells):
        fields = _validated_cell_fields(index, raw_cell)
        cells.append(RegisteredCell(fields, base_dir))
    _reject_reused_cells(cells)
    _require_exact_matrix(cells)
    return cells


def _validated_cell_fields(index: int, raw_cell: Any) -> dict[str, str]:
    _require_object(raw_cell, f"campaign cell {index}")
    if set(raw_cell) != set(CELL_FIELDS):
        raise CampaignContractError(
            f"campaign cell {index} fields must be exactly: " + ", ".join(CELL_FIELDS)
        )
    fields: dict[str, str] = {}
    for field in CELL_FIELDS:
        value = raw_cell[field]
        if not isinstance(value, str) or not value.strip():
            raise CampaignContractError(f"campaign cell {index} field {field} must be a string")
        fields[field] = value.strip()
    for field in CELL_DIGEST_FIELDS:
        if not _is_sha256(fields[field]):
            raise CampaignContractError(
                f"campaign cell {index} field {field} must be 64 lowercase hex"
            )
    if not _CELL_ID_PATTERN.fullmatch(fields["cell_id"]):
        raise CampaignContractError(
            f"campaign cell {index} cell_id must match [a-z0-9][a-z0-9_.-]{{0,79}}"
        )
    if fields["host"] not in SUPPORTED_CANDIDATE_HOSTS:
        raise CampaignContractError(
            f"campaign cell {index} host must be one of: "
            + ", ".join(sorted(SUPPORTED_CANDIDATE_HOSTS))
        )
    for field in ("pack_key_env", "runner_key_env"):
        if not _ENV_NAME_PATTERN.fullmatch(fields[field]):
            raise CampaignContractError(
                f"campaign cell {index} field {field} must be an environment variable name"
            )
    return fields


def _reject_reused_cells(cells: list[RegisteredCell]) -> None:
    seen: dict[str, set[str]] = {}
    pack_identities: set[tuple[str, str]] = set()
    pack_paths: set[Path] = set()
    for cell in cells:
        for field in (
            "cell_id",
            "pack_id",
            "pack_sha256",
            "runner_id",
            "runner_key_env",
        ):
            value = cell[field]
            values = seen.setdefault(field, set())
            if value in values:
                raise CampaignContractError(
                    f"campaign reuses {field} {value} (also declared by another cell)"
                )
            values.add(value)
        identity = (cell["pack_issuer"], cell["pack_id"])
        if identity in pack_identities:
            raise CampaignContractError(
                f"campaign reuses pack {cell['pack_id']} from {cell['pack_issuer']}"
            )
        pack_identities.add(identity)
        if cell.pack_path in pack_paths:
            raise CampaignContractError(f"campaign reuses sealed pack path {cell.pack_path}")
        pack_paths.add(cell.pack_path)
    _require_evaluator_key_identity(cells)


def _require_evaluator_key_identity(cells: list[RegisteredCell]) -> None:
    """One stable signing-key commitment per evaluator, distinct across them.

    The same declared evaluator may reuse one pack-key environment variable
    and secret across all its cells; two different evaluators must never
    share a commitment or a pack-key environment variable, because a shared
    secret would make them one grading authority in fact.
    """

    commitments: dict[str, set[str]] = {}
    pack_envs: dict[str, set[str]] = {}
    for cell in cells:
        commitments.setdefault(cell["evaluator"], set()).add(cell["evaluator_key_sha256"])
        pack_envs.setdefault(cell["evaluator"], set()).add(cell["pack_key_env"])
    for evaluator, keys in sorted(commitments.items()):
        if len(keys) != 1:
            raise CampaignContractError(
                f"evaluator {evaluator} must preregister exactly one "
                "evaluator_key_sha256 across all its cells"
            )
    for evaluator, envs in sorted(pack_envs.items()):
        if len(envs) != 1:
            raise CampaignContractError(
                f"evaluator {evaluator} must use one pack_key_env across all its cells"
            )
    seen_commitments: dict[str, str] = {}
    for evaluator, keys in sorted(commitments.items()):
        commitment = next(iter(keys))
        owner = seen_commitments.setdefault(commitment, evaluator)
        if owner != evaluator:
            raise CampaignContractError(
                f"evaluators {owner} and {evaluator} share signing-key "
                "commitment; distinct evaluators must have distinct commitments"
            )
    seen_envs: dict[str, str] = {}
    for evaluator, envs in sorted(pack_envs.items()):
        env = next(iter(envs))
        owner = seen_envs.setdefault(env, evaluator)
        if owner != evaluator:
            raise CampaignContractError(
                f"evaluators {owner} and {evaluator} share pack key environment variable {env}"
            )


def _require_one_model_per_family(cells: list[RegisteredCell]) -> None:
    """Every host and evaluator must run the exact same model per family.

    Without this, a complete family x host x evaluator matrix could silently
    hide a different exact candidate model on every host. Host-specific
    runtime digests may still differ because adapters differ.
    """

    models: dict[str, set[str]] = {}
    for cell in cells:
        models.setdefault(cell["family"], set()).add(cell["model"])
    for family, family_models in sorted(models.items()):
        if len(family_models) != 1:
            raise CampaignContractError(
                f"family {family} must preregister one exact model across "
                f"all hosts and evaluators; declared: {sorted(family_models)}"
            )


def _require_exact_matrix(cells: list[RegisteredCell]) -> None:
    families = {cell["family"] for cell in cells}
    evaluators = {cell["evaluator"] for cell in cells}
    if len(families) < MIN_MODEL_FAMILIES:
        raise CampaignContractError(
            f"campaign must declare at least {MIN_MODEL_FAMILIES} local model families"
        )
    if len(evaluators) < MIN_EVALUATORS:
        raise CampaignContractError(f"campaign must declare at least {MIN_EVALUATORS} evaluators")
    hosts = {cell["host"] for cell in cells}
    if hosts != set(SUPPORTED_CANDIDATE_HOSTS):
        raise CampaignContractError(
            "campaign must cover every supported host exactly: "
            + ", ".join(sorted(SUPPORTED_CANDIDATE_HOSTS))
        )
    logical: dict[tuple[str, str, str], RegisteredCell] = {}
    for cell in cells:
        key = (cell["family"], cell["host"], cell["evaluator"])
        if key in logical:
            raise CampaignContractError(f"campaign declares logical cell {key} more than once")
        logical[key] = cell
    expected = {
        (family, host, evaluator)
        for family in families
        for host in SUPPORTED_CANDIDATE_HOSTS
        for evaluator in evaluators
    }
    if set(logical) != expected:
        missing = sorted(expected - set(logical))
        raise CampaignContractError(
            "campaign matrix must be the complete product of families x hosts "
            f"x evaluators; missing: {missing}"
        )
    _require_one_model_per_family(cells)
    _require_consistent_logical_contracts(cells)


def _require_consistent_logical_contracts(cells: list[RegisteredCell]) -> None:
    by_logical: dict[tuple[str, str], list[RegisteredCell]] = {}
    for cell in cells:
        by_logical.setdefault((cell["family"], cell["host"]), []).append(cell)
    contract_owners: dict[str, tuple[str, str]] = {}
    contract_path_owners: dict[Path, tuple[str, str]] = {}
    for logical, replicates in by_logical.items():
        shared = {replicate["contract_sha256"] for replicate in replicates}
        paths = {replicate.contract_path for replicate in replicates}
        models = {replicate["model"] for replicate in replicates}
        runtimes = {replicate["runtime_sha256"] for replicate in replicates}
        if len(shared) != 1 or len(paths) != 1 or len(models) != 1 or len(runtimes) != 1:
            raise CampaignContractError(
                f"evaluator replicates of {logical} must preregister the same "
                "inner contract, model, and runtime digest"
            )
        digest = next(iter(shared))
        path = next(iter(paths))
        owner = contract_owners.setdefault(digest, logical)
        path_owner = contract_path_owners.setdefault(path, logical)
        if owner != logical or path_owner != logical:
            raise CampaignContractError(
                "each inner parity contract belongs to exactly one logical "
                f"family/host cell; {digest} is bound to both {owner} and {logical}"
            )


def _require_utc_timestamp(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CampaignContractError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise CampaignContractError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.utcoffset() != timedelta(0):
        raise CampaignContractError(f"{field} must carry a UTC (+00:00 / Z) offset")


def _require_object(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise CampaignContractError(f"{label} must be a JSON object")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()

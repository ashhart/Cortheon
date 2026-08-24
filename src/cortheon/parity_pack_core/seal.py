"""Issue one authenticated evaluator challenge pack.

The order of the checks below is the contract: identity and key material are
validated before any file is read, the destinations are refused before any
case is normalized, and the seal is computed only once the manifest is
complete. Nothing is written until every check has passed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cortheon.parity import (
    evaluation_schedule_hash,
    load_parity_contract,
    public_case_projection,
    public_task_hash,
)
from cortheon.parity_benchmark_core.oracle_web import validate_pack_web_authority
from cortheon.parity_pack_core.artifacts import write_private_json
from cortheon.parity_pack_core.clock import issued_at, require_future_expiry
from cortheon.parity_pack_core.keys import (
    _canonical_signed_payload,
    read_signing_keys,
    signature,
)
from cortheon.parity_pack_core.manifest import build_manifest, public_payload
from cortheon.parity_pack_core.selection import (
    normalize_and_select,
    selection_sha256,
    validate_task_class_coverage,
)


def _declared_evaluator(issuer: str, evaluator: str | None) -> str:
    if evaluator is not None and not evaluator.strip():
        raise ValueError("evaluator cannot be blank; omit it to default to the issuer")
    declared = (evaluator.strip() if evaluator is not None else issuer).strip()
    if not declared:
        raise ValueError("evaluator cannot be empty")
    return declared


def _normalized_authors(authors: list[str]) -> list[str]:
    normalized = sorted({value.strip() for value in authors if value.strip()})
    if not normalized:
        raise ValueError("at least one non-empty author is required")
    return normalized


def _resolved_destinations(output_path: Path, public_output_path: Path, overwrite: bool):
    destination = output_path.expanduser().resolve()
    public_destination = public_output_path.expanduser().resolve()
    if destination.exists() and not overwrite:
        raise ValueError(f"refusing to overwrite existing pack: {destination}")
    if public_destination.exists() and not overwrite:
        raise ValueError(f"refusing to overwrite existing public pack: {public_destination}")
    if public_destination == destination:
        raise ValueError("private and public pack paths must differ")
    return destination, public_destination


def _submitted_cases(input_path: Path) -> list[Any]:
    payload = json.loads(input_path.expanduser().read_text(encoding="utf-8"))
    raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list):
        raise ValueError("input must be an object with a cases array")
    return raw_cases


def seal_case_pack(
    input_path: Path,
    output_path: Path,
    *,
    public_output_path: Path,
    contract_path: Path,
    pack_id: str,
    issuer: str,
    runner_id: str,
    authors: list[str],
    key_env: str,
    runner_key_env: str,
    seed: int,
    holdout_fraction: float,
    rotation_index: int,
    rotation_size: int,
    expires_at: str,
    overwrite: bool,
    evaluator: str | None = None,
) -> dict[str, Any]:
    """Validate, bind, authenticate, and write one evaluator challenge pack."""

    if not pack_id.strip() or not issuer.strip() or not runner_id.strip():
        raise ValueError("pack-id, issuer, and runner-id cannot be empty")
    declared_evaluator = _declared_evaluator(issuer, evaluator)
    normalized_authors = _normalized_authors(authors)
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout-fraction must be between zero and one")
    if rotation_index < 0 or rotation_size < 0:
        raise ValueError("rotation values cannot be negative")
    key, runner_key = read_signing_keys(key_env, runner_key_env)
    destination, public_destination = _resolved_destinations(
        output_path, public_output_path, overwrite
    )
    raw_cases = _submitted_cases(input_path)
    contract, contract_sha256 = load_parity_contract(contract_path)
    normalized_cases, selection = normalize_and_select(
        raw_cases,
        seed=seed,
        holdout_fraction=holdout_fraction,
        rotation_index=rotation_index,
        rotation_size=rotation_size,
        min_cases_per_task_class=int(contract["thresholds"]["min_cases_per_task_class"]),
    )
    validate_task_class_coverage(selection, int(contract["thresholds"]["min_cases_per_task_class"]))
    validate_pack_web_authority({"evaluator": declared_evaluator}, normalized_cases)
    execution_repetitions = int(contract["thresholds"]["min_repetitions"])
    scheduled_contenders = [
        str(contract["candidate"]),
        *[str(value) for value in contract["frontiers"]],
    ]
    schedule_sha256 = evaluation_schedule_hash(
        [str(case["id"]) for case in selection],
        scheduled_contenders,
        execution_repetitions,
        seed,
    )
    created_at = issued_at()
    require_future_expiry(expires_at)
    manifest = build_manifest(
        pack_id=pack_id.strip(),
        issuer=issuer.strip(),
        evaluator=declared_evaluator,
        runner_id=runner_id.strip(),
        runner_key_env=runner_key_env,
        runner_key=runner_key,
        created_at=created_at,
        expires_at=expires_at,
        authors=normalized_authors,
        contract_sha256=contract_sha256,
        selection_sha256=selection_sha256(selection),
        public_tasks_sha256=public_task_hash(selection),
        seed=seed,
        execution_repetitions=execution_repetitions,
        scheduled_contenders=scheduled_contenders,
        schedule_sha256=schedule_sha256,
        holdout_fraction=holdout_fraction,
        rotation_index=rotation_index,
        rotation_size=rotation_size,
        domains=sorted({str(case["domain"]) for case in normalized_cases}),
    )
    maintainers = {str(value).casefold() for value in contract["substrate_maintainers"]}
    if maintainers & {value.casefold() for value in normalized_authors}:
        raise ValueError("pack authors must be independent of substrate maintainers")
    manifest["signature"] = signature(key_env, key, _canonical_signed_payload(manifest, raw_cases))
    write_private_json(destination, {"manifest": manifest, "cases": raw_cases})
    write_private_json(
        public_destination,
        public_payload(manifest, public_case_projection(selection)),
    )
    return {
        "ok": True,
        "path": str(destination),
        "public_path": str(public_destination),
        "pack_id": manifest["pack_id"],
        "contract_sha256": contract_sha256,
        "selection_sha256": manifest["selection_sha256"],
        "public_tasks_sha256": manifest["public_tasks_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "selected_cases": len(selection),
        "total_cases": len(normalized_cases),
        "domains": manifest["domains"],
        "seal": {
            "algorithm": "hmac-sha256",
            "key_id": manifest["signature"]["key_id"],
        },
    }

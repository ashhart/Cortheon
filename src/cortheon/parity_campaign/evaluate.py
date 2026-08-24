"""Regrade a replication campaign from original evidence. Fail closed.

Every preregistered cell is re-evaluated by calling the independent blind
grader over its sealed private pack, attested submission, and inner parity
contract, using each evaluator's stable pack-key environment variable and
the per-cell runner-attestation secret. Each pack's HMAC signature carries
a SHA-256 key commitment that must match the evaluator's preregistered
``evaluator_key_sha256`` and the supplied secret. The stored report's own
``passed`` boolean is never trusted: the campaign decision comes from the
recomputed report and gate, and the stored report must match the recomputed
report byte-for-byte on the canonical evaluation receipt (everything except
``generated_at``). Pack validity and the preregistration sequence are
judged at each submission's attested execution time, never at regrade
wall-clock time.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from cortheon.blind_evaluator import grade_blind_submission
from cortheon.parity import load_parity_contract
from cortheon.parity_campaign.artifacts import file_digest, read_json_object, resolve_artifact
from cortheon.parity_campaign.errors import CampaignContractError
from cortheon.parity_campaign.receipt import evaluation_receipt_sha256
from cortheon.parity_campaign.results import results_digest, validate_results
from cortheon.parity_campaign.schema import (
    CAMPAIGN_CLAIM,
    DECISION_SCHEMA_VERSION,
    IDENTITY_FIELDS,
    INNER_CLAIM,
    REPORT_SCHEMA_VERSION,
    RegisteredCell,
    registration_digest,
    validate_registration,
)
from cortheon.parity_timestamps import ordering_holds

MIN_SECRET_BYTES = 32
OPERATIONAL_REQUIREMENTS = (
    "Prior public registration of this campaign contract (registration_sha256) "
    "is an operational requirement that this repository cannot prove by itself; "
    "independence claims must be audited against the external publication record.",
    "Evaluator independence is a declared, pack-attested identity. Human and "
    "organizational independence of the evaluators requires externally auditable "
    "operation and cannot be established by this code.",
    "Pack and runner authentication is HMAC under shared secrets, and the "
    "evaluator signing identity is a SHA-256 commitment to that secret. This "
    "binds identity only as far as each secret-holder's honesty and custody; "
    "it is not public-key proof and still requires external audit.",
    "The pack issuer is the declared grading authority for each cell; the "
    "evaluator identity is bound by that authority's sealed pack.",
)


def evaluate_replication_campaign(
    registration_path: Path,
    results_path: Path,
) -> dict[str, Any]:
    """Evaluate and regrade a preregistered replication campaign."""

    registration_path = registration_path.expanduser().resolve()
    results_path = results_path.expanduser().resolve()
    registration = read_json_object(registration_path, "registration")
    cells = validate_registration(
        registration,
        base_dir=registration_path.parent,
    )
    digest_of_registration = registration_digest(registration)
    results = read_json_object(results_path, "results")
    mapped = validate_results(
        results,
        cells=cells,
        expected_campaign_id=str(registration["campaign_id"]),
        expected_registration_digest=digest_of_registration,
        base_dir=results_path.parent,
    )

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, **evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), **evidence})

    records: list[dict[str, Any]] = []
    runner_secrets: list[str] = []
    observed_commitments: dict[str, set[str]] = {}
    chain_head = hashlib.sha256(b"").hexdigest()
    declared_at = registration["declared_at"]
    for cell in cells:
        pack_secret = _secret(cell["pack_key_env"], cell["cell_id"])
        observed_commitment = hashlib.sha256(pack_secret.encode("utf-8")).hexdigest()
        observed_commitments.setdefault(cell["evaluator"], set()).add(observed_commitment)
        check(
            f"pack_secret_commitment:{cell['cell_id']}",
            observed_commitment == cell["evaluator_key_sha256"],
            expected=cell["evaluator_key_sha256"],
            observed=observed_commitment,
        )
        runner_secrets.append(_secret(cell["runner_key_env"], cell["cell_id"]))
        record, chain_head = _evaluate_cell(
            cell,
            mapped[cell["cell_id"]],
            results_path.parent,
            chain_head,
            declared_at,
            check,
        )
        records.append(record)

    single_commitment_per_evaluator = all(
        len(commitments) == 1 for commitments in observed_commitments.values()
    )
    distinct_across_evaluators = len(
        {next(iter(commitments)) for commitments in observed_commitments.values()}
    ) == len(observed_commitments)
    check(
        "evaluator_signing_identity",
        bool(observed_commitments)
        and single_commitment_per_evaluator
        and distinct_across_evaluators,
        observed={
            evaluator: sorted(commitments)
            for evaluator, commitments in sorted(observed_commitments.items())
        },
    )
    check(
        "distinct_runner_secrets",
        len(set(runner_secrets)) == len(runner_secrets),
        declared=len(runner_secrets),
        distinct=len(set(runner_secrets)),
    )

    families = sorted({cell["family"] for cell in cells})
    hosts = sorted({cell["host"] for cell in cells})
    evaluators = sorted({cell["evaluator"] for cell in cells})
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "claim": CAMPAIGN_CLAIM,
        "campaign_id": str(registration["campaign_id"]),
        "registration_sha256": digest_of_registration,
        "results_sha256": results_digest(results),
        "chain_head_sha256": chain_head,
        "coverage": {
            "model_families": families,
            "hosts": hosts,
            "evaluators": evaluators,
            "logical_cells": len(families) * len(hosts),
            "regraded_reports": len(records),
        },
        "cells": records,
        "scope": (
            "Replication of the single-model broad-frontier-parity claim across "
            "the preregistered matrix of local model families, supported hosts, "
            "and evaluators. This is not a claim of literal universal parity."
        ),
        "operational_requirements": list(OPERATIONAL_REQUIREMENTS),
        "passed": bool(checks) and all(item["passed"] for item in checks),
        "checks": checks,
        "failure_reasons": [str(item["name"]) for item in checks if item["passed"] is not True],
    }


def _evaluate_cell(
    cell: RegisteredCell,
    result: dict[str, str],
    results_base: Path,
    previous_head: str,
    declared_at: str,
    check: Any,
) -> tuple[dict[str, Any], str]:
    cell_id = cell["cell_id"]
    submission_path = resolve_artifact(results_base, result["submission_path"])
    report_path = resolve_artifact(results_base, result["report_path"])
    submission_sha256 = file_digest(submission_path, label="submission", cell_id=cell_id)
    stored_report_sha256 = file_digest(report_path, label="report", cell_id=cell_id)
    check(
        f"submission_digest:{cell_id}",
        submission_sha256 == result["submission_sha256"],
        expected=result["submission_sha256"],
        observed=submission_sha256,
    )
    check(
        f"report_digest:{cell_id}",
        stored_report_sha256 == result["report_sha256"],
        expected=result["report_sha256"],
        observed=stored_report_sha256,
    )

    try:
        inner_contract, inner_digest = load_parity_contract(cell.contract_path)
    except (OSError, ValueError) as exc:
        raise CampaignContractError(
            f"cannot load inner contract for cell {cell_id}: {exc}"
        ) from exc
    check(
        f"contract_digest:{cell_id}",
        inner_digest == cell["contract_sha256"],
        expected=cell["contract_sha256"],
        observed=inner_digest,
    )
    observed_pack_sha256 = file_digest(cell.pack_path, label="sealed pack", cell_id=cell_id)
    check(
        f"pack_digest:{cell_id}",
        observed_pack_sha256 == cell["pack_sha256"],
        expected=cell["pack_sha256"],
        observed=observed_pack_sha256,
    )
    try:
        recomputed = grade_blind_submission(
            cell.pack_path,
            submission_path,
            contract_path=cell.contract_path,
            key_env=cell["pack_key_env"],
            runner_key_env=cell["runner_key_env"],
        )
    except (OSError, ValueError) as exc:
        raise CampaignContractError(
            f"cannot regrade cell {cell_id} from original evidence: {exc}"
        ) from exc

    stored_report = read_json_object(report_path, f"stored report for {cell_id}")
    if stored_report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise CampaignContractError(
            f"stored report for cell {cell_id} must use schema version {REPORT_SCHEMA_VERSION}"
        )
    expected_identity = {field: cell[field] for field in IDENTITY_FIELDS}
    observed_identity = recomputed.get("release_identity")
    identity_bound = isinstance(observed_identity, dict) and all(
        str(observed_identity.get(field) or "") == expected_identity[field]
        for field in IDENTITY_FIELDS
    )
    check(
        f"cell_identity_bound:{cell_id}",
        identity_bound,
        declared=expected_identity,
        recomputed=(
            {field: str(observed_identity.get(field) or "") for field in IDENTITY_FIELDS}
            if isinstance(observed_identity, dict)
            else None
        ),
    )

    recomputed_bank = recomputed.get("case_bank")
    recomputed_bank = recomputed_bank if isinstance(recomputed_bank, dict) else {}
    seal = recomputed_bank.get("seal")
    seal = seal if isinstance(seal, dict) else {}
    check(
        f"pack_commitment_bound:{cell_id}",
        seal.get("verified") is True
        and str(seal.get("key_commitment") or "") == cell["evaluator_key_sha256"],
        expected=cell["evaluator_key_sha256"],
        observed=str(seal.get("key_commitment") or ""),
        seal_verified=seal.get("verified"),
    )
    recomputed_methodology = recomputed.get("methodology")
    recomputed_methodology = (
        recomputed_methodology if isinstance(recomputed_methodology, dict) else {}
    )
    check(
        f"preregistration_sequence:{cell_id}",
        ordering_holds(
            recomputed_bank.get("created_at"),
            declared_at,
            recomputed_methodology.get("execution_completed_at"),
        ),
        pack_created_at=recomputed_bank.get("created_at"),
        declared_at=declared_at,
        execution_completed_at=recomputed_methodology.get("execution_completed_at"),
    )

    stored_receipt = evaluation_receipt_sha256(stored_report)
    recomputed_receipt = evaluation_receipt_sha256(recomputed)
    check(
        f"report_receipt_match:{cell_id}",
        stored_receipt == recomputed_receipt,
        stored=stored_receipt,
        recomputed=recomputed_receipt,
    )

    gate = recomputed.get("frontier_parity_gate")
    gate = gate if isinstance(gate, dict) else {}
    gate_passed = bool(
        gate.get("claim") == INNER_CLAIM
        and gate.get("passed") is True
        and gate.get("contract_sha256") == cell["contract_sha256"]
    )
    check(
        f"inner_gate_passed:{cell_id}",
        gate_passed,
        gate_claim=gate.get("claim"),
        gate_passed_observed=gate.get("passed"),
        gate_contract_sha256=gate.get("contract_sha256"),
    )

    inner_contract_host = str(inner_contract.get("candidate_host") or "")
    chain_head = hashlib.sha256(
        ":".join(
            (
                previous_head,
                cell_id,
                cell["contract_sha256"],
                cell["pack_sha256"],
                submission_sha256,
                stored_report_sha256,
                recomputed_receipt,
            )
        ).encode("utf-8")
    ).hexdigest()
    record = {
        "cell_id": cell_id,
        "family": cell["family"],
        "host": cell["host"],
        "evaluator": cell["evaluator"],
        "model": cell["model"],
        "evaluator_key_sha256": cell["evaluator_key_sha256"],
        "declared_authority": cell["pack_issuer"],
        "contract_host": inner_contract_host,
        "contract_sha256": cell["contract_sha256"],
        "pack_id": cell["pack_id"],
        "pack_sha256": cell["pack_sha256"],
        "runner_id": cell["runner_id"],
        "submission_sha256": submission_sha256,
        "report_sha256": stored_report_sha256,
        "regraded_receipt_sha256": recomputed_receipt,
        "gate_passed": gate_passed,
        "chain_head_sha256": chain_head,
    }
    return record, chain_head


def _secret(env_name: str, cell_id: str) -> str:
    value = os.environ.get(env_name, "")
    if len(value.encode("utf-8")) < MIN_SECRET_BYTES:
        raise CampaignContractError(
            f"cell {cell_id} requires {env_name} to hold at least "
            f"{MIN_SECRET_BYTES} bytes of secret"
        )
    return value

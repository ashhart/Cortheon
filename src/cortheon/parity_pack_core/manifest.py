"""The sealed manifest and the public projection of it.

The private manifest binds the pack to its contract, its selection, its
execution schedule, and the two evaluator identities. The public projection
republishes only what a contender may see: the same bindings and digests,
never the case bank, the graders, or the expected verdicts.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Any

from cortheon.parity_benchmark_core.oracle_taxonomy import TAXONOMY_VERSION
from cortheon.parity_pack_core.keys import runner_attestation

SCHEMA_VERSION = 2

# Keys the public manifest republishes verbatim from the sealed one. Anything
# outside this list stays with the evaluator.
_PUBLIC_MANIFEST_KEYS = (
    "pack_id",
    "issuer",
    "evaluator",
    "execution_authority",
    "runner_id",
    "runner_attestation",
    "created_at",
    "expires_at",
    "contract_sha256",
    "selection_sha256",
    "public_tasks_sha256",
    "execution_seed",
    "execution_repetitions",
    "scheduled_contenders",
    "schedule_sha256",
)


def build_manifest(
    *,
    pack_id: str,
    issuer: str,
    evaluator: str,
    runner_id: str,
    runner_key_env: str,
    runner_key: str,
    created_at: str,
    expires_at: str,
    authors: list[str],
    contract_sha256: str,
    selection_sha256: str,
    public_tasks_sha256: str,
    seed: int,
    execution_repetitions: int,
    scheduled_contenders: list[str],
    schedule_sha256: str,
    holdout_fraction: float,
    rotation_index: int,
    rotation_size: int,
    domains: list[str],
) -> dict[str, Any]:
    """Assemble the unsigned manifest; the seal is added by the caller."""

    return {
        "schema_version": SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "pack_id": pack_id,
        "issuer": issuer,
        "evaluator": evaluator,
        "execution_authority": "independent_evaluator_managed",
        "runner_id": runner_id,
        "runner_attestation": runner_attestation(runner_key_env, runner_key),
        "created_at": created_at,
        "expires_at": expires_at,
        "authored_by": authors,
        "oracle_mode": "frozen_external",
        "contract_sha256": contract_sha256,
        "selection_sha256": selection_sha256,
        "public_tasks_sha256": public_tasks_sha256,
        "execution_seed": seed,
        "execution_repetitions": execution_repetitions,
        "scheduled_contenders": sorted(scheduled_contenders),
        "schedule_sha256": schedule_sha256,
        "selection": {
            "split": "heldout",
            "seed": seed,
            "holdout_fraction": holdout_fraction,
            "rotation_index": rotation_index,
            "rotation_size": rotation_size,
        },
        "domains": domains,
        "nonce_commitment": hashlib.sha256(secrets.token_bytes(32)).hexdigest(),
    }


def public_payload(
    manifest: dict[str, Any],
    public_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """The contender-visible pack: bindings and digests, no oracle material."""

    return {
        "manifest": {
            "schema_version": SCHEMA_VERSION,
            **{key: manifest[key] for key in _PUBLIC_MANIFEST_KEYS},
            "split": "heldout",
        },
        "cases": public_cases,
    }

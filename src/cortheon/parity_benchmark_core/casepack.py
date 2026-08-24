from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from pathlib import Path
from typing import Any

from cortheon.engine import CortheonEngine
from cortheon.parity_benchmark_core.cases_builtin import _builtin_cases
from cortheon.parity_benchmark_core.models import LoadedCasePack
from cortheon.parity_benchmark_core.oracle_sources import validate_visible_source_bindings
from cortheon.parity_benchmark_core.oracle_taxonomy import (
    ALL_GRADER_TYPES,
    PROOF_GRADER_TYPES,
    TAXONOMY_VERSION,
    validate_case_oracle_binding,
)
from cortheon.parity_benchmark_core.oracle_web import validate_pack_web_authority
from cortheon.parity_benchmark_core.patch_oracle import validate_patch_oracle
from cortheon.parity_benchmark_core.structured_oracles import validate_private_oracle


def _load_cases(path: Path | None) -> list[dict[str, Any]]:
    """Compatibility wrapper for callers that only need normalized cases."""

    return _load_case_pack(
        path,
        key_env="CORTHEON_BENCH_PACK_KEY",
    ).cases


def _load_case_pack(
    path: Path | None,
    *,
    key_env: str,
) -> LoadedCasePack:
    built_in = path is None
    if built_in:
        payload: Any = {"cases": _builtin_cases()}
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    else:
        raw = path.expanduser().read_bytes()
        payload = json.loads(raw)
    metadata = _case_pack_metadata(
        payload,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        built_in=built_in,
        key_env=key_env,
    )
    seal = metadata.get("seal")
    seal_verified = bool(isinstance(seal, dict) and seal.get("verified") is True)
    cases = _normalize_cases(
        payload,
        built_in=built_in,
        allow_external_patch_tests=seal_verified,
    )
    if not built_in:
        validate_pack_web_authority(metadata, cases)
    metadata["oracle_independent"] = bool(
        not built_in
        and metadata.get("oracle_mode") == "frozen_external"
        and all(_case_has_frozen_oracle(case) for case in cases)
    )
    return LoadedCasePack(cases=cases, metadata=metadata)


def _normalize_cases(
    payload: Any,
    *,
    built_in: bool,
    allow_external_patch_tests: bool,
) -> list[dict[str, Any]]:
    raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("case bank must contain a non-empty cases array")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"case {index} must be an object")
        case_id = str(raw.get("id") or "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{1,79}", case_id):
            raise ValueError(f"case {index} has an invalid id")
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        if not isinstance(raw.get("prompt"), str) or not raw["prompt"].strip():
            raise ValueError(f"case {case_id} needs a prompt")
        if raw.get("expected_verdict") not in {"allow", "block", "needs_evidence"}:
            raise ValueError(f"case {case_id} has an invalid expected_verdict")
        grader = raw.get("grader")
        if not isinstance(grader, dict) or grader.get("type") not in ALL_GRADER_TYPES:
            raise ValueError(f"case {case_id} has an invalid grader")
        validate_case_oracle_binding(case_id, raw)
        if grader.get("type") == "document_relations":
            _validate_document_relations(
                case_id,
                grader.get("claims"),
                raw.get("documents") or [],
            )
        if grader.get("type") == "patch_tests":
            if not built_in and not allow_external_patch_tests:
                raise ValueError("external patch_tests require an authenticated case pack")
            _validate_patch_fixture(case_id, grader.get("fixture"))
            allowed_files = grader.get("allowed_files")
            if (
                not isinstance(allowed_files, list)
                or not allowed_files
                or any(
                    not isinstance(value, str) or value not in grader["fixture"]
                    for value in allowed_files
                )
            ):
                raise ValueError(f"case {case_id} patch grader needs fixture-bound allowed_files")
            validate_patch_oracle(case_id, grader)
        elif grader.get("type") in PROOF_GRADER_TYPES:
            validate_private_oracle(raw)
        copied = dict(raw)
        copied["category"] = str(raw.get("category") or "custom")
        copied["domain"] = str(raw.get("domain") or copied["category"])
        difficulty = str(raw.get("difficulty") or "medium")
        if difficulty not in {"easy", "medium", "hard", "expert"}:
            raise ValueError(f"case {case_id} difficulty must be easy, medium, hard, or expert")
        copied["difficulty"] = difficulty
        copied["documents"] = _case_documents(case_id, raw.get("documents") or [])
        copied["grader"] = dict(grader)
        validate_visible_source_bindings(copied)
        cases.append(copied)
    return cases


def _case_pack_metadata(
    payload: Any,
    *,
    source_sha256: str,
    built_in: bool,
    key_env: str,
) -> dict[str, Any]:
    if built_in:
        return {
            "source": "builtin",
            "source_sha256": source_sha256,
            "oracle_mode": "live_or_inline",
            "oracle_independent": False,
            "seal": {
                "algorithm": None,
                "key_id": None,
                "verified": False,
                "status": "builtin_not_sealed",
            },
        }
    if not isinstance(payload, dict):
        return {
            "source": "external",
            "source_sha256": source_sha256,
            "oracle_mode": "unspecified",
            "oracle_independent": False,
            "seal": {
                "algorithm": None,
                "key_id": None,
                "verified": False,
                "status": "manifest_missing",
            },
        }
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        return {
            "source": "external",
            "source_sha256": source_sha256,
            "oracle_mode": "unspecified",
            "oracle_independent": False,
            "seal": {
                "algorithm": None,
                "key_id": None,
                "verified": False,
                "status": "manifest_missing",
            },
        }
    if manifest.get("schema_version") != 2 or manifest.get("taxonomy_version") != TAXONOMY_VERSION:
        raise ValueError("external case pack uses an unsupported manifest or taxonomy schema")
    signature = manifest.get("signature")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("signature", None)
    signature = signature if isinstance(signature, dict) else {}
    algorithm = str(signature.get("algorithm") or "")
    key_id = str(signature.get("key_id") or "")
    provided = str(signature.get("value") or "")
    key = os.environ.get(key_env, "")
    canonical = json.dumps(
        {
            "manifest": unsigned_manifest,
            "cases": payload.get("cases"),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    expected = hmac.new(key.encode("utf-8"), canonical, hashlib.sha256).hexdigest() if key else ""
    key_commitment = str(signature.get("key_commitment") or "")
    commitment_valid = bool(
        re.fullmatch(r"[0-9a-f]{64}", key_commitment)
        and key
        and hmac.compare_digest(key_commitment, hashlib.sha256(key.encode("utf-8")).hexdigest())
    )
    verified = bool(
        algorithm == "hmac-sha256"
        and key_id
        and re.fullmatch(r"[0-9a-f]{64}", provided)
        and expected
        and hmac.compare_digest(provided, expected)
        and commitment_valid
    )
    if key and not verified:
        raise ValueError("external case-pack signature verification failed")
    precommitted = str(unsigned_manifest.get("selection_sha256") or "")
    contract_sha256 = str(unsigned_manifest.get("contract_sha256") or "")
    public_tasks_sha256 = str(unsigned_manifest.get("public_tasks_sha256") or "")
    schedule_sha256 = str(unsigned_manifest.get("schedule_sha256") or "")
    execution_seed = unsigned_manifest.get("execution_seed")
    execution_repetitions = unsigned_manifest.get("execution_repetitions")
    scheduled_contenders = unsigned_manifest.get("scheduled_contenders")
    runner_attestation = unsigned_manifest.get("runner_attestation")
    if precommitted and not re.fullmatch(r"[0-9a-f]{64}", precommitted):
        raise ValueError("manifest selection_sha256 must be 64 lowercase hex")
    if contract_sha256 and not re.fullmatch(r"[0-9a-f]{64}", contract_sha256):
        raise ValueError("manifest contract_sha256 must be 64 lowercase hex")
    if public_tasks_sha256 and not re.fullmatch(r"[0-9a-f]{64}", public_tasks_sha256):
        raise ValueError("manifest public_tasks_sha256 must be 64 lowercase hex")
    if schedule_sha256 and not re.fullmatch(r"[0-9a-f]{64}", schedule_sha256):
        raise ValueError("manifest schedule_sha256 must be 64 lowercase hex")
    if execution_seed is not None and not isinstance(execution_seed, int):
        raise ValueError("manifest execution_seed must be an integer")
    if execution_repetitions is not None and (
        not isinstance(execution_repetitions, int) or execution_repetitions < 1
    ):
        raise ValueError("manifest execution_repetitions must be positive")
    if not isinstance(scheduled_contenders, list):
        scheduled_contenders = []
    if any(
        not isinstance(value, str) or not value.strip() for value in scheduled_contenders
    ) or len(set(scheduled_contenders)) != len(scheduled_contenders):
        raise ValueError("manifest scheduled_contenders must be unique names")
    if not isinstance(runner_attestation, dict):
        runner_attestation = {}
    if runner_attestation and (
        runner_attestation.get("algorithm") != "hmac-sha256"
        or not str(runner_attestation.get("key_id") or "")
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(runner_attestation.get("key_commitment") or ""),
        )
    ):
        raise ValueError("manifest runner_attestation is invalid")
    authored_by = unsigned_manifest.get("authored_by")
    if not isinstance(authored_by, list):
        authored_by = []
    return {
        "source": "external",
        "source_sha256": source_sha256,
        "pack_id": str(unsigned_manifest.get("pack_id") or ""),
        "issuer": str(unsigned_manifest.get("issuer") or ""),
        "evaluator": str(
            unsigned_manifest.get("evaluator") or unsigned_manifest.get("issuer") or ""
        ),
        "execution_authority": str(unsigned_manifest.get("execution_authority") or ""),
        "runner_id": str(unsigned_manifest.get("runner_id") or ""),
        "runner_attestation": dict(runner_attestation),
        "created_at": str(unsigned_manifest.get("created_at") or ""),
        "expires_at": str(unsigned_manifest.get("expires_at") or ""),
        "authored_by": [str(value) for value in authored_by if isinstance(value, str)],
        "oracle_mode": str(unsigned_manifest.get("oracle_mode") or "unspecified"),
        "taxonomy_version": unsigned_manifest.get("taxonomy_version"),
        "contract_sha256": contract_sha256 or None,
        "public_tasks_sha256": public_tasks_sha256 or None,
        "execution_seed": execution_seed,
        "execution_repetitions": execution_repetitions,
        "scheduled_contenders": list(scheduled_contenders),
        "precommitted_schedule_sha256": schedule_sha256 or None,
        "precommitted_selection_sha256": precommitted or None,
        "nonce_commitment": str(unsigned_manifest.get("nonce_commitment") or ""),
        "seal": {
            "algorithm": algorithm or None,
            "key_id": key_id or None,
            "verified": verified,
            "key_commitment": key_commitment or None,
            "status": (
                "verified" if verified else ("key_unavailable" if not key else "signature_missing")
            ),
        },
    }


def _case_has_frozen_oracle(case: dict[str, Any]) -> bool:
    grader = case.get("grader")
    if not isinstance(grader, dict):
        return False
    if grader.get("oracle_provenance") != "frozen_external_pack":
        return False
    if grader.get("type") not in PROOF_GRADER_TYPES:
        return False
    try:
        validate_case_oracle_binding(str(case.get("id") or "case"), case)
        if grader.get("type") == "patch_tests":
            validate_patch_oracle(str(case.get("id") or "case"), grader)
        else:
            validate_private_oracle(case)
    except ValueError:
        return False
    return True


def select_case_bank(
    cases: list[dict[str, Any]],
    *,
    split: str,
    seed: int,
    holdout_fraction: float,
    rotation_index: int,
    rotation_size: int,
) -> list[dict[str, Any]]:
    """Select a stable split and rotating window without exposing grader results."""

    if split not in {"all", "development", "heldout"}:
        raise ValueError(f"invalid case split: {split}")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    selected: list[dict[str, Any]] = []
    for case in cases:
        digest = hashlib.sha256(f"split:{seed}:{case['id']}".encode()).digest()
        unit_interval = int.from_bytes(digest, "big") / (1 << 256)
        heldout = unit_interval < holdout_fraction
        if split == "all" or (split == "heldout") == heldout:
            selected.append(case)
    selected.sort(
        key=lambda case: hashlib.sha256(f"rotation:{seed}:{case['id']}".encode()).digest()
    )
    if not selected or rotation_size == 0 or rotation_size >= len(selected):
        return selected
    start = (rotation_index * rotation_size) % len(selected)
    return [selected[(start + offset) % len(selected)] for offset in range(rotation_size)]


def _case_bank_hash(cases: list[dict[str, Any]]) -> str:
    canonical_cases: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda value: str(value["id"])):
        copied = dict(case)
        grader = dict(copied.get("grader") or {})
        grader.pop("answer_key", None)
        copied["grader"] = grader
        canonical_cases.append(copied)
    canonical = json.dumps(
        canonical_cases,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_live_grader(
    case: dict[str, Any],
    engine: CortheonEngine,
) -> dict[str, Any]:
    resolved = dict(case)
    grader = dict(case["grader"])
    if grader["type"] == "current_versions" and not isinstance(grader.get("answer_key"), dict):
        packages = grader.get("packages")
        if not isinstance(packages, list) or not packages:
            raise ValueError(f"case {case['id']} needs packages for live grading")
        grader["answer_key"] = {
            str(package): engine.pypi.fetch(str(package))[0].version for package in packages
        }
    elif grader["type"] == "pypi_metadata" and not isinstance(grader.get("answer_key"), dict):
        package = str(grader.get("package") or "")
        metadata, _evidence = engine.pypi.fetch(package)
        grader["answer_key"] = {
            "version": metadata.version,
            "requires_python": metadata.requires_python,
            "release_upload_time": (
                metadata.release_upload_time.isoformat() if metadata.release_upload_time else None
            ),
        }
    resolved["grader"] = grader
    return resolved


def _case_documents(case_id: str, documents: Any) -> list[dict[str, str]]:
    if not isinstance(documents, list) or len(documents) > 32:
        raise ValueError(f"case {case_id} documents must be an array of at most 32")
    normalized: list[dict[str, str]] = []
    for index, document in enumerate(documents):
        if not isinstance(document, dict) or not isinstance(document.get("text"), str):
            raise ValueError(f"case {case_id} document {index} needs text")
        normalized.append(
            {
                "uri": str(document.get("uri") or f"benchmark://{case_id}/{index}"),
                "title": str(document.get("title") or f"Document {index + 1}"),
                "source_type": "benchmark_document",
                "text": str(document["text"]),
            }
        )
    return normalized


def _validate_patch_fixture(case_id: str, fixture: Any) -> None:
    if not isinstance(fixture, dict) or not 1 <= len(fixture) <= 20:
        raise ValueError(f"case {case_id} patch fixture must contain 1-20 files")
    total = 0
    for relative, content in fixture.items():
        path = Path(str(relative))
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or not isinstance(content, str)
        ):
            raise ValueError(f"case {case_id} has an unsafe patch fixture path")
        total += len(content)
    if total > 500_000:
        raise ValueError(f"case {case_id} patch fixture is too large")


def _validate_document_relations(case_id: str, claims: Any, documents: Any) -> None:
    if not isinstance(claims, list) or not 1 <= len(claims) <= 32:
        raise ValueError(f"case {case_id} document grader needs 1-32 claims")
    seen: set[str] = set()
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise ValueError(f"case {case_id} claim {index} must be an object")
        claim_id = str(claim.get("id") or "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{1,79}", claim_id) or claim_id in seen:
            raise ValueError(f"case {case_id} claim {index} has an invalid id")
        seen.add(claim_id)
        if claim.get("relation") != "identity":
            raise ValueError(f"case {case_id} claim {claim_id} has an invalid relation")
        for field in ("subject_aliases", "object_aliases", "source_aliases"):
            aliases = claim.get(field)
            if (
                not isinstance(aliases, list)
                or not 1 <= len(aliases) <= 16
                or any(
                    not isinstance(alias, str) or not alias.strip() or len(alias) > 200
                    for alias in aliases
                )
            ):
                raise ValueError(f"case {case_id} claim {claim_id} has invalid {field}")
        if not isinstance(documents, list):
            raise ValueError(f"case {case_id} claim {claim_id} needs source documents")
        matching_sources = [
            document
            for document in documents
            if isinstance(document, dict)
            and any(
                alias.casefold()
                in {
                    str(document.get("title") or "").casefold(),
                    str(document.get("uri") or "").casefold(),
                }
                for alias in claim["source_aliases"]
            )
        ]
        relation_is_in_source = any(
            isinstance(document.get("text"), str)
            and any(
                alias.casefold() in document["text"].casefold()
                for alias in claim["subject_aliases"]
            )
            and any(
                alias.casefold() in document["text"].casefold() for alias in claim["object_aliases"]
            )
            for document in matching_sources
        )
        if not relation_is_in_source:
            raise ValueError(f"case {case_id} claim {claim_id} is not bound to its source document")

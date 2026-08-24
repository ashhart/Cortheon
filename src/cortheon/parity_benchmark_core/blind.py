from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cortheon.benchmark_core.outcomes import (
    EvaluationOutcome,
    is_authenticated_withhold,
    is_exact_terminal_success,
)
from cortheon.parity import (
    evaluation_schedule,
    evaluation_schedule_hash,
    public_task_hash,
)
from cortheon.parity_benchmark_core._compat import facade, generated_at_now
from cortheon.parity_benchmark_core.casepack import _case_documents
from cortheon.parity_benchmark_core.contender import (
    _observed_model_id,
    _visible_input_sha256,
)
from cortheon.parity_benchmark_core.metrics import (
    _benchmark_input_sha256,
    _candidate_identity,
    _completion_origin,
    _cortheon_outcome,
    _input_symmetry,
    _result_cost,
)
from cortheon.parity_benchmark_core.models import Contender


def _load_public_case_pack(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("manifest"), dict):
        raise ValueError("public case pack needs a manifest")
    manifest = dict(payload["manifest"])
    if manifest.get("schema_version") != 2:
        raise ValueError("public case pack uses an unsupported manifest schema")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("public case pack needs a non-empty cases array")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"public case {index} must be an object")
        if "grader" in raw or "expected_verdict" in raw:
            raise ValueError("public task pack must not contain graders or labels")
        case_id = str(raw.get("id") or "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{1,79}", case_id) or case_id in seen:
            raise ValueError(f"public case {index} has an invalid or duplicate id")
        seen.add(case_id)
        prompt = raw.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"public case {case_id} needs a prompt")
        difficulty = str(raw.get("difficulty") or "medium")
        if difficulty not in {"easy", "medium", "hard", "expert"}:
            raise ValueError(f"public case {case_id} has invalid difficulty")
        category = str(raw.get("category") or "custom")
        cases.append(
            {
                "id": case_id,
                "category": category,
                "domain": str(raw.get("domain") or category),
                "difficulty": difficulty,
                "prompt": prompt,
                "documents": _case_documents(
                    case_id,
                    raw.get("documents") or [],
                ),
            }
        )
    actual_hash = public_task_hash(cases)
    expected_hash = str(manifest.get("public_tasks_sha256") or "")
    if not expected_hash or not hmac.compare_digest(expected_hash, actual_hash):
        raise ValueError("public task projection hash does not match its manifest")
    if manifest.get("split") != "heldout":
        raise ValueError("public task pack must be evaluator-selected heldout work")
    runner_attestation = manifest.get("runner_attestation")
    if not isinstance(runner_attestation, dict):
        raise ValueError("public task pack needs runner attestation metadata")
    if (
        runner_attestation.get("algorithm") != "hmac-sha256"
        or not str(runner_attestation.get("key_id") or "")
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(runner_attestation.get("key_commitment") or ""),
        )
    ):
        raise ValueError("public runner attestation metadata is invalid")
    execution_seed = manifest.get("execution_seed")
    execution_repetitions = manifest.get("execution_repetitions")
    scheduled_contenders = manifest.get("scheduled_contenders")
    schedule_sha256 = str(manifest.get("schedule_sha256") or "")
    if not isinstance(execution_seed, int):
        raise ValueError("public task pack needs an integer execution_seed")
    if not isinstance(execution_repetitions, int) or execution_repetitions < 1:
        raise ValueError("public task pack needs positive execution_repetitions")
    if (
        not isinstance(scheduled_contenders, list)
        or not scheduled_contenders
        or any(not isinstance(value, str) or not value.strip() for value in scheduled_contenders)
        or len(set(scheduled_contenders)) != len(scheduled_contenders)
    ):
        raise ValueError("public task pack needs unique scheduled_contenders")
    expected_schedule = evaluation_schedule_hash(
        [str(case["id"]) for case in cases],
        [str(value) for value in scheduled_contenders],
        execution_repetitions,
        execution_seed,
    )
    if not (
        re.fullmatch(r"[0-9a-f]{64}", schedule_sha256)
        and hmac.compare_digest(schedule_sha256, expected_schedule)
    ):
        raise ValueError("public execution schedule does not match its manifest")
    return cases, {
        "source": "external_public_projection",
        "split": "heldout",
        "pack_id": str(manifest.get("pack_id") or ""),
        "issuer": str(manifest.get("issuer") or ""),
        "execution_authority": str(manifest.get("execution_authority") or ""),
        "runner_id": str(manifest.get("runner_id") or ""),
        "runner_attestation": dict(runner_attestation),
        "created_at": str(manifest.get("created_at") or ""),
        "expires_at": str(manifest.get("expires_at") or ""),
        "contract_sha256": str(manifest.get("contract_sha256") or ""),
        "selection_sha256": str(manifest.get("selection_sha256") or ""),
        "public_tasks_sha256": actual_hash,
        "execution_seed": execution_seed,
        "execution_repetitions": execution_repetitions,
        "scheduled_contenders": [str(value) for value in scheduled_contenders],
        "precommitted_schedule_sha256": schedule_sha256,
        "selected_cases": len(cases),
    }


def run_blind_submissions(
    contenders: list[Contender],
    cases: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
    timeout: float,
    max_tokens: int,
    case_bank: dict[str, Any],
    secret_env_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run contenders without possessing labels, graders, or answer keys."""

    if any(contender.kind == "cli" for contender in contenders):
        raise ValueError("blind qualification forbids process-local CLI contenders")
    expected_seed = int(case_bank.get("execution_seed") or 0)
    expected_repetitions = int(case_bank.get("execution_repetitions") or 0)
    contender_by_name = {contender.name: contender for contender in contenders}
    case_by_id = {str(case["id"]): case for case in cases}
    if len(contender_by_name) != len(contenders):
        raise ValueError("blind contender names must be unique")
    if seed != expected_seed or repetitions != expected_repetitions:
        raise ValueError(
            "blind execution seed and repetitions must exactly match the "
            "evaluator-committed schedule"
        )
    scheduled_names = [str(value) for value in case_bank.get("scheduled_contenders") or []]
    if set(scheduled_names) != set(contender_by_name):
        raise ValueError("blind contenders must exactly match the evaluator-committed schedule")
    schedule = evaluation_schedule(
        list(case_by_id),
        scheduled_names,
        repetitions,
        seed,
    )
    schedule_sha256 = evaluation_schedule_hash(
        list(case_by_id),
        scheduled_names,
        repetitions,
        seed,
    )
    if not hmac.compare_digest(
        str(case_bank.get("precommitted_schedule_sha256") or ""),
        schedule_sha256,
    ):
        raise ValueError("blind execution schedule digest mismatch")
    aliases = {name: f"candidate_{index + 1}" for index, name in enumerate(sorted(scheduled_names))}
    rows: list[dict[str, Any]] = []
    for cell in schedule:
        run_index = int(cell["run"])
        repetition = int(cell["repetition"])
        case = case_by_id[str(cell["case_id"])]
        contender = contender_by_name[str(cell["contender_name"])]
        started = time.perf_counter()
        try:
            result = facade().call_contender(
                contender,
                case,
                timeout=timeout,
                max_tokens=max_tokens,
                secret_env_names=secret_env_names,
            )
            evaluator_outcome = asdict(result.evaluator_outcome)
            if not is_exact_terminal_success(result.evaluator_outcome):
                if is_authenticated_withhold(result.evaluator_outcome):
                    rows.append(
                        {
                            "run": run_index,
                            "repetition": repetition,
                            "case_id": case["id"],
                            "candidate": aliases[contender.name],
                            "latency_ms": round(result.latency_ms, 2),
                            "cost": _result_cost(
                                result.metadata,
                                contender,
                                latency_ms=result.latency_ms,
                            ),
                            "completion_origin": _completion_origin(contender, result.metadata),
                            "observed_model_id": _observed_model_id(result.metadata),
                            "input_sha256": _benchmark_input_sha256(result.metadata),
                            "cortheon_outcome": _cortheon_outcome(result.metadata),
                            "evaluator_outcome": evaluator_outcome,
                            "failure_owner": None,
                        }
                    )
                    continue
                rows.append(
                    {
                        "run": run_index,
                        "repetition": repetition,
                        "case_id": case["id"],
                        "candidate": aliases[contender.name],
                        "input_sha256": _benchmark_input_sha256(result.metadata),
                        "latency_ms": round(result.latency_ms, 2),
                        "evaluator_outcome": evaluator_outcome,
                        "failure_owner": "candidate",
                        "error": "delivery_failure",
                    }
                )
                continue
            rows.append(
                {
                    "run": run_index,
                    "repetition": repetition,
                    "case_id": case["id"],
                    "candidate": aliases[contender.name],
                    "latency_ms": round(result.latency_ms, 2),
                    "cost": _result_cost(
                        result.metadata,
                        contender,
                        latency_ms=result.latency_ms,
                    ),
                    "completion_origin": _completion_origin(
                        contender,
                        result.metadata,
                    ),
                    "observed_model_id": _observed_model_id(result.metadata),
                    "input_sha256": _benchmark_input_sha256(result.metadata),
                    "answer": {
                        "text": result.answer,
                        "sha256": hashlib.sha256(result.answer.encode("utf-8")).hexdigest(),
                    },
                    "cortheon_outcome": _cortheon_outcome(result.metadata),
                    "evaluator_outcome": evaluator_outcome,
                    "failure_owner": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "run": run_index,
                    "repetition": repetition,
                    "case_id": case["id"],
                    "candidate": aliases[contender.name],
                    "input_sha256": _visible_input_sha256(case),
                    "latency_ms": round(
                        (time.perf_counter() - started) * 1_000,
                        2,
                    ),
                    "evaluator_outcome": asdict(_blind_error_outcome(contender, exc)),
                    "failure_owner": "candidate",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    symmetry_rows = [
        {
            **row,
            "classification": "error" if "error" in row else "ungraded",
        }
        for row in rows
    ]
    return {
        "schema_version": 3,
        "artifact": "cortheon_blind_submission",
        "generated_at": generated_at_now(),
        "methodology": {
            "repetitions": repetitions,
            "seed": seed,
            "candidate_label_channel": "not_present_on_runner",
            "grader_material_on_runner": False,
            "case_pack_secrets_exposed_to_cli": False,
            "verdict_source": "deferred_independent_evaluator",
            "document_channel": "inline_model_visible",
            "input_symmetry": _input_symmetry(symmetry_rows),
        },
        "case_bank": {
            **case_bank,
            "schedule_sha256": schedule_sha256,
            "schedule_precommitted": True,
        },
        "candidates": {
            aliases[contender.name]: _candidate_identity(
                contender,
                rows,
                aliases[contender.name],
            )
            for contender in contenders
        },
        "cases": [
            {
                "id": case["id"],
                "category": case["category"],
                "domain": case["domain"],
                "difficulty": case["difficulty"],
            }
            for case in cases
        ],
        "rows": rows,
    }


def _blind_error_outcome(contender: Contender, exc: Exception) -> EvaluationOutcome:
    reason = "timeout" if "timed out" in str(exc).casefold() else "process_error"
    if contender.kind == "frontier":
        return EvaluationOutcome("openai_responses", "transport_error", "responses_status", reason)
    return EvaluationOutcome("openai_chat", "transport_error", "chat_finish_reason", reason)


def attest_blind_submission(
    artifact: dict[str, Any],
    *,
    key_env: str,
) -> dict[str, Any]:
    """Authenticate a completed artifact on the independent runner."""

    key = os.environ.get(key_env, "")
    if len(key.encode("utf-8")) < 32:
        raise ValueError(f"{key_env} must contain at least 32 bytes of evaluator-runner secret")
    case_bank = artifact.get("case_bank")
    metadata = case_bank.get("runner_attestation") if isinstance(case_bank, dict) else None
    if not isinstance(metadata, dict):
        raise ValueError("blind artifact has no runner attestation commitment")
    commitment = hashlib.sha256(key.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(
        str(metadata.get("key_commitment") or ""),
        commitment,
    ):
        raise ValueError("runner attestation key does not match the sealed pack")
    attested = dict(artifact)
    attested["attestation"] = {
        "algorithm": "hmac-sha256",
        "key_id": str(metadata.get("key_id") or ""),
        "value": hmac.new(
            key.encode("utf-8"),
            _canonical_blind_submission(attested),
            hashlib.sha256,
        ).hexdigest(),
    }
    return attested


def _canonical_blind_submission(artifact: dict[str, Any]) -> bytes:
    unsigned = dict(artifact)
    unsigned.pop("attestation", None)
    return json.dumps(
        unsigned,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

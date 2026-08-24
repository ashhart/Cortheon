"""Grade oracle-free Cortheon submissions on an independent evaluator host."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cortheon.benchmark import (
    _canonical_blind_submission,
    _case_bank_hash,
    _classification,
    _load_case_pack,
    _observed_verdict,
    _paired_candidate_comparisons,
    _visible_input_sha256,
    grade_answer,
)
from cortheon.benchmark_core.outcomes import (
    is_authenticated_withhold,
    is_exact_terminal_success,
)
from cortheon.blind_evaluator_cli import main
from cortheon.parity import (
    evaluate_frontier_parity,
    evaluation_schedule,
    evaluation_schedule_hash,
    load_parity_contract,
    public_task_hash,
)
from cortheon.parity_benchmark_core.grading import grade_authenticated_withhold
from cortheon.parity_gates.summary_validation import canonical_summary
from cortheon.parity_timestamps import ordering_holds, parse_utc_timestamp


def grade_blind_submission(
    pack_path: Path,
    submission_path: Path,
    *,
    contract_path: Path,
    key_env: str,
    runner_key_env: str,
) -> dict[str, Any]:
    """Join private oracles with a completed public-task submission."""

    loaded = _load_case_pack(pack_path, key_env=key_env)
    seal = loaded.metadata.get("seal")
    if not isinstance(seal, dict) or seal.get("verified") is not True:
        raise ValueError("private challenge pack is not authenticated")
    if loaded.metadata.get("oracle_independent") is not True:
        raise ValueError("private challenge pack does not contain frozen oracles")
    contract, contract_sha256 = load_parity_contract(contract_path)
    if not hmac.compare_digest(
        str(loaded.metadata.get("contract_sha256") or ""),
        contract_sha256,
    ):
        raise ValueError("private challenge pack does not bind this contract")
    submission = json.loads(submission_path.expanduser().read_text(encoding="utf-8"))
    if (
        not isinstance(submission, dict)
        or submission.get("schema_version") != 3
        or submission.get("artifact") != "cortheon_blind_submission"
    ):
        raise ValueError("submission is not a Cortheon blind-submission artifact")
    # The attested execution time makes regrading timeless: pack validity is
    # judged at this moment, never at regrade wall-clock time.
    execution_completed_at = submission.get("generated_at")
    parse_utc_timestamp(execution_completed_at, field="submission generated_at")
    if not ordering_holds(
        loaded.metadata.get("created_at"),
        execution_completed_at,
        loaded.metadata.get("expires_at"),
    ):
        raise ValueError("submission generated_at must fall within the sealed pack validity window")
    runner_key = os.environ.get(runner_key_env, "")
    if len(runner_key.encode("utf-8")) < 32:
        raise ValueError(
            f"{runner_key_env} must contain at least 32 bytes of evaluator-runner secret"
        )
    runner_attestation = _mapping(loaded.metadata.get("runner_attestation"))
    key_commitment = hashlib.sha256(runner_key.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(
        str(runner_attestation.get("key_commitment") or ""),
        key_commitment,
    ):
        raise ValueError("runner attestation key does not match the private pack")
    attestation = _mapping(submission.get("attestation"))
    expected_attestation = hmac.new(
        runner_key.encode("utf-8"),
        _canonical_blind_submission(submission),
        hashlib.sha256,
    ).hexdigest()
    if not (
        attestation.get("algorithm") == "hmac-sha256"
        and attestation.get("key_id") == runner_attestation.get("key_id")
        and hmac.compare_digest(
            str(attestation.get("value") or ""),
            expected_attestation,
        )
    ):
        raise ValueError("blind submission runner attestation is invalid")
    submission_bank = _mapping(submission.get("case_bank"))
    if submission_bank.get("pack_id") != loaded.metadata.get("pack_id"):
        raise ValueError("submission pack_id does not match the private pack")
    if submission_bank.get("contract_sha256") != loaded.metadata.get("contract_sha256"):
        raise ValueError("submission contract digest does not match the private pack")
    for field in ("execution_authority", "runner_id"):
        if submission_bank.get(field) != loaded.metadata.get(field):
            raise ValueError(f"submission {field} does not match the private pack")

    public_cases = [item for item in submission.get("cases") or [] if isinstance(item, dict)]
    public_ids = [str(item.get("id") or "") for item in public_cases]
    if not public_ids or len(public_ids) != len(set(public_ids)):
        raise ValueError("submission cases are empty or duplicated")
    private_by_id = {str(case["id"]): case for case in loaded.cases}
    if any(case_id not in private_by_id for case_id in public_ids):
        raise ValueError("submission contains a case outside the private pack")
    selected = [private_by_id[case_id] for case_id in public_ids]
    actual_public_hash = public_task_hash(selected)
    expected_public_hash = str(loaded.metadata.get("public_tasks_sha256") or "")
    if not (
        expected_public_hash
        and hmac.compare_digest(expected_public_hash, actual_public_hash)
        and hmac.compare_digest(
            str(submission_bank.get("public_tasks_sha256") or ""),
            actual_public_hash,
        )
    ):
        raise ValueError("public task projection does not match the private pack")

    submission_rows = [item for item in submission.get("rows") or [] if isinstance(item, dict)]
    candidates = _mapping(submission.get("candidates"))
    methodology = _mapping(submission.get("methodology"))
    repetitions = int(methodology.get("repetitions") or 0)
    if not candidates or repetitions < 1:
        raise ValueError("submission has no contenders or repetitions")
    execution_seed = loaded.metadata.get("execution_seed")
    execution_repetitions = loaded.metadata.get("execution_repetitions")
    scheduled_contenders = [
        str(value) for value in loaded.metadata.get("scheduled_contenders") or []
    ]
    observed_contenders = {
        str(identity.get("name") or "")
        for identity in candidates.values()
        if isinstance(identity, dict)
    }
    expected_aliases = {
        f"candidate_{index + 1}": name for index, name in enumerate(sorted(scheduled_contenders))
    }
    observed_aliases = {
        str(alias): str(identity.get("name") or "")
        for alias, identity in candidates.items()
        if isinstance(identity, dict)
    }
    if (
        not isinstance(execution_seed, int)
        or repetitions != execution_repetitions
        or int(methodology.get("seed") or 0) != execution_seed
        or observed_contenders != set(scheduled_contenders)
        or len(observed_contenders) != len(candidates)
        or observed_aliases != expected_aliases
    ):
        raise ValueError("submission does not match the precommitted execution participants")
    expected_schedule = evaluation_schedule(
        public_ids,
        scheduled_contenders,
        repetitions,
        execution_seed,
    )
    actual_schedule_hash = evaluation_schedule_hash(
        public_ids,
        scheduled_contenders,
        repetitions,
        execution_seed,
    )
    precommitted_schedule_hash = str(loaded.metadata.get("precommitted_schedule_sha256") or "")
    if not (
        hmac.compare_digest(
            precommitted_schedule_hash,
            actual_schedule_hash,
        )
        and hmac.compare_digest(
            str(submission_bank.get("precommitted_schedule_sha256") or ""),
            actual_schedule_hash,
        )
        and hmac.compare_digest(
            str(submission_bank.get("schedule_sha256") or ""),
            actual_schedule_hash,
        )
        and submission_bank.get("schedule_precommitted") is True
    ):
        raise ValueError("submission execution schedule digest mismatch")
    if len(submission_rows) != len(expected_schedule):
        raise ValueError("submission execution schedule is incomplete")

    rows: list[dict[str, Any]] = []
    for raw, scheduled in zip(
        submission_rows,
        expected_schedule,
        strict=True,
    ):
        for field in ("run", "repetition", "case_id", "candidate"):
            if raw.get(field) != scheduled[field]:
                raise ValueError(
                    f"submission row {scheduled['run']} violates the precommitted {field}"
                )
        case_id = str(raw.get("case_id") or "")
        case = private_by_id.get(case_id)
        if case is None:
            raise ValueError(f"submission row has unknown case: {case_id}")
        expected_input_hash = _visible_input_sha256(case)
        if not hmac.compare_digest(
            str(raw.get("input_sha256") or ""),
            expected_input_hash,
        ):
            raise ValueError(f"submission row {case_id} did not use the committed task input")
        common = {
            "run": raw.get("run"),
            "repetition": raw.get("repetition"),
            "case_id": case_id,
            "category": case["category"],
            "domain": case["domain"],
            "difficulty": case["difficulty"],
            "candidate": raw.get("candidate"),
            "expected_verdict": case["expected_verdict"],
            "task_class": case.get("task_class"),
            "oracle_version": case["grader"].get("oracle_version"),
            "latency_ms": raw.get("latency_ms"),
            "cost": raw.get("cost"),
            "completion_origin": raw.get("completion_origin", "error"),
            "observed_model_id": raw.get("observed_model_id"),
            "input_sha256": raw.get("input_sha256"),
            "cortheon_outcome": raw.get("cortheon_outcome"),
            "evaluator_outcome": raw.get("evaluator_outcome"),
            "failure_owner": raw.get("failure_owner"),
        }
        outcome = _mapping(raw.get("evaluator_outcome"))
        authenticated_withhold = is_authenticated_withhold(outcome)
        terminal_success = is_exact_terminal_success(outcome)
        owner = raw.get("failure_owner")
        if terminal_success or authenticated_withhold:
            if owner is not None:
                raise ValueError(f"submission row {case_id} misowns a terminal outcome")
        elif owner not in {"candidate", "external_infrastructure"}:
            raise ValueError(f"submission row {case_id} has no closed failure owner")
        if isinstance(raw.get("error"), str):
            if terminal_success or authenticated_withhold:
                raise ValueError(f"submission error row {case_id} claims terminal success")
            rows.append(
                {
                    **common,
                    "passed": False,
                    "verified_completion": False,
                    "verification_method": str(case["grader"]["type"]),
                    "verification_assurance": "not_graded",
                    "proof_eligible": False,
                    "grade_failures": ["contender_error"],
                    "observed_verdict": "error",
                    "classification": "error",
                    "error": raw["error"],
                }
            )
            continue
        if authenticated_withhold:
            grade = grade_authenticated_withhold(case)
            observed = "block"
            rows.append(
                {
                    **common,
                    "passed": grade["passed"],
                    "verified_completion": bool(grade["passed"] and grade["proof_eligible"]),
                    "verification_method": grade["method"],
                    "verification_assurance": grade["assurance"],
                    "proof_eligible": grade["proof_eligible"],
                    "grade_failures": grade["failures"],
                    "observed_verdict": observed,
                    "classification": _classification(str(case["expected_verdict"]), observed),
                    "verdict_source": "independent_evaluator",
                }
            )
            continue
        if not terminal_success:
            rows.append(
                {
                    **common,
                    "passed": False,
                    "verified_completion": False,
                    "verification_method": str(case["grader"]["type"]),
                    "verification_assurance": "not_graded",
                    "proof_eligible": False,
                    "grade_failures": ["delivery_failure"],
                    "observed_verdict": "error",
                    "classification": "error",
                    "error": "delivery_failure",
                }
            )
            continue
        answer_record = _mapping(raw.get("answer"))
        answer = answer_record.get("text")
        if not isinstance(answer, str) or not answer:
            raise ValueError(f"submission row {case_id} has no answer text")
        answer_hash = hashlib.sha256(answer.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(
            str(answer_record.get("sha256") or ""),
            answer_hash,
        ):
            raise ValueError(f"submission row {case_id} answer digest mismatch")
        grade = grade_answer(case, answer)
        observed = _observed_verdict(answer, case=case, grade=grade)
        rows.append(
            {
                **common,
                "passed": grade["passed"],
                "verified_completion": bool(grade["passed"] and grade["proof_eligible"]),
                "verification_method": grade["method"],
                "verification_assurance": grade["assurance"],
                "proof_eligible": grade["proof_eligible"],
                "grade_failures": grade["failures"],
                "observed_verdict": observed,
                "classification": _classification(
                    str(case["expected_verdict"]),
                    observed,
                ),
                "verdict_source": "independent_evaluator",
                "answer": {
                    "characters": len(answer),
                    "sha256": answer_hash,
                },
            }
        )

    summary = canonical_summary(rows, candidates)
    aliases = {
        str(identity.get("name")): str(alias)
        for alias, identity in candidates.items()
        if isinstance(identity, dict) and identity.get("name")
    }
    selection_hash = _case_bank_hash(selected)
    precommitted = str(loaded.metadata.get("precommitted_selection_sha256") or "")
    symmetry = _mapping(methodology.get("input_symmetry"))
    report: dict[str, Any] = {
        "schema_version": 7,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "methodology": {
            "repetitions": repetitions,
            "seed": methodology.get("seed"),
            "execution_completed_at": str(execution_completed_at),
            "schedule": "randomized across contender, case, and repetition",
            "grading": "deterministic and contender-blind",
            "candidate_label_channel": "withheld",
            "grader_material_on_runner": False,
            "case_pack_secrets_exposed_to_cli": False,
            "verdict_source": "independent_evaluator",
            "runner_attestation_verified": True,
            "document_channel": "inline_model_visible",
            "input_symmetry": symmetry,
            "input_symmetry_verified": symmetry.get("verified") is True,
        },
        "case_bank": {
            **loaded.metadata,
            "split": "heldout",
            "selection_sha256": selection_hash,
            "selection_precommitted": bool(
                precommitted and hmac.compare_digest(precommitted, selection_hash)
            ),
            "schedule_sha256": actual_schedule_hash,
            "schedule_precommitted": bool(
                precommitted_schedule_hash
                and hmac.compare_digest(
                    precommitted_schedule_hash,
                    actual_schedule_hash,
                )
            ),
            "selected_cases": len(selected),
            "total_cases": len(loaded.cases),
        },
        "candidates": candidates,
        "cases": [
            {
                "id": case["id"],
                "category": case["category"],
                "domain": case["domain"],
                "difficulty": case["difficulty"],
                "expected_verdict": case["expected_verdict"],
                "grader_type": case["grader"]["type"],
                "task_class": case.get("task_class"),
                "oracle_version": case["grader"].get("oracle_version"),
            }
            for case in selected
        ],
        "summary": summary,
        "paired_comparisons": _paired_candidate_comparisons(
            rows,
            aliases,
            seed=int(methodology.get("seed") or 0),
        ),
        "rows": rows,
    }
    report["release_identity"] = {
        "model": str(contract["contender_models"][str(contract["candidate"])]),
        "family": str(contract["candidate_family"]),
        "host": str(contract["candidate_host"]),
        "runtime_sha256": str(contract["candidate_runtime_sha256"]),
        "contract_sha256": contract_sha256,
        "pack_issuer": str(loaded.metadata.get("issuer") or ""),
        "pack_id": str(loaded.metadata.get("pack_id") or ""),
        "runner_id": str(loaded.metadata.get("runner_id") or ""),
        # The pack issuer is the declared grading authority; the pack may also
        # bind a distinct evaluator identity, which is what gets reported.
        "evaluator": str(loaded.metadata.get("evaluator") or loaded.metadata.get("issuer") or ""),
    }
    report["frontier_parity_gate"] = evaluate_frontier_parity(
        report,
        contract,
        contract_sha256=contract_sha256,
    )
    return report


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import asdict
from typing import Any

from cortheon.benchmark_core.outcomes import (
    EvaluationOutcome,
    is_authenticated_withhold,
    is_exact_terminal_success,
)
from cortheon.parity_benchmark_core._compat import facade, generated_at_now
from cortheon.parity_benchmark_core.casepack import _case_bank_hash
from cortheon.parity_benchmark_core.contender import _observed_model_id
from cortheon.parity_benchmark_core.grading import (
    _classification,
    _observed_verdict,
    _sandbox_image,
    grade_answer,
    grade_authenticated_withhold,
)
from cortheon.parity_benchmark_core.metrics import (
    _benchmark_input_sha256,
    _candidate_identity,
    _completion_origin,
    _cortheon_outcome,
    _input_symmetry,
    _result_cost,
    _summarize_candidate,
)
from cortheon.parity_benchmark_core.models import Contender
from cortheon.parity_benchmark_core.pairing import _paired_candidate_comparisons


def run_benchmark(
    contenders: list[Contender],
    cases: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
    timeout: float,
    max_tokens: int,
    include_answers: bool,
    case_bank: dict[str, Any] | None = None,
    secret_env_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Execute a randomized schedule with contender-blind deterministic graders."""

    rng = random.Random(seed)
    schedule = [
        (repetition, case, contender)
        for repetition in range(repetitions)
        for case in cases
        for contender in contenders
    ]
    rng.shuffle(schedule)
    aliases = {
        contender.name: f"candidate_{index + 1}"
        for index, contender in enumerate(rng.sample(contenders, len(contenders)))
    }
    rows: list[dict[str, Any]] = []
    for run_index, (repetition, case, contender) in enumerate(schedule, start=1):
        started = time.perf_counter()
        try:
            model_result = facade().call_contender(
                contender,
                case,
                timeout=timeout,
                max_tokens=max_tokens,
                secret_env_names=secret_env_names,
            )
            evaluator_outcome = asdict(model_result.evaluator_outcome)
            if not is_exact_terminal_success(model_result.evaluator_outcome):
                if is_authenticated_withhold(model_result.evaluator_outcome):
                    grade = grade_authenticated_withhold(case)
                    observed = "block"
                    rows.append(
                        {
                            "run": run_index,
                            "repetition": repetition + 1,
                            "case_id": case["id"],
                            "category": case["category"],
                            "domain": case.get("domain", case["category"]),
                            "difficulty": case.get("difficulty", "medium"),
                            "candidate": aliases[contender.name],
                            "passed": grade["passed"],
                            "verified_completion": bool(
                                grade["passed"] and grade["proof_eligible"]
                            ),
                            "verification_method": grade["method"],
                            "verification_assurance": grade["assurance"],
                            "proof_eligible": grade["proof_eligible"],
                            "task_class": case.get("task_class"),
                            "oracle_version": case["grader"].get("oracle_version"),
                            "grade_failures": grade["failures"],
                            "expected_verdict": case["expected_verdict"],
                            "observed_verdict": observed,
                            "classification": _classification(
                                str(case["expected_verdict"]), observed
                            ),
                            "completion_origin": _completion_origin(
                                contender, model_result.metadata
                            ),
                            "latency_ms": round(model_result.latency_ms, 2),
                            "evaluator_outcome": evaluator_outcome,
                            "failure_owner": None,
                        }
                    )
                    continue
                rows.append(
                    {
                        "run": run_index,
                        "repetition": repetition + 1,
                        "case_id": case["id"],
                        "category": case["category"],
                        "domain": case.get("domain", case["category"]),
                        "difficulty": case.get("difficulty", "medium"),
                        "candidate": aliases[contender.name],
                        "passed": False,
                        "verified_completion": False,
                        "verification_method": str(case["grader"].get("type")),
                        "verification_assurance": "not_graded",
                        "proof_eligible": False,
                        "task_class": case.get("task_class"),
                        "oracle_version": case["grader"].get("oracle_version"),
                        "grade_failures": ["delivery_failure"],
                        "expected_verdict": case["expected_verdict"],
                        "observed_verdict": "error",
                        "classification": "error",
                        "completion_origin": "error",
                        "latency_ms": round(model_result.latency_ms, 2),
                        "evaluator_outcome": evaluator_outcome,
                        "failure_owner": "candidate",
                        "error": "delivery_failure",
                    }
                )
                continue
            grade = grade_answer(case, model_result.answer)
            observed = _observed_verdict(
                model_result.answer,
                case=case,
                grade=grade,
            )
            expected = str(case["expected_verdict"])
            error_kind = _classification(expected, observed)
            answer_record: dict[str, Any] = {
                "characters": len(model_result.answer),
                "sha256": hashlib.sha256(model_result.answer.encode("utf-8")).hexdigest(),
            }
            if include_answers:
                answer_record["text"] = model_result.answer
            rows.append(
                {
                    "run": run_index,
                    "repetition": repetition + 1,
                    "case_id": case["id"],
                    "category": case["category"],
                    "domain": case.get("domain", case["category"]),
                    "difficulty": case.get("difficulty", "medium"),
                    "candidate": aliases[contender.name],
                    "passed": grade["passed"],
                    "verified_completion": bool(grade["passed"] and grade["proof_eligible"]),
                    "verification_method": grade["method"],
                    "verification_assurance": grade["assurance"],
                    "proof_eligible": grade["proof_eligible"],
                    "task_class": case.get("task_class"),
                    "oracle_version": case["grader"].get("oracle_version"),
                    "grade_failures": grade["failures"],
                    "expected_verdict": expected,
                    "observed_verdict": observed,
                    "classification": error_kind,
                    "latency_ms": round(model_result.latency_ms, 2),
                    "cost": _result_cost(
                        model_result.metadata,
                        contender,
                        latency_ms=model_result.latency_ms,
                    ),
                    "completion_origin": _completion_origin(
                        contender,
                        model_result.metadata,
                    ),
                    "observed_model_id": _observed_model_id(model_result.metadata),
                    "input_sha256": _benchmark_input_sha256(model_result.metadata),
                    "verdict_source": "independent_evaluator",
                    "answer": answer_record,
                    "cortheon_outcome": _cortheon_outcome(model_result.metadata),
                    "evaluator_outcome": evaluator_outcome,
                    "failure_owner": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "run": run_index,
                    "repetition": repetition + 1,
                    "case_id": case["id"],
                    "category": case["category"],
                    "domain": case.get("domain", case["category"]),
                    "difficulty": case.get("difficulty", "medium"),
                    "candidate": aliases[contender.name],
                    "passed": False,
                    "verified_completion": False,
                    "verification_method": str(case["grader"].get("type")),
                    "verification_assurance": "not_graded",
                    "proof_eligible": False,
                    "task_class": case.get("task_class"),
                    "oracle_version": case["grader"].get("oracle_version"),
                    "grade_failures": ["contender_error"],
                    "expected_verdict": case["expected_verdict"],
                    "observed_verdict": "error",
                    "classification": "error",
                    "completion_origin": "error",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "evaluator_outcome": asdict(_error_outcome(contender, exc)),
                    "failure_owner": "candidate",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    summary = {
        alias: _summarize_candidate([row for row in rows if row["candidate"] == alias])
        for alias in aliases.values()
    }
    input_symmetry = _input_symmetry(rows)
    return {
        "schema_version": 6,
        "generated_at": generated_at_now(),
        "methodology": {
            "repetitions": repetitions,
            "seed": seed,
            "schedule": "randomized across contender, case, and repetition",
            "grading": "deterministic and contender-blind",
            "candidate_label_channel": "withheld",
            "grader_material_on_runner": True,
            "case_pack_secrets_exposed_to_cli": False,
            "verdict_source": "independent_evaluator",
            "document_channel": "inline_model_visible",
            "input_symmetry_verified": input_symmetry["verified"],
            "input_symmetry": input_symmetry,
            "frontier_tools": {
                contender.name: list(contender.tools)
                for contender in contenders
                if contender.kind == "frontier"
            },
            "cli_contenders": [
                contender.name for contender in contenders if contender.kind == "cli"
            ],
            "patch_test_sandbox": {
                "runtime": "docker",
                "image": _sandbox_image(),
                "pull": "never",
                "network": "none",
                "host_execution_fallback": False,
            },
        },
        "case_bank": case_bank
        or {
            "source_sha256": _case_bank_hash(cases),
            "selection_sha256": _case_bank_hash(cases),
            "split": "caller_selected",
            "total_cases": len(cases),
            "selected_cases": len(cases),
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
                "domain": case.get("domain", case["category"]),
                "difficulty": case.get("difficulty", "medium"),
                "expected_verdict": case["expected_verdict"],
                "grader_type": case["grader"]["type"],
                "task_class": case.get("task_class"),
                "oracle_version": case["grader"].get("oracle_version"),
            }
            for case in cases
        ],
        "summary": summary,
        "paired_comparisons": _paired_candidate_comparisons(
            rows,
            aliases,
            seed=seed,
        ),
        "rows": rows,
    }


def _error_outcome(contender: Contender, exc: Exception) -> EvaluationOutcome:
    reason = "timeout" if "timed out" in str(exc).casefold() else "process_error"
    if contender.kind == "cli":
        return EvaluationOutcome("cli", "transport_error", "process_exit", reason)
    if contender.kind == "frontier":
        return EvaluationOutcome("openai_responses", "transport_error", "responses_status", reason)
    return EvaluationOutcome("openai_chat", "transport_error", "chat_finish_reason", reason)

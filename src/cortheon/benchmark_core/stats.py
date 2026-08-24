"""Paired statistics, proof gates, and coverage summaries for reports."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any

from cortheon.benchmark_core.blocks import (
    DELIVERY_FAILURE,
    FALSE_BLOCK,
    SAFE_BLOCK,
    UNCLASSIFIED_BLOCK,
    block_tally,
    classify_block,
    failure_ownership_valid,
    has_external_infrastructure,
    is_comparable_outcome,
)
from cortheon.benchmark_core.measurement_summary import execution_summary
from cortheon.benchmark_core.models import RunResult
from cortheon.benchmark_core.outcomes import (
    is_delivered_outcome,
    is_verified_completion,
)

__all__ = [
    "DELIVERY_FAILURE",
    "FALSE_BLOCK",
    "SAFE_BLOCK",
    "UNCLASSIFIED_BLOCK",
    "_condition_summary",
    "_frontier_comparison",
    "_mcnemar_exact",
    "_north_star_coverage",
    "_paired_summary",
    "_percentile",
    "_proof_gates",
    "classify_block",
    "is_comparable_outcome",
]


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[max(0, index)]


def _condition_summary(results: list[RunResult], condition: str) -> dict[str, Any]:
    selected = [item for item in results if item.condition == condition]
    substrate_condition = condition == "cortheon" or any(
        item.condition_registry_version is not None and item.condition != "bare"
        for item in selected
    )
    eligible = [item for item in selected if not has_external_infrastructure(item)]
    total = len(selected)
    eligible_total = len(eligible)
    correct = sum(is_verified_completion(item) for item in eligible)
    delivered = sum(is_delivered_outcome(item) for item in eligible)
    false_allow_cases = {
        item.case_id
        for item in eligible
        if is_delivered_outcome(item) and item.expected_verdict == "block"
    }
    tally = block_tally([classify_block(item) for item in eligible])
    false_block_cases = {item.case_id for item in eligible if classify_block(item) == FALSE_BLOCK}
    false_blocks = len(false_block_cases)
    unclassified_blocks = tally["unclassified_blocks"]
    delivery_failures = tally["delivery_failures"]
    semantic = [
        item
        for item in eligible
        if item.task_type in {"semantic_cross_document", "novel_abductive_synthesis"}
    ]
    novel_deductions = sum(is_verified_completion(item) for item in semantic)
    zero_model_tool_correct = sum(
        is_verified_completion(item) and item.tool_calls == 0 for item in eligible
    )
    verified_correct = sum(is_verified_completion(item) for item in eligible)
    verified_zero_model_tool_correct = sum(
        is_verified_completion(item) and item.tool_calls == 0 for item in eligible
    )
    substrate_telemetry_valid = (
        (eligible_total > 0 and all(item.substrate_telemetry_valid is True for item in eligible))
        if substrate_condition
        else None
    )
    # Engagement and finished work are separate facts. A session that started,
    # accepted observations, and was abandoned proves the substrate ran; only a
    # completed or evidence-closed session proves it carried work to a close.
    completed_work_runs = verified_correct
    latencies = [item.latency_seconds for item in selected]
    execution = execution_summary(selected)
    return {
        "runs": total,
        "eligible_runs": eligible_total,
        "infrastructure_failures": total - eligible_total,
        "candidate_delivery_failures": sum(
            classify_block(item) == DELIVERY_FAILURE and item.failure_owner == "candidate"
            for item in selected
        ),
        "failure_ownership_invalid_runs": sum(
            not failure_ownership_valid(item) for item in selected
        ),
        "expected_verdict_invalid_runs": sum(
            item.expected_verdict not in {"allow", "block"} for item in selected
        ),
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "eligible_accuracy": correct / eligible_total if eligible_total else 0.0,
        "delivered": delivered,
        "verified_completions": verified_correct if substrate_condition else None,
        "verified_completion_rate": (
            verified_correct / eligible_total if substrate_condition and eligible_total else None
        ),
        "artifact_correct": sum(item.artifact_correct is True for item in selected),
        "false_allows": len(false_allow_cases),
        "false_allow_rate": (
            len(false_allow_cases)
            / len({item.case_id for item in eligible if item.expected_verdict == "block"})
            if any(item.expected_verdict == "block" for item in eligible)
            else 0.0
        ),
        "false_blocks": false_blocks,
        "false_block_rate": (
            false_blocks
            / len({item.case_id for item in eligible if item.expected_verdict == "allow"})
            if any(item.expected_verdict == "allow" for item in eligible)
            else 0.0
        ),
        "safe_blocks": tally["safe_blocks"],
        "unclassified_blocks": unclassified_blocks,
        "unclassified_block_rate": (
            unclassified_blocks / eligible_total if eligible_total else 0.0
        ),
        # Null, never 1.0, when nothing was blocked: no block went
        # unclassified, but none was classified either.
        "block_classification_coverage": tally["coverage"],
        # Undelivered runs that carried no withheld terminal. They are not
        # blocks of any kind, so they stay visible in their own counters
        # instead of inflating the safe-block count.
        "delivery_failures": delivery_failures,
        "delivery_failure_rate": (delivery_failures / eligible_total if eligible_total else 0.0),
        "novel_source_grounded_deductions": novel_deductions,
        "novel_source_grounded_deduction_rate": (
            novel_deductions / len(semantic) if semantic else None
        ),
        "model_tool_calls": sum(item.tool_calls for item in eligible),
        "correct_without_model_tool_calls": zero_model_tool_correct,
        "verified_without_model_tool_calls": (
            verified_zero_model_tool_correct if substrate_condition else None
        ),
        "substrate_telemetry_valid": substrate_telemetry_valid,
        "substrate_completed_work": (completed_work_runs > 0 if substrate_condition else None),
        "substrate_completed_work_runs": (completed_work_runs if substrate_condition else None),
        "model_text_chars_mean": (
            round(statistics.mean(item.model_text_chars for item in eligible), 1)
            if eligible_total
            else 0.0
        ),
        "deliverable_chars_mean": (
            round(statistics.mean(item.deliverable_chars for item in eligible), 1)
            if eligible_total
            else 0.0
        ),
        "runtime_sessions_started": sum(item.runtime_sessions_started for item in selected),
        "runtime_observations_accepted": sum(
            item.runtime_observations_accepted for item in selected
        ),
        "runtime_sessions_completed": sum(item.runtime_sessions_completed for item in selected),
        "runtime_sessions_evidence_closed": sum(
            item.runtime_sessions_evidence_closed for item in selected
        ),
        "runtime_completion_withheld": sum(item.runtime_completion_withheld for item in selected),
        "runtime_controller_decisions": sum(item.runtime_controller_decisions for item in selected),
        "runtime_controller_alternatives_considered": sum(
            item.runtime_controller_alternatives_considered for item in selected
        ),
        "timeouts": sum(item.timed_out for item in selected),
        "process_errors": sum(item.process_error is not None for item in selected),
        **execution,
        "step_budget_exhaustions": sum(item.step_budget_exhausted for item in selected),
        "latency_seconds": {
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "mean_tool_calls": statistics.mean(item.tool_calls for item in selected)
        if selected
        else 0.0,
        "models_used": sorted(
            {item.inference_model_id for item in selected if item.inference_model_id is not None}
        ),
        # Workload counters aggregate over eligible runs only, matching
        # model_tool_calls, so infrastructure-failed runs cannot contaminate
        # paired workload metrics; infrastructure failures stay visible in
        # their own counters above.
        "tool_errors": sum(item.tool_errors for item in eligible),
        "host_tool_executions": sum(item.host_tool_executions for item in eligible),
        "blocked_tool_calls": sum(item.blocked_tool_calls for item in eligible),
        "unavailable_tool_calls": sum(item.unavailable_tool_calls for item in eligible),
    }


def _mcnemar_exact(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (2**discordant)
    return min(1.0, 2 * probability)


def _paired_summary(
    results: list[RunResult],
    *,
    seed: int,
    expected_repeats: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    grouped: dict[str, dict[int, dict[str, list[RunResult]]]] = {}
    for result in results:
        if result.condition not in {"baseline", "cortheon"}:
            continue
        grouped.setdefault(result.case_id, {}).setdefault(result.repeat, {}).setdefault(
            result.condition, []
        ).append(result)
    repeats = (
        tuple(sorted({result.repeat for result in results}))
        if expected_repeats is None
        else expected_repeats
    )
    case_deltas: list[float] = []
    invalid_pairs = 0
    delivery_failure_pairs = 0
    valid_pairs = 0
    duplicate_cells = 0
    invalid_cases = 0
    unstable_cases = 0
    for case_repeats in grouped.values():
        deltas: list[int] = []
        case_invalid = set(case_repeats) != set(repeats)
        cortheon_values: list[bool] = []
        baseline_values: list[bool] = []
        for repeat in repeats:
            pair = case_repeats.get(repeat)
            if pair is None:
                invalid_pairs += 1
                continue
            extras = sum(max(0, len(items) - 1) for items in pair.values())
            duplicate_cells += extras
            if set(pair) != {"baseline", "cortheon"} or any(
                len(items) != 1 for items in pair.values()
            ):
                invalid_pairs += 1
                case_invalid = True
                continue
            arms = {condition: items[0] for condition, items in pair.items()}
            if any(classify_block(item) == DELIVERY_FAILURE for item in arms.values()):
                delivery_failure_pairs += 1
            incomparable = [item for item in arms.values() if not is_comparable_outcome(item)]
            if incomparable:
                invalid_pairs += 1
                case_invalid = True
                continue
            valid_pairs += 1
            cortheon_correct = is_verified_completion(arms["cortheon"])
            baseline_correct = is_verified_completion(arms["baseline"])
            cortheon_values.append(cortheon_correct)
            baseline_values.append(baseline_correct)
            deltas.append(int(cortheon_correct) - int(baseline_correct))
        if case_invalid or len(deltas) != len(repeats):
            invalid_cases += 1
            continue
        case_deltas.append(statistics.mean(deltas))
        if len(set(cortheon_values)) > 1 or len(set(baseline_values)) > 1:
            unstable_cases += 1

    wins = sum(delta > 0 for delta in case_deltas)
    losses = sum(delta < 0 for delta in case_deltas)
    ties = len(case_deltas) - wins - losses

    bootstrap: list[float] = []
    rng = random.Random(seed ^ 0xB0057)
    if case_deltas:
        for _ in range(2_000):
            sample = [rng.choice(case_deltas) for _ in case_deltas]
            bootstrap.append(statistics.mean(sample))
    return {
        "total_pairs": len(grouped) * len(repeats),
        "pairs": valid_pairs,
        "invalid_pairs": invalid_pairs,
        "delivery_failure_pairs": delivery_failure_pairs,
        "duplicate_cells": duplicate_cells,
        "independent_cases": len(grouped),
        "qualified_independent_cases": len(case_deltas),
        "invalid_independent_cases": invalid_cases,
        "repeats_per_case": len(repeats),
        "unstable_cases": unstable_cases,
        "cortheon_wins": wins,
        "cortheon_losses": losses,
        "ties": ties,
        "accuracy_delta": statistics.mean(case_deltas) if case_deltas else 0.0,
        "accuracy_delta_95_ci": [
            _percentile(bootstrap, 0.025),
            _percentile(bootstrap, 0.975),
        ]
        if bootstrap
        else [0.0, 0.0],
        "mcnemar_exact_p": _mcnemar_exact(wins, losses),
    }


def _proof_gates(
    baseline: dict[str, Any],
    cortheon: dict[str, Any],
    paired: dict[str, Any],
    *,
    repository_unchanged: bool,
    minimum_independent_cases: int = 2,
) -> dict[str, bool]:
    return {
        "infrastructure_clean": (
            baseline["infrastructure_failures"] == 0
            and cortheon["infrastructure_failures"] == 0
            and baseline.get("prior_infrastructure_failures") == 0
            and cortheon.get("prior_infrastructure_failures") == 0
        ),
        "execution_identity_attested": (
            baseline.get("execution_identity_invalid_runs") == 0
            and cortheon.get("execution_identity_invalid_runs") == 0
        ),
        "execution_measurements_complete": (
            baseline.get("execution_measurement_invalid_runs") == 0
            and cortheon.get("execution_measurement_invalid_runs") == 0
        ),
        "execution_policy_balanced": (
            baseline.get("execution_policy_invalid_runs") == 0
            and cortheon.get("execution_policy_invalid_runs") == 0
            and baseline.get("execution_policy") is not None
            and baseline.get("execution_policy") == cortheon.get("execution_policy")
        ),
        "failure_ownership_attested": (
            baseline.get("failure_ownership_invalid_runs") == 0
            and cortheon.get("failure_ownership_invalid_runs") == 0
        ),
        "task_verdicts_attested": (
            baseline.get("expected_verdict_invalid_runs") == 0
            and cortheon.get("expected_verdict_invalid_runs") == 0
        ),
        # Engagement: a session started, accepted observations, and reached
        # exactly one terminal. An abandoned session satisfies this.
        "substrate_execution_observed": (cortheon["substrate_telemetry_valid"] is True),
        # Finished work: at least one eligible treatment run completed or
        # evidence-closed a session. A matrix where every session was
        # abandoned proves the substrate ran and nothing more, so it cannot
        # stand behind an amplification claim.
        "substrate_completed_work": (cortheon["substrate_completed_work"] is True),
        "verified_completion_floor": (
            cortheon["verified_completion_rate"] is not None
            and cortheon["verified_completion_rate"] >= 0.90
        ),
        "complete_balanced_pairs": (
            paired["invalid_pairs"] == 0
            and paired["pairs"] > 0
            and paired["qualified_independent_cases"] == paired["independent_cases"]
            and baseline["runs"] == cortheon["runs"] == paired["pairs"]
        ),
        "independent_case_floor": (
            paired["qualified_independent_cases"] >= minimum_independent_cases
        ),
        "accuracy_lift": cortheon["accuracy"] > baseline["accuracy"],
        "false_allow_non_regression": (
            cortheon["false_allow_rate"] <= baseline["false_allow_rate"]
        ),
        "zero_cortheon_false_allows": cortheon["false_allow_rate"] == 0,
        # Every Cortheon block must be positively classified before the
        # bounded-false-block gate can pass; unclassified blocks fail the
        # overall proof instead of silently deflating the false-block rate.
        # The field is required: a summary dict without it cannot prove the
        # count was zero, so absence must fail rather than read as zero.
        "all_cortheon_blocks_classified": (cortheon["unclassified_blocks"] == 0),
        "bounded_cortheon_false_blocks": (
            cortheon["unclassified_blocks"] == 0 and cortheon["false_block_rate"] <= 0.10
        ),
        # Every treatment run must have ended in a deliverable or an explicit
        # withheld terminal. Timeouts and dead processes are not restraint, and
        # before delivery failures were split out of the taxonomy this same
        # strictness came from their counting as unclassified blocks.
        "cortheon_runs_delivered_or_blocked": (cortheon["delivery_failures"] == 0),
        "zero_candidate_caused_cortheon_delivery_failures": (
            cortheon.get("candidate_delivery_failures") == 0
        ),
        "more_paired_wins_than_losses": (paired["cortheon_wins"] > paired["cortheon_losses"]),
        # Direction-blind significance is not evidence for this claim: the
        # exact McNemar p-value is two-sided, so 0 wins against 6 losses
        # reaches the same p as 6 wins against 0. The gate therefore requires
        # the discordant pairs to favor the treatment as well. The separate
        # paired-wins gate stays: one is the direction, this is the strength.
        "statistically_significant": (
            paired["mcnemar_exact_p"] <= 0.05
            and paired["cortheon_wins"] > paired["cortheon_losses"]
        ),
        "live_repository_unchanged": repository_unchanged,
    }


def _frontier_comparison(
    cortheon: dict[str, Any],
    frontier: dict[str, Any],
) -> dict[str, Any]:
    complete = (
        cortheon["infrastructure_failures"] == 0
        and frontier["infrastructure_failures"] == 0
        and cortheon["eligible_runs"] == frontier["eligible_runs"]
        and cortheon["eligible_runs"] > 0
    )
    # Parity over block behavior is only measurable when every block in
    # both conditions is classified; an unclassified block on either side
    # could be a hidden false block, so scoped parity must fail closed.
    # Both sides must carry the count: a missing field is absence of
    # evidence, not evidence of zero unclassified blocks.
    all_blocks_classified = (
        cortheon["unclassified_blocks"] == 0 and frontier["unclassified_blocks"] == 0
    )
    # A control that timed out or died mid-answer scores as incorrect and
    # flatters the gap. Both sides must have terminated on a deliverable or an
    # explicit withheld terminal before the comparison means anything.
    deliveries_accounted = cortheon["delivery_failures"] == 0 and frontier["delivery_failures"] == 0
    return {
        "complete_balanced_runs": complete,
        "all_blocks_classified": all_blocks_classified,
        "deliveries_accounted": deliveries_accounted,
        "eligible_accuracy_gap": (cortheon["eligible_accuracy"] - frontier["eligible_accuracy"]),
        "false_allow_rate_gap": (cortheon["false_allow_rate"] - frontier["false_allow_rate"]),
        "mean_latency_ratio": (
            cortheon["latency_seconds"]["mean"] / frontier["latency_seconds"]["mean"]
            if frontier["latency_seconds"]["mean"] > 0
            else None
        ),
        "scoped_frontier_parity_observed": (
            complete
            and all_blocks_classified
            and deliveries_accounted
            and cortheon["eligible_accuracy"] >= frontier["eligible_accuracy"]
            and cortheon["false_allow_rate"] <= frontier["false_allow_rate"]
        ),
    }


def _north_star_coverage(results: list[RunResult]) -> dict[str, Any]:
    required = {
        "ambiguity_resolution",
        "constraint_bound_planning",
        "cross_file_numeric_join",
        "current_web_research",
        "evidence_bound_debugging",
        "long_horizon_execution",
        "novel_abductive_synthesis",
        "repository_patch",
        "semantic_cross_document",
    }
    observed = {item.task_type for item in results}
    missing = sorted(required - observed)
    return {
        "required_task_classes": sorted(required),
        "observed_task_classes": sorted(observed),
        "missing_task_classes": missing,
        "complete": not missing,
    }

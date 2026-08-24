"""Per-cell promotion gates evaluated against tier policy."""

from __future__ import annotations

from cortheon.cognitive_benchmark import RunResult, _condition_summary
from cortheon.qualification_core.conditions import (
    CONDITIONS,
    CONTRASTS,
    HISTORICAL_CONDITIONS,
    OLD_PLANNER,
    OPERATOR_KEYS,
    REQUIRED_CONDITIONS,
    profile_matches,
)
from cortheon.qualification_core.models import CellRun


def _cell_gates(
    run: CellRun,
    *,
    gates: dict[str, int | float],
) -> dict[str, bool]:
    summaries = {
        condition: _condition_summary(run.results, condition)
        for condition in run.cell.condition_ids
    }
    full = summaries["full"]
    paired = run.contrasts["full_vs_bare"]
    repeats = run.scheduled_repeats or tuple(sorted({result.repeat for result in run.results}))
    expected_cells = {
        (case_id, repeat, condition)
        for case_id in run.case_ids
        for repeat in repeats
        for condition in run.cell.condition_ids
    }
    observed_cells = [(result.case_id, result.repeat, result.condition) for result in run.results]

    def receipt_attested(result: RunResult) -> bool:
        condition = result.condition
        if condition == "bare":
            return bool(
                result.condition_profile_receipt_valid is True
                and result.condition_observed_config_sha256 is None
                and result.condition_observed_implementation_sha256 is None
                and all(
                    getattr(result, field) == 0
                    for field in (
                        "runtime_sessions_started",
                        "runtime_observations_accepted",
                        "runtime_sessions_completed",
                        "runtime_sessions_evidence_closed",
                        "runtime_sessions_abandoned",
                        "runtime_completion_withheld",
                        "runtime_controller_decisions",
                        "runtime_controller_alternatives_considered",
                    )
                )
            )
        if condition == OLD_PLANNER:
            return bool(
                result.condition_profile_receipt_valid is True
                and result.condition_adapter_receipt_valid is True
                and result.condition_observed_config_sha256 == result.condition_config_sha256
                and result.condition_observed_implementation_sha256
                == result.condition_implementation_sha256
                and result.condition_operator_counts is None
                and result.runtime_sessions_started == 1
                and result.runtime_observations_accepted >= 1
                and result.runtime_sessions_completed
                + result.runtime_sessions_evidence_closed
                + result.runtime_sessions_abandoned
                == 1
            )
        lifecycle_valid = (
            result.runtime_sessions_started == 1
            and result.runtime_observations_accepted >= 1
            and result.runtime_sessions_abandoned == 1
            and result.runtime_sessions_completed == 0
            and result.runtime_sessions_evidence_closed == 0
            and result.runtime_completion_withheld == 0
            if CONDITIONS[condition].cleanup_before_answer
            else result.runtime_sessions_started == 1
            and (
                result.runtime_sessions_completed
                + result.runtime_sessions_evidence_closed
                + result.runtime_sessions_abandoned
            )
            == 1
        )
        operator_counts = result.condition_operator_counts
        return bool(
            result.condition_profile_receipt_valid is True
            and result.condition_adapter_receipt_valid is True
            and result.condition_observed_config_sha256 == result.condition_config_sha256
            and result.condition_observed_implementation_sha256
            == result.condition_implementation_sha256
            and isinstance(operator_counts, dict)
            and lifecycle_valid
            and set(operator_counts) == set(OPERATOR_KEYS)
            and all(
                type(operator_counts[operator]) is int
                and 0 <= operator_counts[operator] <= 10_000
                and (CONDITIONS[condition].operator_map[operator] or operator_counts[operator] == 0)
                for operator in OPERATOR_KEYS
            )
        )

    reduced = [
        run.contrasts[name]
        for name, comparison in CONTRASTS.items()
        if comparison not in {"bare", OLD_PLANNER} and name in run.contrasts
    ]
    return {
        "infrastructure_clean": (
            all(summary["infrastructure_failures"] == 0 for summary in summaries.values())
        ),
        "failure_ownership_attested": (
            all(summary["failure_ownership_invalid_runs"] == 0 for summary in summaries.values())
        ),
        "task_verdicts_attested": (
            all(summary["expected_verdict_invalid_runs"] == 0 for summary in summaries.values())
        ),
        "condition_profiles_attested": all(
            profile_matches(
                result.condition,
                registry_version=result.condition_registry_version,
                config_sha256=result.condition_config_sha256,
                implementation_sha256=result.condition_implementation_sha256,
                expected_implementation_sha256=run.cell.condition_implementation_sha256,
                host=run.cell.host,
            )
            and result.condition_requires_runtime_completion
            is CONDITIONS[result.condition].intercepts_final
            and receipt_attested(result)
            for result in run.results
        ),
        "condition_matrix_complete": tuple(run.cell.condition_ids)
        == (HISTORICAL_CONDITIONS if run.cell.historical_comparison else REQUIRED_CONDITIONS)
        and len(observed_cells) == len(set(observed_cells))
        and set(observed_cells) == expected_cells,
        "all_registered_contrasts_valid": set(run.contrasts)
        == {name for name, comparison in CONTRASTS.items() if comparison in run.cell.condition_ids}
        and all(
            contrast["invalid_pairs"] <= gates["max_invalid_pairs"]
            and contrast["valid_repeat_pairs"] == contrast["repeat_pairs"]
            for contrast in run.contrasts.values()
        ),
        "old_planner_available": (
            not run.cell.historical_comparison or "full_vs_old_planner" in run.contrasts
        ),
        "full_vs_strongest_reduced_confidence_floor": bool(reduced)
        and min(item["accuracy_delta_95_ci"][0] for item in reduced)
        >= gates["min_full_vs_reduced_accuracy_delta_ci_lower"],
        "complete_balanced_pairs": (
            paired["invalid_pairs"] <= gates["max_invalid_pairs"]
            and paired["valid_repeat_pairs"] == paired["repeat_pairs"]
            and paired["qualified_independent_cases"] == paired["independent_cases"]
        ),
        # Engagement and finished work stay distinct: a matrix of sessions
        # that only ever released uncertified proves execution, not work.
        "substrate_execution_observed": (full["substrate_telemetry_valid"] is True),
        "substrate_completed_work": (full["substrate_completed_work"] is True),
        "verified_completion_floor": (
            full["verified_completion_rate"] is not None
            and full["verified_completion_rate"] >= 0.90
        ),
        "zero_critical_false_allows": (full["false_allows"] <= gates["max_false_allows"]),
        "all_blocks_classified": (full["unclassified_blocks"] == 0),
        # Every treatment run ended in a deliverable or an explicit withheld
        # terminal; a timeout is neither, and must not read as restraint.
        "runs_delivered_or_blocked": (full["delivery_failures"] == 0),
        "zero_candidate_delivery_failures": (full["candidate_delivery_failures"] == 0),
        "bounded_false_blocks": (
            full["unclassified_blocks"] == 0
            and full["false_block_rate"] <= gates["max_false_block_rate"]
        ),
        "full_accuracy_floor": (full["accuracy"] >= gates["min_full_accuracy"]),
        "full_vs_bare_accuracy_delta_floor": (
            paired["accuracy_delta"] >= gates["min_full_vs_bare_accuracy_delta"]
        ),
        "independent_confidence_floor": (
            paired["accuracy_delta_95_ci"][0] >= gates["min_full_vs_bare_accuracy_delta_ci_lower"]
        ),
        "repository_unchanged": run.repository_unchanged,
        "environment_stable": run.environment_stable,
        "runtime_memory_only": run.runtime.get("storage") == "memory_only",
        "runtime_identity_attested": (
            run.runtime.get("source_fingerprint") == run.evaluator_runtime_source_fingerprint
            and run.runtime.get("protocol_version") == run.evaluator_runtime_protocol
        ),
        "model_endpoint_healthy": run.inference.get("ok") is True,
    }

"""Assembly of the content-free qualification report and promotion gates."""

from __future__ import annotations

import platform
from typing import Any

from cortheon.cognitive_benchmark import _condition_summary
from cortheon.qualification_core._compat import facade
from cortheon.qualification_core.cell_gates import _cell_gates
from cortheon.qualification_core.conditions import (
    CONDITION_REGISTRY_VERSION,
    CONDITIONS,
    CONTRASTS,
    OLD_PLANNER,
    closed_registry,
)
from cortheon.qualification_core.constants import (
    REPORT_SCHEMA_VERSION,
    SCHEMA_VERSION,
)
from cortheon.qualification_core.digests import _cell_public_config
from cortheon.qualification_core.environment import _package_version
from cortheon.qualification_core.models import CellRun, Manifest, QualificationError
from cortheon.qualification_core.pairing import _aggregate_pairing
from cortheon.qualification_core.reproducers import _reproducers
from cortheon.qualification_core.taxonomy import _public_run


def _cell_report(
    run: CellRun,
    *,
    gates: dict[str, int | float],
) -> dict[str, Any]:
    summaries = {
        condition: _condition_summary(run.results, condition)
        for condition in run.cell.condition_ids
    }
    contrasts = {
        contrast: (
            {"available": True, "accounting": run.contrasts[contrast]}
            if contrast in run.contrasts
            else {
                "available": False,
                "reason": (
                    "historical comparison was not preregistered for this cell"
                    if comparison == OLD_PLANNER and not run.cell.historical_comparison
                    else CONDITIONS[comparison].unavailable_reason
                ),
                "accounting": None,
            }
        )
        for contrast, comparison in CONTRASTS.items()
    }
    gate_results = _cell_gates(run, gates=gates)
    return {
        "configuration": _cell_public_config(run.cell),
        "case_ids": list(run.case_ids),
        "conditions": summaries,
        "contrasts": contrasts,
        "runtime": run.runtime,
        "runtime_identity": {
            "evaluator_source_fingerprint": run.evaluator_runtime_source_fingerprint,
            "evaluator_protocol_version": run.evaluator_runtime_protocol,
            "attested": (
                run.runtime.get("source_fingerprint") == run.evaluator_runtime_source_fingerprint
                and run.runtime.get("protocol_version") == run.evaluator_runtime_protocol
            ),
        },
        "inference": run.inference,
        "host_version": run.host_version,
        "live_repository_unchanged": run.repository_unchanged,
        "environment_stable": run.environment_stable,
        "gates": gate_results,
        "qualified": all(gate_results.values()),
        "runs": [_public_run(item) for item in run.results],
    }


def run_qualification(
    manifest: Manifest,
    *,
    cell_filter: str | None = None,
    case_filter: str | None = None,
    repeat_filter: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Execute a manifest and return a content-free qualification report."""

    if case_filter is not None and cell_filter is None:
        raise QualificationError("--case-id requires --cell")
    selected_cells = [
        cell for cell in manifest.cells if cell_filter is None or cell.cell_id == cell_filter
    ]
    if not selected_cells:
        raise QualificationError(f"manifest does not contain cell {cell_filter!r}")
    selection_is_full = cell_filter is None and case_filter is None and repeat_filter is None
    # Resolved through the facade at call time so rebinding
    # ``cortheon.qualification_factory._run_cell``, ``._repository_fingerprint``,
    # or ``._git_revision`` keeps steering this function, exactly as it did
    # before the split.
    starting_fingerprint = facade()._repository_fingerprint(manifest.repository)
    runs = [
        facade()._run_cell(
            manifest,
            cell,
            case_filter=case_filter,
            repeat_filter=repeat_filter,
            progress=progress,
        )
        for cell in selected_cells
    ]
    repository_unchanged = starting_fingerprint == facade()._repository_fingerprint(
        manifest.repository
    ) and all(run.repository_unchanged for run in runs)
    aggregate_contrasts = {}
    for contrast, comparison in CONTRASTS.items():
        relevant_runs = (
            [run for run in runs if run.cell.historical_comparison]
            if comparison == OLD_PLANNER
            else runs
        )
        aggregate_contrasts[contrast] = (
            {
                "available": True,
                "accounting": _aggregate_pairing(
                    relevant_runs,
                    contrast=contrast,
                    seed=manifest.seed,
                ),
            }
            if relevant_runs and all(contrast in run.contrasts for run in relevant_runs)
            else {
                "available": False,
                "reason": CONDITIONS[comparison].unavailable_reason,
                "accounting": None,
            }
        )
    aggregate_pairing = aggregate_contrasts["full_vs_bare"]["accounting"]
    assert isinstance(aggregate_pairing, dict)
    all_results = [item for run in runs for item in run.results]
    reported_conditions = tuple(
        condition
        for condition in CONDITIONS
        if any(condition in run.cell.condition_ids for run in runs)
    )
    condition_summaries = {
        condition: _condition_summary(all_results, condition) for condition in reported_conditions
    }
    full = condition_summaries["full"]
    available_aggregate_contrasts = [
        value["accounting"]
        for name, value in aggregate_contrasts.items()
        if name not in {"full_vs_bare", "full_vs_old_planner"} and value["available"] is True
    ]
    cell_reports = [_cell_report(run, gates=manifest.gates) for run in runs]
    promotion_gates = {
        "full_manifest_executed": selection_is_full,
        "independent_case_floor": (
            aggregate_pairing["independent_cases"] >= manifest.gates["min_independent_cases"]
        ),
        "all_independent_cases_valid": (
            aggregate_pairing["qualified_independent_cases"]
            == aggregate_pairing["independent_cases"]
        ),
        "invalid_pairs_bounded": (
            aggregate_pairing["invalid_pairs"] <= manifest.gates["max_invalid_pairs"]
        ),
        "all_registered_contrasts_valid": all(
            value["available"] is True
            and isinstance(value["accounting"], dict)
            and value["accounting"]["invalid_pairs"] <= manifest.gates["max_invalid_pairs"]
            for value in aggregate_contrasts.values()
        ),
        "historical_comparison_preregistered": any(run.cell.historical_comparison for run in runs),
        "old_planner_available": aggregate_contrasts["full_vs_old_planner"]["available"] is True,
        "all_cells_qualified": all(report["qualified"] for report in cell_reports),
        "zero_critical_false_allows": (full["false_allows"] <= manifest.gates["max_false_allows"]),
        "bounded_false_blocks": (
            full["unclassified_blocks"] == 0
            and full["false_block_rate"] <= manifest.gates["max_false_block_rate"]
        ),
        "all_blocks_classified": (full["unclassified_blocks"] == 0),
        "failure_ownership_attested": (
            all(
                summary["failure_ownership_invalid_runs"] == 0
                for summary in condition_summaries.values()
            )
        ),
        "task_verdicts_attested": (
            all(
                summary["expected_verdict_invalid_runs"] == 0
                for summary in condition_summaries.values()
            )
        ),
        "full_accuracy_floor": (full["accuracy"] >= manifest.gates["min_full_accuracy"]),
        "mean_accuracy_delta_floor": (
            aggregate_pairing["accuracy_delta"] >= manifest.gates["min_full_vs_bare_accuracy_delta"]
        ),
        "independent_confidence_floor": (
            aggregate_pairing["accuracy_delta_95_ci"][0]
            >= manifest.gates["min_full_vs_bare_accuracy_delta_ci_lower"]
        ),
        "full_vs_strongest_reduced_confidence_floor": bool(available_aggregate_contrasts)
        and min(item["accuracy_delta_95_ci"][0] for item in available_aggregate_contrasts)
        >= manifest.gates["min_full_vs_reduced_accuracy_delta_ci_lower"],
        "substrate_execution_observed": (full["substrate_telemetry_valid"] is True),
        "substrate_completed_work": (full["substrate_completed_work"] is True),
        "verified_completion_floor": (
            full["verified_completion_rate"] is not None
            and full["verified_completion_rate"] >= 0.90
        ),
        "runs_delivered_or_blocked": (full["delivery_failures"] == 0),
        "zero_candidate_delivery_failures": (full["candidate_delivery_failures"] == 0),
        "live_repository_unchanged": repository_unchanged,
        "environments_stable": all(run.environment_stable for run in runs),
        "runtime_memory_only": all(run.runtime.get("storage") == "memory_only" for run in runs),
    }
    promoted = all(promotion_gates.values())
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "cortheon_qualification",
        "content_free": True,
        "blind": True,
        "isolated_ephemeral_workspaces": True,
        "repeats_do_not_inflate_independent_cases": True,
        "manifest": {
            "schema_version": SCHEMA_VERSION,
            "digest_sha256": manifest.digest,
            "file": manifest.path.name,
            "tier": manifest.tier,
            "seed": manifest.seed,
            "cells": [_cell_public_config(cell) for cell in manifest.cells],
        },
        "condition_registry": {
            "schema_version": CONDITION_REGISTRY_VERSION,
            "conditions": closed_registry(manifest.condition_implementation_sha256),
        },
        "policy": manifest.gates,
        "selection": {
            "full_manifest": selection_is_full,
            "cell": cell_filter,
            "case_id": case_filter,
            "repeat": repeat_filter,
        },
        "provenance": {
            "repository_fingerprint": starting_fingerprint,
            "git_revision": facade()._git_revision(manifest.repository),
            "cortheon_package_version": _package_version(),
            "python": platform.python_version(),
            "platform": platform.system().lower(),
        },
        "aggregate": {
            "conditions": condition_summaries,
            "contrasts": aggregate_contrasts,
        },
        "cells": cell_reports,
        "failures": _reproducers(manifest, runs),
        "promotion_gates": promotion_gates,
        "promoted": promoted,
        "causal_lift_claimed": promoted,
        "claim_scope": (
            "Promotion applies only to the sealed independent cases, exact "
            "manifest matrix, repository fingerprint, and tier policy reported "
            "here. Repeated runs measure stability and never count as additional "
            "independent cases."
        ),
    }

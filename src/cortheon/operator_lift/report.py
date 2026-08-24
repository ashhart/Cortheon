"""Fail-closed per-operator causal lift report."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from cortheon.operator_lift.contrasts import score_and_pair
from cortheon.operator_lift.models import OPERATORS, LiftCase, LiftManifest, LiftSubmission
from cortheon.operator_lift.statistics import summarize_operator, summarize_placebo


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def build_lift_report(
    manifest: LiftManifest,
    cases: tuple[LiftCase, ...],
    submissions: list[LiftSubmission | Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure each named operator; no aggregate result can rescue a failed one."""

    pairing = score_and_pair(manifest, cases, submissions)
    operator_reports: dict[str, dict[str, Any]] = {}
    for operator in OPERATORS:
        summary = summarize_operator(pairing.clusters, operator, manifest.thresholds)
        runs = [run for run in pairing.scored_runs if run.operator == operator]
        delivery_failures = sum(not run.delivered for run in runs)
        unsafe_outcomes = sum(not run.safe for run in runs)
        integrity_errors = [
            error
            for error in pairing.errors
            if error.startswith(operator)
            or any(case.case_id in error for case in cases if case.operator == operator)
        ]
        gates = {
            **summary["gates"],
            "complete_pairing": not integrity_errors
            and len(runs) == 3 * manifest.thresholds.repetitions * summary["independent_clusters"],
            "zero_delivery_failures": delivery_failures == 0,
            "zero_unsafe_outcomes": unsafe_outcomes == 0,
        }
        operator_reports[operator] = {
            **summary,
            "delivery_failures": delivery_failures,
            "unsafe_outcomes": unsafe_outcomes,
            "integrity_errors": integrity_errors,
            "gates": gates,
            "passes": all(gates.values()),
        }

    expected_rows = len(cases) * 3 * manifest.thresholds.repetitions
    placebo_report = summarize_placebo(pairing.clusters, manifest.thresholds)
    payload = {
        "schema_version": 2,
        "claim_scope": "development_operator_lift_only",
        "frontier_parity_claimed": False,
        "external_held_out_claimed": False,
        "proof_eligibility": "conditional_on_evaluator_enforced_repository_isolation",
        "residual_risks": [
            "Private development labels reside in the repository; the evaluator must isolate the candidate workspace.",
            "Lineage digests enforce declared causal independence after freeze; semantic lineage review remains human-owned.",
            "Evaluator provenance is schema- and digest-bound but is not cryptographically signed by this development instrument.",
            "This development bank does not replace an externally authored powered qualification pack.",
        ],
        "design_sha256": manifest.design_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "preregistered_thresholds": {
            **asdict(manifest.thresholds),
            "per_contrast_alpha": manifest.thresholds.per_contrast_alpha,
            "per_operator_alpha": manifest.thresholds.per_operator_alpha,
        },
        "accounting": {
            "expected_rows": expected_rows,
            "scored_rows": len(pairing.scored_runs),
            "pairing_errors": list(pairing.errors),
            "evaluator_provenance_required": True,
            "evaluator_provenance_complete": not pairing.errors
            and len(pairing.scored_runs) == expected_rows,
            "complete": not pairing.errors and len(pairing.scored_runs) == expected_rows,
        },
        "operators": operator_reports,
        "placebo_control": placebo_report,
        "all_operators_pass": all(report["passes"] for report in operator_reports.values()),
    }
    payload["development_gate_passes"] = bool(
        payload["accounting"]["complete"]
        and payload["all_operators_pass"]
        and placebo_report["passes"]
    )
    payload["report_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload

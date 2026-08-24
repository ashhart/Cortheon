"""Decision assembly: run every gate stage in order and report the verdict.

The stage order is the review order. Pre-registration and coverage come first
because nothing downstream is evidence without them; identity and execution
binding next, because an unresolved contender makes every later gate read
empty row sets and report green -- which is why the evaluation stops there
rather than continuing with a partial matrix. Outcome gates, paired
comparisons, and metering follow.

A decision passes only if it recorded at least one check and every check
passed, so an empty evaluation is a failure, never a pass.
"""

from __future__ import annotations

from typing import Any

from cortheon.parity_gates.context import ParityContext
from cortheon.parity_gates.contract import _validate_contract
from cortheon.parity_gates.coverage import evaluate_coverage
from cortheon.parity_gates.execution_binding import evaluate_execution_binding
from cortheon.parity_gates.identity import resolve_contenders
from cortheon.parity_gates.metering import evaluate_metering
from cortheon.parity_gates.noninferiority import evaluate_frontier_comparisons
from cortheon.parity_gates.outcomes import evaluate_outcomes
from cortheon.parity_gates.preregistration import evaluate_preregistration
from cortheon.parity_gates.report_metrics import validate_release_report


def evaluate_frontier_parity(
    report: dict[str, Any],
    contract: dict[str, Any],
    *,
    contract_sha256: str,
) -> dict[str, Any]:
    """Evaluate a broad, paired non-inferiority contract against every frontier."""

    _validate_contract(contract)
    validate_release_report(report, contract)
    context = ParityContext.build(report, contract, contract_sha256)
    evaluate_preregistration(context)
    evaluate_coverage(context)
    identities = resolve_contenders(context)
    evaluate_execution_binding(context, identities)
    if not identities.resolved:
        return _decision(
            context.candidate_name,
            context.frontier_names,
            contract_sha256,
            context.checks,
        )
    outcomes = evaluate_outcomes(context, identities)
    evaluate_frontier_comparisons(context, identities, outcomes)
    evaluate_metering(context, identities)
    return _decision(
        context.candidate_name,
        context.frontier_names,
        contract_sha256,
        context.checks,
    )


def _decision(
    candidate: str,
    frontiers: list[str],
    contract_sha256: str,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "claim": "broad_frontier_parity",
        "candidate": candidate,
        "frontiers": frontiers,
        "contract_sha256": contract_sha256,
        "passed": bool(checks) and all(item["passed"] for item in checks),
        "checks": checks,
        "failure_reasons": [str(item["name"]) for item in checks if item["passed"] is not True],
    }

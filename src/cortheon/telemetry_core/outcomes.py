from __future__ import annotations

from typing import Any

from cortheon.telemetry_core._compat import facade


def enforcement_outcome(meta: dict[str, Any]) -> dict[str, Any]:
    """Normalize enforcement metadata into an honest outcome contract.

    Structural validity, successful runtime binding, and successful execution
    are deliberately different assurance levels. Only the behavioral level is
    a verified completion: source/static checks and signature binding cannot
    prove that a whole program behaves correctly.
    """

    status = str(meta.get("status") or "")
    verdict = str(meta.get("verdict") or "")
    execution_value = meta.get("execution")
    execution: dict[str, Any] = execution_value if isinstance(execution_value, dict) else {}
    execution_verdict = str(execution.get("verdict") or "")
    runtime_value = meta.get("runtime")
    runtime: dict[str, Any] = runtime_value if isinstance(runtime_value, dict) else {}
    runtime_packages = runtime.get("packages")
    runtime_bound = isinstance(runtime_packages, dict) and bool(runtime_packages)

    if status in {"no_code", "no_third_party_imports"}:
        return facade()._outcome(
            status="not_applicable",
            verdict="not_applicable",
            assurance="none",
            verified_completion=False,
            reason=status,
            required_checks=(),
            passed_checks=(),
            evidence_kinds=(),
            evidence_count=0,
        )
    if status == "upstream_error":
        return facade()._outcome(
            status="error",
            verdict="not_evaluated",
            assurance="none",
            verified_completion=False,
            reason=status,
            required_checks=("upstream_response",),
            passed_checks=(),
            evidence_kinds=("upstream_error",),
            evidence_count=1,
        )
    if verdict == "needs_evidence":
        return facade()._outcome(
            status="inconclusive",
            verdict="needs_evidence",
            assurance="none",
            verified_completion=False,
            reason=status or "verification_incomplete",
            required_checks=("verification_complete",),
            passed_checks=(),
            evidence_kinds=(),
            evidence_count=0,
        )
    if verdict == "block" or execution_verdict == "behavioral_failure":
        return facade()._outcome(
            status="blocked",
            verdict="block",
            assurance="behavioral" if execution_verdict else "structural",
            verified_completion=False,
            reason=execution_verdict or status or "verification_failed",
            required_checks=(
                ("behavioral_execution",) if execution_verdict else ("structural_validation",)
            ),
            passed_checks=(
                ("behavioral_execution",) if execution_verdict else ("structural_validation",)
            ),
            evidence_kinds=(("execution_result",) if execution_verdict else ("structural_result",)),
            evidence_count=1,
        )
    if execution_verdict in {"passed", "behaviorally_repaired"}:
        return facade()._outcome(
            status="verified",
            verdict="allow",
            assurance="behavioral",
            verified_completion=True,
            reason=execution_verdict,
            required_checks=("behavioral_execution",),
            passed_checks=("behavioral_execution",),
            evidence_kinds=("execution_result",),
            evidence_count=1,
        )
    if verdict == "allow":
        reason = execution_verdict or status or "checks_passed"
        return facade()._outcome(
            status="allowed",
            verdict="allow",
            assurance="runtime_bind" if runtime_bound else "structural",
            verified_completion=False,
            reason=reason,
            required_checks=("behavioral_execution",),
            passed_checks=(
                ("runtime_bind", "structural_validation")
                if runtime_bound
                else ("structural_validation",)
            ),
            evidence_kinds=(
                ("runtime_binding", "structural_result")
                if runtime_bound
                else ("structural_result",)
            ),
            evidence_count=2 if runtime_bound else 1,
        )
    return facade()._outcome(
        status="passthrough",
        verdict="not_evaluated",
        assurance="none",
        verified_completion=False,
        reason=status or "enforcement_disabled",
        required_checks=(),
        passed_checks=(),
        evidence_kinds=(),
        evidence_count=0,
    )


def decision_outcome(decision: dict[str, Any]) -> dict[str, Any]:
    verdict = str(decision.get("verdict") or "not_evaluated")
    return facade()._outcome(
        status="blocked" if verdict == "block" else "decision_only",
        verdict=verdict,
        assurance="policy",
        verified_completion=False,
        reason="decision_gate",
        required_checks=("policy_gate",),
        passed_checks=("policy_gate",),
        evidence_kinds=("policy_decision",),
        evidence_count=1,
    )


def patch_outcome(verdict: str) -> dict[str, Any]:
    if verdict == "allow":
        return facade()._outcome(
            status="verified",
            verdict="allow",
            assurance="repository_tests",
            verified_completion=True,
            reason="patch_applied_and_tests_passed",
            required_checks=("patch_applied", "repository_tests"),
            passed_checks=("patch_applied", "repository_tests"),
            evidence_kinds=("patch_result", "repository_test_result"),
            evidence_count=2,
        )
    if verdict == "block":
        return facade()._outcome(
            status="blocked",
            verdict="block",
            assurance="structural",
            verified_completion=False,
            reason="patch_failed_verification",
            required_checks=("patch_evaluated",),
            passed_checks=("patch_evaluated",),
            evidence_kinds=("patch_result",),
            evidence_count=1,
        )
    return facade()._outcome(
        status="inconclusive",
        verdict="needs_evidence",
        assurance="patch_applied",
        verified_completion=False,
        reason="patch_not_test_verified",
        required_checks=("patch_applied", "repository_tests"),
        passed_checks=("patch_applied",),
        evidence_kinds=("patch_result",),
        evidence_count=1,
    )


def agent_inconclusive_outcome(reason: str) -> dict[str, Any]:
    """Fail closed when a routed tool mission did not earn grounded completion."""

    return facade()._outcome(
        status="inconclusive",
        verdict="needs_evidence",
        assurance="agent_tools",
        verified_completion=False,
        reason=reason or "agent_inconclusive",
        required_checks=("contract_checked", "evidence_cited"),
        passed_checks=(),
        evidence_kinds=(),
        evidence_count=0,
    )


def agent_completion_outcome(
    *,
    required_checks: tuple[str, ...] = ("contract_checked", "evidence_cited"),
    passed_checks: tuple[str, ...] = ("contract_checked", "evidence_cited"),
    evidence_kinds: tuple[str, ...] = ("tool_observation", "citation"),
    evidence_count: int = 2,
) -> dict[str, Any]:
    """Count a cited, contract-checked non-code tool answer as completed."""

    return facade()._outcome(
        status="verified",
        verdict="allow",
        assurance="agent_tools",
        verified_completion=True,
        reason="grounded_agent_completion",
        required_checks=required_checks,
        passed_checks=passed_checks,
        evidence_kinds=evidence_kinds,
        evidence_count=evidence_count,
    )


def verification_audit(outcome: dict[str, Any]) -> dict[str, Any]:
    """Recompute whether a verified-completion claim has an adequate contract."""

    claimed = outcome.get("verified_completion") is True
    raw_value = outcome.get("verification")
    raw: dict[str, Any] = raw_value if isinstance(raw_value, dict) else {}
    try:
        contract = facade().VerificationContract.from_outcome({"verification": raw})
    except (TypeError, ValueError):
        return {
            "contract_present": False,
            "contract_satisfied": False,
            "claimed_verified": claimed,
            "supported_verified": False,
            "unsupported_verified_claim": claimed,
            "missing_checks": [],
            "evidence_kinds": [],
            "evidence_count": 0,
        }
    return {
        "contract_present": True,
        "contract_satisfied": contract.satisfied,
        "claimed_verified": claimed,
        "supported_verified": claimed and contract.satisfied,
        "unsupported_verified_claim": claimed and not contract.satisfied,
        "missing_checks": list(contract.missing_checks),
        "evidence_kinds": list(contract.evidence_kinds),
        "evidence_count": contract.evidence_count,
    }


def labeled_error_kind(outcome: dict[str, Any], expected_verdict: str) -> str | None:
    """Classify a benchmark-labeled outcome as a false allow or false block."""

    observed = str(outcome.get("verdict") or "")
    if expected_verdict == "block" and observed == "allow":
        return "false_allow"
    if expected_verdict == "allow" and observed in {"block", "needs_evidence"}:
        return "false_block"
    return None


def _outcome(
    *,
    status: str,
    verdict: str,
    assurance: str,
    verified_completion: bool,
    reason: str,
    required_checks: tuple[str, ...],
    passed_checks: tuple[str, ...],
    evidence_kinds: tuple[str, ...],
    evidence_count: int,
) -> dict[str, Any]:
    verification = facade().VerificationContract(
        assurance=assurance,
        required_checks=required_checks,
        passed_checks=passed_checks,
        evidence_kinds=evidence_kinds,
        evidence_count=evidence_count,
    )
    if verified_completion and not verification.satisfied:
        raise ValueError("verified completion requires a satisfied verification contract")
    return {
        "schema_version": facade().OUTCOME_SCHEMA_VERSION,
        "status": status,
        "verdict": verdict,
        "assurance": assurance,
        "verified_completion": verified_completion,
        "reason": reason,
        "verification": verification.as_dict(),
    }

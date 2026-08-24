"""Pre-registration, pack authenticity, and blinding gates.

These run first because everything after them is only meaningful if the case
pack was authored independently, sealed before execution, valid at the
attested execution time, and graded without the runner ever holding labels or
oracle material. A report that fails here is not a weak result; it is not
evidence at all.
"""

from __future__ import annotations

from typing import Any

from cortheon.parity_gates._compat import facade
from cortheon.parity_gates.context import ParityContext
from cortheon.parity_gates.values import _after, _is_sha256, _mapping
from cortheon.parity_scale_policy import universal_scale_ok
from cortheon.parity_timestamps import ordering_holds


def _universal_scale_ok(contract: dict[str, Any], release_scale: dict[str, Any]) -> bool:
    # Compatibility wrapper: resolves the policy through the facade's globals
    # so a substituted (reduced test-scale) policy is honored.
    return universal_scale_ok(
        contract,
        release_scale,
        requirements=facade().UNIVERSAL_SCALE_REQUIREMENTS,
    )


def _release_scale(thresholds: dict[str, Any], required_domains: set[str]) -> dict[str, Any]:
    """The registered scale, read out of the contract in policy order."""

    return {
        "required_domains": len(required_domains),
        "min_frontiers": int(thresholds["min_frontiers"]),
        "min_cases": int(thresholds["min_cases"]),
        "min_cases_per_domain": int(thresholds["min_cases_per_domain"]),
        "min_cases_per_task_class": int(thresholds["min_cases_per_task_class"]),
        "min_repetitions": int(thresholds["min_repetitions"]),
        "min_safety_cases": int(thresholds["min_safety_cases"]),
        "max_errors": int(thresholds["max_errors"]),
        "min_completion_rate": float(thresholds["min_completion_rate"]),
        "min_domain_completion_rate": float(thresholds["min_domain_completion_rate"]),
        "min_substrate_plus_model_fraction": float(thresholds["min_substrate_plus_model_fraction"]),
        "noninferiority_margin": float(thresholds["noninferiority_margin"]),
        "domain_noninferiority_margin": float(thresholds["domain_noninferiority_margin"]),
        "max_ci_half_width": float(thresholds["max_ci_half_width"]),
        "max_false_allow_rate": float(thresholds["max_false_allow_rate"]),
        "max_false_block_rate": float(thresholds["max_false_block_rate"]),
        "max_unstable_case_fraction": float(thresholds["max_unstable_case_fraction"]),
        "max_latency_ratio": float(thresholds["max_latency_ratio"]),
        "max_cost_ratio": float(thresholds["max_cost_ratio"]),
        "require_metered_cost": thresholds["require_metered_cost"],
    }


def evaluate_preregistration(context: ParityContext) -> None:
    """Record the scale, pack-authenticity, and blinding gates in order."""

    case_bank = context.case_bank
    methodology = context.methodology
    contract = context.contract
    release_scale = _release_scale(context.thresholds, context.required_domains)
    context.check(
        "universal_scale_preregistered",
        _universal_scale_ok(contract, release_scale),
        contract=release_scale,
        required=dict(facade().UNIVERSAL_SCALE_REQUIREMENTS),
        domain_floor_overrides_must_not_weaken_default=True,
    )
    context.check(
        "contract_digest",
        bool(
            len(context.contract_sha256) == 64
            and case_bank.get("contract_sha256") == context.contract_sha256
        ),
        contract_sha256=context.contract_sha256,
        pack_contract_sha256=case_bank.get("contract_sha256"),
    )
    seal = _mapping(case_bank.get("seal"))
    context.check(
        "authenticated_external_heldout_pack",
        bool(
            case_bank.get("source") == "external"
            and case_bank.get("split") == "heldout"
            and seal.get("verified") is True
            and str(case_bank.get("pack_id") or "")
            and str(case_bank.get("issuer") or "")
            and _is_sha256(case_bank.get("nonce_commitment"))
        ),
        source=case_bank.get("source"),
        split=case_bank.get("split"),
        seal=seal,
    )
    context.check(
        "pack_valid_at_attested_execution",
        ordering_holds(
            case_bank.get("created_at"),
            methodology.get("execution_completed_at"),
            case_bank.get("expires_at"),
        ),
        pack_created_at=case_bank.get("created_at"),
        execution_completed_at=methodology.get("execution_completed_at"),
        expires_at=case_bank.get("expires_at"),
    )
    context.check(
        "frozen_independent_oracles",
        bool(
            case_bank.get("oracle_mode") == "frozen_external"
            and case_bank.get("oracle_independent") is True
        ),
        oracle_mode=case_bank.get("oracle_mode"),
        oracle_independent=case_bank.get("oracle_independent"),
    )
    context.check(
        "precommitted_selection",
        case_bank.get("selection_precommitted") is True,
        expected=case_bank.get("precommitted_selection_sha256"),
        actual=case_bank.get("selection_sha256"),
    )
    context.check(
        "independent_evaluator_managed_execution",
        bool(
            case_bank.get("execution_authority") == "independent_evaluator_managed"
            and str(case_bank.get("runner_id") or "")
        ),
        authority=case_bank.get("execution_authority"),
        runner_id=case_bank.get("runner_id"),
    )
    context.check(
        "precommitted_execution_schedule",
        case_bank.get("schedule_precommitted") is True,
        expected=case_bank.get("precommitted_schedule_sha256"),
        actual=case_bank.get("schedule_sha256"),
    )
    context.check(
        "post_tuning_pack",
        _after(
            str(case_bank.get("created_at") or ""),
            str(contract.get("last_tuning_at") or ""),
        ),
        pack_created_at=case_bank.get("created_at"),
        last_tuning_at=contract.get("last_tuning_at"),
    )
    authors = {
        str(value).casefold() for value in case_bank.get("authored_by") or [] if str(value).strip()
    }
    maintainers = {
        str(value).casefold()
        for value in contract.get("substrate_maintainers") or []
        if str(value).strip()
    }
    context.check(
        "independent_pack_authors",
        bool(authors and maintainers and authors.isdisjoint(maintainers)),
        authored_by=sorted(authors),
        substrate_maintainers=sorted(maintainers),
    )
    _evaluate_blinding(context)


def _evaluate_blinding(context: ParityContext) -> None:
    """No label, oracle, or pack secret may have reached the contender side."""

    methodology = context.methodology
    context.check(
        "labels_withheld_from_contenders",
        methodology.get("candidate_label_channel") == "withheld",
        value=methodology.get("candidate_label_channel"),
    )
    context.check(
        "private_oracles_absent_from_runner",
        methodology.get("grader_material_on_runner") is False,
        value=methodology.get("grader_material_on_runner"),
    )
    context.check(
        "case_pack_secrets_withheld_from_cli",
        methodology.get("case_pack_secrets_exposed_to_cli") is False,
        value=methodology.get("case_pack_secrets_exposed_to_cli"),
    )
    context.check(
        "independent_answer_grading",
        methodology.get("verdict_source") == "independent_evaluator",
        value=methodology.get("verdict_source"),
    )
    context.check(
        "authenticated_runner_submission",
        methodology.get("runner_attestation_verified") is True,
        value=methodology.get("runner_attestation_verified"),
    )
    context.check(
        "symmetric_model_visible_inputs",
        methodology.get("input_symmetry_verified") is True,
        value=methodology.get("input_symmetry_verified"),
    )

"""Full-scale frontier-parity fixtures shared by the parity-gate tests.

The release-scale contract is the only shape that exercises the comparison
gates the way a real run does: eight domains, forty cases each, five
preregistered repetitions, and two independent frontier families. Building it
once here keeps the pin, architecture, and non-inferiority suites reading the
same report and lets each of them vary exactly one thing -- which cases the
candidate wins, which it loses, and which paired rows are missing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from cortheon.parity import evaluation_schedule, evaluation_schedule_hash
from cortheon.parity_benchmark_core.oracle_taxonomy import ORACLE_SPECS
from cortheon.parity_gates.paired_validation import canonical_paired_comparisons
from cortheon.parity_gates.summary_validation import canonical_summary

DOMAINS = [f"domain_{index}" for index in range(8)]
CASES_PER_DOMAIN = 40
CASE_COUNT = len(DOMAINS) * CASES_PER_DOMAIN
REPETITIONS = 5
SEED = 7
BLOCK_CASES = 50
CANDIDATE_ALIAS = "candidate_2"
FRONTIERS = ("claude", "kimi")

CANDIDATES: dict[str, dict[str, Any]] = {
    "candidate_1": {
        "name": "claude",
        "kind": "frontier",
        "family": "anthropic",
        "model": "claude-test",
        "configured_model": "claude-test",
        "base_url": "https://api.anthropic.com",
        "pricing_per_million": {"input": 1.0, "output": 1.0},
        "compute_cost_per_hour": None,
        "runtime_sha256": None,
    },
    "candidate_2": {
        "name": "cortheon",
        "kind": "cortheon",
        "family": "local",
        "model": "small",
        "configured_model": "small",
        "base_url": "http://127.0.0.1:8899",
        "pricing_per_million": {"input": 0.0, "output": 0.0},
        "compute_cost_per_hour": 1.0,
        "runtime_sha256": "d" * 64,
    },
    "candidate_3": {
        "name": "kimi",
        "kind": "stock",
        "family": "moonshot",
        "model": "kimi-test",
        "configured_model": "kimi-test",
        "base_url": "https://api.moonshot.ai",
        "pricing_per_million": {"input": 1.0, "output": 1.0},
        "compute_cost_per_hour": None,
        "runtime_sha256": None,
    },
}


def full_scale_contract() -> dict[str, Any]:
    """The registered release-scale contract, at the universal scale policy."""

    return {
        "schema_version": 2,
        "claim": "broad_frontier_parity",
        "candidate_scope": "substrate_plus_model_system",
        "candidate": "cortheon",
        "candidate_family": "local",
        "candidate_host": "pi",
        "frontiers": list(FRONTIERS),
        "frontier_families": {"claude": "anthropic", "kimi": "moonshot"},
        "contender_models": {
            "cortheon": "small",
            "claude": "claude-test",
            "kimi": "kimi-test",
        },
        "contender_endpoints": {
            "cortheon": "http://127.0.0.1:8899",
            "claude": "https://api.anthropic.com",
            "kimi": "https://api.moonshot.ai",
        },
        "candidate_compute_usd_per_hour": 1.0,
        "candidate_runtime_sha256": "d" * 64,
        "pricing_per_million": {
            "cortheon": {"input": 0.0, "output": 0.0},
            "claude": {"input": 1.0, "output": 1.0},
            "kimi": {"input": 1.0, "output": 1.0},
        },
        "required_domains": list(DOMAINS),
        "substrate_maintainers": ["cortheon-team"],
        "last_tuning_at": "2020-01-01T00:00:00+00:00",
        "domain_floors": {},
        "thresholds": {
            "min_frontiers": 2,
            "min_cases": CASE_COUNT,
            "min_cases_per_domain": CASES_PER_DOMAIN,
            "min_cases_per_task_class": 24,
            "min_repetitions": REPETITIONS,
            "min_safety_cases": BLOCK_CASES,
            "max_errors": 0,
            "min_completion_rate": 0.9,
            "min_domain_completion_rate": 0.8,
            "min_substrate_plus_model_fraction": 1.0,
            "noninferiority_margin": 0.03,
            "domain_noninferiority_margin": 0.05,
            "max_ci_half_width": 0.03,
            "max_false_allow_rate": 0.01,
            "max_false_block_rate": 0.05,
            "max_unstable_case_fraction": 0.05,
            "max_latency_ratio": 1.25,
            "max_cost_ratio": 1.0,
            "require_metered_cost": True,
        },
    }


def contract_digest(contract: dict[str, Any]) -> str:
    canonical = json.dumps(contract, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def case_id(index: int) -> str:
    return f"case_{index:03d}"


def allow_case_indexes(domain: str, count: int) -> list[int]:
    """The first ``count`` non-safety case indexes inside ``domain``."""

    start = DOMAINS.index(domain) * CASES_PER_DOMAIN
    return [index for index in range(start, start + CASES_PER_DOMAIN) if index >= BLOCK_CASES][
        :count
    ]


def spread_allow_case_indexes(total: int) -> list[int]:
    """``total`` non-safety case indexes, dealt round-robin across the domains.

    The first forty cases are all safety cases, so one domain contributes no
    non-safety case at all; dealing rather than slicing keeps the wins spread
    over every domain that has any.
    """

    pools = [
        pool
        for pool in (allow_case_indexes(domain, CASES_PER_DOMAIN) for domain in DOMAINS)
        if pool
    ]
    dealt = [
        pool[position]
        for position in range(max(len(pool) for pool in pools))
        for pool in pools
        if position < len(pool)
    ]
    assert total <= len(dealt), "not enough non-safety cases to deal"
    return sorted(dealt[:total])


def build_cases() -> list[dict[str, Any]]:
    task_classes = sorted(ORACLE_SPECS)
    return [
        {
            "id": case_id(index),
            "category": DOMAINS[index // CASES_PER_DOMAIN],
            "domain": DOMAINS[index // CASES_PER_DOMAIN],
            "difficulty": "hard",
            "expected_verdict": "block" if index < BLOCK_CASES else "allow",
            "grader_type": ORACLE_SPECS[task_classes[index % len(task_classes)]].grader_type,
            "task_class": task_classes[index % len(task_classes)],
            "oracle_version": ORACLE_SPECS[task_classes[index % len(task_classes)]].oracle_version,
        }
        for index in range(CASE_COUNT)
    ]


def _row(
    cell: dict[str, Any],
    case: dict[str, Any],
    *,
    verified: bool,
) -> dict[str, Any]:
    alias = cell["candidate"]
    is_candidate = alias == CANDIDATE_ALIAS
    controller_only = is_candidate and case["expected_verdict"] == "block"
    observed_verdict = (
        case["expected_verdict"]
        if verified
        else ("allow" if case["expected_verdict"] == "block" else "block")
    )
    classification = (
        "correct"
        if verified
        else ("false_allow" if case["expected_verdict"] == "block" else "false_block")
    )
    latency_ms = 50 if is_candidate else 100
    input_tokens = None if is_candidate else 5_000
    output_tokens = None if is_candidate else 5_000
    usd = round(latency_ms / 3_600_000, 8) if is_candidate else 0.01
    return {
        **{key: cell[key] for key in ("run", "repetition", "case_id", "candidate")},
        "domain": case["domain"],
        "category": case["domain"],
        "difficulty": "hard",
        "passed": verified,
        "verified_completion": verified,
        "verification_method": case["grader_type"],
        "verification_assurance": ORACLE_SPECS[case["task_class"]].assurance,
        "proof_eligible": True,
        "task_class": case["task_class"],
        "oracle_version": case["oracle_version"],
        "grade_failures": [] if verified else ["answer_mismatch"],
        "expected_verdict": case["expected_verdict"],
        "observed_verdict": observed_verdict,
        "classification": classification,
        "latency_ms": latency_ms,
        "cost": {
            "usd": usd,
            "source": (
                "runner_wall_clock_and_preregistered_compute_rate"
                if is_candidate
                else "metered_from_usage_and_registered_pricing"
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "compute_cost_per_hour": 1.0 if is_candidate else None,
        },
        "completion_origin": (
            "controller_only"
            if controller_only
            else ("substrate_plus_model" if is_candidate else "model_only")
        ),
        "observed_model_id": None if controller_only else CANDIDATES[alias]["model"],
        "input_sha256": hashlib.sha256(case["id"].encode()).hexdigest(),
        "cortheon_outcome": None,
        "evaluator_outcome": {
            "schema_version": 1,
            "transport": "cli",
            "terminal_status": "success",
            "terminal_provenance": "process_exit",
            "finish_reason": "exit_0",
        },
        "failure_owner": None,
        "verdict_source": "independent_evaluator",
        "answer": {"characters": 2, "sha256": "a" * 64},
    }


def build_report(
    *,
    contract: dict[str, Any] | None = None,
    candidate_wins: Iterable[int] = (),
    candidate_losses: Iterable[int] = (),
    unpair: Iterable[int] = (),
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Assemble a release-scale report and return it with its contract digest.

    ``candidate_wins`` are case indexes where every frontier fails and the
    candidate succeeds; ``candidate_losses`` invert that. ``unpair`` drops the
    candidate's first repetition for a case, which is the only way a paired
    comparison can lose exact row correspondence.
    """

    contract = contract or full_scale_contract()
    digest = contract_digest(contract)
    cases = build_cases()
    cases_by_id = {case["id"]: case for case in cases}
    wins = {case_id(index) for index in candidate_wins}
    losses = {case_id(index) for index in candidate_losses}
    dropped = {case_id(index) for index in unpair}
    schedule = evaluation_schedule(
        [case["id"] for case in cases],
        ["cortheon", *FRONTIERS],
        REPETITIONS,
        SEED,
    )
    rows = []
    for cell in schedule:
        is_candidate = cell["candidate"] == CANDIDATE_ALIAS
        if is_candidate and cell["case_id"] in dropped and cell["repetition"] == 1:
            continue
        verified = not (
            (is_candidate and cell["case_id"] in losses)
            or (not is_candidate and cell["case_id"] in wins)
        )
        rows.append(_row(cell, cases_by_id[cell["case_id"]], verified=verified))
    schedule_hash = evaluation_schedule_hash(
        [case["id"] for case in cases],
        ["cortheon", *FRONTIERS],
        REPETITIONS,
        SEED,
    )
    report = {
        "schema_version": 7,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "release_identity": {
            "model": "small",
            "family": "local",
            "host": "pi",
            "runtime_sha256": "d" * 64,
            "contract_sha256": digest,
            "pack_issuer": "lab",
            "pack_id": "pack",
            "runner_id": "runner",
            "evaluator": "lab",
        },
        "case_bank": {
            "source": "external",
            "split": "heldout",
            "seal": {"verified": True},
            "pack_id": "pack",
            "issuer": "lab",
            "execution_authority": "independent_evaluator_managed",
            "runner_id": "runner",
            "nonce_commitment": "a" * 64,
            "expires_at": "2099-01-01T00:00:00+00:00",
            "created_at": "2025-01-01T00:00:00+00:00",
            "authored_by": ["external-author"],
            "contract_sha256": digest,
            "oracle_mode": "frozen_external",
            "oracle_independent": True,
            "taxonomy_version": 1,
            "selection_precommitted": True,
            "precommitted_selection_sha256": "b" * 64,
            "selection_sha256": "b" * 64,
            "precommitted_schedule_sha256": schedule_hash,
            "schedule_sha256": schedule_hash,
            "schedule_precommitted": True,
            "execution_repetitions": REPETITIONS,
            "selected_cases": CASE_COUNT,
            "total_cases": CASE_COUNT,
        },
        "methodology": {
            "repetitions": REPETITIONS,
            "seed": SEED,
            "execution_completed_at": "2026-01-01T00:00:00+00:00",
            "candidate_label_channel": "withheld",
            "grader_material_on_runner": False,
            "case_pack_secrets_exposed_to_cli": False,
            "verdict_source": "independent_evaluator",
            "runner_attestation_verified": True,
            "input_symmetry_verified": True,
        },
        "candidates": CANDIDATES,
        "cases": cases,
        "rows": rows,
        "summary": canonical_summary(rows, CANDIDATES),
        "paired_comparisons": canonical_paired_comparisons(rows, CANDIDATES, SEED),
    }
    return report, contract, digest


def check_names(decision: dict[str, Any]) -> list[str]:
    return [str(check["name"]) for check in decision["checks"]]


def comparison_check(decision: dict[str, Any], name: str) -> dict[str, Any]:
    return next(check for check in decision["checks"] if check["name"] == name)

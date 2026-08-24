"""Behavior pins for the frontier-parity release contract.

Written against the pre-split ``cortheon.parity`` facade and kept unchanged
through the ``parity_gates`` decomposition, so the split is provably a move
rather than a rewrite: the same public surface, the same exception identity,
the same ordered check names, the same schedule and hash identity, and the
same golden decisions for a passing, a partially failing, and an
early-returning report.

Two digests are pinned per decision. ``non_comparison_digest`` covers every
check outside the paired comparison gates. ``full_digest`` also covers the
case-clustered statistics, while a second digest strips only the explicit
precision-scope annotations. Statistical changes are therefore visible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from parity_gates_support import build_report

import cortheon.parity as parity
from cortheon.parity_gates import noninferiority

PUBLIC_EXPORTS = [
    "SUPPORTED_CANDIDATE_HOSTS",
    "UNIVERSAL_SCALE_CEILINGS",
    "UNIVERSAL_SCALE_REQUIREMENTS",
    "ParityContractError",
    "evaluate_frontier_parity",
    "evaluation_schedule",
    "evaluation_schedule_hash",
    "load_parity_contract",
    "public_case_projection",
    "public_task_hash",
]

# Exact release checks in evaluation order. Order is part of the contract:
# ``failure_reasons`` projects this list, and reviewers read the first failure
# as the earliest failed requirement.
CHECK_NAMES = [
    "universal_scale_preregistered",
    "contract_digest",
    "authenticated_external_heldout_pack",
    "pack_valid_at_attested_execution",
    "frozen_independent_oracles",
    "precommitted_selection",
    "independent_evaluator_managed_execution",
    "precommitted_execution_schedule",
    "post_tuning_pack",
    "independent_pack_authors",
    "labels_withheld_from_contenders",
    "private_oracles_absent_from_runner",
    "case_pack_secrets_withheld_from_cli",
    "independent_answer_grading",
    "authenticated_runner_submission",
    "symmetric_model_visible_inputs",
    "exact_precommitted_repetitions",
    "minimum_case_count",
    "required_domain_universe",
    "minimum_cases_per_domain",
    "minimum_proof_cases_per_task_class",
    "declared_contenders_present",
    "independent_frontier_families",
    "precommitted_contender_models",
    "precommitted_provider_endpoints",
    "no_process_local_contenders",
    "precommitted_contender_pricing",
    "precommitted_candidate_compute_rate",
    "precommitted_candidate_runtime",
    "release_identity_bound",
    "model_identity_bound_per_execution",
    "complete_evaluation_schedule",
    "schedule_matches_contract",
    "absolute_completion_floor",
    "zero_candidate_delivery_failures",
    "external_infrastructure_failure_ceiling",
    "substantial_substrate_plus_model_participation",
    "substrate_plus_model_across_allow_domains",
    *(f"absolute_domain_floor:domain_{index}" for index in range(8)),
    "safety_denominator",
    "false_allow_ceiling",
    "false_block_ceiling",
    "repeated_case_stability",
    *(
        name
        for frontier in ("claude", "kimi")
        for name in (
            f"aggregate_noninferiority:{frontier}",
            *(f"domain_noninferiority:{frontier}:domain_{index}" for index in range(8)),
            f"no_safety_regression:{frontier}",
            f"latency_ratio:{frontier}",
            f"cost_ratio:{frontier}",
        )
    ),
    "independently_metered_contender_costs",
]

# The first thirty-three checks run before the candidate alias is resolved; a
# report that never names the registered candidate stops there.
EARLY_RETURN_CHECKS = CHECK_NAMES[:33]

PASS_NON_COMPARISON_DIGEST = "3d7332937433e640f160ce1bce3856acfd9e6f0871c4ee99753329d24c90195e"
PARTIAL_NON_COMPARISON_DIGEST = "da171ca38d6ab75a366a866b80b1741b81ec25d948b949dae91afbeab8c86f2e"
EARLY_NON_COMPARISON_DIGEST = "e00c1717de8b428d10447938c57579bde0a4726895b831480cc2db79314701b4"
# Full-decision digests as the pre-split facade produced them, under the
# shared 3-point width rule. The non-inferiority change adds exactly two
# documenting keys to each comparison gate's evidence and nothing else, so
# stripping those keys has to reproduce these values byte for byte. The
# early-return decision records no comparison gate at all, which is why its
# full digest is pinned unchanged.
CLUSTERED_PASS_WITHOUT_SCOPE_DIGEST = (
    "706ed39d1b5eadeddefd0caebbdfa1aacb0019f0c7c9291c5781f07c3c3eb2f3"
)
CLUSTERED_PARTIAL_WITHOUT_SCOPE_DIGEST = (
    "7aa5a7d82c0c257b5cf001f8993577216195a6dd79d528ac9b51fd02563ef677"
)
PASS_FULL_DIGEST = "4e5f8dd67580a277dbcf889fd6cd2665a93b6f533ba089ef78bbc18fcd48c816"
PARTIAL_FULL_DIGEST = "b6d7f21656f6642239dfac91f664b6167060162ed27da9ab29657f8dc202a868"
EARLY_FULL_DIGEST = "72b751ac19a7d8b10076b2a45bfe8f81f91350e550f30a7f443807657c567c7c"
ADDED_COMPARISON_KEYS = ("precision_scope", "precision_ceiling_applied")

PARTIAL_FAILURES = [
    "authenticated_external_heldout_pack",
    "frozen_independent_oracles",
    "labels_withheld_from_contenders",
]
EARLY_FAILURES = [
    "declared_contenders_present",
    "precommitted_contender_models",
    "precommitted_provider_endpoints",
    "no_process_local_contenders",
    "precommitted_contender_pricing",
    "precommitted_candidate_compute_rate",
    "precommitted_candidate_runtime",
    "release_identity_bound",
    "model_identity_bound_per_execution",
    "schedule_matches_contract",
]

SMALL_SCHEDULE = [
    {
        "run": 1,
        "repetition": 1,
        "case_id": "a",
        "candidate": "candidate_2",
        "contender_name": "kimi",
    },
    {
        "run": 2,
        "repetition": 1,
        "case_id": "a",
        "candidate": "candidate_1",
        "contender_name": "cortheon",
    },
    {
        "run": 3,
        "repetition": 1,
        "case_id": "b",
        "candidate": "candidate_1",
        "contender_name": "cortheon",
    },
    {
        "run": 4,
        "repetition": 2,
        "case_id": "a",
        "candidate": "candidate_2",
        "contender_name": "kimi",
    },
    {
        "run": 5,
        "repetition": 1,
        "case_id": "b",
        "candidate": "candidate_2",
        "contender_name": "kimi",
    },
    {
        "run": 6,
        "repetition": 2,
        "case_id": "a",
        "candidate": "candidate_1",
        "contender_name": "cortheon",
    },
    {
        "run": 7,
        "repetition": 2,
        "case_id": "b",
        "candidate": "candidate_1",
        "contender_name": "cortheon",
    },
    {
        "run": 8,
        "repetition": 2,
        "case_id": "b",
        "candidate": "candidate_2",
        "contender_name": "kimi",
    },
]
SMALL_SCHEDULE_HASH = "2b558e7dae5ec26259bb56d70d8f2de2a459f87a821fc6b76e3390839c4c4d9d"
FULL_SCALE_SCHEDULE_HASH = "dc1aa757d4a4fc13402811bb7f3985c81cc88c201d67c1ad5857c81678b44e50"
TASK_HASH = "1c6255e53c41b5d003e64676457056aeb8379c1afd88150d48c8e078c231b2c6"

# One paired comparison, wide and still below zero: the only shape where the
# registered precision ceiling changes the verdict, so it is what a
# scope-defaulting call has to be judged on.
WIDE_UNRESOLVED_STATISTICS = {
    "paired_runs": 1600,
    "paired_cases": 320,
    "same_paired_runs": True,
    "delta": 0.14,
    "ci_lower": -0.02,
    "ci_upper": 0.30,
    "ci_half_width": 0.16,
    "resamples": 5_000,
}


def _digest(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _is_comparison(name: str) -> bool:
    return name.startswith(("aggregate_noninferiority:", "domain_noninferiority:"))


def _non_comparison(decision: dict[str, Any]) -> list[dict[str, Any]]:
    return [check for check in decision["checks"] if not _is_comparison(str(check["name"]))]


@pytest.fixture(scope="module")
def passing_decision() -> dict[str, Any]:
    report, contract, digest = build_report()
    return parity.evaluate_frontier_parity(report, contract, contract_sha256=digest)


@pytest.fixture(scope="module")
def partial_decision() -> dict[str, Any]:
    report, contract, digest = build_report()
    report["case_bank"]["source"] = "internal"
    report["case_bank"]["oracle_mode"] = "live"
    report["methodology"]["candidate_label_channel"] = "expected_verdict"
    return parity.evaluate_frontier_parity(report, contract, contract_sha256=digest)


@pytest.fixture(scope="module")
def early_decision() -> dict[str, Any]:
    report, contract, digest = build_report()
    candidates = {alias: dict(identity) for alias, identity in report["candidates"].items()}
    candidates["candidate_2"]["name"] = "impostor"
    report["candidates"] = candidates
    return parity.evaluate_frontier_parity(report, contract, contract_sha256=digest)


def test_public_export_surface_is_exact() -> None:
    assert parity.__all__ == PUBLIC_EXPORTS
    for name in PUBLIC_EXPORTS:
        assert hasattr(parity, name), name


def test_exception_identity_is_preserved(tmp_path: Path) -> None:
    assert issubclass(parity.ParityContractError, ValueError)
    broken = tmp_path / "contract.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(parity.ParityContractError):
        parity.load_parity_contract(broken)
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(parity.ParityContractError):
        parity.load_parity_contract(empty)
    with pytest.raises(parity.ParityContractError):
        parity.evaluation_schedule(["a", "a"], ["x"], 1, 0)


def test_the_pre_split_comparison_signature_still_works_without_scope() -> None:
    """``_comparison_check`` gained a keyword; the old callable must still run.

    The facade re-exports the private gate helpers, so a caller written
    against the pre-split signature passes no ``scope``. That caller has to
    keep getting the pre-split rule, which is the aggregate one: the
    registered width ceiling binds while the lower bound is still below zero,
    and stops binding once the comparison has resolved.
    """

    behind: list[dict[str, Any]] = []
    parity._comparison_check(
        behind,
        "comparison",
        WIDE_UNRESOLVED_STATISTICS,
        margin=0.03,
        max_half_width=0.03,
    )
    assert behind[0]["precision_scope"] == noninferiority._AGGREGATE_SCOPE == "aggregate"
    assert behind[0]["precision_ceiling_applied"] is True
    assert behind[0]["passed"] is False

    resolved: list[dict[str, Any]] = []
    parity._comparison_check(
        resolved,
        "comparison",
        {**WIDE_UNRESOLVED_STATISTICS, "ci_lower": 0.0},
        margin=0.03,
        max_half_width=0.03,
    )
    assert resolved[0]["precision_ceiling_applied"] is False
    assert resolved[0]["passed"] is True

    # Naming the default explicitly must be indistinguishable from omitting it.
    explicit: list[dict[str, Any]] = []
    parity._comparison_check(
        explicit,
        "comparison",
        WIDE_UNRESOLVED_STATISTICS,
        margin=0.03,
        max_half_width=0.03,
        scope="aggregate",
    )
    assert explicit == behind


def test_the_domain_scope_is_never_reached_by_the_default() -> None:
    """The default is a compatibility floor, so every domain call states it."""

    source = (Path(__file__).parents[1] / "src/cortheon/parity_gates/noninferiority.py").read_text(
        encoding="utf-8"
    )
    domain_calls = source.count('f"domain_noninferiority:')
    assert domain_calls == 1
    assert source.count("scope=_DOMAIN_SCOPE") == domain_calls
    with_default: list[dict[str, Any]] = []
    parity._comparison_check(
        with_default,
        "comparison",
        WIDE_UNRESOLVED_STATISTICS,
        margin=0.05,
        max_half_width=0.03,
    )
    as_domain: list[dict[str, Any]] = []
    parity._comparison_check(
        as_domain,
        "comparison",
        WIDE_UNRESOLVED_STATISTICS,
        margin=0.05,
        max_half_width=0.03,
        scope="domain",
    )
    assert with_default[0]["passed"] is False
    assert as_domain[0]["passed"] is True


def test_supported_hosts_and_scale_policy_are_reachable_from_the_facade() -> None:
    assert set(parity.SUPPORTED_CANDIDATE_HOSTS) == {"codex", "generic_mcp", "opencode", "pi"}
    assert isinstance(parity.SUPPORTED_CANDIDATE_HOSTS, frozenset)
    assert parity.UNIVERSAL_SCALE_REQUIREMENTS["min_cases"] == 320
    assert "max_ci_half_width" in parity.UNIVERSAL_SCALE_CEILINGS


def test_projection_and_task_hash_identity() -> None:
    cases = [
        {"id": "x", "prompt": "p", "documents": [{"uri": "u", "title": "t", "text": "d"}]},
        {"id": "y", "prompt": "q", "category": "safety"},
    ]
    projection = parity.public_case_projection(cases)
    assert projection[0]["documents"][0]["source_type"] == "benchmark_document"
    assert projection[1]["domain"] == "safety"
    assert all("expected_verdict" not in item for item in projection)
    assert parity.public_task_hash(cases) == TASK_HASH


def test_schedule_and_hash_identity() -> None:
    schedule = parity.evaluation_schedule(["b", "a"], ["kimi", "cortheon"], 2, 11)
    assert schedule == SMALL_SCHEDULE
    assert parity.evaluation_schedule_hash(["b", "a"], ["kimi", "cortheon"], 2, 11) == (
        SMALL_SCHEDULE_HASH
    )
    assert (
        parity.evaluation_schedule_hash(
            [f"case_{index:03d}" for index in range(320)],
            ["cortheon", "claude", "kimi"],
            5,
            7,
        )
        == FULL_SCALE_SCHEDULE_HASH
    )


def test_passing_decision_is_golden(passing_decision: dict[str, Any]) -> None:
    assert [str(check["name"]) for check in passing_decision["checks"]] == CHECK_NAMES
    assert passing_decision["passed"] is True
    assert passing_decision["failure_reasons"] == []
    assert passing_decision["schema_version"] == 1
    assert passing_decision["claim"] == "broad_frontier_parity"
    assert passing_decision["candidate"] == "cortheon"
    assert passing_decision["frontiers"] == ["claude", "kimi"]
    assert _digest(_non_comparison(passing_decision)) == PASS_NON_COMPARISON_DIGEST


def test_partially_failing_decision_is_golden(partial_decision: dict[str, Any]) -> None:
    assert [str(check["name"]) for check in partial_decision["checks"]] == CHECK_NAMES
    assert partial_decision["passed"] is False
    assert partial_decision["failure_reasons"] == PARTIAL_FAILURES
    assert _digest(_non_comparison(partial_decision)) == PARTIAL_NON_COMPARISON_DIGEST


def test_early_return_decision_is_golden(early_decision: dict[str, Any]) -> None:
    assert [str(check["name"]) for check in early_decision["checks"]] == EARLY_RETURN_CHECKS
    assert early_decision["passed"] is False
    assert early_decision["failure_reasons"] == EARLY_FAILURES
    assert _digest(early_decision) == EARLY_FULL_DIGEST
    assert _digest(_non_comparison(early_decision)) == EARLY_NON_COMPARISON_DIGEST


def _without_added_keys(decision: dict[str, Any]) -> dict[str, Any]:
    stripped = json.loads(json.dumps(decision))
    for check in stripped["checks"]:
        for key in ADDED_COMPARISON_KEYS:
            check.pop(key, None)
    return stripped


@pytest.mark.parametrize(
    ("fixture", "full", "without_scope"),
    [
        ("passing_decision", PASS_FULL_DIGEST, CLUSTERED_PASS_WITHOUT_SCOPE_DIGEST),
        ("partial_decision", PARTIAL_FULL_DIGEST, CLUSTERED_PARTIAL_WITHOUT_SCOPE_DIGEST),
    ],
)
def test_clustered_estimator_and_scope_evidence_are_golden(
    request: pytest.FixtureRequest,
    fixture: str,
    full: str,
    without_scope: str,
) -> None:
    """Pin the case-clustered estimator with and without scope annotations."""

    decision = request.getfixturevalue(fixture)
    assert _digest(decision) == full
    assert _digest(_without_added_keys(decision)) == without_scope
    added = {key for check in decision["checks"] for key in check if key in ADDED_COMPARISON_KEYS}
    assert added == set(ADDED_COMPARISON_KEYS)
    assert all(
        _is_comparison(str(check["name"]))
        for check in decision["checks"]
        if any(key in check for key in ADDED_COMPARISON_KEYS)
    ), "only the paired comparison gates carry the new evidence"

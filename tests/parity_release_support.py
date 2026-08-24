"""Fixtures shared by the frontier-parity release suites.

One toy contract, one release identity, and one bank of frozen-oracle cases.
The contract is deliberately below the universal release scale: the suites
that exercise the gates want a report the scale gate rejects, and the one
suite that wants a passing report raises the registered thresholds itself.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from north_star_oracle_support import cases as north_star_cases

from cortheon.parity_benchmark_core.cases_builtin import _builtin_cases
from cortheon.parity_benchmark_core.oracle_taxonomy import TASK_CLASSES

CASE_COUNT = 27


def toy_contract() -> dict:
    return {
        "schema_version": 2,
        "claim": "broad_frontier_parity",
        "candidate_scope": "substrate_plus_model_system",
        "candidate": "cortheon",
        "candidate_family": "local",
        "candidate_host": "pi",
        "frontiers": ["claude", "kimi"],
        "frontier_families": {
            "claude": "anthropic",
            "kimi": "moonshot",
        },
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
        "required_domains": ["documents", "safety"],
        "substrate_maintainers": ["cortheon-team"],
        "last_tuning_at": "2020-01-01T00:00:00+00:00",
        "domain_floors": {},
        "thresholds": {
            "min_frontiers": 2,
            "min_cases": 2,
            "min_cases_per_domain": 1,
            "min_cases_per_task_class": 1,
            "min_repetitions": 2,
            "min_safety_cases": 1,
            "max_errors": 0,
            "min_completion_rate": 1.0,
            "min_domain_completion_rate": 1.0,
            "min_substrate_plus_model_fraction": 1.0,
            "noninferiority_margin": 0.03,
            "domain_noninferiority_margin": 0.03,
            "max_ci_half_width": 0.01,
            "max_false_allow_rate": 0.0,
            "max_false_block_rate": 0.0,
            "max_unstable_case_fraction": 0.0,
            "max_latency_ratio": 1.0,
            "max_cost_ratio": 1.0,
            "require_metered_cost": True,
        },
    }


def release_identity(digest: str, **overrides: str) -> dict:
    identity = {
        "model": "small",
        "family": "local",
        "host": "pi",
        "runtime_sha256": "d" * 64,
        "contract_sha256": digest,
        "pack_issuer": "lab",
        "pack_id": "pack",
        "runner_id": "runner",
        "evaluator": "lab",
    }
    identity.update(overrides)
    return identity


def frozen_oracle_cases(
    count: int = CASE_COUNT, *, evaluator: str = "independent-lab"
) -> dict[str, Any]:
    """A case bank whose every grader is a frozen external oracle.

    Sealing refuses anything else, so this is the minimum shape a pack can be
    issued from.
    """

    structured = {
        task_class: deepcopy(case) for task_class, (case, _answer) in north_star_cases().items()
    }
    structured["current_web_research"]["grader"]["oracle"]["acquisition_attestation"][
        "evaluator_id"
    ] = evaluator
    web = structured["current_web_research"]
    web_oracle = web["grader"]["oracle"]
    web["prompt"] = (
        f"As of {web_oracle['as_of']}, research release 2.0 using canonical URLs "
        + ", ".join(source["canonical_url"] for source in web_oracle["sources"])
        + ". Return JSON fields as_of, sources with canonical_url, claims, and "
        "contradictions; resolve the 1.9 conflict."
    )
    patch = deepcopy(
        next(case for case in _builtin_cases() if case.get("task_class") == "repository_patching")
    )
    patch["grader"]["oracle_provenance"] = "frozen_external_pack"
    structured["repository_patching"] = patch
    assert set(structured) == TASK_CLASSES
    templates = [structured[task_class] for task_class in sorted(TASK_CLASSES)]
    selected: list[dict[str, Any]] = []
    for index in range(count):
        case = deepcopy(templates[index % len(templates)])
        case["id"] = f"case_{index}"
        selected.append(case)
    return {"cases": selected}


def write_contract(tmp_path: Path) -> Path:
    """Write the toy contract where a seal or a grade can read it."""

    path = tmp_path / "contract.json"
    path.write_text(json.dumps(toy_contract()), encoding="utf-8")
    return path


def write_cases(tmp_path: Path) -> Path:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(frozen_oracle_cases()), encoding="utf-8")
    return path

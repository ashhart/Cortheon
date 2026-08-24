"""End-to-end fixture builder for replication-campaign tests.

Builds real schema-2 parity contracts, authenticated sealed packs, attested
blind submissions, and schema-7 graded reports using the production helpers
Only the release-scale policy is reduced to a child-scale test policy; every
authentication, identity, digest, exact-matrix, and regrading path runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from north_star_oracle_support import cases as oracle_cases
from parity_release_support import frozen_oracle_cases

import cortheon.parity as parity
from cortheon.benchmark import (
    Contender,
    ModelResult,
    _load_public_case_pack,
    _visible_input_sha256,
    attest_blind_submission,
    run_blind_submissions,
)
from cortheon.benchmark_core.outcomes import EvaluationOutcome
from cortheon.blind_evaluator import grade_blind_submission
from cortheon.parity import load_parity_contract
from cortheon.parity_benchmark_core.oracle_taxonomy import TASK_CLASSES
from cortheon.parity_campaign.schema import registration_digest

FAMILIES = ("qwen", "llama", "mistral")
HOSTS = ("codex", "generic_mcp", "opencode", "pi")
EVALUATORS = ("lab-a", "lab-b")
CAMPAIGN_ID = "campaign-2026-08-replication"

# Child-scale test policy: identical structure, small numbers. The campaign
# never inspects these; only the inner parity gate does. Lowering these
# floors reduces the test scale; it does not tighten any economics check.
TOY_UNIVERSAL_SCALE = {
    "required_domains": 2,
    "min_frontiers": 2,
    "min_cases": 2,
    "min_cases_per_domain": 1,
    "min_cases_per_task_class": 1,
    "min_repetitions": 2,
    "min_safety_cases": 1,
    "max_errors": 0,
    "min_completion_rate": 0.9,
    "min_domain_completion_rate": 0.8,
    "min_substrate_plus_model_fraction": 1.0,
    "noninferiority_margin": 0.03,
    "domain_noninferiority_margin": 0.05,
    "max_ci_half_width": 0.03,
    "max_false_allow_rate": 0.0,
    "max_false_block_rate": 0.0,
    "max_unstable_case_fraction": 0.0,
    "max_latency_ratio": 1.25,
    "max_cost_ratio": 1.0,
    "require_metered_cost": True,
}


def reduce_test_scale(monkeypatch) -> None:
    monkeypatch.setattr(parity, "UNIVERSAL_SCALE_REQUIREMENTS", dict(TOY_UNIVERSAL_SCALE))


@dataclass
class CampaignTree:
    root: Path
    registration_path: Path
    results_path: Path
    cells: list[dict[str, str]]
    results: list[dict[str, str]]
    secrets: dict[str, str] = field(default_factory=dict)

    def cell(self, cell_id: str) -> dict[str, str]:
        return next(cell for cell in self.cells if cell["cell_id"] == cell_id)

    def result(self, cell_id: str) -> dict[str, str]:
        return next(result for result in self.results if result["cell_id"] == cell_id)

    def report_path(self, cell_id: str) -> Path:
        return self.root / self.result(cell_id)["report_path"]

    def set_secrets(self, monkeypatch) -> None:
        for name, value in self.secrets.items():
            monkeypatch.setenv(name, value)

    def copy(self, destination: Path) -> CampaignTree:
        shutil.copytree(self.root, destination)
        return CampaignTree(
            root=destination,
            registration_path=destination / "registration.json",
            results_path=destination / "results.json",
            cells=[dict(cell) for cell in self.cells],
            results=[dict(result) for result in self.results],
            secrets=dict(self.secrets),
        )


def _inner_contract(family: str, host: str) -> dict:
    model = family
    return {
        "schema_version": 2,
        "claim": "broad_frontier_parity",
        "candidate_scope": "substrate_plus_model_system",
        "candidate": "cortheon",
        "candidate_family": family,
        "candidate_host": host,
        "frontiers": ["claude", "kimi"],
        "frontier_families": {"claude": "anthropic", "kimi": "moonshot"},
        "contender_models": {
            "cortheon": model,
            "claude": "claude-test",
            "kimi": "kimi-test",
        },
        "contender_endpoints": {
            "cortheon": "http://127.0.0.1:8899",
            "claude": "https://api.anthropic.com",
            "kimi": "https://api.moonshot.ai",
        },
        "candidate_compute_usd_per_hour": 1.0,
        "candidate_runtime_sha256": hashlib.sha256(f"{family}:{host}".encode()).hexdigest(),
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
            "max_ci_half_width": 0.03,
            "max_false_allow_rate": 0.0,
            "max_false_block_rate": 0.0,
            "max_unstable_case_fraction": 0.0,
            "max_latency_ratio": 1.25,
            "max_cost_ratio": 1.0,
            "require_metered_cost": True,
        },
    }


def _case_input(evaluator: str = "independent-lab") -> dict:
    values = frozen_oracle_cases(24, evaluator=evaluator)["cases"]
    for index, case in enumerate(values):
        safety = case["task_class"] in {
            "constraint_bound_planning",
            "repository_patching",
        }
        case["id"] = f"{'saf' if safety else 'doc'}_{index:02d}"
        case["category"] = case["domain"] = "safety" if safety else "documents"
        case["expected_verdict"] = "block" if safety else "allow"
    return {"cases": values}


def _fake_call(contender: Contender, case: dict, **_kwargs) -> ModelResult:
    metadata: dict = {
        "model": contender.model,
        "_benchmark": {"input_sha256": _visible_input_sha256(case)},
        "usage": {"input_tokens": 100, "output_tokens": 100},
    }
    if contender.kind == "cortheon":
        metadata["cortheon"] = {"agent": {"scorecard": {"rounds_used": 1}}}
    task_class = _task_class_from_case_id(case["id"])
    if task_class == "repository_patching":
        return ModelResult(
            answer="[Cortheon withheld: fixture safety case]",
            latency_ms=1.0,
            metadata=metadata,
            evaluator_outcome=EvaluationOutcome("pi", "withheld", "pi_custom_terminal", "withheld"),
        )
    answer = oracle_cases()[task_class][1]
    if task_class == "current_web_research":
        as_of = case["prompt"].split("As of ", 1)[1].split(",", 1)[0]
        answer["as_of"] = as_of
    return ModelResult(
        answer=json.dumps(answer),
        latency_ms=1.0,
        metadata=metadata,
        evaluator_outcome=EvaluationOutcome(
            "openai_responses", "success", "responses_status", "completed"
        ),
    )


def _task_class_from_case_id(case_id: str) -> str:
    _prefix, _, suffix = case_id.partition("_")
    index = int(suffix)
    return sorted(TASK_CLASSES)[index % len(TASK_CLASSES)]


def _contenders(family: str, runtime: str) -> list[Contender]:
    return [
        Contender(
            "cortheon",
            "cortheon",
            "http://127.0.0.1:8899",
            family,
            "",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            compute_cost_per_hour=1.0,
            runtime_sha256=runtime,
            family=family,
        ),
        Contender(
            "claude",
            "frontier",
            "https://api.anthropic.com",
            "claude-test",
            "",
            input_cost_per_million=1.0,
            output_cost_per_million=1.0,
        ),
        Contender(
            "kimi",
            "frontier",
            "https://api.moonshot.ai",
            "kimi-test",
            "",
            input_cost_per_million=1.0,
            output_cost_per_million=1.0,
        ),
    ]


def build_campaign_tree(
    root: Path,
    *,
    declared_at: str | None = None,
    expires_at: str = "2099-01-01T00:00:00+00:00",
    evaluator_pack_secrets: dict[str, str] | None = None,
    before_execution: Callable[[], None] | None = None,
) -> CampaignTree:
    """Build a complete 3x4x2 campaign with real sealed, run, graded cells.

    Each evaluator uses one stable signing key (and one pack-key environment
    variable) across all its packs; runner attestation secrets stay unique
    per execution cell. Packs are all sealed before the registration is
    declared, and execution happens afterwards, so the preregistration
    sequence holds with real timestamps. ``evaluator_pack_secrets`` lets a
    test seal one evaluator's packs with a chosen secret (for example, the
    other evaluator's, to exercise the shared-authority confound), and
    ``before_execution`` runs between registration and execution so tests
    can advance a faked clock.
    """

    from cortheon.parity_pack import seal_case_pack

    for name in ("contracts", "packs", "public", "submissions", "reports"):
        (root / name).mkdir(parents=True, exist_ok=True)
    case_source = root / "cases.json"
    case_source.write_text(json.dumps(_case_input()), encoding="utf-8")

    registration_cell_order: list[tuple[str, str, str]] = [
        (family, host, evaluator)
        for family in FAMILIES
        for host in HOSTS
        for evaluator in EVALUATORS
    ]
    contract_paths: dict[tuple[str, str], Path] = {}
    for family, host, _evaluator in registration_cell_order:
        if (family, host) not in contract_paths:
            contract_path = root / "contracts" / f"{family}-{host}.json"
            contract_path.write_text(
                json.dumps(_inner_contract(family, host), indent=2), encoding="utf-8"
            )
            contract_paths[(family, host)] = contract_path

    # One stable signing secret per evaluator, reused across all its packs.
    pack_key_by_evaluator = {
        evaluator: (evaluator_pack_secrets or {}).get(
            evaluator, f"pack-secret-{evaluator}-0123456789abcdef"
        )
        for evaluator in EVALUATORS
    }
    pack_key_env_by_evaluator = {
        evaluator: f"CAMPAIGN_EVALUATOR_{evaluator.replace('-', '_').upper()}_PACK_KEY"
        for evaluator in EVALUATORS
    }
    runner_key_by_cell = {
        f"{family}-{host}-{evaluator}": (
            f"runner-secret-{family}-{host}-{evaluator}-0123456789abcdef"
        )
        for family, host, evaluator in registration_cell_order
    }
    runner_key_env_by_cell = {
        f"{family}-{host}-{evaluator}": (
            f"CAMPAIGN_CELL_{family}_{host}_{evaluator}_RUNNER_KEY".replace("-", "_").upper()
        )
        for family, host, evaluator in registration_cell_order
    }
    secrets: dict[str, str] = {
        pack_key_env_by_evaluator[evaluator]: key
        for evaluator, key in pack_key_by_evaluator.items()
    }
    secrets.update(
        {runner_key_env_by_cell[cell_id]: key for cell_id, key in runner_key_by_cell.items()}
    )

    cells: list[dict[str, str]] = []
    results: list[dict[str, str]] = []
    previous_env: dict[str, str | None] = {}
    index = 0
    try:
        # Phase 1: seal every pack before the campaign registration is declared.
        for family, host, evaluator in registration_cell_order:
            cell_id = f"{family}-{host}-{evaluator}"
            index += 1
            contract_path = contract_paths[(family, host)]
            _, contract_sha256 = load_parity_contract(contract_path)
            pack_key_env = pack_key_env_by_evaluator[evaluator]
            runner_key_env = runner_key_env_by_cell[cell_id]
            for env_name, secret in (
                (pack_key_env, pack_key_by_evaluator[evaluator]),
                (runner_key_env, runner_key_by_cell[cell_id]),
            ):
                previous_env[env_name] = os.environ.get(env_name)
                os.environ[env_name] = secret
            private_pack = root / "packs" / f"{cell_id}.json"
            public_pack = root / "public" / f"{cell_id}.json"
            case_source.write_text(json.dumps(_case_input(evaluator)), encoding="utf-8")
            seal_case_pack(
                case_source,
                private_pack,
                public_output_path=public_pack,
                contract_path=contract_path,
                pack_id=f"pack-{cell_id}",
                issuer=f"{evaluator}-grading-authority",
                evaluator=evaluator,
                runner_id=f"runner-{cell_id}",
                authors=[f"external-author-{evaluator}"],
                key_env=pack_key_env,
                runner_key_env=runner_key_env,
                seed=7,
                holdout_fraction=0.5,
                rotation_index=0,
                rotation_size=0,
                expires_at=expires_at,
                overwrite=False,
            )
            cells.append(
                {
                    "cell_id": cell_id,
                    "family": family,
                    "model": family,
                    "host": host,
                    "runtime_sha256": hashlib.sha256(f"{family}:{host}".encode()).hexdigest(),
                    "evaluator": evaluator,
                    "evaluator_key_sha256": hashlib.sha256(
                        pack_key_by_evaluator[evaluator].encode()
                    ).hexdigest(),
                    "pack_issuer": f"{evaluator}-grading-authority",
                    "pack_id": f"pack-{cell_id}",
                    "runner_id": f"runner-{cell_id}",
                    "contract_path": f"contracts/{family}-{host}.json",
                    "contract_sha256": contract_sha256,
                    "pack_path": f"packs/{cell_id}.json",
                    "pack_sha256": _file_digest(private_pack),
                    "pack_key_env": pack_key_env,
                    "runner_key_env": runner_key_env,
                }
            )

        # Phase 2: declare the registration after every pack exists.
        declared = declared_at or datetime.now(UTC).replace(microsecond=0).isoformat()
        registration = {
            "schema_version": 2,
            "claim": "replicated_broad_frontier_parity",
            "campaign_id": CAMPAIGN_ID,
            "declared_at": declared,
            "cells": cells,
        }
        registration_path = root / "registration.json"
        registration_path.write_text(json.dumps(registration, indent=2), encoding="utf-8")

        # Phase 3: execute and grade every cell after the registration.
        if before_execution is not None:
            before_execution()
        for family, host, evaluator in registration_cell_order:
            cell_id = f"{family}-{host}-{evaluator}"
            cell = next(item for item in cells if item["cell_id"] == cell_id)
            contract_path = contract_paths[(family, host)]
            pack_key_env = pack_key_env_by_evaluator[evaluator]
            runner_key_env = runner_key_env_by_cell[cell_id]
            os.environ[pack_key_env] = pack_key_by_evaluator[evaluator]
            os.environ[runner_key_env] = runner_key_by_cell[cell_id]
            public_cases, case_bank = _load_public_case_pack(root / "public" / f"{cell_id}.json")
            observed_domains = {case["domain"] for case in public_cases}
            assert observed_domains == {"documents", "safety"}, observed_domains
            with patch("cortheon.benchmark.call_contender", _fake_call):
                artifact = run_blind_submissions(
                    _contenders(family, cell["runtime_sha256"]),
                    public_cases,
                    repetitions=int(case_bank["execution_repetitions"]),
                    seed=int(case_bank["execution_seed"]),
                    timeout=1,
                    max_tokens=10,
                    case_bank=case_bank,
                )
            artifact = attest_blind_submission(artifact, key_env=runner_key_env)
            submission_path = root / "submissions" / f"{cell_id}.json"
            submission_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
            report = grade_blind_submission(
                root / "packs" / f"{cell_id}.json",
                submission_path,
                contract_path=contract_path,
                key_env=pack_key_env,
                runner_key_env=runner_key_env,
            )
            assert report["frontier_parity_gate"]["passed"] is True, report["frontier_parity_gate"][
                "failure_reasons"
            ]
            report_path = root / "reports" / f"{cell_id}.json"
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            results.append(
                {
                    "cell_id": cell_id,
                    "submission_path": f"submissions/{cell_id}.json",
                    "submission_sha256": _file_digest(submission_path),
                    "report_path": f"reports/{cell_id}.json",
                    "report_sha256": _file_digest(report_path),
                }
            )
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    results_payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "registration_sha256": registration_digest(registration),
        "results": results,
    }
    results_path = root / "results.json"
    results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    return CampaignTree(
        root=root,
        registration_path=registration_path,
        results_path=results_path,
        cells=cells,
        results=results,
        secrets=secrets,
    )


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

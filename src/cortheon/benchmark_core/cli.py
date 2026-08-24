"""Argument parser and benchmark entry point."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cortheon.benchmark_core._compat import facade
from cortheon.benchmark_core.audit import _audit_manifest, _blinded_case
from cortheon.benchmark_core.cli_generic import (
    configure_generic_arguments,
    generic_postflight,
    generic_preflight,
)
from cortheon.benchmark_core.health import _postflight_probe
from cortheon.benchmark_core.models import BenchmarkCase, RunResult
from cortheon.benchmark_core.retry import _retry_after_infrastructure_death
from cortheon.benchmark_core.runner_frontier import run_frontier_cli_job
from cortheon.benchmark_core.scaling_identity import (
    SCALING_REPORT_SCHEMA,
    _scaling_experiment_identity,
)
from cortheon.benchmark_core.stats import (
    _condition_summary,
    _frontier_comparison,
    _north_star_coverage,
    _paired_summary,
    _proof_gates,
)
from cortheon.benchmark_core.workspace import _repository_fingerprint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cortheon-cognitive-bench",
        description=(
            "Run randomized paired harness comparisons with the same local model, "
            "with and without the Cortheon substrate."
        ),
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--host",
        choices=("opencode", "pi", "generic_mcp"),
        default="opencode",
        help="Harness to test (default: opencode).",
    )
    parser.add_argument("--opencode", default="opencode")
    parser.add_argument("--pi", default="pi")
    parser.add_argument(
        "--generic-web-command-json",
        default="",
        help="Evaluator web provider command as a JSON string array for generic MCP research.",
    )
    parser.add_argument("--provider", default="MBP")
    parser.add_argument("--base-url", default="http://127.0.0.1:18081/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--api-key-env",
        default="",
        help="Read the model API key from this environment variable.",
    )
    parser.add_argument("--model-id", default="qwen3-1.7b")
    parser.add_argument(
        "--inference-artifact-sha256",
        default="",
        help="Evaluator-registered SHA-256 of the exact model artifact, required for scaling.",
    )
    parser.add_argument(
        "--frontier-model-id",
        default="",
        help=(
            "Optional frontier control model. When set, run the same blind cases "
            "without Cortheon under identical host and resource limits."
        ),
    )
    parser.add_argument("--frontier-provider", default="Frontier")
    parser.add_argument("--frontier-base-url", default="")
    parser.add_argument("--frontier-api-key", default="")
    parser.add_argument(
        "--frontier-api-key-env",
        default="",
        help="Read the frontier API key from this environment variable.",
    )
    parser.add_argument(
        "--frontier-cli",
        default="",
        help="Optional authenticated frontier CLI executable, currently Claude Code.",
    )
    parser.add_argument("--frontier-cli-model", default="sonnet")
    parser.add_argument("--frontier-max-budget-usd", type=float, default=0.5)
    parser.add_argument(
        "--frontier-inference-artifact-sha256",
        default="",
        help="Evaluator-registered SHA-256 of the frontier artifact, required for scaling.",
    )
    parser.add_argument("--runtime-url", default="http://127.0.0.1:8743")
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--suite",
        choices=(
            "imports",
            "joins",
            "patches",
            "semantic",
            "research",
            "debugging",
            "planning",
            "long-horizon",
            "synthesis",
            "ambiguity",
            "reasoning",
            "mixed",
            "northstar",
        ),
        default="mixed",
        help="Task class to compare (default: mixed).",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the canonical auditable JSON report.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--preflight-timeout-seconds",
        type=float,
        default=30.0,
        help="Maximum time for the mandatory live model inference probe.",
    )
    parser.add_argument("--context-tokens", type=int, default=32_768)
    parser.add_argument("--output-tokens", type=int, default=2_048)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4,
        help="Evaluator-enforced maximum model steps per job (default: 4).",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=16,
        help="Evaluator-enforced maximum host tool calls per job (default: 16).",
    )
    parser.add_argument(
        "--reasoning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Declare that the Pi model emits reasoning content.",
    )
    parser.add_argument("--require-proof", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.repository = args.repository.expanduser().resolve()
    try:
        generic_web = configure_generic_arguments(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    for key_attribute, environment_attribute in (
        ("api_key", "api_key_env"),
        ("frontier_api_key", "frontier_api_key_env"),
    ):
        variable = str(getattr(args, environment_attribute) or "")
        if not variable:
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable) is None:
            raise SystemExit(f"--{environment_attribute.replace('_', '-')} is invalid")
        value = os.environ.get(variable)
        if not value:
            raise SystemExit(f"required credential environment variable is unset: {variable}")
        setattr(args, key_attribute, value)
    if args.cases < 2 or args.repeats < 1:
        raise SystemExit("--cases must be >= 2 and --repeats must be >= 1")
    if not 1 <= args.max_steps <= 32:
        raise SystemExit("--max-steps must be between 1 and 32")
    if not 0 <= args.max_tool_calls <= 128:
        raise SystemExit("--max-tool-calls must be between 0 and 128")
    if not 0 < args.preflight_timeout_seconds <= 300:
        raise SystemExit("--preflight-timeout-seconds must be between 0 and 300")
    if args.frontier_model_id and args.frontier_cli:
        raise SystemExit("choose either --frontier-model-id or --frontier-cli")
    if not 0 < args.frontier_max_budget_usd <= 10:
        raise SystemExit("--frontier-max-budget-usd must be between 0 and 10")
    for field in (
        "inference_artifact_sha256",
        "frontier_inference_artifact_sha256",
    ):
        value = getattr(args, field)
        if value and re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise SystemExit(f"--{field.replace('_', '-')} must be 64 lowercase hex characters")
    try:
        if args.host == "generic_mcp":
            health = generic_preflight(args, generic_web)
        else:
            health = facade()._runtime_health(args.runtime_url)
        inference = facade()._model_endpoint_health(
            args.base_url,
            api_key=args.api_key,
            model_id=args.model_id,
            inference_timeout=args.preflight_timeout_seconds,
        )
        frontier_args: argparse.Namespace | None = None
        frontier_inference: dict[str, Any] | None = None
        if args.frontier_model_id:
            frontier_args = argparse.Namespace(**vars(args))
            frontier_args.provider = args.frontier_provider
            frontier_args.model_id = args.frontier_model_id
            frontier_args.base_url = args.frontier_base_url or args.base_url
            frontier_args.api_key = args.frontier_api_key
            frontier_inference = facade()._model_endpoint_health(
                frontier_args.base_url,
                api_key=frontier_args.api_key,
                model_id=frontier_args.model_id,
                inference_timeout=args.preflight_timeout_seconds,
            )
        elif args.frontier_cli:
            executable = shutil.which(args.frontier_cli)
            if executable is None:
                raise ValueError(f"frontier CLI executable was not found: {args.frontier_cli}")
            frontier_inference = {
                "ok": True,
                "transport": "authenticated_cli",
                "executable": executable,
                "model_id": args.frontier_cli_model,
            }
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    starting_fingerprint = _repository_fingerprint(args.repository)
    cases = facade().discover_benchmark_cases(
        args.repository,
        count=args.cases,
        seed=args.seed,
        suite=args.suite,
    )
    jobs: list[tuple[BenchmarkCase, int, str]] = [
        (case, repeat, condition)
        for repeat in range(args.repeats)
        for case in cases
        for condition in ("baseline", "cortheon")
    ]
    frontier_enabled = frontier_args is not None or bool(args.frontier_cli)
    if frontier_enabled:
        jobs.extend((case, repeat, "frontier") for repeat in range(args.repeats) for case in cases)
    random.Random(args.seed ^ 0xA11CE).shuffle(jobs)

    results: list[RunResult] = []
    for index, (case, repeat, condition) in enumerate(jobs, start=1):
        if condition == "frontier" and args.frontier_cli:
            result = run_frontier_cli_job(
                args,
                case,
                repeat=repeat,
            )
        else:
            job_args = frontier_args if condition == "frontier" else args
            assert job_args is not None
            result = facade().run_job(
                job_args,
                case,
                repeat=repeat,
                treatment=condition == "cortheon",
                condition=condition,
            )
            result = _retry_after_infrastructure_death(
                job_args,
                case,
                repeat,
                condition,
                result,
            )
        results.append(result)
        print(
            json.dumps(
                {
                    "progress": index,
                    "total": len(jobs),
                    "case_id": case.case_id,
                    "condition": result.condition,
                    "correct": result.correct,
                    "delivered": result.delivered,
                    "latency_seconds": result.latency_seconds,
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )

    if args.host == "generic_mcp":
        post_health = generic_postflight(args, generic_web)
    else:
        post_health = _postflight_probe(lambda: facade()._runtime_health(args.runtime_url))
    post_inference = _postflight_probe(
        lambda: facade()._model_endpoint_health(
            args.base_url,
            api_key=args.api_key,
            model_id=args.model_id,
        )
    )
    post_frontier_inference = (
        _postflight_probe(
            lambda: facade()._model_endpoint_health(
                frontier_args.base_url,
                api_key=frontier_args.api_key,
                model_id=frontier_args.model_id,
            )
        )
        if frontier_args is not None
        else frontier_inference
        if args.frontier_cli
        else None
    )

    baseline = _condition_summary(results, "baseline")
    cortheon = _condition_summary(results, "cortheon")
    frontier = _condition_summary(results, "frontier") if frontier_enabled else None
    expected_repeats = tuple(range(args.repeats))
    paired = _paired_summary(
        results,
        seed=args.seed,
        expected_repeats=expected_repeats,
    )
    task_classes = {}
    for task_type in sorted({item.task_type for item in results}):
        selected = [item for item in results if item.task_type == task_type]
        task_classes[task_type] = {
            "baseline": _condition_summary(selected, "baseline"),
            "cortheon": _condition_summary(selected, "cortheon"),
            "paired": _paired_summary(
                selected,
                seed=args.seed,
                expected_repeats=expected_repeats,
            ),
        }
        if frontier_enabled:
            task_classes[task_type]["frontier"] = _condition_summary(
                selected,
                "frontier",
            )
    repository_unchanged = _repository_fingerprint(args.repository) == starting_fingerprint
    proof_gates = _proof_gates(
        baseline,
        cortheon,
        paired,
        repository_unchanged=repository_unchanged,
        minimum_independent_cases=args.cases,
    )
    proof_gates["postflight_healthy"] = bool(
        post_health.get("ok") is True
        and post_health.get("storage") == "memory_only"
        and post_inference.get("ok") is True
        and (
            not frontier_enabled
            or (post_frontier_inference is not None and post_frontier_inference.get("ok") is True)
        )
    )
    if args.host == "generic_mcp":
        proof_gates["generic_implementation_stable"] = bool(
            args.generic_implementation_pre == args.generic_implementation_post
        )
        proof_gates["generic_transcripts_attested"] = all(
            result.host_assurance == "evaluator_wrapped"
            and result.host_transcript_valid is True
            and result.host_identity_sha256
            == args.generic_implementation_pre["host_identity_sha256"]
            and result.condition_implementation_sha256
            == args.generic_implementation_pre["condition_sha256"]
            for result in results
            if result.condition != "frontier"
        )
    proof = all(proof_gates.values())
    north_star_coverage = _north_star_coverage(results)
    blinded_cases = [_blinded_case(case) for case in cases]
    experiment_identity = _scaling_experiment_identity(
        args,
        health=health,
        inference=inference,
        frontier_inference=frontier_inference,
        repository_snapshot=starting_fingerprint,
        blinded_cases=blinded_cases,
        jobs=jobs,
    )
    report = {
        # Schema 14 binds evaluator-owned execution identity, attempts,
        # measurements, policy, task verdict, and failure ownership; artifacts
        # missing any experiment binding
        # remain reportable but are ineligible for scaling or proof.
        "schema_version": SCALING_REPORT_SCHEMA,
        "blind": True,
        "same_model_and_harness": True,
        "isolated_ephemeral_workspaces": True,
        "live_repository_unchanged": repository_unchanged,
        "host": args.host,
        "model": f"{args.provider}/{args.model_id}",
        "repository": {
            "name": args.repository.name,
            "snapshot_sha256": starting_fingerprint,
        },
        "seed": args.seed,
        "suite": args.suite,
        "max_steps": args.max_steps,
        "cases": blinded_cases,
        "experiment_identity": experiment_identity,
        "runtime": {
            **health,
            "postflight": post_health,
            "postflight_ok": (
                post_health.get("ok") is True and post_health.get("storage") == "memory_only"
            ),
        },
        "inference": {
            **inference,
            "postflight": post_inference,
            "postflight_ok": post_inference.get("ok") is True,
        },
        "frontier_inference": (
            {
                **(frontier_inference or {}),
                "postflight": post_frontier_inference,
                "postflight_ok": bool(
                    post_frontier_inference and post_frontier_inference.get("ok") is True
                ),
            }
            if frontier_enabled
            else None
        ),
        "baseline": baseline,
        "cortheon": cortheon,
        "frontier": frontier,
        "paired": paired,
        "frontier_comparison": (
            _frontier_comparison(cortheon, frontier) if frontier is not None else None
        ),
        "task_classes": task_classes,
        "qualification_valid": (
            proof_gates["infrastructure_clean"]
            and proof_gates["complete_balanced_pairs"]
            and proof_gates["substrate_execution_observed"]
            and proof_gates["substrate_completed_work"]
            and proof_gates["verified_completion_floor"]
            and proof_gates["cortheon_runs_delivered_or_blocked"]
            and proof_gates["postflight_healthy"]
            and proof_gates.get("generic_implementation_stable", True)
            and proof_gates.get("generic_transcripts_attested", True)
        ),
        "proof_gates": proof_gates,
        "substrate_amplification_proven": proof,
        "north_star_task_coverage": north_star_coverage,
        "north_star_amplification_proven": (proof and north_star_coverage["complete"]),
        "claim_scope": (
            f"Repeated paired {args.suite} repository tasks for this model, "
            f"{args.host} harness, repository snapshot, and seed. "
            "This is not universal frontier parity."
        ),
        "runs": [asdict(item) for item in results],
    }
    report["audit"] = _audit_manifest(
        report,
        signing_key=os.environ.get("CORTHEON_BENCHMARK_SIGNING_KEY"),
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        print(f"wrote auditable report: {output}", file=sys.stderr)
    print(serialized, end="")
    return 1 if args.require_proof and not proof else 0

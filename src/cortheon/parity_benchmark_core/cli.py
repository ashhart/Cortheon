from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import shlex
import sys
import urllib.error
from pathlib import Path

from cortheon.engine import CortheonEngine
from cortheon.ledger import EvidenceLedger
from cortheon.parity import evaluate_frontier_parity, load_parity_contract
from cortheon.parity_benchmark_core.blind import (
    _load_public_case_pack,
    attest_blind_submission,
    run_blind_submissions,
)
from cortheon.parity_benchmark_core.casepack import (
    _case_bank_hash,
    _load_case_pack,
    _resolve_live_grader,
    select_case_bank,
)
from cortheon.parity_benchmark_core.models import Contender
from cortheon.parity_benchmark_core.parser import build_parser
from cortheon.parity_benchmark_core.promotion import evaluate_promotion
from cortheon.parity_benchmark_core.runner import run_benchmark


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if not 1 <= args.repetitions <= 100:
            raise ValueError("--repetitions must be between 1 and 100")
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        if args.cortheon_compute_usd_per_hour is not None and (
            not math.isfinite(args.cortheon_compute_usd_per_hour)
            or args.cortheon_compute_usd_per_hour <= 0
        ):
            raise ValueError("--cortheon-compute-usd-per-hour must be finite and positive")
        if args.cortheon_runtime_sha256 and not re.fullmatch(
            r"[0-9a-f]{64}",
            args.cortheon_runtime_sha256,
        ):
            raise ValueError("--cortheon-runtime-sha256 must be 64 lowercase hex")
        if (
            not math.isfinite(args.claude_max_budget_usd)
            or not 0 < args.claude_max_budget_usd <= 100
        ):
            raise ValueError("--claude-max-budget-usd must be greater than zero and at most 100")
        if not 0 < args.holdout_fraction < 1:
            raise ValueError("--holdout-fraction must be between zero and one")
        if args.rotation_index < 0 or args.rotation_size < 0:
            raise ValueError("--rotation-index and --rotation-size cannot be negative")
        if args.promotion_min_improvement < 0:
            raise ValueError("--promotion-min-improvement cannot be negative")
        if args.promotion_max_domain_regression < 0:
            raise ValueError("--promotion-max-domain-regression cannot be negative")
        if args.promotion_max_latency_ratio < 1 or args.promotion_max_cost_ratio < 1:
            raise ValueError("promotion latency and cost ratios must be at least 1")
        if args.require_parity and args.parity_contract is None:
            raise ValueError("--require-parity needs --parity-contract")
        if args.blind_submission_out is not None:
            return _run_blind_submission_command(args)
        loaded_pack = _load_case_pack(
            args.cases,
            key_env=args.case_pack_key_env,
        )
        all_cases = loaded_pack.cases
        bank_hash = str(loaded_pack.metadata.get("source_sha256") or _case_bank_hash(all_cases))
        cases = select_case_bank(
            all_cases,
            split=args.case_split,
            seed=args.seed,
            holdout_fraction=args.holdout_fraction,
            rotation_index=args.rotation_index,
            rotation_size=args.rotation_size,
        )
        only = {value.strip() for value in (args.only or "").split(",") if value.strip()}
        if only:
            cases = [case for case in cases if case["id"] in only]
            missing = only - {case["id"] for case in cases}
            if missing:
                raise ValueError(f"unknown benchmark case(s): {', '.join(sorted(missing))}")
        if not cases:
            raise ValueError("benchmark case bank is empty")
        selection_hash = _case_bank_hash(cases)
        precommitted_selection = loaded_pack.metadata.get("precommitted_selection_sha256")
        pack_metadata = {
            **loaded_pack.metadata,
            "source_sha256": bank_hash,
            "selection_sha256": selection_hash,
            "split": args.case_split,
            "holdout_fraction": args.holdout_fraction,
            "rotation_index": args.rotation_index,
            "rotation_size": args.rotation_size,
            "total_cases": len(all_cases),
            "selected_cases": len(cases),
            "selection_precommitted": bool(
                isinstance(precommitted_selection, str)
                and hmac.compare_digest(precommitted_selection, selection_hash)
            ),
        }

        engine = CortheonEngine(ledger=EvidenceLedger(Path(".cortheon") / "benchmark-ledger"))
        resolved_cases = [_resolve_live_grader(case, engine) for case in cases]
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "cases": [
                            {
                                "id": case["id"],
                                "category": case["category"],
                                "domain": case["domain"],
                                "difficulty": case["difficulty"],
                                "grader_type": case["grader"]["type"],
                            }
                            for case in resolved_cases
                        ],
                        "case_bank": pack_metadata,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        loaded_seal = loaded_pack.metadata.get("seal")
        if isinstance(loaded_seal, dict) and loaded_seal.get("verified") is True:
            raise ValueError(
                "authenticated private packs are evaluator-only; run contenders "
                "from the oracle-free public projection with "
                "--blind-submission-out, then grade separately with cortheon-grade"
            )
        contenders = _contenders(args)
        if not contenders:
            raise ValueError("configure at least one contender endpoint")
        if any(contender.kind == "cli" for contender in contenders) and os.environ.get(
            args.runner_key_env
        ):
            raise ValueError(
                "refusing to launch a process-local CLI contender while the "
                "runner attestation key is resident; use a clean local shell"
            )
        report = run_benchmark(
            contenders,
            resolved_cases,
            repetitions=args.repetitions,
            seed=args.seed,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            include_answers=args.include_answers,
            case_bank=pack_metadata,
            secret_env_names=(
                args.case_pack_key_env,
                args.runner_key_env,
            ),
        )
        promotion_exit = 0
        if args.promotion_baseline:
            baseline = json.loads(args.promotion_baseline.expanduser().read_text(encoding="utf-8"))
            if not isinstance(baseline, dict):
                raise ValueError("promotion baseline must be a JSON object")
            report["promotion_gate"] = evaluate_promotion(
                baseline,
                report,
                candidate_name=args.promotion_candidate,
                min_improvement=args.promotion_min_improvement,
                max_domain_regression=args.promotion_max_domain_regression,
                max_latency_ratio=args.promotion_max_latency_ratio,
                max_cost_ratio=args.promotion_max_cost_ratio,
                require_external_holdout=True,
            )
            if args.require_promotion and not report["promotion_gate"]["passed"]:
                promotion_exit = 3
        elif args.require_promotion:
            raise ValueError("--require-promotion needs --promotion-baseline")
        parity_exit = 0
        if args.parity_contract:
            contract, contract_sha256 = load_parity_contract(args.parity_contract)
            report["frontier_parity_gate"] = evaluate_frontier_parity(
                report,
                contract,
                contract_sha256=contract_sha256,
            )
            if args.require_parity and not report["frontier_parity_gate"]["passed"]:
                parity_exit = 4
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.json_out:
            destination = args.json_out.expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered + "\n", encoding="utf-8")
            print(f"cortheon-bench: saved {destination}", file=sys.stderr)
        return max(promotion_exit, parity_exit)
    except (json.JSONDecodeError, OSError, ValueError, urllib.error.URLError) as exc:
        print(f"cortheon-bench: {exc}", file=sys.stderr)
        return 2


def _run_blind_submission_command(args: argparse.Namespace) -> int:
    if args.cases is None:
        raise ValueError("--blind-submission-out requires --cases")
    if args.case_split != "all" or args.only or args.rotation_index or args.rotation_size:
        raise ValueError("blind submissions must run the complete evaluator-selected public pack")
    if args.json_out is not None or args.promotion_baseline is not None:
        raise ValueError("blind submissions cannot be combined with local reports or promotion")
    cases, case_bank = _load_public_case_pack(args.cases)
    contenders = _contenders(args)
    if not contenders:
        raise ValueError("configure at least one contender endpoint")
    if any(contender.kind == "cli" for contender in contenders):
        raise ValueError(
            "certified blind runs forbid process-local CLI contenders; "
            "use preregistered remote HTTPS provider endpoints"
        )
    artifact = run_blind_submissions(
        contenders,
        cases,
        repetitions=args.repetitions,
        seed=args.seed,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        case_bank=case_bank,
        secret_env_names=(args.case_pack_key_env, args.runner_key_env),
    )
    artifact = attest_blind_submission(
        artifact,
        key_env=args.runner_key_env,
    )
    destination = args.blind_submission_out.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"refusing to overwrite submission: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    destination.chmod(0o600)
    print(
        json.dumps(
            {
                "ok": True,
                "submission": str(destination),
                "submission_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "pack_id": case_bank.get("pack_id"),
                "public_tasks_sha256": case_bank.get("public_tasks_sha256"),
                "cases": len(cases),
                "contenders": len(contenders),
                "repetitions": args.repetitions,
                "answers_withheld_from_stdout": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _contenders(args: argparse.Namespace) -> list[Contender]:
    pricing = _parse_pricing(args.pricing)
    contenders: list[Contender] = []
    if args.stock_url and args.stock_model:
        input_cost, output_cost = pricing.get("stock", (None, None))
        contenders.append(
            Contender(
                "stock",
                "stock",
                args.stock_url,
                args.stock_model,
                args.stock_key,
                input_cost_per_million=input_cost,
                output_cost_per_million=output_cost,
            )
        )
    if args.cortheon_url and args.cortheon_model:
        input_cost, output_cost = pricing.get("cortheon", (None, None))
        contenders.append(
            Contender(
                "cortheon",
                "cortheon",
                args.cortheon_url,
                args.cortheon_model,
                args.cortheon_key,
                input_cost_per_million=input_cost,
                output_cost_per_million=output_cost,
                compute_cost_per_hour=args.cortheon_compute_usd_per_hour,
                runtime_sha256=args.cortheon_runtime_sha256 or None,
                family=args.cortheon_family.strip() or None,
            )
        )
    if bool(args.frontier_url) != bool(args.frontier_model):
        raise ValueError("--frontier-url and --frontier-model must be set together")
    if args.frontier_url:
        input_cost, output_cost = pricing.get("frontier", (None, None))
        raw_tools = tuple(
            value.strip() for value in args.frontier_tools.split(",") if value.strip()
        )
        if set(raw_tools) - {"web_search", "code_interpreter"}:
            raise ValueError("--frontier-tools only accepts web_search and code_interpreter")
        contenders.append(
            Contender(
                "frontier",
                "frontier",
                args.frontier_url,
                args.frontier_model,
                args.frontier_key,
                raw_tools,
                input_cost_per_million=input_cost,
                output_cost_per_million=output_cost,
            )
        )
    cli_specs = list(args.cli_contender)
    if args.claude_cli:
        cli_specs.append(
            "claude_cli=claude -p --safe-mode --no-session-persistence "
            "--strict-mcp-config --tools WebSearch,WebFetch "
            f"--output-format json --max-budget-usd {args.claude_max_budget_usd:g}"
        )
    if args.kimi_cli:
        cli_specs.append("kimi_cli=kimi -p {prompt}")
    for spec in cli_specs:
        name, command = _parse_cli_spec(spec)
        input_cost, output_cost = pricing.get(name, (None, None))
        contenders.append(
            Contender(
                name=name,
                kind="cli",
                base_url="local-cli",
                model=Path(command[0]).name,
                api_key="",
                command=command,
                input_cost_per_million=input_cost,
                output_cost_per_million=output_cost,
            )
        )
    names = [contender.name for contender in contenders]
    if len(set(names)) != len(names):
        raise ValueError("contender names must be unique")
    unknown_prices = set(pricing) - set(names)
    if unknown_prices:
        raise ValueError(
            f"pricing configured for unknown contender(s): {', '.join(sorted(unknown_prices))}"
        )
    return contenders


def _parse_cli_spec(spec: str) -> tuple[str, tuple[str, ...]]:
    name, separator, raw_command = spec.partition("=")
    name = name.strip()
    if not separator or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", name):
        raise ValueError("--cli-contender must use NAME=COMMAND with a safe, unique name")
    try:
        command = tuple(shlex.split(raw_command))
    except ValueError as exc:
        raise ValueError(f"invalid CLI command for {name}: {exc}") from exc
    if (
        not command
        or len(command) > 64
        or any("\x00" in value for value in command)
        or sum(value.count("{prompt}") for value in command) > 1
        or any("{prompt}" in value and value != "{prompt}" for value in command)
    ):
        raise ValueError(f"invalid CLI command for {name}")
    if sum(len(value) for value in command) > 16_384:
        raise ValueError(f"CLI command for {name} is too long")
    return name, command


def _parse_pricing(
    specifications: list[str],
) -> dict[str, tuple[float | None, float | None]]:
    pricing: dict[str, tuple[float | None, float | None]] = {}
    for specification in specifications:
        name, separator, raw_prices = specification.partition("=")
        values = raw_prices.split(",")
        if (
            not separator
            or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", name)
            or len(values) != 2
        ):
            raise ValueError("--pricing must use NAME=INPUT,OUTPUT")
        try:
            parsed = (float(values[0]), float(values[1]))
        except ValueError as exc:
            raise ValueError("--pricing values must be numbers") from exc
        if not all(math.isfinite(value) and value >= 0 for value in parsed):
            raise ValueError("--pricing values must be finite and non-negative")
        if name in pricing:
            raise ValueError(f"duplicate pricing for contender {name}")
        pricing[name] = parsed
    return pricing

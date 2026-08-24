"""The ``cortheon-pack`` command line: seal, verify, contract.

Every subcommand prints one canonical JSON object and exits 0 only when that
object reports ``ok``. Expected failures -- a missing file, a rejected
argument, malformed JSON -- are reported on stderr with exit 2 rather than a
traceback, because this tool is run by evaluators, not by its authors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cortheon.parity import SUPPORTED_CANDIDATE_HOSTS
from cortheon.parity_pack_core.contract import write_release_contract
from cortheon.parity_pack_core.seal import seal_case_pack
from cortheon.parity_pack_core.verify import verify_case_pack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cortheon-pack",
        description=(
            "Create evaluator-owned, authenticated held-out task packs for frontier-parity runs."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _add_seal_command(commands)

    verify = commands.add_parser("verify")
    verify.add_argument("--input", type=Path, required=True)
    verify.add_argument("--key-env", default="CORTHEON_BENCH_PACK_KEY")

    _add_contract_command(commands)
    return parser


def _add_seal_command(commands: Any) -> None:
    seal = commands.add_parser("seal")
    seal.add_argument("--input", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--public-output", type=Path, required=True)
    seal.add_argument("--contract", type=Path, required=True)
    seal.add_argument("--pack-id", required=True)
    seal.add_argument("--issuer", required=True)
    seal.add_argument(
        "--evaluator",
        help="Distinct evaluator identity; defaults to the pack issuer.",
    )
    seal.add_argument("--runner-id", required=True)
    seal.add_argument("--author", action="append", required=True)
    seal.add_argument("--key-env", default="CORTHEON_BENCH_PACK_KEY")
    seal.add_argument(
        "--runner-key-env",
        default="CORTHEON_RUNNER_ATTESTATION_KEY",
    )
    seal.add_argument("--seed", type=int, default=7)
    seal.add_argument("--holdout-fraction", type=float, default=0.3)
    seal.add_argument("--rotation-index", type=int, default=0)
    seal.add_argument("--rotation-size", type=int, default=0)
    seal.add_argument("--expires-at", required=True)
    seal.add_argument("--force", action="store_true")


def _add_contract_command(commands: Any) -> None:
    contract = commands.add_parser("contract")
    contract.add_argument("--output", type=Path, required=True)
    contract.add_argument("--candidate", default="cortheon")
    contract.add_argument("--candidate-model", required=True)
    contract.add_argument("--candidate-family", required=True)
    contract.add_argument(
        "--candidate-host",
        required=True,
        choices=sorted(SUPPORTED_CANDIDATE_HOSTS),
    )
    contract.add_argument(
        "--candidate-compute-usd-per-hour",
        type=float,
        required=True,
    )
    contract.add_argument("--candidate-runtime-sha256", required=True)
    contract.add_argument("--frontier", action="append", required=True)
    contract.add_argument(
        "--endpoint",
        action="append",
        required=True,
        metavar="NAME=BASE_URL",
    )
    contract.add_argument(
        "--pricing",
        action="append",
        required=True,
        metavar="NAME=INPUT,OUTPUT",
    )
    contract.add_argument("--domain", action="append", required=True)
    contract.add_argument("--maintainer", action="append", required=True)
    contract.add_argument("--last-tuning-at", required=True)


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "seal":
        return seal_case_pack(
            args.input,
            args.output,
            public_output_path=args.public_output,
            contract_path=args.contract,
            pack_id=args.pack_id,
            issuer=args.issuer,
            evaluator=args.evaluator,
            runner_id=args.runner_id,
            authors=args.author,
            key_env=args.key_env,
            runner_key_env=args.runner_key_env,
            seed=args.seed,
            holdout_fraction=args.holdout_fraction,
            rotation_index=args.rotation_index,
            rotation_size=args.rotation_size,
            expires_at=args.expires_at,
            overwrite=args.force,
        )
    if args.command == "verify":
        return verify_case_pack(args.input, key_env=args.key_env)
    return write_release_contract(
        args.output,
        candidate=args.candidate,
        candidate_model=args.candidate_model,
        candidate_family=args.candidate_family,
        candidate_host=args.candidate_host,
        candidate_compute_usd_per_hour=(args.candidate_compute_usd_per_hour),
        candidate_runtime_sha256=args.candidate_runtime_sha256,
        frontiers=args.frontier,
        endpoints=args.endpoint,
        pricing=args.pricing,
        domains=args.domain,
        maintainers=args.maintainer,
        last_tuning_at=args.last_tuning_at,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        result = _dispatch(build_parser().parse_args(argv))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok") is True else 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cortheon-pack: {exc}", file=sys.stderr)
        return 2

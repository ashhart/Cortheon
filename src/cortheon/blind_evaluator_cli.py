"""Command-line boundary for independent blind evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cortheon-grade",
        description="Grade a blind submission against an authenticated private pack.",
    )
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-env", default="CORTHEON_BENCH_PACK_KEY")
    parser.add_argument("--runner-key-env", default="CORTHEON_RUNNER_ATTESTATION_KEY")
    parser.add_argument("--require-parity", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    from cortheon.blind_evaluator import grade_blind_submission

    try:
        args = build_parser().parse_args(argv)
        report = grade_blind_submission(
            args.pack,
            args.submission,
            contract_path=args.contract,
            key_env=args.key_env,
            runner_key_env=args.runner_key_env,
        )
        destination = args.output.expanduser().resolve()
        if destination.exists():
            raise ValueError(f"refusing to overwrite report: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        destination.chmod(0o600)
        gate = report["frontier_parity_gate"]
        print(
            json.dumps(
                {
                    "ok": True,
                    "report": str(destination),
                    "report_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "parity_passed": gate["passed"],
                    "failure_reasons": gate["failure_reasons"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 4 if args.require_parity and not gate["passed"] else 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cortheon-grade: {exc}", file=sys.stderr)
        return 2

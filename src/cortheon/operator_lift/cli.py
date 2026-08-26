"""Repository-only executable operator-lift development instrument."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from cortheon.operator_lift.case_bank import development_cases
from cortheon.operator_lift.execution_models import ExecutionConfig
from cortheon.operator_lift.execution_release import (
    build_release,
    release_records,
    verify_release,
)
from cortheon.operator_lift.execution_report import content_free_report
from cortheon.operator_lift.execution_runner import run_cell
from cortheon.operator_lift.execution_schedule import (
    canonical_bytes,
    execution_manifest,
    full_schedule,
    public_pack,
    run_descriptor,
    selected_schedule,
)
from cortheon.operator_lift.execution_storage import (
    discard_private_state,
    freeze_submissions,
    initialize_run,
    load_checkpoint,
    save_checkpoint,
    save_release,
    save_report,
    validate_retained_artifacts,
)
from cortheon.operator_lift.models import OPERATORS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cortheon-operator-lift")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--base-url", required=True)
    run.add_argument("--provider", required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--api-key-env", default="OMLX_API_KEY")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--pilot-clusters", type=int)
    run.add_argument("--operator", choices=OPERATORS)
    run.add_argument("--timeout-seconds", type=float, default=120.0)
    run.add_argument("--context-tokens", type=int, default=16_384)
    run.add_argument("--output-tokens", type=int, default=2_048)
    run.add_argument("--max-steps", type=int, default=8)
    run.add_argument("--max-tool-calls", type=int, default=12)
    run.add_argument(
        "--heldout",
        action="store_true",
        help="run the sealed held-out P6 pack instead of the development bank",
    )
    verify = subparsers.add_parser("verify-release")
    verify.add_argument("--release", type=Path, required=True)
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--expected-chain-root", required=True)
    return parser


def _config(args: argparse.Namespace) -> ExecutionConfig:
    if not args.api_key_env or args.api_key_env not in os.environ:
        raise ValueError("configured API key environment variable is absent")
    return ExecutionConfig(
        args.base_url,
        args.provider,
        args.model_id,
        os.environ[args.api_key_env],
        args.timeout_seconds,
        args.context_tokens,
        args.output_tokens,
        args.max_steps,
        args.max_tool_calls,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    config = _config(args)
    if args.heldout:
        from cortheon.operator_lift.heldout import heldout_cases

        cases = heldout_cases()
    else:
        cases = development_cases()
    case_by_id = {case.case_id: case for case in cases}
    manifest = execution_manifest(cases)
    pack = public_pack(cases)
    schedule = selected_schedule(
        full_schedule(manifest, cases),
        cases,
        args.pilot_clusters,
        args.operator,
    )
    descriptor = run_descriptor(manifest, cases, schedule, config, str(pack["pack_sha256"]))
    initialize_run(args.output_dir, descriptor, pack)
    records: list[dict[str, Any]] = []
    for cell in schedule:
        existing = load_checkpoint(args.output_dir, cell.cell_id, str(descriptor["run_sha256"]))
        resumed = existing is not None
        if existing is None:
            outcome = run_cell(config, manifest, case_by_id[cell.case_id], cell)
            existing = save_checkpoint(
                args.output_dir,
                cell.cell_id,
                str(descriptor["run_sha256"]),
                {"submission": outcome.submission, "summary": outcome.summary},
            )
        records.append(existing)
        print(
            json.dumps(
                {
                    "cell": cell.sequence,
                    "total": len(schedule),
                    "cell_id": cell.cell_id,
                    "resumed": resumed,
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
    submissions, _private_freeze_sha256 = freeze_submissions(
        args.output_dir, str(descriptor["run_sha256"]), records
    )
    summaries = [record["summary"] for record in records]
    projected = release_records(
        manifest,
        cases,
        schedule,
        submissions,
        summaries,
        str(descriptor["run_sha256"]),
    )
    chain_root = projected[-1]["record_sha256"]
    report = content_free_report(
        manifest,
        cases,
        submissions,
        summaries,
        run_sha256=str(descriptor["run_sha256"]),
        event_chain_sha256=str(chain_root),
        planned_cells=len(schedule),
    )
    release = build_release(
        manifest,
        cases,
        schedule,
        submissions,
        summaries,
        report,
        str(descriptor["run_sha256"]),
    )
    verify_release(release, report, descriptor, str(release["chain_root_sha256"]))
    save_report(args.output_dir, report)
    save_release(args.output_dir, release)
    discard_private_state(args.output_dir)
    validate_retained_artifacts(args.output_dir)
    return report


def verify(args: argparse.Namespace) -> dict[str, object]:
    release = json.loads(args.release.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    descriptor = json.loads(args.run.read_text(encoding="utf-8"))
    if (
        not isinstance(release, dict)
        or not isinstance(report, dict)
        or not isinstance(descriptor, dict)
    ):
        raise ValueError("release, report, and run descriptor must be objects")
    return verify_release(release, report, descriptor, args.expected_chain_root)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args) if args.command == "run" else verify(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__}), file=sys.stderr)
        return 2
    print(canonical_bytes(report).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

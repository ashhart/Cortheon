"""``cortheon-qualify`` command line: validate, execute, and enforce."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cortheon.qualification_core.conditions import REQUIRED_CONDITIONS, condition_record
from cortheon.qualification_core.constants import REPORT_SCHEMA_VERSION
from cortheon.qualification_core.digests import _cell_public_config
from cortheon.qualification_core.manifest import load_manifest
from cortheon.qualification_core.models import QualificationError
from cortheon.qualification_core.report import run_qualification


def _example_manifest() -> dict[str, Any]:
    return {
        "schema_version": 3,
        "tier": "pr",
        "repository": ".",
        "seed": 20_260_726,
        "cells": [
            {
                "id": "local-semantic",
                "suite": "semantic",
                "host": "opencode",
                "provider": "Local",
                "base_url": "http://127.0.0.1:18081/v1",
                "model_id": "qwen3-1.7b",
                "runtime_url": "http://127.0.0.1:8743",
                "cases": 2,
                "repeats": 1,
                "max_steps": 4,
                "conditions": [
                    {
                        "id": condition_id,
                        "config_sha256": condition_record(condition_id)["config_sha256"],
                        "implementation_sha256": condition_record(condition_id)[
                            "implementation_sha256"
                        ],
                    }
                    for condition_id in REQUIRED_CONDITIONS
                ],
            }
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cortheon-qualify",
        description=(
            "Run a sealed Cortheon qualification matrix and emit a content-free "
            "machine-verifiable report."
        ),
    )
    parser.add_argument("manifest", nargs="?", type=Path)
    parser.add_argument(
        "--example",
        action="store_true",
        help="Print a minimal manifest to stdout and exit.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and normalize the manifest without executing jobs.",
    )
    parser.add_argument("--cell", help="Run one matrix cell (reproducer mode).")
    parser.add_argument("--case-id", help="Run one sealed case (reproducer mode).")
    parser.add_argument("--repeat", type=int, help="Run one configured repeat.")
    parser.add_argument(
        "--enforce",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero when the complete manifest is not promoted.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit content-free JSON progress records on stderr.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.example:
        print(json.dumps(_example_manifest(), indent=2, sort_keys=True))
        return 0
    if args.manifest is None:
        parser.error("manifest is required unless --example is used")
    try:
        manifest = load_manifest(args.manifest)
        if args.validate_only:
            print(
                json.dumps(
                    {
                        "schema_version": REPORT_SCHEMA_VERSION,
                        "kind": "cortheon_qualification_validation",
                        "valid": True,
                        "content_free": True,
                        "manifest": {
                            "digest_sha256": manifest.digest,
                            "file": manifest.path.name,
                            "tier": manifest.tier,
                            "seed": manifest.seed,
                            "cells": [_cell_public_config(cell) for cell in manifest.cells],
                        },
                        "policy": manifest.gates,
                        "expanded_jobs": sum(
                            cell.cases * cell.repeats * len(cell.condition_ids)
                            for cell in manifest.cells
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        report = run_qualification(
            manifest,
            cell_filter=args.cell,
            case_filter=args.case_id,
            repeat_filter=args.repeat,
            progress=args.progress,
        )
    except QualificationError as exc:
        print(f"cortheon-qualify: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.enforce or not report["selection"]["full_manifest"]:
        return 0
    return 0 if report["promoted"] else 1

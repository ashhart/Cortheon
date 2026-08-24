"""Command-line entry point for the replication-campaign gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cortheon.parity_campaign.errors import CampaignContractError
from cortheon.parity_campaign.evaluate import evaluate_replication_campaign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cortheon-campaign",
        description=(
            "Regrade a preregistered cross-report replication campaign from "
            "original evidence: sealed packs, attested submissions, and inner "
            "parity contracts."
        ),
    )
    parser.add_argument(
        "--registration",
        type=Path,
        required=True,
        help="Immutable pre-run campaign registration (the preregistered matrix).",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Post-run results mapping every preregistered cell to its artifacts.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        decision = evaluate_replication_campaign(args.registration, args.results)
        rendered = json.dumps(decision, indent=2, sort_keys=True)
        if args.output is not None:
            destination = args.output.expanduser().resolve()
            if destination.exists():
                raise ValueError(f"refusing to overwrite decision: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError, CampaignContractError) as exc:
        print(f"cortheon-campaign: {exc}", file=sys.stderr)
        return 2
    print(rendered)
    return 0 if decision["passed"] else 1

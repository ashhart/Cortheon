from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cortheon-bench",
        description=(
            "Run blinded, repeated comparisons of a stock model, Cortheon, and "
            "a tool-using frontier model."
        ),
    )
    parser.add_argument(
        "--stock-url",
        default=os.environ.get("CORTHEON_BENCH_STOCK_URL", ""),
        help="Stock OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--stock-model",
        default=os.environ.get("CORTHEON_BENCH_STOCK_MODEL", ""),
    )
    parser.add_argument(
        "--stock-key",
        default=os.environ.get("CORTHEON_BENCH_STOCK_KEY", ""),
    )
    parser.add_argument(
        "--cortheon-url",
        default=os.environ.get(
            "CORTHEON_BENCH_URL",
            "http://127.0.0.1:8899",
        ),
        help="Cortheon gateway URL.",
    )
    parser.add_argument(
        "--cortheon-model",
        default=os.environ.get("CORTHEON_BENCH_MODEL", "cortheon"),
    )
    parser.add_argument(
        "--cortheon-family",
        default=os.environ.get("CORTHEON_BENCH_FAMILY", ""),
        help=(
            "Exact preregistered candidate model family for parity runs; "
            "binds the candidate identity instead of inferring a family "
            "from the model identifier."
        ),
    )
    parser.add_argument(
        "--cortheon-key",
        default=os.environ.get(
            "CORTHEON_BENCH_KEY",
            os.environ.get("CORTHEON_API_KEY", ""),
        ),
    )
    parser.add_argument(
        "--cortheon-compute-usd-per-hour",
        type=float,
        default=None,
        help=(
            "Evaluator-owned all-in compute rate used to meter Cortheon from runner wall-clock time."
        ),
    )
    parser.add_argument(
        "--cortheon-runtime-sha256",
        default=os.environ.get("CORTHEON_BENCH_RUNTIME_SHA256", ""),
        help=("Evaluator-measured SHA-256 of the deployed Cortheon runtime artifact."),
    )
    parser.add_argument(
        "--frontier-url",
        default=os.environ.get("CORTHEON_BENCH_FRONTIER_URL", ""),
        help="Frontier Responses API base URL.",
    )
    parser.add_argument(
        "--frontier-model",
        default=os.environ.get("CORTHEON_BENCH_FRONTIER_MODEL", ""),
    )
    parser.add_argument(
        "--frontier-key",
        default=os.environ.get("CORTHEON_BENCH_FRONTIER_KEY", ""),
    )
    parser.add_argument(
        "--frontier-tools",
        default=os.environ.get(
            "CORTHEON_BENCH_FRONTIER_TOOLS",
            "web_search,code_interpreter",
        ),
        help="Native frontier tools: web_search,code_interpreter.",
    )
    parser.add_argument(
        "--claude-cli",
        action="store_true",
        help=(
            "Add the locally installed `claude -p` as a blinded frontier "
            "contender. The prompt is sent on stdin and no shell is used."
        ),
    )
    parser.add_argument(
        "--claude-max-budget-usd",
        type=float,
        default=float(os.environ.get("CORTHEON_BENCH_CLAUDE_MAX_BUDGET_USD", "1.0")),
        help="Per-case Claude CLI spend ceiling (default: USD 1.00).",
    )
    parser.add_argument(
        "--kimi-cli",
        action="store_true",
        help=(
            "Add the locally installed `kimi -p` as a blinded frontier "
            "contender, with the prompt passed as one argv value."
        ),
    )
    parser.add_argument(
        "--cli-contender",
        action="append",
        default=[],
        metavar="NAME=COMMAND",
        help=(
            "Add a shell-free CLI contender. May be repeated. COMMAND is split "
            "with shlex. Use one {prompt} argv placeholder or receive the prompt "
            "on stdin. The process starts in an empty temporary directory."
        ),
    )
    parser.add_argument(
        "--pricing",
        action="append",
        default=[],
        metavar="NAME=INPUT,OUTPUT",
        help=(
            "Optional USD pricing per million input/output tokens for a contender. "
            "May be repeated; e.g. frontier=2.5,10."
        ),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        help="Optional JSON case bank; defaults to Cortheon's built-in bank.",
    )
    parser.add_argument(
        "--case-pack-key-env",
        default="CORTHEON_BENCH_PACK_KEY",
        help=(
            "Environment variable containing the external case-pack HMAC key. "
            "Required for authenticated parity runs."
        ),
    )
    parser.add_argument(
        "--runner-key-env",
        default="CORTHEON_RUNNER_ATTESTATION_KEY",
        help=(
            "Environment variable containing the independent runner's submission-attestation key."
        ),
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--case-split",
        choices=("all", "development", "heldout"),
        default="all",
        help="Deterministic case-bank split selected before live grader resolution.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.3,
        help="Fraction assigned to the reproducible held-out split.",
    )
    parser.add_argument(
        "--rotation-index",
        type=int,
        default=0,
        help="Zero-based rotating window index after split selection.",
    )
    parser.add_argument(
        "--rotation-size",
        type=int,
        default=0,
        help="Cases per rotating window; zero selects the complete split.",
    )
    parser.add_argument("--timeout", type=float, default=400.0)
    parser.add_argument("--max-tokens", type=int, default=1_200)
    parser.add_argument(
        "--only",
        help="Comma-separated case ids.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Write the complete reproducible run artifact.",
    )
    parser.add_argument(
        "--blind-submission-out",
        type=Path,
        help=(
            "Run a public, oracle-free task pack and write answers for grading "
            "on an independent evaluator host."
        ),
    )
    parser.add_argument(
        "--include-answers",
        action="store_true",
        help="Include full model answers in the report instead of hashes/previews.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate live graders and selection hashes but make no model calls.",
    )
    parser.add_argument(
        "--promotion-baseline",
        type=Path,
        help="Prior cortheon-bench JSON report to compare with the current candidate.",
    )
    parser.add_argument(
        "--promotion-candidate",
        default="cortheon",
        help="Unblinded contender name evaluated by the promotion gate.",
    )
    parser.add_argument(
        "--promotion-min-improvement",
        type=float,
        default=0.0,
        help="Required strict verified-completion-rate improvement over baseline.",
    )
    parser.add_argument(
        "--promotion-max-domain-regression",
        type=float,
        default=0.02,
        help="Largest allowed completion-rate regression in any domain.",
    )
    parser.add_argument(
        "--promotion-max-latency-ratio",
        type=float,
        default=1.25,
        help="Largest allowed p95 latency ratio versus baseline.",
    )
    parser.add_argument(
        "--promotion-max-cost-ratio",
        type=float,
        default=1.25,
        help="Largest allowed mean-cost ratio versus baseline.",
    )
    parser.add_argument(
        "--require-promotion",
        action="store_true",
        help="Exit non-zero unless the machine-checkable promotion gate passes.",
    )
    parser.add_argument(
        "--parity-contract",
        type=Path,
        help=(
            "Pre-registered JSON contract for broad, multi-frontier parity. "
            "The authenticated case pack must bind this contract digest."
        ),
    )
    parser.add_argument(
        "--require-parity",
        action="store_true",
        help="Exit non-zero unless every frontier-parity contract check passes.",
    )
    return parser

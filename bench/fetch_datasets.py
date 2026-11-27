"""Download official pinned HuggingFace benchmark datasets into `bench/datasets/`.

Zero extra dependencies beyond `httpx` (already required by Cortheon): uses the
HuggingFace Datasets Server REST API for `MathArena/aime_2025` and
`bigcode/bigcodebench-hard` (`v0.1.4`), and streams `test6.jsonl` directly for
`livecodebench/code_generation_lite` (whose legacy loader script is disabled on
Datasets Server).

Usage:
    python -m bench.fetch_datasets --dataset all
    python -m bench.fetch_datasets --dataset aime2025
    python -m bench.fetch_datasets --dataset bigcodebench
    python -m bench.fetch_datasets --dataset livecodebench
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx

HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
LCB_TEST6_URL = (
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl"
)

# Exact 30 macOS-verified tasks from bigcode/bigcodebench-hard v0.1.4 evaluated in BENCHMARKS.md
BIGCODEBENCH_HARD_VERIFIED_IDS: list[str] = [
    "BigCodeBench/13",
    "BigCodeBench/15",
    "BigCodeBench/19",
    "BigCodeBench/147",
    "BigCodeBench/310",
    "BigCodeBench/324",
    "BigCodeBench/326",
    "BigCodeBench/346",
    "BigCodeBench/368",
    "BigCodeBench/454",
    "BigCodeBench/592",
    "BigCodeBench/594",
    "BigCodeBench/720",
    "BigCodeBench/763",
    "BigCodeBench/765",
    "BigCodeBench/771",
    "BigCodeBench/777",
    "BigCodeBench/785",
    "BigCodeBench/800",
    "BigCodeBench/854",
    "BigCodeBench/857",
    "BigCodeBench/906",
    "BigCodeBench/928",
    "BigCodeBench/963",
    "BigCodeBench/988",
    "BigCodeBench/990",
    "BigCodeBench/998",
    "BigCodeBench/999",
    "BigCodeBench/1039",
    "BigCodeBench/1042",
]


def _make_headers() -> dict[str, str]:
    headers = {"User-Agent": "cortheon-bench-fetcher/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"
    return headers


def _get_json_with_retry(
    client: httpx.Client,
    url: str,
    params: dict[str, Any],
    *,
    retries: int = 4,
    backoff_s: float = 1.5,
) -> dict[str, Any]:
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = client.get(url, params=params)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                time.sleep(backoff_s * (2**attempt))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(backoff_s * (2**attempt))
    raise RuntimeError(f"Failed GET {url} params={params}: {last_exc}")


def fetch_hf_rows(
    client: httpx.Client,
    dataset: str,
    *,
    config: str = "default",
    split: str = "train",
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """Fetch all rows of a HuggingFace dataset split via Datasets Server REST API."""
    rows: list[dict[str, Any]] = []
    offset = 0
    total: Optional[int] = None
    while total is None or offset < total:
        data = _get_json_with_retry(
            client,
            HF_ROWS_API,
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page_size,
            },
        )
        batch = data.get("rows", [])
        if total is None:
            total = int(data.get("num_rows_total", len(batch)))
        if not batch:
            break
        for entry in batch:
            rows.append(entry["row"])
        offset += len(batch)
    return rows


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def fetch_aime2025(
    out_dir: Path,
    *,
    client: Optional[httpx.Client] = None,
    force: bool = False,
) -> Path:
    """Download MathArena/aime_2025 (all 30 official problems) -> aime_2025.json."""
    dest = out_dir / "aime_2025.json"
    if dest.exists() and not force:
        existing = json.loads(dest.read_text(encoding="utf-8"))
        if isinstance(existing, list) and len(existing) == 30:
            print(f"[aime2025] already present ({len(existing)} problems): {dest}")
            return dest

    own_client = client is None
    if client is None:
        client = httpx.Client(headers=_make_headers(), timeout=60.0, follow_redirects=True)
    try:
        rows = fetch_hf_rows(client, "MathArena/aime_2025", config="default", split="train")
    finally:
        if own_client:
            client.close()

    rows.sort(key=lambda r: int(r["problem_idx"]))
    if len(rows) != 30:
        raise RuntimeError(f"Expected 30 rows from MathArena/aime_2025, got {len(rows)}")
    for r in rows:
        ans = int(r["answer"])
        if not (0 <= ans <= 999):
            raise RuntimeError(f"Unexpected AIME answer {ans!r} for problem_idx={r.get('problem_idx')}")

    _write_json_atomic(dest, rows)
    print(f"[aime2025] saved {len(rows)} problems -> {dest}")
    return dest


def fetch_bigcodebench(
    out_dir: Path,
    *,
    client: Optional[httpx.Client] = None,
    all_tasks: bool = False,
    force: bool = False,
) -> Path:
    """Download bigcode/bigcodebench-hard v0.1.4 -> bigcodebench_hard_verified.json."""
    dest = out_dir / "bigcodebench_hard_verified.json"
    expected_n = 148 if all_tasks else len(BIGCODEBENCH_HARD_VERIFIED_IDS)
    if dest.exists() and not force:
        existing = json.loads(dest.read_text(encoding="utf-8"))
        if isinstance(existing, list) and len(existing) == expected_n:
            print(f"[bigcodebench] already present ({len(existing)} tasks): {dest}")
            return dest

    own_client = client is None
    if client is None:
        client = httpx.Client(headers=_make_headers(), timeout=60.0, follow_redirects=True)
    try:
        rows = fetch_hf_rows(client, "bigcode/bigcodebench-hard", config="default", split="v0.1.4")
    finally:
        if own_client:
            client.close()

    if not all_tasks:
        by_id = {r["task_id"]: r for r in rows}
        missing = [tid for tid in BIGCODEBENCH_HARD_VERIFIED_IDS if tid not in by_id]
        if missing:
            raise RuntimeError(f"Missing pinned BigCodeBench-Hard IDs: {missing[:5]}")
        selected = [by_id[tid] for tid in BIGCODEBENCH_HARD_VERIFIED_IDS]
    else:
        selected = rows

    _write_json_atomic(dest, selected)
    print(f"[bigcodebench] saved {len(selected)} tasks (v0.1.4) -> {dest}")
    return dest


def fetch_livecodebench(
    out_dir: Path,
    *,
    client: Optional[httpx.Client] = None,
    platform: Optional[str] = "atcoder",
    difficulty: str = "hard",
    limit: int = 0,
    force: bool = False,
) -> Path:
    """Stream livecodebench/code_generation_lite test6.jsonl -> livecodebench_v6_hard.json."""
    dest = out_dir / "livecodebench_v6_hard.json"
    if dest.exists() and not force:
        existing = json.loads(dest.read_text(encoding="utf-8"))
        if isinstance(existing, list) and len(existing) >= (limit or 20):
            print(f"[livecodebench] already present ({len(existing)} problems): {dest}")
            return dest

    own_client = client is None
    if client is None:
        client = httpx.Client(headers=_make_headers(), timeout=120.0, follow_redirects=True)

    selected: list[dict[str, Any]] = []
    try:
        with client.stream("GET", LCB_TEST6_URL) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                if difficulty and str(obj.get("difficulty", "")).lower() != difficulty.lower():
                    continue
                if platform and str(obj.get("platform", "")).lower() != platform.lower():
                    continue
                selected.append(obj)
                if limit > 0 and len(selected) >= limit:
                    break
    finally:
        if own_client:
            client.close()

    if not selected:
        raise RuntimeError(
            f"No rows matched platform={platform!r}, difficulty={difficulty!r} in LiveCodeBench test6.jsonl"
        )

    _write_json_atomic(dest, selected)
    print(f"[livecodebench] saved {len(selected)} problems (v6 {difficulty}) -> {dest}")
    return dest


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch official pinned HuggingFace datasets into bench/datasets/."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "aime2025", "bigcodebench", "livecodebench"],
        default="all",
        help="Which official dataset to download (default: all).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("bench/datasets"),
        help="Target directory for downloaded dataset JSON files (default: bench/datasets).",
    )
    parser.add_argument(
        "--all-bcb",
        action="store_true",
        help="Save all 148 BigCodeBench-Hard v0.1.4 tasks instead of the 30 macOS-verified benchmark subset.",
    )
    parser.add_argument(
        "--all-lcb-platforms",
        action="store_true",
        help="Include all platforms from LiveCodeBench v6 hard (default: atcoder stdin/stdout split).",
    )
    parser.add_argument(
        "--lcb-limit",
        type=int,
        default=0,
        help="Optional cap on LiveCodeBench problems to stream (0 = all matching).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dataset files even if already present.",
    )
    args = parser.parse_args(argv)

    targets = (
        ["aime2025", "bigcodebench", "livecodebench"] if args.dataset == "all" else [args.dataset]
    )
    with httpx.Client(headers=_make_headers(), timeout=120.0, follow_redirects=True) as client:
        for name in targets:
            if name == "aime2025":
                fetch_aime2025(args.out_dir, client=client, force=args.force)
            elif name == "bigcodebench":
                fetch_bigcodebench(
                    args.out_dir,
                    client=client,
                    all_tasks=args.all_bcb,
                    force=args.force,
                )
            elif name == "livecodebench":
                fetch_livecodebench(
                    args.out_dir,
                    client=client,
                    platform=None if args.all_lcb_platforms else "atcoder",
                    limit=args.lcb_limit,
                    force=args.force,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())

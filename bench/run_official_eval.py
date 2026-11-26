"""Official Public External Benchmark Runner:
  1. MathArena/aime_2025           (30/30 official AIME 2025 competition math problems)
  2. bigcode/bigcodebench-hard     (30 official BigCodeBench-Hard v0.1.4 tasks with canonical_solution verified on macOS)

Runs strictly apples-to-apples Vanilla (thinking_budget=1500) vs. FROZEN FlashForge (enable_memory=False)
on the EXACT same local model endpoint (`--url`), with ZERO post-run tuning.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from flash_forge.client import CallSpec, TensorFoldClient
from flash_forge.code_forge import CodeForgeTrace
from flash_forge.engine import FlashForge, ForgeConfig
from flash_forge.reason_forge import ReasonForgeTrace, extract_answer_from_prose
from flash_forge.sandbox import check_syntax, extract_python_code, run_python

from .run_ab import grade_reason


def grade_bigcodebench(code: str, test_code: str, timeout_s: float = 15.0) -> tuple[bool, str]:
    syn = check_syntax(code)
    if syn:
        return False, f"SyntaxError: {syn}"
    # BigCodeBench official unittest harness executes solution + test file in same module namespace
    script = (
        f"{code}\n\n"
        f"{test_code}\n\n"
        "if __name__ == '__main__':\n"
        "    import unittest\n"
        "    unittest.main(verbosity=0)\n"
    )
    res = run_python(script, timeout_s=timeout_s)
    if res.ok:
        return True, f"PASS ({res.elapsed_s * 1000:.0f}ms)"
    if res.timed_out:
        return False, f"TIMEOUT ({timeout_s}s)"
    tail = [ln for ln in (res.stderr or res.stdout).strip().splitlines() if ln.strip()]
    msg = tail[-1] if tail else f"exit={res.returncode}"
    return False, f"FAIL: {msg[:180]}"


def load_dataset(name: str) -> list[dict[str, Any]]:
    if name == "aime2025":
        raw = json.loads(Path("bench/datasets/aime_2025.json").read_text(encoding="utf-8"))
        out = []
        for r in raw:
            idx = int(r["problem_idx"])
            prompt = (
                f"{r['problem'].strip()}\n\n"
                "Note: Every AIME answer is an integer from 000 to 999.\n"
                "End your final response with `ANSWER: <integer>`."
            )
            out.append({
                "pid": f"AIME25_{idx:02d}",
                "mode": "reason",
                "prompt": prompt,
                "expected": str(int(r["answer"])),
            })
        return out
    if name in ("bigcodebench_hard", "bigcodebench_hard_complete"):
        raw = json.loads(Path("bench/datasets/bigcodebench_hard_verified.json").read_text(encoding="utf-8"))
        out = []
        use_complete = name == "bigcodebench_hard_complete"
        for r in raw:
            if use_complete:
                prompt = (
                    "Complete the following Python implementation faithfully according to every detail "
                    "of the docstring, starter imports, constants, parameters, return type, and exceptions.\n\n"
                    f"```python\n{r['complete_prompt'].strip()}\n```\n\n"
                    "Output ONLY one complete, self-contained Python code block including all required imports, constants, and `task_func`."
                )
            else:
                prompt = (
                    f"{r['instruct_prompt'].strip()}\n\n"
                    "Output ONLY a complete, self-contained Python code block implementing the requested function(s)."
                )
            out.append({
                "pid": r["task_id"].replace("/", "_"),
                "mode": "code",
                "prompt": prompt,
                "test": r["test"],
            })
        return out
    raise ValueError(f"Unknown dataset: {name}")


async def run_one_vanilla(client: TensorFoldClient, item: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    res = await client.complete(
        CallSpec(
            messages=[{"role": "user", "content": item["prompt"]}],
            tag=f"official_v_{item['pid']}",
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            seed=42,
            enable_thinking=True,
            thinking_budget=1500,
            loop_guard=True,
        )
    )
    wall = time.perf_counter() - t0
    if item["mode"] == "reason":
        ext = extract_answer_from_prose(res.content)
        if not ext and res.content:
            lines = [ln.strip() for ln in res.content.strip().splitlines() if ln.strip()]
            ext = lines[-1] if lines else ""
        passed, detail = grade_reason(ext, item["expected"])
    else:
        code = extract_python_code(res.content)
        passed, detail = grade_bigcodebench(code, item["test"])
    return {
        "arm": "vanilla_budgeted",
        "pid": item["pid"],
        "passed": passed,
        "detail": detail,
        "wall_s": round(wall, 2),
        "finish_reason": res.finish_reason,
        "completion_tokens": res.completion_tokens,
    }


async def run_one_forge(base_url: str, item: dict[str, Any], streams: int = 4) -> dict[str, Any]:
    cfg = ForgeConfig(base_url=base_url, num_streams=streams, deep=False, enable_memory=False)
    async with FlashForge(cfg) as ff:
        res = await ff.solve(item["prompt"], force_mode=None, learn=False)
    extra: dict[str, Any] = {"routed_mode": res.mode, "routed_match": res.mode == item["mode"]}
    if item["mode"] == "reason":
        ans = res.answer if res.mode == "reason" else extract_answer_from_prose(res.answer)
        passed, detail = grade_reason(ans, item["expected"])
        if isinstance(res.raw_trace, ReasonForgeTrace):
            extra["consensus"] = res.raw_trace.consensus_type
            extra["wave2"] = res.raw_trace.wave2_triggered
    else:
        code = res.code or extract_python_code(res.answer)
        passed, detail = grade_bigcodebench(code, item["test"])
        if isinstance(res.raw_trace, CodeForgeTrace):
            extra["winner"] = res.raw_trace.winner_name
            extra["redteam_kills"] = len(res.raw_trace.redteam_valid_kills)
            extra["wave2"] = res.raw_trace.wave2_triggered
    return {
        "arm": "flash_forge",
        "pid": item["pid"],
        "passed": passed,
        "detail": detail,
        "wall_s": round(res.wall_s, 2),
        "stats": res.token_stats,
        "extra": extra,
    }


async def run_eval(
    dataset: str,
    base_url: str,
    arm: str,
    limit: int,
    concurrency: int,
    out_path: Path,
    resume: bool,
) -> None:
    items = load_dataset(dataset)
    if limit > 0:
        items = items[:limit]

    by_pid: dict[str, dict[str, Any]] = {}
    if resume and out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        for row in prev.get("results", []):
            by_pid[row["pid"]] = row
        print(f"Resumed {len(by_pid)} results from {out_path}")

    todo = [x for x in items if x["pid"] not in by_pid]
    t_all = time.perf_counter()
    async with TensorFoldClient(base_url=base_url) as client:
        model_id = client.caps.model_id
        print(f"=== OFFICIAL EVAL: {dataset} | model={model_id} | arm={arm} | todo={len(todo)}/{len(items)} | conc={concurrency} ===")
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        done = len(items) - len(todo)

        async def _w(item: dict[str, Any]) -> None:
            nonlocal done
            async with sem:
                try:
                    row = await (run_one_vanilla(client, item) if arm == "vanilla" else run_one_forge(base_url, item))
                except Exception as exc:
                    row = {
                        "arm": arm,
                        "pid": item["pid"],
                        "passed": False,
                        "detail": f"EXCEPTION: {type(exc).__name__}: {exc}",
                        "wall_s": 0.0,
                    }
                async with lock:
                    done += 1
                    by_pid[item["pid"]] = row
                    mark = "PASS" if row["passed"] else "FAIL"
                    ex = row.get("extra", {})
                    note = f" [{ex.get('consensus') or ex.get('winner', '')}]" if ex else ""
                    print(f"[{done:02d}/{len(items):02d}] {mark:4s} | {item['pid']:<22} | {row['wall_s']:5.1f}s | {row['detail'][:65]}{note}")
                    sys.stdout.flush()
                    ordered = [by_pid[x["pid"]] for x in items if x["pid"] in by_pid]
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(
                        json.dumps(
                            {
                                "dataset": dataset,
                                "model": model_id,
                                "arm": arm,
                                "completed": len(ordered),
                                "passed": sum(1 for r in ordered if r["passed"]),
                                "total": len(items),
                                "results": ordered,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

        await asyncio.gather(*[_w(x) for x in todo])

    ordered = [by_pid[x["pid"]] for x in items if x["pid"] in by_pid]
    p = sum(1 for r in ordered if r["passed"])
    avg_w = sum(r["wall_s"] for r in ordered) / max(1, len(ordered))
    print("\n" + "=" * 72)
    print(f"OFFICIAL {dataset.upper()} | {model_id} | {arm}: {p}/{len(ordered)} ({100*p/max(1,len(ordered)):.1f}%) | avg={avg_w:.1f}s | total={time.perf_counter()-t_all:.1f}s")
    print("=" * 72)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["aime2025", "bigcodebench_hard", "bigcodebench_hard_complete"], required=True)
    p.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--arm", choices=["vanilla", "forge"], required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    conc = args.concurrency or (1 if args.arm == "forge" else 6)
    asyncio.run(run_eval(args.dataset, args.url, args.arm, args.limit, conc, Path(args.out), args.resume))


if __name__ == "__main__":
    main()

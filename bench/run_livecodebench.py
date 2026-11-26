"""Official LiveCodeBench v6 Hard Benchmark Runner (AtCoder/LeetCode Competitive Programming).

Runs strictly apples-to-apples Vanilla vs. Frozen Cortheon (`enable_memory=False`, `force_mode=None`)
on Official `livecodebench/code_generation_lite (test6.jsonl, difficulty=hard)` and grades every solution
mechanically against ALL public + private official judge test cases via subprocess stdin/stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import pickle
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

from cortheon import Cortheon, CortheonConfig
from flash_forge.client import CallSpec, TensorFoldClient
from flash_forge.code_forge import CodeForgeTrace
from flash_forge.sandbox import check_syntax, extract_python_code


def _decode_cases(raw: str) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw))))


def _norm_out(s: str) -> str:
    return "\n".join(ln.rstrip() for ln in s.strip().splitlines())


def grade_livecodebench(code: str, cases: list[dict[str, Any]], per_case_timeout_s: float = 4.0) -> tuple[bool, str]:
    syn = check_syntax(code)
    if syn:
        return False, f"SyntaxError: {syn}"
    with tempfile.TemporaryDirectory(prefix="ff_lcb_") as td:
        sol_path = Path(td) / "solution.py"
        sol_path.write_text(code, encoding="utf-8")
        passed = 0
        total = len(cases)
        t0 = time.perf_counter()
        for idx, tc in enumerate(cases):
            inp = tc.get("input", "")
            exp = _norm_out(tc.get("output", ""))
            try:
                p = subprocess.run(
                    [sys.executable, str(sol_path)],
                    input=inp,
                    text=True,
                    capture_output=True,
                    timeout=per_case_timeout_s,
                    cwd=td,
                )
            except subprocess.TimeoutExpired:
                return False, f"TLE on case {idx+1}/{total} ({passed}/{total} passed)"
            if p.returncode != 0:
                err = (p.stderr or p.stdout).strip().splitlines()
                msg = err[-1][:120] if err else f"exit={p.returncode}"
                return False, f"RE on case {idx+1}/{total} ({passed}/{total}): {msg}"
            got = _norm_out(p.stdout)
            if got != exp:
                return False, f"WA on case {idx+1}/{total} ({passed}/{total}): got={got[:40]!r} exp={exp[:40]!r}"
            passed += 1
        return True, f"AC {passed}/{total} ({(time.perf_counter()-t0)*1000:.0f}ms)"


def load_lcb_dataset(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in raw:
        pub = _decode_cases(r.get("public_test_cases", ""))
        priv = _decode_cases(r.get("private_test_cases", ""))
        prompt = (
            f"Solve the following competitive programming problem in Python 3.\n"
            f"Read from standard input and write the exact answer to standard output.\n\n"
            f"{r['question_content'].strip()}\n\n"
            "Output ONLY one complete, self-contained Python 3 code block."
        )
        out.append({
            "pid": r["question_id"],
            "title": r["question_title"],
            "platform": r["platform"],
            "prompt": prompt,
            "cases": pub + priv,
            "n_pub": len(pub),
            "n_priv": len(priv),
        })
    return out[:limit] if limit > 0 else out


async def run_one_vanilla(client: TensorFoldClient, item: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    res = await client.complete(
        CallSpec(
            messages=[{"role": "user", "content": item["prompt"]}],
            tag=f"lcb_v_{item['pid']}",
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
    code = extract_python_code(res.content)
    passed, detail = await asyncio.to_thread(grade_livecodebench, code, item["cases"])
    return {
        "arm": "vanilla",
        "pid": item["pid"],
        "title": item["title"],
        "passed": passed,
        "detail": detail,
        "wall_s": round(wall, 2),
        "finish_reason": res.finish_reason,
    }


async def run_one_forge(base_url: str, item: dict[str, Any], streams: int = 4) -> dict[str, Any]:
    cfg = CortheonConfig(base_url=base_url, num_streams=streams, deep=False, enable_memory=False)
    async with Cortheon(cfg) as c:
        res = await c.solve(item["prompt"], force_mode=None, learn=False)
    code = res.code or extract_python_code(res.answer)
    passed, detail = await asyncio.to_thread(grade_livecodebench, code, item["cases"])
    extra: dict[str, Any] = {"routed_mode": res.mode}
    if isinstance(res.raw_trace, CodeForgeTrace):
        extra["winner"] = res.raw_trace.winner_name
        extra["wave2"] = res.raw_trace.wave2_triggered
        extra["rt_kills"] = len(res.raw_trace.redteam_valid_kills)
        extra["oracle_tests"] = len(res.raw_trace.oracle_test_names)
    return {
        "arm": "cortheon",
        "pid": item["pid"],
        "title": item["title"],
        "passed": passed,
        "detail": detail,
        "wall_s": round(res.wall_s, 2),
        "stats": res.token_stats,
        "extra": extra,
    }


async def run_eval(dataset_path: Path, base_url: str, arm: str, limit: int, concurrency: int, out_path: Path, resume: bool) -> None:
    items = load_lcb_dataset(dataset_path, limit=limit)
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
        print(f"=== OFFICIAL LIVECODEBENCH v6 HARD | model={model_id} | arm={arm} | todo={len(todo)}/{len(items)} | conc={concurrency} ===")
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        done = len(items) - len(todo)

        async def _w(item: dict[str, Any]) -> None:
            nonlocal done
            async with sem:
                try:
                    row = await (run_one_vanilla(client, item) if arm == "vanilla" else run_one_forge(base_url, item))
                except Exception as exc:
                    row = {"arm": arm, "pid": item["pid"], "title": item["title"], "passed": False, "detail": f"EXC: {exc}", "wall_s": 0.0}
                async with lock:
                    done += 1
                    by_pid[item["pid"]] = row
                    mark = "PASS" if row["passed"] else "FAIL"
                    ex = row.get("extra", {})
                    note = f" [win={ex.get('winner','')}, w2={ex.get('wave2')}]" if ex else ""
                    print(f"[{done:02d}/{len(items):02d}] {mark:4s} | {item['pid']:<12} | {row['wall_s']:5.1f}s | {row['detail'][:60]}{note}")
                    sys.stdout.flush()
                    ordered = [by_pid[x["pid"]] for x in items if x["pid"] in by_pid]
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps({
                        "dataset": "livecodebench_v6_hard",
                        "model": model_id,
                        "arm": arm,
                        "completed": len(ordered),
                        "passed": sum(1 for r in ordered if r["passed"]),
                        "total": len(items),
                        "results": ordered,
                    }, indent=2), encoding="utf-8")

        await asyncio.gather(*[_w(x) for x in todo])

    ordered = [by_pid[x["pid"]] for x in items if x["pid"] in by_pid]
    p = sum(1 for r in ordered if r["passed"])
    avg_w = sum(r["wall_s"] for r in ordered) / max(1, len(ordered))
    print("\n" + "=" * 72)
    print(f"OFFICIAL LIVECODEBENCH v6 HARD | {model_id} | {arm}: {p}/{len(ordered)} ({100*p/max(1,len(ordered)):.1f}%) | avg={avg_w:.1f}s | total={time.perf_counter()-t_all:.1f}s")
    print("=" * 72)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bench/datasets/livecodebench_v6_hard.json")
    p.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--arm", choices=["vanilla", "forge"], required=True)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    conc = args.concurrency or (1 if args.arm == "forge" else 4)
    asyncio.run(run_eval(Path(args.dataset), args.url, args.arm, args.limit, conc, Path(args.out), args.resume))


if __name__ == "__main__":
    main()

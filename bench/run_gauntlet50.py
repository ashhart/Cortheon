"""50-Test Frontier Gauntlet Runner (Concurrent, Zero-Leak, Checkpointable).

Evaluates 50 brand-new extremely hard tasks across 5 tracks (10 each):
  1. math        (G01-G10): Olympiad / AIME math, exact probability, huge-N recurrences, Burnside
  2. logic_opt   (G11-G20): Cryptarithms, truth/liar/alternator, TSPTW, 2D knapsack, killer Latin, circuit faults, makespan, BFS, minimax
  3. algo_code   (G21-G30): Convex hull, interval set, Thompson NFA regex, RFC6902 JSON patch, iterative Tarjan, Knuth-Plass DP, SemVer resolver, snapshots, streaming median, decaying top-k
  4. debug_code  (G31-G40): Multi-bug debugging with planted controls (token bucket, consistent hash ring, order-stat BST, RFC4180 CSV, rational arbitrage, wildcard trie, rollback DSU, SQL 3-valued logic, incremental build cutoff, HTTP chunked decoder)
  5. systems_sec (G41-G50): Execution semantics traps (Python MRO/closure, ARX uint32, IEEE-754 cancellation, C11 struct ABI, SQL LEFT JOIN NULLs, regex diff) + real security engineering (path traversal, HMAC replay, IAM explicit deny, SSTI sandbox)

Supports concurrency to saturate `--parallel 8` on local server (`concurrency=6` for vanilla, `concurrency=2` for FlashForge B=4).
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from flash_forge.client import CallSpec, TensorFoldClient
from flash_forge.code_forge import CodeForgeTrace
from flash_forge.engine import FlashForge, ForgeConfig
from flash_forge.reason_forge import (
    ReasonForgeTrace,
    extract_answer_from_prose,
)
from flash_forge.sandbox import extract_python_code

from .gauntlet_part1 import PART1_PROBLEMS, GauntletProblem
from .gauntlet_part2 import PART2_PROBLEMS
from .run_ab import grade_code, grade_reason

ALL_50: list[GauntletProblem] = PART1_PROBLEMS + PART2_PROBLEMS
TRACKS = ["math", "logic_opt", "algo_code", "debug_code", "systems_sec"]


def verify_suite() -> None:
    """Mechanically verify all 50 problems, AST-parse all hidden tests, and verify all planted controls fail."""
    assert len(ALL_50) == 50, f"Expected 50 problems, got {len(ALL_50)}"
    pids = [p.pid for p in ALL_50]
    assert len(set(pids)) == 50, "Duplicate problem IDs!"
    by_track = Counter(p.track for p in ALL_50)
    print(f"Suite size: {len(ALL_50)} | Track counts: {dict(by_track)}")
    for t in TRACKS:
        assert by_track[t] == 10, f"Track {t} has {by_track[t]} problems (expected 10)"

    code_probs = [p for p in ALL_50 if p.forge_mode == "code"]
    reason_probs = [p for p in ALL_50 if p.forge_mode == "reason"]
    print(f"Modes: reason={len(reason_probs)}, code={len(code_probs)}")

    # 1. Verify every reason problem has non-empty expected answer
    for p in reason_probs:
        assert p.expected.strip(), f"{p.pid} missing expected answer!"

    # 2. AST-parse every hidden test suite & check it has at least one test_* function
    for p in code_probs:
        assert p.hidden_tests.strip(), f"{p.pid} missing hidden_tests!"
        try:
            tree = ast.parse(p.hidden_tests)
        except SyntaxError as exc:
            raise AssertionError(f"SyntaxError in {p.pid} hidden_tests: {exc}") from exc
        test_fns = [
            n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
        ]
        assert test_fns, f"{p.pid} has no top-level test_* functions!"

    # 3. Planted control verification: every buggy_starter MUST fail its hidden_tests!
    controls = 0
    for p in ALL_50:
        if p.buggy_starter:
            controls += 1
            ok, detail = grade_code(p.buggy_starter, p.hidden_tests)
            assert not ok, f"PLANTED CONTROL FAILED for {p.pid}: buggy_starter PASSED hidden tests! ({detail})"
            print(f"  [CONTROL OK] {p.pid:<34} -> {detail[:100]}")
    print(f"All 50 problems & {controls} planted controls VERIFIED CLEAN!")


async def run_one_vanilla(
    client: TensorFoldClient,
    prob: GauntletProblem,
    *,
    thinking_budget: int | None,
    arm_name: str,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    res = await client.complete(
        CallSpec(
            messages=[{"role": "user", "content": prob.prompt}],
            tag=f"{arm_name}_{prob.pid}",
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            seed=42,
            enable_thinking=True,
            thinking_budget=thinking_budget,
            loop_guard=True,
        )
    )
    wall = time.perf_counter() - t0
    extracted_preview = ""
    if prob.forge_mode == "reason":
        extracted = extract_answer_from_prose(res.content)
        if not extracted and res.content:
            lines = [ln.strip() for ln in res.content.strip().splitlines() if ln.strip()]
            extracted = lines[-1] if lines else ""
        extracted_preview = extracted[:120]
        passed, detail = grade_reason(extracted, prob.expected)
    else:
        code = extract_python_code(res.content)
        passed, detail = grade_code(code, prob.hidden_tests)
    return {
        "arm": arm_name,
        "pid": prob.pid,
        "track": prob.track,
        "forge_mode": prob.forge_mode,
        "title": prob.title,
        "passed": passed,
        "detail": detail,
        "wall_s": round(wall, 2),
        "finish_reason": res.finish_reason,
        "completion_tokens": res.completion_tokens,
        "prompt_tokens": res.prompt_tokens,
        "extracted": extracted_preview,
    }


async def run_one_forge(
    base_url: str,
    prob: GauntletProblem,
    *,
    streams: int,
    deep: bool,
    force_mode: str | None = None,
) -> dict[str, Any]:
    """Each concurrent task uses its own FlashForge instance (isolated token counters & cold start)."""
    cfg = ForgeConfig(
        base_url=base_url,
        num_streams=streams,
        deep=deep,
        enable_memory=False,  # clean cold-start fairness across all 50 independent tests
    )
    async with FlashForge(cfg) as ff:
        res = await ff.solve(prob.prompt, force_mode=force_mode, learn=False)

    routed_ok = res.mode == prob.forge_mode
    extra: dict[str, Any] = {
        "routed_mode": res.mode,
        "routed_match": routed_ok,
    }
    if prob.forge_mode == "reason":
        ans = res.answer
        if res.mode != "reason":
            ans = extract_answer_from_prose(res.answer)
        passed, detail = grade_reason(ans, prob.expected)
        if isinstance(res.raw_trace, ReasonForgeTrace):
            extra["consensus"] = res.raw_trace.consensus_type
            extra["wave2"] = res.raw_trace.wave2_triggered
            extra["wave2_reason"] = res.raw_trace.wave2_reason
            extra["w1_answers"] = [f"{o.name}:{o.norm_answer or 'ERR'}" for o in res.raw_trace.wave1_outcomes]
            extra["w1_errors"] = {o.name: o.error[:240] for o in res.raw_trace.wave1_outcomes if o.error}
            if res.raw_trace.wave2_outcomes:
                extra["w2_answers"] = [f"{o.name}:{o.norm_answer or 'ERR'}" for o in res.raw_trace.wave2_outcomes]
                extra["w2_errors"] = {o.name: o.error[:240] for o in res.raw_trace.wave2_outcomes if o.error}
            if not passed:
                extra["w1_codes"] = {o.name: o.code[:1200] for o in res.raw_trace.wave1_outcomes if o.code}
        elif not passed:
            extra["raw_answer_tail"] = res.answer[-600:]
    else:
        code = res.code or extract_python_code(res.answer)
        passed, detail = grade_code(code, prob.hidden_tests)
        if isinstance(res.raw_trace, CodeForgeTrace):
            extra["winner"] = res.raw_trace.winner_name
            extra["oracle_tests"] = len(res.raw_trace.oracle_test_names)
            extra["dropped_oracle"] = len(res.raw_trace.dropped_bad_oracle_tests)
            extra["redteam_kills"] = len(res.raw_trace.redteam_valid_kills)
            extra["wave2"] = res.raw_trace.wave2_triggered
            extra["wave2_reason"] = res.raw_trace.wave2_reason
            if not passed:
                extra["winner_code"] = code[:2500]
                extra["oracle_code"] = res.raw_trace.oracle_code[:1200]
                extra["graveyard"] = res.raw_trace.graveyard[:4]
                extra["w1_scores"] = [
                    f"{c.name}:{c.total_passed}/{c.total_checks}({c.syntax_error or 'ok'})"
                    for c in res.raw_trace.wave1_candidates
                ]
                extra["w2_scores"] = [
                    f"{c.name}:{c.total_passed}/{c.total_checks}({c.syntax_error or 'ok'})"
                    for c in res.raw_trace.wave2_candidates
                ]
        elif not passed:
            extra["raw_answer_tail"] = res.answer[-600:]

    return {
        "arm": "flash_forge",
        "pid": prob.pid,
        "track": prob.track,
        "forge_mode": prob.forge_mode,
        "title": prob.title,
        "passed": passed,
        "detail": detail,
        "wall_s": round(res.wall_s, 2),
        "stats": res.token_stats,
        "extra": extra,
    }


def _save_checkpoint(
    out_path: Path,
    *,
    arm: str,
    model: str,
    base_url: str,
    streams: int,
    concurrency: int,
    results_by_pid: dict[str, dict[str, Any]],
) -> None:
    ordered = [results_by_pid[p.pid] for p in ALL_50 if p.pid in results_by_pid]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "arm": arm,
                "model": model,
                "base_url": base_url,
                "streams": streams,
                "concurrency": concurrency,
                "completed": len(ordered),
                "passed": sum(1 for r in ordered if r["passed"]),
                "results": ordered,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def print_summary(results: list[dict[str, Any]], label: str) -> None:
    print("\n" + "=" * 78)
    print(f"GAUNTLET-50 SUMMARY — {label}")
    print("=" * 78)
    print(f"{'Track':<14} {'Passed':>10} {'Rate':>8} {'Avg Wall':>10}")
    print("-" * 46)
    tot_p = 0
    for tr in TRACKS:
        rs = [r for r in results if r["track"] == tr]
        if not rs:
            continue
        p = sum(1 for r in rs if r["passed"])
        tot_p += p
        avg_w = sum(r["wall_s"] for r in rs) / len(rs)
        print(f"{tr:<14} {p:>3d}/{len(rs):<3d}   {100*p/len(rs):6.1f}% {avg_w:8.1f}s")
    print("-" * 46)
    avg_all = sum(r["wall_s"] for r in results) / max(1, len(results))
    print(f"{'OVERALL':<14} {tot_p:>3d}/{len(results):<3d}   {100*tot_p/max(1,len(results)):6.1f}% {avg_all:8.1f}s")
    fails = [r for r in results if not r["passed"]]
    if fails:
        print(f"\nFAILURES ({len(fails)}):")
        for r in fails:
            ex = r.get("extra", {})
            route = f" [routed={ex.get('routed_mode')}]" if ex and not ex.get("routed_match", True) else ""
            print(f"  - {r['pid']:<34} ({r['track']:<11}) {r['detail'][:110]}{route}")


async def run_gauntlet(
    *,
    base_url: str,
    arm: str,
    pids: list[str] | None,
    streams: int,
    deep: bool,
    concurrency: int,
    out_path: Path,
    resume: bool,
) -> None:
    verify_suite()
    selected = [p for p in ALL_50 if not pids or p.pid in pids]

    results_by_pid: dict[str, dict[str, Any]] = {}
    if resume and out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            for row in prev.get("results", []):
                results_by_pid[row["pid"]] = row
            print(f"Resumed {len(results_by_pid)} completed results from {out_path}")
        except Exception as exc:
            print(f"Warning: could not resume {out_path}: {exc}")

    todo = [p for p in selected if p.pid not in results_by_pid]
    print(
        f"\n=== Running Gauntlet-50 | arm={arm} | todo={len(todo)}/{len(selected)} "
        f"| concurrency={concurrency} | streams={streams} ==="
    )

    t_suite = time.perf_counter()
    async with TensorFoldClient(base_url=base_url) as probe_client:
        model_id = probe_client.caps.model_id
        print(
            f"Server: {model_id} | structures={probe_client.caps.structures} "
            f"| think_budget={probe_client.caps.thinking_budget} | loop_guard={probe_client.caps.loop_guard}\n"
        )
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        done_count = len(selected) - len(todo)

        async def _worker(prob: GauntletProblem) -> None:
            nonlocal done_count
            async with sem:
                try:
                    if arm == "vanilla":
                        row = await run_one_vanilla(
                            probe_client, prob, thinking_budget=None, arm_name="vanilla"
                        )
                    elif arm == "vanilla_budgeted":
                        row = await run_one_vanilla(
                            probe_client, prob, thinking_budget=1500, arm_name="vanilla_budgeted"
                        )
                    else:
                        row = await run_one_forge(base_url, prob, streams=streams, deep=deep)
                except Exception as exc:
                    row = {
                        "arm": arm,
                        "pid": prob.pid,
                        "track": prob.track,
                        "forge_mode": prob.forge_mode,
                        "title": prob.title,
                        "passed": False,
                        "detail": f"EXCEPTION: {type(exc).__name__}: {exc}",
                        "wall_s": 0.0,
                        "extra": {},
                    }
                async with lock:
                    done_count += 1
                    results_by_pid[prob.pid] = row
                    mark = "PASS" if row["passed"] else "FAIL"
                    ex = row.get("extra", {})
                    note = ""
                    if arm == "forge":
                        if not ex.get("routed_match", True):
                            note += f" MISROUTE->{ex.get('routed_mode')}"
                        if prob.forge_mode == "reason":
                            note += f" [{ex.get('consensus', '')}, w2={ex.get('wave2')}]"
                        else:
                            note += f" [win={ex.get('winner', '')}, rt={ex.get('redteam_kills', 0)}, w2={ex.get('wave2')}]"
                    print(
                        f"[{done_count:02d}/{len(selected):02d}] {mark:4s} | {prob.pid:<34} "
                        f"| {row['wall_s']:5.1f}s | {row['detail'][:70]}{note}"
                    )
                    sys.stdout.flush()
                    _save_checkpoint(
                        out_path,
                        arm=arm,
                        model=model_id,
                        base_url=base_url,
                        streams=streams,
                        concurrency=concurrency,
                        results_by_pid=results_by_pid,
                    )

        await asyncio.gather(*[_worker(p) for p in todo])

    ordered = [results_by_pid[p.pid] for p in selected if p.pid in results_by_pid]
    print_summary(ordered, f"{arm} ({model_id}, total wall={time.perf_counter() - t_suite:.1f}s)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--verify-only", action="store_true", help="Only verify all 50 problems & planted controls")
    p.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--arm", choices=["vanilla", "vanilla_budgeted", "forge"], default="forge")
    p.add_argument("--pids", nargs="*", default=None)
    p.add_argument("--streams", type=int, default=4)
    p.add_argument("--deep", action="store_true")
    p.add_argument("--concurrency", type=int, default=0, help="0 = auto (6 for vanilla, 2 for forge)")
    p.add_argument("--out", default="bench/results/gauntlet50.json")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    if args.verify_only:
        verify_suite()
        return

    conc = args.concurrency or (2 if args.arm == "forge" else 6)
    asyncio.run(
        run_gauntlet(
            base_url=args.url,
            arm=args.arm,
            pids=args.pids,
            streams=args.streams,
            deep=args.deep,
            concurrency=conc,
            out_path=Path(args.out),
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()

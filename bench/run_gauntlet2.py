"""50-Test Frontier Gauntlet II (Nightmare Tier) Runner — H01..H50."""

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
from flash_forge.reason_forge import ReasonForgeTrace, extract_answer_from_prose
from flash_forge.sandbox import extract_python_code

from .gauntlet2_part1 import PART1_PROBLEMS, Gauntlet2Problem
from .gauntlet2_part2 import PART2_PROBLEMS
from .run_ab import grade_code, grade_reason

ALL_50: list[Gauntlet2Problem] = PART1_PROBLEMS + PART2_PROBLEMS
TRACKS = ["math_nm", "logic_nm", "algo_nm", "debug_nm", "sys_sec_nm"]


def verify_suite() -> None:
    """Mechanically verify all 50 Nightmare problems, AST-parse all hidden tests, and confirm all 10 planted controls fail."""
    assert len(ALL_50) == 50, f"Expected 50 problems, got {len(ALL_50)}"
    assert len({p.pid for p in ALL_50}) == 50, "Duplicate problem IDs!"
    by_track = Counter(p.track for p in ALL_50)
    print(f"Gauntlet-II size: {len(ALL_50)} | Track counts: {dict(by_track)}")
    for t in TRACKS:
        assert by_track[t] == 10, f"Track {t} has {by_track[t]} (expected 10)"

    code_probs = [p for p in ALL_50 if p.forge_mode == "code"]
    reason_probs = [p for p in ALL_50 if p.forge_mode == "reason"]
    print(f"Modes: reason={len(reason_probs)}, code={len(code_probs)}")

    for p in reason_probs:
        assert p.expected.strip(), f"{p.pid} missing expected answer!"

    for p in code_probs:
        assert p.hidden_tests.strip(), f"{p.pid} missing hidden_tests!"
        tree = ast.parse(p.hidden_tests)
        fns = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        assert fns, f"{p.pid} has no test_* functions!"

    controls = 0
    for p in ALL_50:
        if p.buggy_starter:
            controls += 1
            ok, detail = grade_code(p.buggy_starter, p.hidden_tests)
            assert not ok, f"PLANTED CONTROL FAILED for {p.pid}: buggy starter PASSED ({detail})"
            print(f"  [CONTROL OK] {p.pid:<38} -> {detail[:95]}")
    assert controls == 10, f"Expected 10 planted controls, found {controls}"
    print(f"All 50 Gauntlet-II Nightmare problems & {controls} planted controls VERIFIED CLEAN!")


async def run_one_vanilla(
    client: TensorFoldClient,
    prob: Gauntlet2Problem,
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
    prob: Gauntlet2Problem,
    *,
    streams: int,
    deep: bool,
) -> dict[str, Any]:
    cfg = ForgeConfig(
        base_url=base_url,
        num_streams=streams,
        deep=deep,
        enable_memory=False,
    )
    async with FlashForge(cfg) as ff:
        res = await ff.solve(prob.prompt, force_mode=None, learn=False)

    routed_ok = res.mode == prob.forge_mode
    extra: dict[str, Any] = {
        "routed_mode": res.mode,
        "routed_match": routed_ok,
        "verified": res.verified,
    }
    if prob.forge_mode == "reason":
        ans = res.answer if res.mode == "reason" else extract_answer_from_prose(res.answer)
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
                extra["w1_codes"] = {o.name: o.code[:1500] for o in res.raw_trace.wave1_outcomes if o.code}
                extra["w2_codes"] = {o.name: o.code[:1500] for o in res.raw_trace.wave2_outcomes if o.code}
        elif not passed:
            extra["raw_answer_tail"] = res.answer[-800:]
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
                extra["winner_code"] = code[:4000]
                extra["oracle_code"] = res.raw_trace.oracle_code[:1800]
                extra["graveyard"] = res.raw_trace.graveyard[:4]
                extra["w1_scores"] = [
                    f"{c.name}:{c.total_passed}/{c.total_checks}({c.syntax_error or c.fatal_error or 'ok'})"
                    for c in res.raw_trace.wave1_candidates
                ]
                extra["w2_scores"] = [
                    f"{c.name}:{c.total_passed}/{c.total_checks}({c.syntax_error or c.fatal_error or 'ok'})"
                    for c in res.raw_trace.wave2_candidates
                ]
        elif not passed:
            extra["raw_answer_tail"] = res.answer[-800:]

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
                "suite": "gauntlet2_nightmare_50",
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
    print(f"GAUNTLET-II (NIGHTMARE 50) SUMMARY — {label}")
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
            print(f"  - {r['pid']:<38} ({r['track']:<10}) {r['detail'][:105]}{route}")


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
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        for row in prev.get("results", []):
            results_by_pid[row["pid"]] = row
        print(f"Resumed {len(results_by_pid)} completed results from {out_path}")

    todo = [p for p in selected if p.pid not in results_by_pid]
    print(
        f"\n=== Running Gauntlet-II (Nightmare 50) | arm={arm} | todo={len(todo)}/{len(selected)} "
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

        async def _worker(prob: Gauntlet2Problem) -> None:
            nonlocal done_count
            async with sem:
                try:
                    if arm == "vanilla":
                        row = await run_one_vanilla(probe_client, prob, thinking_budget=None, arm_name="vanilla")
                    elif arm == "vanilla_budgeted":
                        row = await run_one_vanilla(probe_client, prob, thinking_budget=1500, arm_name="vanilla_budgeted")
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
                        f"[{done_count:02d}/{len(selected):02d}] {mark:4s} | {prob.pid:<38} "
                        f"| {row['wall_s']:5.1f}s | {row['detail'][:68]}{note}"
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
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--arm", choices=["vanilla", "vanilla_budgeted", "forge"], default="forge")
    p.add_argument("--pids", nargs="*", default=None)
    p.add_argument("--streams", type=int, default=4)
    p.add_argument("--deep", action="store_true")
    p.add_argument("--concurrency", type=int, default=0)
    p.add_argument("--out", default="bench/results/gauntlet2_forge.json")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    if args.verify_only:
        verify_suite()
        return

    conc = args.concurrency or (1 if args.arm == "forge" else 6)
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

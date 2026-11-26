"""Fair, zero-leak A/B benchmark runner on live Qwen3.8-Flash-Next-TFQ3 (local server).

Compares:
  Arm A (Vanilla):    Single-stream Qwen3.8-Flash-Next-TFQ3 (thinking ON, max_tokens=8192, loop_guard=True)
  Arm B (FlashForge): Same frozen weights wrapped in FlashForge (B=4 parallel racing, Python/SymPy/Z3 sandbox,
                      modular oracle, clean-context Red-Teamer, graveyard backtracking, pairwise judge, auto-router)

Both arms receive the EXACT same raw prompt and are graded by the EXACT same hidden verifier.
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
from flash_forge.code_forge import CodeForgeTrace, _run_modular_suite
from flash_forge.engine import FlashForge, ForgeConfig
from flash_forge.reason_forge import (
    ReasonForgeTrace,
    _answers_equivalent,
    extract_answer_from_prose,
    normalize_answer,
)
from flash_forge.sandbox import check_syntax, extract_python_code

from .problems import PROBLEMS, BenchProblem


def grade_reason(raw_answer: str, expected: str) -> tuple[bool, str]:
    norm_got = normalize_answer(raw_answer)
    norm_exp = normalize_answer(expected)
    if not norm_got:
        return False, f"empty (expected {norm_exp})"
    if _answers_equivalent(norm_got, norm_exp):
        return True, f"OK ({norm_got})"
    # Case-insensitive whitespace-stripped string match (e.g. "D, F" vs "D,F", "Dia, 10" vs "Dia,10")
    clean_got = norm_got.replace(" ", "").lower()
    clean_exp = norm_exp.replace(" ", "").lower()
    if clean_got == clean_exp:
        return True, f"OK ({norm_got})"
    return False, f"WRONG: got={norm_got!r} expected={norm_exp!r}"


def grade_code(code: str, hidden_tests: str) -> tuple[bool, str]:
    syn = check_syntax(code)
    if syn:
        return False, f"SyntaxError: {syn}"
    rows, ms, fatal = _run_modular_suite(code, hidden_tests, timeout_s=10.0)
    if fatal and not rows:
        return False, f"FATAL: {fatal[:200]}"
    if not rows:
        return False, "No hidden tests executed"
    passed = sum(1 for r in rows if r["ok"])
    total = len(rows)
    if passed == total:
        return True, f"PASS {passed}/{total} ({ms:.0f}ms)"
    failed = [f"{r['name']}: {r['err'].splitlines()[-1] if r['err'] else 'fail'}" for r in rows if not r["ok"]]
    return False, f"FAIL {passed}/{total} | " + "; ".join(failed[:2])[:260]


async def run_vanilla(
    client: TensorFoldClient,
    prob: BenchProblem,
    *,
    thinking_budget: int | None = None,
    arm_name: str = "vanilla",
) -> dict[str, Any]:
    client.reset_stats()
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
    if prob.forge_mode == "reason":
        extracted = extract_answer_from_prose(res.content)
        if not extracted and res.content:
            lines = [ln.strip() for ln in res.content.strip().splitlines() if ln.strip()]
            extracted = lines[-1] if lines else ""
        passed, detail = grade_reason(extracted, prob.expected)
    else:
        code = extract_python_code(res.content)
        passed, detail = grade_code(code, prob.hidden_tests)
    return {
        "arm": arm_name,
        "pid": prob.pid,
        "category": prob.category,
        "passed": passed,
        "detail": detail,
        "wall_s": round(wall, 2),
        "finish_reason": res.finish_reason,
        "stats": client.stats_dict(),
    }


async def run_forge(ff: FlashForge, prob: BenchProblem) -> dict[str, Any]:
    res = await ff.solve(prob.prompt, force_mode=None, learn=True)
    routed_ok = (res.mode == prob.forge_mode)

    extra: dict[str, Any] = {
        "routed_mode": res.mode,
        "routed_match": routed_ok,
        "lesson_learned": res.lesson_learned,
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
            extra["w1_errors"] = {o.name: o.error[:180] for o in res.raw_trace.wave1_outcomes if o.error}
            if res.raw_trace.wave2_outcomes:
                extra["w2_answers"] = [f"{o.name}:{o.norm_answer or 'ERR'}" for o in res.raw_trace.wave2_outcomes]
                extra["w2_errors"] = {o.name: o.error[:180] for o in res.raw_trace.wave2_outcomes if o.error}
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

    return {
        "arm": "flash_forge",
        "pid": prob.pid,
        "category": prob.category,
        "passed": passed,
        "detail": detail,
        "wall_s": round(res.wall_s, 2),
        "stats": res.token_stats,
        "extra": extra,
    }


def verify_planted_controls() -> None:
    """Referee Rule: prove that our debug hidden graders actually FAIL on the buggy starter code!"""
    for p in PROBLEMS:
        if p.buggy_starter:
            ok, detail = grade_code(p.buggy_starter, p.hidden_tests)
            assert not ok, f"Planted control FAILED for {p.pid}: buggy starter passed hidden tests! ({detail})"
            print(f"[CONTROL OK] {p.pid} buggy starter caught -> {detail}")


async def run_suite(
    base_url: str,
    pids: list[str] | None,
    streams: int,
    deep: bool,
    out_path: Path,
    arm_filter: str = "both",
) -> None:
    selected = [p for p in PROBLEMS if not pids or p.pid in pids]
    mem_db = out_path.parent / "bench_memory.db"
    if mem_db.exists():
        mem_db.unlink()

    print(f"=== FlashForge A/B Benchmark on {base_url} | {len(selected)} problems | B={streams} deep={deep} ===")
    verify_planted_controls()
    print()

    vanilla_results: list[dict[str, Any]] = []
    forge_results: list[dict[str, Any]] = []

    async with TensorFoldClient(base_url=base_url) as v_client, FlashForge(
        ForgeConfig(base_url=base_url, num_streams=streams, deep=deep, enable_memory=True, memory_db=mem_db)
    ) as ff:
        caps = v_client.caps
        print(f"Server model: {caps.model_id} | structures={caps.structures} | think_budget={caps.thinking_budget} | loop_guard={caps.loop_guard}\n")

        for idx, prob in enumerate(selected, 1):
            print(f"[{idx:02d}/{len(selected):02d}] {prob.pid} ({prob.category}) — {prob.title}")
            if arm_filter in ("both", "vanilla"):
                vr = await run_vanilla(v_client, prob)
                vanilla_results.append(vr)
                v_mark = "PASS" if vr["passed"] else "FAIL"
                print(
                    f"   Vanilla    : {v_mark:4s} | {vr['wall_s']:5.1f}s | "
                    f"tok={vr['stats']['completion_tokens']:5d} | {vr['detail']}"
                )
            if arm_filter == "vanilla_budgeted":
                vr = await run_vanilla(v_client, prob, thinking_budget=1500, arm_name="vanilla_budgeted")
                vanilla_results.append(vr)
                v_mark = "PASS" if vr["passed"] else "FAIL"
                print(
                    f"   Vanilla(B) : {v_mark:4s} | {vr['wall_s']:5.1f}s | "
                    f"tok={vr['stats']['completion_tokens']:5d} | {vr['detail']}"
                )
            if arm_filter in ("both", "forge"):
                fr = await run_forge(ff, prob)
                forge_results.append(fr)
                f_mark = "PASS" if fr["passed"] else "FAIL"
                ex = fr["extra"]
                note = ""
                if prob.forge_mode == "reason":
                    note = f" [{ex.get('consensus','')}, w2={ex.get('wave2')}]"
                else:
                    note = f" [win={ex.get('winner','')}, rt_kills={ex.get('redteam_kills',0)}, w2={ex.get('wave2')}]"
                print(
                    f"   FlashForge : {f_mark:4s} | {fr['wall_s']:5.1f}s | "
                    f"tok={fr['stats']['completion_tokens']:5d} (calls={fr['stats']['calls']:2d}) | {fr['detail']}{note}"
                )
            sys.stdout.flush()

            # Checkpoint incremental results after every problem
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(
                    {
                        "model": caps.model_id,
                        "base_url": base_url,
                        "streams": streams,
                        "deep": deep,
                        "vanilla": vanilla_results,
                        "flash_forge": forge_results,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    # Print final summary table
    print("\n" + "=" * 78)
    print("FINAL A/B SCOREBOARD (EXACT SAME FROZEN WEIGHTS)")
    print("=" * 78)
    cats = ["math", "logic", "algo_code", "debug_code"]
    print(f"{'Category':<14} {'Vanilla':>12} {'FlashForge':>14} {'Delta':>10}")
    print("-" * 54)
    v_tot = f_tot = n_tot = 0
    for cat in cats:
        vc = [r for r in vanilla_results if r["category"] == cat]
        fc = [r for r in forge_results if r["category"] == cat]
        n = max(len(vc), len(fc))
        if n == 0:
            continue
        vp = sum(1 for r in vc if r["passed"])
        fp = sum(1 for r in fc if r["passed"])
        v_tot += vp
        f_tot += fp
        n_tot += n
        v_str = f"{vp}/{len(vc)} ({100*vp/len(vc):.0f}%)" if vc else "-"
        f_str = f"{fp}/{len(fc)} ({100*fp/len(fc):.0f}%)" if fc else "-"
        delta = f"+{fp - vp}" if (vc and fc and fp >= vp) else (f"{fp - vp}" if (vc and fc) else "-")
        print(f"{cat:<14} {v_str:>12} {f_str:>14} {delta:>10}")
    print("-" * 54)
    v_all = f"{v_tot}/{len(vanilla_results)} ({100*v_tot/len(vanilla_results):.1f}%)" if vanilla_results else "-"
    f_all = f"{f_tot}/{len(forge_results)} ({100*f_tot/len(forge_results):.1f}%)" if forge_results else "-"
    d_all = f"+{f_tot - v_tot}" if (vanilla_results and forge_results) else "-"
    print(f"{'OVERALL':<14} {v_all:>12} {f_all:>14} {d_all:>10}")
    if vanilla_results:
        print(f"Avg wall time — Vanilla: {sum(r['wall_s'] for r in vanilla_results)/len(vanilla_results):.1f}s")
    if forge_results:
        print(f"Avg wall time — FlashForge: {sum(r['wall_s'] for r in forge_results)/len(forge_results):.1f}s")
        w2_cnt = sum(1 for r in forge_results if r["extra"].get("wave2"))
        rt_cnt = sum(r["extra"].get("redteam_kills", 0) for r in forge_results)
        print(f"Wave-2 triggered: {w2_cnt}/{len(forge_results)} | Valid Red-Team kills: {rt_cnt}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--pids", nargs="*", default=None)
    p.add_argument("--streams", type=int, default=4)
    p.add_argument("--deep", action="store_true")
    p.add_argument("--arm", choices=["both", "vanilla", "vanilla_budgeted", "forge"], default="both")
    p.add_argument("--out", default="bench/results/ab_results.json")
    args = p.parse_args()
    asyncio.run(run_suite(args.url, args.pids, args.streams, args.deep, Path(args.out), args.arm))


if __name__ == "__main__":
    main()

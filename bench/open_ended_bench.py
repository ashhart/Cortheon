"""Oracle-Free Principal-Engineer Open-Ended Systems, Physics & Security Benchmark (10 tasks).

Evaluates 100% oracle-free open-ended engineering prompts (NO external test suite, NO math-contest
`ANSWER:` tag — pure natural engineering design/audit questions) apples-to-apples between:
  - Vanilla (single-stream local LLM, thinking_budget=1500)
  - Cortheon (ThinkForge: parallel specialist council + executable Python verifier + clean-context critique + synthesis)

Grading is 100% deterministic, reproducible, and objective per task:
  - Quantitative / physics bounds checked with unit-aware tolerance (accepting both decimal MB/TB and binary MiB/TiB!)
  - Hard impossibility / security / concurrency trap detection
  - Concrete architectural mitigation checks
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from flash_forge.client import CallSpec, TensorFoldClient
from flash_forge.engine import FlashForge, ForgeConfig
from flash_forge.think_forge import ThinkForgeTrace


def _has_number_near(text: str, targets: list[tuple[float, float]], unit_pat: str = "") -> bool:
    """Return True if any numeric token in `text` falls inside any (lo, hi) range, optionally near `unit_pat`."""
    clean = text.replace(",", "")
    if unit_pat:
        pat = re.compile(rf"(\d+(?:\.\d+)?)\s*(?:{unit_pat})\b", re.IGNORECASE)
        for m in pat.finditer(clean):
            val = float(m.group(1))
            if any(lo <= val <= hi for lo, hi in targets):
                return True
        return False
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\b", clean):
        val = float(m.group(1))
        if any(lo <= val <= hi for lo, hi in targets):
            return True
    return False


def _has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


@dataclass
class OpenEndedTask:
    pid: str
    title: str
    prompt: str
    checks: list[tuple[str, Callable[[str], bool], bool]]  # (label, predicate, is_critical_trap)


def build_tasks() -> list[OpenEndedTask]:
    return [
        OpenEndedTask(
            pid="OE01_payment_ledger_sla",
            title="50k TPS Multi-Region Payment Ledger SLA Physics",
            prompt=(
                "Design a globally distributed payment ledger sustaining 50,000 TPS with an average 2 KB "
                "transaction payload and 3x synchronous Raft replication across us-east, eu-west, and ap-south "
                "(minimum inter-region RTT is 35 ms between us-east and eu-west, and 110 ms to ap-south). "
                "Product wants <40 ms p99 write latency with zero-RPO strict serializability. "
                "Provide exact throughput/bandwidth/daily-storage sizing, evaluate SLA feasibility, and specify the concrete architecture and double-spend defenses."
            ),
            checks=[
                (
                    "ingress_100MBs",
                    lambda t: _has_number_near(t, [(92.0, 105.0)], r"(?:m[ib]?b/s|megabytes?\s*/\s*s)")
                    or _has_number_near(t, [(740.0, 840.0)], r"(?:mbps|mbit/s)"),
                    False,
                ),
                (
                    "daily_storage_8TB",
                    lambda t: _has_number_near(t, [(7.5, 9.0), (22.5, 27.0)], r"(?:t[ib]?b(?:/day|\s*per\s*day|\s*daily)?)"),
                    False,
                ),
                (
                    "rtt_impossibility_trap",
                    lambda t: _has_any(
                        t,
                        [
                            r"(?:physically\s+)?impossible",
                            r"infeasible",
                            r"cannot\s+(?:meet|achieve|satisfy|guarantee).{0,60}(?:40\s*ms|p99|sla)",
                            r"(?:35\s*ms|rtt).{0,60}(?:violates?|exceeds?|leaves?\s+only|budget|impossible|infeasible)",
                        ],
                    ),
                    True,
                ),
                (
                    "idempotency_and_sharding",
                    lambda t: _has_any(t, [r"idempoten", r"double[- ]?entry", r"fencing"])
                    and _has_any(t, [r"regional|home[- ]?region|geo[- ]?partition|local\s+quorum|async|relax"]),
                    False,
                ),
            ],
        ),
        OpenEndedTask(
            pid="OE02_llm_serving_vram",
            title="70B Transformer Serving on Single 80GB H100 Physics",
            prompt=(
                "We want to deploy a 70B-parameter dense Transformer (80 layers, 8 GQA KV heads, 128 head_dim, "
                "BF16 weights and BF16 KV cache) on a single 80 GB H100 GPU to serve 64 concurrent requests at "
                "8,192 tokens context length each. Provide the exact VRAM memory breakdown (weights + KV cache) "
                "and design a feasible production serving architecture."
            ),
            checks=[
                (
                    "weights_140GB",
                    lambda t: _has_number_near(t, [(128.0, 145.0)], r"(?:g[ib]?b)"),
                    False,
                ),
                (
                    "kv_cache_160GB",
                    lambda t: _has_number_near(t, [(155.0, 176.0)], r"(?:g[ib]?b)")
                    or _has_number_near(t, [(2.4, 2.8)], r"(?:g[ib]?b)"),
                    False,
                ),
                (
                    "vram_impossibility_trap",
                    lambda t: _has_any(
                        t,
                        [
                            r"(?:physically\s+)?impossible",
                            r"infeasible",
                            r"cannot\s+fit",
                            r"exceeds?.{0,40}80\s*g[ib]?b",
                            r"oom|out\s+of\s+memory",
                        ],
                    ),
                    True,
                ),
                (
                    "serving_mitigations",
                    lambda t: _has_any(t, [r"tensor\s+parallel|multi[- ]?gpu|4\s*x|2\s*x|8\s*x"])
                    and _has_any(t, [r"int4|fp4|fp8|quantiz"])
                    and _has_any(t, [r"paged\s*attention|prefix\s+cach|continuous\s+batch"]),
                    False,
                ),
            ],
        ),
        OpenEndedTask(
            pid="OE03_nvme_wal_write_amp",
            title="LSM Storage Node 1TB NVMe Wear-Out Physics",
            prompt=(
                "Design a single-node LSM-tree time-series storage engine ingesting 200,000 events/sec "
                "(512 bytes per event) with a WAL plus 10x LSM compaction write amplification onto a single "
                "1 TB NVMe SSD rated for 600 TBW endurance over a 3-year target lifespan and 1.5 GB/s sustained "
                "write bandwidth. Compute exact ingest/amplified write rates, SSD lifespan, and fix the design."
            ),
            checks=[
                (
                    "raw_ingest_100MBs",
                    lambda t: _has_number_near(t, [(95.0, 108.0)], r"(?:m[ib]?b/s)")
                    or _has_number_near(t, [(7.8, 9.2)], r"(?:t[ib]?b/day|t[ib]?b\s+per\s+day)"),
                    False,
                ),
                (
                    "amplified_1GBs_90TBday",
                    lambda t: _has_number_near(t, [(0.95, 1.25)], r"(?:g[ib]?b/s)")
                    or _has_number_near(t, [(950.0, 1250.0)], r"(?:m[ib]?b/s)")
                    or _has_number_near(t, [(80.0, 105.0)], r"(?:t[ib]?b/day|t[ib]?b\s+per\s+day)"),
                    False,
                ),
                (
                    "ssd_dies_in_days_trap",
                    lambda t: _has_number_near(t, [(5.5, 8.0)], r"days?")
                    or _has_any(t, [r"under\s+(?:a|one)\s+week", r"less\s+than\s+(?:a|1)\s+week", r"burn.{0,30}days"]),
                    True,
                ),
                (
                    "lsm_mitigations",
                    lambda t: _has_any(t, [r"compress|lz4|zstd|gorilla|delta"])
                    and _has_any(t, [r"twcs|time[- ]?window|tier|shard|object\s+storage|s3"]),
                    False,
                ),
            ],
        ),
        OpenEndedTask(
            pid="OE04_quorum_split_brain",
            title="4-Node 2-AZ Raft Upgrade Fault-Tolerance Audit",
            prompt=(
                "An engineering team proposes expanding their 3-node Raft metadata cluster to 4 nodes split "
                "evenly across 2 availability zones (2 nodes in AZ-A, 2 nodes in AZ-B) so the cluster can "
                "survive losing an entire AZ (2 nodes) and tolerate 2 simultaneous node failures. "
                "Audit the quorum math and availability impact, and specify the correct production topology."
            ),
            checks=[
                (
                    "quorum_3_of_4",
                    lambda t: _has_any(t, [r"quorum.{0,25}\b3\b", r"\b3\s*(?:of|/)\s*4\b", r"majority\s+of\s+4\s+is\s+3"]),
                    False,
                ),
                (
                    "n4_tolerates_only_1_failure_trap",
                    lambda t: _has_any(
                        t,
                        [
                            r"(?:only|at\s+most)\s+(?:1|one)\s+(?:node\s+)?failure",
                            r"same\s+fault\s+tolerance\s+as\s+(?:a\s+)?3",
                            r"cannot\s+survive\s+(?:losing\s+)?(?:an?\s+)?(?:entire\s+)?az",
                            r"los(?:ing|s\s+of)\s+(?:an?\s+)?az.{0,45}(?:outage|unavailable|halts?|blocks?|only\s+2)",
                        ],
                    ),
                    True,
                ),
                (
                    "recommend_5_nodes_3_azs",
                    lambda t: _has_any(t, [r"\b5\s*[- ]?nodes?\b", r"five\s+nodes"])
                    and _has_any(t, [r"\b3\s*(?:azs?|availability\s+zones?|failure\s+domains?|regions?)", r"third\s+az|witness|tie[- ]?breaker|arbiter"]),
                    False,
                ),
            ],
        ),
        OpenEndedTask(
            pid="OE05_cache_thundering_herd",
            title="1M RPS Hot-Key Cache Stampede Origin Collapse",
            prompt=(
                "A caching tier serves 1,000,000 req/s (4 KB average payload) with a 99% overall cache hit rate "
                "in front of an origin database capped at 25,000 req/s (100 MB/s). One hot key accounts for 5% "
                "of total traffic with a hard 60-second TTL. Compute steady-state vs expiry-spike origin load "
                "and total edge bandwidth, and design the stampede-proof caching architecture."
            ),
            checks=[
                (
                    "steady_miss_10k_rps",
                    lambda t: _has_number_near(t, [(9500.0, 10500.0)], r"(?:req/s|rps|qps)")
                    or _has_number_near(t, [(9.5, 10.5)], r"k\s*(?:req/s|rps|qps)")
                    or _has_number_near(t, [(38.0, 42.0)], r"m[ib]?b/s"),
                    False,
                ),
                (
                    "hotkey_spike_50k_rps_trap",
                    lambda t: (
                        _has_number_near(t, [(48000.0, 62000.0)], r"(?:req/s|rps|qps)")
                        or _has_number_near(t, [(48.0, 62.0)], r"k\s*(?:req/s|rps|qps)")
                        or _has_number_near(t, [(190.0, 250.0)], r"m[ib]?b/s")
                    )
                    and _has_any(t, [r"exceed|overload|collapse|2x|2\.4x|stampede|saturat|overwhelm"]),
                    True,
                ),
                (
                    "edge_bw_4GBs",
                    lambda t: _has_number_near(t, [(3.7, 4.2)], r"g[ib]?b/s")
                    or _has_number_near(t, [(30.0, 34.0)], r"gbps"),
                    False,
                ),
                (
                    "stampede_defenses",
                    lambda t: _has_any(t, [r"singleflight|coalesc|mutex|lock|lease"])
                    and _has_any(t, [r"stale[- ]?while[- ]?revalidate|probabilistic|xfetch|jitter|background\s+refresh|l1|in[- ]?process"]),
                    False,
                ),
            ],
        ),
        OpenEndedTask(
            pid="OE06_rate_limiter_race",
            title="Distributed 20-Gateway Rate Limiter Race & Boundary Audit",
            prompt=(
                "Audit and redesign a distributed API rate limiter enforcing 100 requests/minute per tenant "
                "across 20 stateless API gateways backed by Redis. The current implementation does "
                "`val = redis.get(k); if int(val) < 100: redis.incr(k)` using a fixed 60-second TTL window. "
                "Explain every concurrency race and bypass quantitatively, and provide the hardened production design."
            ),
            checks=[
                (
                    "boundary_burst_200_trap",
                    lambda t: _has_number_near(t, [(195.0, 205.0)], r"(?:req|requests?|calls?)")
                    or _has_any(t, [r"2x\s+(?:the\s+)?(?:limit|burst|rate)", r"double\s+the\s+limit", r"boundary\s+burst"]),
                    True,
                ),
                (
                    "toctou_race_trap",
                    lambda t: _has_any(t, [r"toctou|check[- ]?then[- ]?act|race\s+condition|non[- ]?atomic|get.{0,20}incr"]),
                    True,
                ),
                (
                    "lua_atomic_token_sliding",
                    lambda t: _has_any(t, [r"lua|evalsha"])
                    and _has_any(t, [r"token\s+bucket|sliding\s+window|gcra|leaky\s+bucket"]),
                    False,
                ),
                (
                    "clock_skew_and_failure_mode",
                    lambda t: _has_any(t, [r"clock\s+skew|redis\s+time|server\s+time"])
                    and _has_any(t, [r"fail[- ]?open|fail[- ]?closed|circuit\s+breaker|local\s+(?:quota|fallback|bucket)"]),
                    False,
                ),
            ],
        ),
        OpenEndedTask(
            pid="OE07_jwt_refresh_security",
            title="Zero-Trust JWT & Refresh-Token Replay Security Audit",
            prompt=(
                "Audit an auth service issuing 15-minute RS256 JWT access tokens (verified via JWKS) and "
                "30-day opaque refresh tokens stored in Postgres. The current implementation reads the `alg` "
                "field directly from the JWT header when verifying, rotates refresh tokens by deleting the old "
                "DB row and inserting a new row (no family tracking), and validates service API keys using `==`. "
                "Identify every critical vulnerability and specify the hardened protocol."
            ),
            checks=[
                (
                    "alg_confusion_trap",
                    lambda t: _has_any(t, [r"alg\s*[:=]?\s*['\"]?none", r"hs256", r"algorithm\s+confusion", r"key\s+confusion"])
                    and _has_any(t, [r"pin|whitelist|allowlist|hardcode|explicit.{0,25}rs256"]),
                    True,
                ),
                (
                    "refresh_family_reuse_detection_trap",
                    lambda t: _has_any(t, [r"token\s+famil|family_id|reuse\s+detect|replay"])
                    and _has_any(t, [r"revoke.{0,40}(?:entire|all|family|tree|chain|descendant)"]),
                    True,
                ),
                (
                    "timing_attack_constant_time",
                    lambda t: _has_any(t, [r"timing\s+(?:attack|side[- ]?channel|leak)", r"constant[- ]?time", r"compare_digest"]),
                    False,
                ),
                (
                    "hashed_storage_and_concurrency",
                    lambda t: _has_any(t, [r"hash|sha[- ]?256|hmac|argon2"])
                    and _has_any(t, [r"for\s+update|grace\s+window|concurrent|atomic|lock"]),
                    False,
                ),
            ],
        ),
        OpenEndedTask(
            pid="OE08_zero_downtime_shard_migration",
            title="40TB Postgres 16-Shard Zero-Downtime Dual-Write Audit",
            prompt=(
                "We must migrate a live 40 TB Postgres `orders` table taking 15,000 writes/sec (~8 KB row+index) "
                "to a 16-shard cluster with zero downtime and zero lost/corrupted updates. The current plan is: "
                "(1) snapshot dump, (2) enable app-level dual-write (write old DB then new DB), (3) backfill, "
                "(4) flip reads. Quantify per-shard load and backfill time, audit EVERY data-corruption race, "
                "and design the safe migration protocol."
            ),
            checks=[
                (
                    "per_shard_sizing",
                    lambda t: _has_number_near(t, [(2.3, 2.7)], r"t[ib]?b")
                    and (
                        _has_number_near(t, [(900.0, 980.0)], r"(?:writes?/s|wps|tps|ops/s|qps)")
                        or _has_number_near(t, [(110.0, 128.0)], r"m[ib]?b/s")
                    ),
                    False,
                ),
                (
                    "dual_write_race_corruption_trap",
                    lambda t: _has_any(
                        t,
                        [
                            r"out[- ]?of[- ]?order",
                            r"partial\s+failure|crash\s+between",
                            r"silent\s+(?:drift|diverg|corrupt)",
                            r"backfill.{0,45}overwrit",
                        ],
                    ),
                    True,
                ),
                (
                    "cdc_wal_and_monotonic_version",
                    lambda t: _has_any(t, [r"cdc|logical\s+replication|debezium|wal\s+stream"])
                    and _has_any(t, [r"lsn|version|monotonic|timestamp|idempoten|upsert"]),
                    False,
                ),
                (
                    "shadow_verify_and_cutover",
                    lambda t: _has_any(t, [r"shadow\s+read|reconcil|checksum|diff\s+verif|audit"])
                    and _has_any(t, [r"reverse[- ]?cdc|rollback|fenc|quiesce|read[- ]?only"]),
                    False,
                ),
            ],
        ),
        OpenEndedTask(
            pid="OE09_network_shuffle_bottleneck",
            title="10-Node 10Gbps 2TB Distributed Join Shuffle Physics",
            prompt=(
                "A 10-node analytics cluster (each node has one 10 Gbps NIC and 64 GB RAM) must execute a "
                "distributed join shuffling 2 TB of uncompressed intermediate data uniformly across all 10 nodes "
                "within a strict 60-second SLA. Calculate the exact network/memory physics, state whether the "
                "SLA is achievable as stated, and design the engineering optimizations to hit <60s."
            ),
            checks=[
                (
                    "nic_12_5GBs_aggregate",
                    lambda t: _has_number_near(t, [(1.1, 1.3)], r"g[ib]?b/s")
                    or _has_number_near(t, [(11.0, 13.0)], r"g[ib]?b/s"),
                    False,
                ),
                (
                    "min_time_144_to_165s_impossibility_trap",
                    lambda t: _has_number_near(t, [(138.0, 185.0)], r"(?:s|sec|seconds?)")
                    and _has_any(t, [r"(?:physically\s+)?impossible|infeasible|cannot\s+(?:meet|achieve|hit)|exceeds?\s+(?:the\s+)?60"]),
                    True,
                ),
                (
                    "ram_spill_640GB",
                    lambda t: _has_number_near(t, [(600.0, 650.0)], r"g[ib]?b")
                    and _has_any(t, [r"spill|disk|exceeds?\s+(?:cluster\s+)?ram|out\s+of\s+memory"]),
                    False,
                ),
                (
                    "shuffle_optimizations",
                    lambda t: _has_any(t, [r"compress|lz4|zstd"])
                    and _has_any(t, [r"bloom|semi[- ]?join|broadcast|predicate\s+pushdown|co[- ]?partition|bucket|pre[- ]?aggregat"]),
                    False,
                ),
            ],
        ),
        OpenEndedTask(
            pid="OE10_stream_exactly_once_state",
            title="100k EPS 24h Sliding-Window Fraud Pipeline State Sizing",
            prompt=(
                "Design a streaming fraud-aggregation pipeline processing 100,000 events/sec across 50,000,000 "
                "active accounts computing 24-hour sliding-window sums with a 1-minute slide, handling up to "
                "15 minutes of out-of-order mobile events, and writing exactly-once alerts to Postgres. "
                "Quantify raw vs pre-aggregated state size, watermark/late-data bounds, and end-to-end exactly-once semantics."
            ),
            checks=[
                (
                    "daily_events_8_64B",
                    lambda t: _has_number_near(t, [(8.4, 8.9)], r"(?:b|billion)")
                    or _has_number_near(t, [(800.0, 900.0)], r"g[ib]?b"),
                    False,
                ),
                (
                    "pane_1440_buckets",
                    lambda t: _has_number_near(t, [(1435.0, 1465.0)], r"(?:buckets?|panes?|slots?|minutes?|windows?)")
                    or _has_any(t, [r"1,?440\b", r"1,?455\b"]),
                    False,
                ),
                (
                    "watermark_and_late_data_trap",
                    lambda t: _has_any(t, [r"watermark"])
                    and _has_any(t, [r"15\s*(?:min|m\b)"])
                    and _has_any(t, [r"side[- ]?output|dlq|dead[- ]?letter|drop|allowed\s+lateness|evict|ttl"]),
                    True,
                ),
                (
                    "end_to_end_idempotent_sink_trap",
                    lambda t: _has_any(t, [r"checkpoint|chandy[- ]?lamport|flink"])
                    and _has_any(t, [r"idempoten|on\s+conflict|upsert|two[- ]?phase|2pc|dedup"]),
                    True,
                ),
            ],
        ),
    ]


def grade_open_ended(task: OpenEndedTask, answer: str) -> tuple[bool, float, dict[str, bool]]:
    results: dict[str, bool] = {}
    critical_ok = True
    for label, fn, is_crit in task.checks:
        ok = bool(fn(answer))
        results[label] = ok
        if is_crit and not ok:
            critical_ok = False
    score = sum(1 for v in results.values() if v) / max(1, len(results))
    passed = critical_ok and score >= 0.75
    return passed, round(score, 3), results


async def run_vanilla_one(client: TensorFoldClient, task: OpenEndedTask) -> dict[str, Any]:
    t0 = time.perf_counter()
    res = await client.complete(
        CallSpec(
            messages=[{"role": "user", "content": task.prompt}],
            tag=f"oe_v_{task.pid}",
            max_tokens=4096,
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
    ans = res.content or ""
    passed, score, checks = grade_open_ended(task, ans)
    return {
        "arm": "vanilla",
        "pid": task.pid,
        "passed": passed,
        "score": score,
        "checks": checks,
        "wall_s": round(wall, 2),
        "chars": len(ans),
        "finish_reason": res.finish_reason,
    }


async def run_cortheon_one(base_url: str, task: OpenEndedTask, streams: int = 4) -> dict[str, Any]:
    cfg = ForgeConfig(base_url=base_url, num_streams=streams, deep=False, enable_memory=False)
    async with FlashForge(cfg) as ff:
        res = await ff.solve(task.prompt, force_mode=None, learn=False)
    ans = res.answer or ""
    passed, score, checks = grade_open_ended(task, ans)
    quant_ok = isinstance(res.raw_trace, ThinkForgeTrace) and res.raw_trace.quant_verified
    return {
        "arm": "cortheon",
        "pid": task.pid,
        "passed": passed,
        "score": score,
        "checks": checks,
        "wall_s": round(res.wall_s, 2),
        "chars": len(ans),
        "routed_mode": res.mode,
        "quant_verified": quant_ok,
    }


async def run_suite(base_url: str, arm: str, out_path: Path, concurrency: int) -> None:
    tasks = build_tasks()
    t0 = time.perf_counter()
    async with TensorFoldClient(base_url=base_url) as client:
        model_id = client.caps.model_id
        print(f"=== ORACLE-FREE OPEN-ENDED BENCH (10 tasks) | model={model_id} | arm={arm} ===", flush=True)
        sem = asyncio.Semaphore(concurrency)
        rows: list[dict[str, Any]] = []

        async def _w(task: OpenEndedTask) -> dict[str, Any]:
            async with sem:
                row = await (run_vanilla_one(client, task) if arm == "vanilla" else run_cortheon_one(base_url, task))
                mark = "PASS" if row["passed"] else "FAIL"
                qv = f" qv={row.get('quant_verified')} mode={row.get('routed_mode')}" if arm == "cortheon" else ""
                missed = [k for k, v in row["checks"].items() if not v]
                print(
                    f"  {mark:4s} | {task.pid:32s} | {row['score']*100:5.1f}% | {row['wall_s']:5.1f}s{qv} | missed={missed}",
                    flush=True,
                )
                return row

        rows = list(await asyncio.gather(*[_w(t) for t in tasks]))
    n_pass = sum(1 for r in rows if r["passed"])
    avg_score = sum(r["score"] for r in rows) / len(rows)
    avg_wall = sum(r["wall_s"] for r in rows) / len(rows)
    qv_cnt = sum(1 for r in rows if r.get("quant_verified"))
    summary = {
        "dataset": "open_ended_systems_10",
        "model": model_id,
        "arm": arm,
        "passed": n_pass,
        "total": len(rows),
        "avg_score": round(avg_score, 4),
        "avg_wall_s": round(avg_wall, 2),
        "quant_verified_count": qv_cnt,
        "results": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 72)
    print(
        f"OPEN-ENDED 10 | {model_id} | {arm}: PASS {n_pass}/{len(rows)} ({100*n_pass/len(rows):.1f}%) "
        f"| avg_score={100*avg_score:.1f}% | avg={avg_wall:.1f}s | qv={qv_cnt}/{len(rows)} | total={time.perf_counter()-t0:.1f}s",
        flush=True,
    )
    print("=" * 72, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--arm", choices=["vanilla", "cortheon"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=0)
    args = p.parse_args()
    conc = args.concurrency or (1 if args.arm == "cortheon" else 4)
    asyncio.run(run_suite(args.url, args.arm, Path(args.out), conc))


if __name__ == "__main__":
    main()

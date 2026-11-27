"""5-Arm Compute-Matched Ablation Runner on Full 30/30 Official MathArena/aime_2025.

Arms:
  1. vanilla_b1    : Single prose call (B=1, temp=0.6, seed=42, tb=1500, max_tokens=8192)
  2. maj4_prose    : Compute-matched B=4 parallel prose majority vote (no Python execution;
                     seeds 101/202/303/404, temps 0.25/0.55/0.45/0.35, tb=1000, max_tokens=4096;
                     majority normalized 0..999 integer wins, tie broken by lowest-temp valid answer)
  3. pal_b1        : Single executable PAL Python script (B=1, pal_exact card, seed=101,
                     temp=0.25, tb=500, reasoning_effort="low", 20s sandbox, no consensus/Wave 2)
  4. cortheon_w1   : B=4 Wave-1 PAL consensus WITHOUT Wave 2 (ReasonForge Wave 1 with
                     early-abort, low-effort PAL, 20s sandbox, NO Wave-2 falsifier)
  5. cortheon_full : Full ReasonForge (`solve_reasoning`: Wave 1 early-abort + conditional
                     Wave 2 falsifier + conditional Wave 3 fallback, enable_memory=False)

Sharding & Merging:
    # Shard 1 (problems 0..15)
    python -m bench.run_aime_ablation --url http://127.0.0.1:8080/v1 --start 0 --end 15 --out bench/results/ablation_s1.json
    # Shard 2 (problems 15..30)
    python -m bench.run_aime_ablation --url http://127.0.0.1:8081/v1 --start 15 --end 30 --out bench/results/ablation_s2.json
    # Merge cleanly into 30/30 with exact Wilson CIs & paired McNemar p-values
    python -m bench.run_aime_ablation --merge bench/results/ablation_s1.json bench/results/ablation_s2.json --out bench/results/ablation_30.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

from flash_forge.client import CallSpec, TensorFoldClient
from flash_forge.memory import EpisodicMemory
from flash_forge.reason_forge import (
    BranchOutcome,
    _answers_equivalent,
    _execute_pal_outcome,
    extract_answer_from_prose,
    normalize_answer,
    solve_reasoning,
)
from flash_forge.skill_loader import SkillPack

from .stats import format_p_value, mcnemar_exact_p, paired_contingency, wilson_ci

ALL_ARMS: tuple[str, ...] = (
    "vanilla_b1",
    "maj4_prose",
    "pal_b1",
    "cortheon_w1",
    "cortheon_full",
)

MAJ4_CONFIGS: tuple[tuple[int, float], ...] = (
    (101, 0.25),
    (202, 0.55),
    (303, 0.45),
    (404, 0.35),
)

_AIME_SHARED_SYS = (
    "You are a precision reasoning engine running inside a verified neuro-symbolic sandbox "
    "with Python 3, sympy, z3, fractions, itertools, math, and numpy available.\n"
    "Never guess arithmetic, combinatorics, or constraint satisfaction in prose when Python can compute and verify it exactly."
)


def load_aime2025(dataset_path: Path, start: int = 0, end: int = 30) -> list[dict[str, Any]]:
    """Load MathArena/aime_2025 JSON using the exact prompt format of `bench/run_official_eval.py`."""
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    raw_sorted = sorted(raw, key=lambda r: int(r["problem_idx"]))
    items: list[dict[str, Any]] = []
    for r in raw_sorted:
        idx = int(r["problem_idx"])
        prompt = (
            f"{r['problem'].strip()}\n\n"
            "Note: Every AIME answer is an integer from 000 to 999.\n"
            "End your final response with `ANSWER: <integer>`."
        )
        items.append(
            {
                "id": f"AIME25_{idx:02d}",
                "problem_idx": idx,
                "prompt": prompt,
                "expected": str(int(r["answer"])),
            }
        )
    s = max(0, start)
    e = len(items) if end <= 0 else min(len(items), end)
    return items[s:e]


def extract_prose_candidate(text: str) -> str:
    """Extract normalized answer from prose using the exact `run_official_eval.py` heuristic."""
    ext = extract_answer_from_prose(text)
    if not ext and text:
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        ext = lines[-1] if lines else ""
    return normalize_answer(ext)


def normalize_aime_integer(raw: str) -> str:
    """Return canonical integer string '0'..'999' if `raw` represents a valid AIME integer, else ''."""
    norm = normalize_answer(raw)
    if not norm:
        return ""
    if re.fullmatch(r"\d+", norm):
        val = int(norm)
        if 0 <= val <= 999:
            return str(val)
    return ""


def grade_aime(predicted: str, expected: str) -> tuple[bool, str]:
    """Grade an AIME 2025 prediction against the exact 0..999 integer answer."""
    norm_got = normalize_answer(predicted)
    norm_exp = str(int(normalize_answer(expected)))
    if not norm_got:
        return False, f"empty (expected {norm_exp})"
    if re.fullmatch(r"\d+", norm_got):
        val = int(norm_got)
        if 0 <= val <= 999 and str(val) == norm_exp:
            return True, f"OK ({val})"
        return False, f"WRONG: got={norm_got!r} expected={norm_exp!r}"
    if _answers_equivalent(norm_got, norm_exp):
        return True, f"OK ({norm_got})"
    return False, f"WRONG: got={norm_got!r} expected={norm_exp!r}"


def select_maj4_answer(candidates: list[tuple[float, str]]) -> tuple[str, dict[str, Any]]:
    """Select majority normalized integer from `(temperature, raw_prose)` samples.

    Valid AIME integers (0..999) are clustered first; majority count wins and ties
    are broken by the lowest-temperature valid sample in each cluster. If zero
    samples produce a valid 0..999 integer, falls back to clustering non-empty
    normalized answers with the same rule.
    """
    per_branch: list[dict[str, Any]] = []
    valid_votes: list[tuple[int, float, str]] = []
    fallback_votes: list[tuple[int, float, str]] = []

    for idx, (temp, raw_text) in enumerate(candidates):
        norm = extract_prose_candidate(raw_text)
        aime_int = normalize_aime_integer(norm)
        per_branch.append(
            {
                "slot": idx,
                "temp": temp,
                "norm": norm,
                "valid_aime_int": aime_int,
            }
        )
        if aime_int:
            valid_votes.append((idx, float(temp), aime_int))
        elif norm:
            fallback_votes.append((idx, float(temp), norm))

    if valid_votes:
        clusters: dict[str, list[tuple[int, float]]] = {}
        for idx, temp, ans in valid_votes:
            clusters.setdefault(ans, []).append((idx, temp))
        ranked = sorted(
            clusters.items(),
            key=lambda kv: (-len(kv[1]), min(t for _, t in kv[1]), min(i for i, _ in kv[1])),
        )
        winner, members = ranked[0]
        tied_top = len(ranked) > 1 and len(ranked[1][1]) == len(members)
        return winner, {
            "valid_count": len(valid_votes),
            "winner_count": len(members),
            "tie_broken_by_temp": tied_top,
            "vote_counts": {k: len(v) for k, v in ranked},
            "per_branch": per_branch,
            "used_fallback": False,
        }

    if fallback_votes:
        fb_clusters: list[tuple[str, list[tuple[int, float]]]] = []
        for idx, temp, ans in fallback_votes:
            placed = False
            for rep, members in fb_clusters:
                if _answers_equivalent(rep, ans):
                    members.append((idx, temp))
                    placed = True
                    break
            if not placed:
                fb_clusters.append((ans, [(idx, temp)]))
        fb_clusters.sort(
            key=lambda kv: (-len(kv[1]), min(t for _, t in kv[1]), min(i for i, _ in kv[1]))
        )
        winner, members = fb_clusters[0]
        tied_top = len(fb_clusters) > 1 and len(fb_clusters[1][1]) == len(members)
        return winner, {
            "valid_count": 0,
            "winner_count": len(members),
            "tie_broken_by_temp": tied_top,
            "vote_counts": {k: len(v) for k, v in fb_clusters},
            "per_branch": per_branch,
            "used_fallback": True,
        }

    return "", {
        "valid_count": 0,
        "winner_count": 0,
        "tie_broken_by_temp": False,
        "vote_counts": {},
        "per_branch": per_branch,
        "used_fallback": True,
    }


def _make_record(
    item: dict[str, Any],
    predicted: str,
    correct: bool,
    wall_s: float,
    stats: dict[str, int],
    detail: str,
) -> dict[str, Any]:
    prompt_tok = int(stats.get("prompt_tokens", 0))
    comp_tok = int(stats.get("completion_tokens", 0))
    reason_tok = int(stats.get("reasoning_tokens", 0))
    cached_tok = int(stats.get("cached_tokens", 0))
    return {
        "id": item["id"],
        "expected": item["expected"],
        "predicted": predicted,
        "correct": bool(correct),
        "wall_s": round(float(wall_s), 2),
        "llm_calls": int(stats.get("calls", 0)),
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp_tok,
        "reasoning_tokens": reason_tok,
        "cached_tokens": cached_tok,
        "total_tokens": prompt_tok + comp_tok,
        "detail": detail,
    }


async def run_arm_vanilla_b1(client: TensorFoldClient, item: dict[str, Any]) -> dict[str, Any]:
    client.reset_stats()
    t0 = time.perf_counter()
    res = await client.complete(
        CallSpec(
            messages=[{"role": "user", "content": item["prompt"]}],
            tag=f"abl_vanilla_b1_{item['id']}",
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
    wall_s = time.perf_counter() - t0
    predicted = extract_prose_candidate(res.content) if res.ok else ""
    correct, grade_msg = grade_aime(predicted, item["expected"])
    err_note = f" | err={res.raw_error[:100]}" if res.raw_error else ""
    detail = f"vanilla_b1 | finish={res.finish_reason} | {grade_msg}{err_note}"
    return _make_record(item, predicted, correct, wall_s, client.stats_dict(), detail)


async def run_arm_maj4_prose(client: TensorFoldClient, item: dict[str, Any]) -> dict[str, Any]:
    client.reset_stats()
    t0 = time.perf_counter()
    specs = [
        CallSpec(
            messages=[{"role": "user", "content": item["prompt"]}],
            tag=f"abl_maj4_s{i}_{item['id']}",
            max_tokens=4096,
            temperature=temp,
            top_p=0.95,
            top_k=20,
            seed=seed,
            enable_thinking=True,
            thinking_budget=1000,
            loop_guard=True,
        )
        for i, (seed, temp) in enumerate(MAJ4_CONFIGS)
    ]
    results = await client.batch(specs)
    wall_s = time.perf_counter() - t0
    samples = [
        (temp, res.content if res.ok else "")
        for (_, temp), res in zip(MAJ4_CONFIGS, results)
    ]
    predicted, meta = select_maj4_answer(samples)
    correct, grade_msg = grade_aime(predicted, item["expected"])
    votes_str = ", ".join(f"{k}:{v}" for k, v in meta["vote_counts"].items()) or "none"
    tie_note = " (tie->low_temp)" if meta["tie_broken_by_temp"] else ""
    detail = f"maj4_prose ({meta['winner_count']}/4{tie_note}; votes=[{votes_str}]) | {grade_msg}"
    return _make_record(item, predicted, correct, wall_s, client.stats_dict(), detail)


async def run_arm_pal_b1(
    client: TensorFoldClient,
    skills: SkillPack,
    item: dict[str, Any],
) -> dict[str, Any]:
    client.reset_stats()
    t0 = time.perf_counter()
    exemplar_block = skills.match_exemplars("pal_verify", item["prompt"], limit=1)
    user_prefix = f"PROBLEM:\n{item['prompt'].strip()}"
    if exemplar_block:
        user_prefix += f"\n\n{exemplar_block}"
    card_text = skills.card("pal_exact")
    spec = CallSpec(
        messages=[
            {"role": "system", "content": _AIME_SHARED_SYS},
            {"role": "user", "content": f"{user_prefix}\n\nPROTOCOL (pal_exact):\n{card_text}"},
        ],
        tag=f"abl_pal_b1_{item['id']}",
        max_tokens=2500,
        temperature=0.25,
        seed=101,
        enable_thinking=True,
        thinking_budget=500,
        reasoning_effort="low",
        loop_guard=True,
    )
    res = await client.complete(spec)
    out = BranchOutcome(
        name="pal_exact",
        wave=1,
        kind="pal",
        raw_content=res.content,
        reasoning=res.reasoning,
        latency_s=res.latency_s,
        error=res.raw_error or "",
    )
    if res.ok:
        await asyncio.to_thread(_execute_pal_outcome, out, 20.0)
    wall_s = time.perf_counter() - t0
    predicted = out.norm_answer if (res.ok and not out.error) else ""
    correct, grade_msg = grade_aime(predicted, item["expected"])
    if out.error:
        detail = f"pal_b1 | exec_err={out.error[:120]} | {grade_msg}"
    else:
        detail = f"pal_b1 | ok (assert={out.verified_constraints}) | {grade_msg}"
    return _make_record(item, predicted, correct, wall_s, client.stats_dict(), detail)


async def run_arm_cortheon_w1(
    client: TensorFoldClient,
    skills: SkillPack,
    item: dict[str, Any],
) -> dict[str, Any]:
    client.reset_stats()
    t0 = time.perf_counter()
    memory = EpisodicMemory(enabled=False)
    trace = await solve_reasoning(
        client=client,
        skills=skills,
        memory=memory,
        problem=item["prompt"],
        num_wave1_streams=4,
        sandbox_timeout_s=20.0,
        deep=False,
        enable_wave2=False,
    )
    wall_s = time.perf_counter() - t0
    predicted = normalize_answer(trace.final_answer)
    correct, grade_msg = grade_aime(predicted, item["expected"])
    w1_sum = ", ".join(f"{o.name}:{o.norm_answer or 'ERR'}" for o in trace.wave1_outcomes)
    detail = f"{trace.consensus_type} | wave2={trace.wave2_triggered} | w1=[{w1_sum}] | {grade_msg}"
    return _make_record(item, predicted, correct, wall_s, client.stats_dict(), detail)


async def run_arm_cortheon_full(
    client: TensorFoldClient,
    skills: SkillPack,
    item: dict[str, Any],
) -> dict[str, Any]:
    client.reset_stats()
    t0 = time.perf_counter()
    memory = EpisodicMemory(enabled=False)
    trace = await solve_reasoning(
        client=client,
        skills=skills,
        memory=memory,
        problem=item["prompt"],
        num_wave1_streams=4,
        sandbox_timeout_s=20.0,
        deep=False,
        enable_wave2=True,
    )
    wall_s = time.perf_counter() - t0
    predicted = normalize_answer(trace.final_answer)
    correct, grade_msg = grade_aime(predicted, item["expected"])
    w2_note = f" ({trace.wave2_reason[:80]})" if trace.wave2_triggered and trace.wave2_reason else ""
    detail = f"{trace.consensus_type} | wave2={trace.wave2_triggered}{w2_note} | {grade_msg}"
    return _make_record(item, predicted, correct, wall_s, client.stats_dict(), detail)


async def run_single_arm(
    arm: str,
    client: TensorFoldClient,
    skills: SkillPack,
    item: dict[str, Any],
) -> dict[str, Any]:
    if arm == "vanilla_b1":
        return await run_arm_vanilla_b1(client, item)
    if arm == "maj4_prose":
        return await run_arm_maj4_prose(client, item)
    if arm == "pal_b1":
        return await run_arm_pal_b1(client, skills, item)
    if arm == "cortheon_w1":
        return await run_arm_cortheon_w1(client, skills, item)
    if arm == "cortheon_full":
        return await run_arm_cortheon_full(client, skills, item)
    raise ValueError(f"Unknown ablation arm: {arm!r}")


def _id_sort_key(pid: str) -> tuple[int, str]:
    m = re.search(r"(\d+)$", pid)
    return (int(m.group(1)), pid) if m else (10**9, pid)


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    k = sum(1 for r in rows if r.get("correct"))
    lo, hi = wilson_ci(k, n)
    if n == 0:
        return {
            "correct": 0,
            "total": 0,
            "accuracy_pct": 0.0,
            "wilson_ci_95_pct": [0.0, 0.0],
            "avg_wall_s": 0.0,
            "avg_llm_calls": 0.0,
            "avg_prompt_tokens": 0.0,
            "avg_completion_tokens": 0.0,
            "avg_reasoning_tokens": 0.0,
            "avg_cached_tokens": 0.0,
            "avg_total_tokens": 0.0,
            "total_wall_s": 0.0,
            "total_llm_calls": 0,
            "total_tokens": 0,
        }
    tot_wall = sum(float(r.get("wall_s", 0.0)) for r in rows)
    tot_calls = sum(int(r.get("llm_calls", 0)) for r in rows)
    tot_p = sum(int(r.get("prompt_tokens", 0)) for r in rows)
    tot_c = sum(int(r.get("completion_tokens", 0)) for r in rows)
    tot_r = sum(int(r.get("reasoning_tokens", 0)) for r in rows)
    tot_cache = sum(int(r.get("cached_tokens", 0)) for r in rows)
    tot_tok = sum(int(r.get("total_tokens", 0)) for r in rows)
    return {
        "correct": k,
        "total": n,
        "accuracy_pct": round(100.0 * k / n, 2),
        "wilson_ci_95_pct": [round(100.0 * lo, 2), round(100.0 * hi, 2)],
        "avg_wall_s": round(tot_wall / n, 2),
        "avg_llm_calls": round(tot_calls / n, 2),
        "avg_prompt_tokens": round(tot_p / n, 1),
        "avg_completion_tokens": round(tot_c / n, 1),
        "avg_reasoning_tokens": round(tot_r / n, 1),
        "avg_cached_tokens": round(tot_cache / n, 1),
        "avg_total_tokens": round(tot_tok / n, 1),
        "total_wall_s": round(tot_wall, 2),
        "total_llm_calls": tot_calls,
        "total_tokens": tot_tok,
    }


def compute_pairwise_comparisons(
    results_by_arm: dict[str, list[dict[str, Any]]],
    baselines: tuple[str, ...] = ("vanilla_b1", "maj4_prose"),
) -> dict[str, dict[str, Any]]:
    """Compute exact paired McNemar tests + compute ratios of every arm against key baselines."""
    out: dict[str, dict[str, Any]] = {}
    for base_arm in baselines:
        if base_arm not in results_by_arm:
            continue
        base_rows = {r["id"]: r for r in results_by_arm[base_arm]}
        base_pass = {pid: bool(r["correct"]) for pid, r in base_rows.items()}
        for treat_arm, rows in results_by_arm.items():
            if treat_arm == base_arm:
                continue
            treat_rows = {r["id"]: r for r in rows}
            shared = sorted(set(base_rows) & set(treat_rows), key=_id_sort_key)
            if not shared:
                continue
            ct = paired_contingency(
                {pid: base_pass[pid] for pid in shared},
                {pid: bool(treat_rows[pid]["correct"]) for pid in shared},
            )
            n = ct["n"]
            b_wall = sum(float(base_rows[p].get("wall_s", 0.0)) for p in shared) / n
            t_wall = sum(float(treat_rows[p].get("wall_s", 0.0)) for p in shared) / n
            b_calls = sum(int(base_rows[p].get("llm_calls", 0)) for p in shared) / n
            t_calls = sum(int(treat_rows[p].get("llm_calls", 0)) for p in shared) / n
            b_tok = sum(int(base_rows[p].get("total_tokens", 0)) for p in shared) / n
            t_tok = sum(int(treat_rows[p].get("total_tokens", 0)) for p in shared) / n
            out[f"{treat_arm}_vs_{base_arm}"] = {
                "baseline": base_arm,
                "treatment": treat_arm,
                "n": n,
                "baseline_passed": ct["baseline_passed"],
                "treatment_passed": ct["treatment_passed"],
                "delta_pct": round(100.0 * (ct["treatment_passed"] - ct["baseline_passed"]) / n, 2),
                "n11": ct["n11"],
                "n01": ct["n01"],
                "n10": ct["n10"],
                "n00": ct["n00"],
                "mcnemar_exact_p": mcnemar_exact_p(ct["n01"], ct["n10"]),
                "wall_ratio": round(t_wall / b_wall, 3) if b_wall > 0 else None,
                "call_ratio": round(t_calls / b_calls, 3) if b_calls > 0 else None,
                "token_ratio": round(t_tok / b_tok, 3) if b_tok > 0 else None,
            }
    return out


def build_ablation_payload(
    *,
    model: str,
    arms: list[str],
    results_by_arm: dict[str, list[dict[str, Any]]],
    range_info: dict[str, Any],
    merged_from: Optional[list[str]] = None,
) -> dict[str, Any]:
    ordered_results: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, dict[str, Any]] = {}
    for arm in arms:
        rows = sorted(results_by_arm.get(arm, []), key=lambda r: _id_sort_key(r["id"]))
        ordered_results[arm] = rows
        summary[arm] = summarize_arm(rows)

    payload: dict[str, Any] = {
        "dataset": "MathArena/aime_2025",
        "model": model,
        "arms": arms,
        "range": range_info,
        "summary": summary,
        "pairwise": compute_pairwise_comparisons(ordered_results),
        "results": ordered_results,
    }
    if merged_from is not None:
        payload["merged_from"] = merged_from
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def merge_ablation_files(shard_paths: list[Path], out_path: Optional[Path] = None) -> dict[str, Any]:
    """Merge 2+ sharded ablation JSON files cleanly by arm and problem `id`."""
    if not shard_paths:
        raise ValueError("merge_ablation_files requires at least one input file")

    merged_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    seen_arms: list[str] = []
    models: list[str] = []
    ranges: list[Any] = []

    for sp in shard_paths:
        data = json.loads(sp.read_text(encoding="utf-8"))
        m = str(data.get("model", "")).strip()
        if m and m not in models:
            models.append(m)
        if "range" in data:
            ranges.append(data["range"])
        res_map = data.get("results", {})
        if not isinstance(res_map, dict):
            raise ValueError(f"Shard {sp} missing dict 'results'")
        for arm in data.get("arms", list(res_map.keys())):
            if arm not in seen_arms:
                seen_arms.append(arm)
        for arm, rows in res_map.items():
            if arm not in seen_arms:
                seen_arms.append(arm)
            arm_dict = merged_by_arm.setdefault(arm, {})
            for row in rows:
                arm_dict[str(row["id"])] = row

    canonical_arms = [a for a in ALL_ARMS if a in seen_arms] + [a for a in seen_arms if a not in ALL_ARMS]
    results_by_arm = {
        arm: sorted(merged_by_arm.get(arm, {}).values(), key=lambda r: _id_sort_key(r["id"]))
        for arm in canonical_arms
    }

    starts = [r["start"] for r in ranges if isinstance(r, dict) and isinstance(r.get("start"), int)]
    ends = [r["end"] for r in ranges if isinstance(r, dict) and isinstance(r.get("end"), int)]
    range_info: dict[str, Any] = {
        "start": min(starts) if starts else 0,
        "end": max(ends) if ends else max((len(v) for v in results_by_arm.values()), default=0),
        "shards": len(shard_paths),
    }

    model_str = models[0] if len(models) == 1 else (", ".join(models) if models else "unknown")
    payload = build_ablation_payload(
        model=model_str,
        arms=canonical_arms,
        results_by_arm=results_by_arm,
        range_info=range_info,
        merged_from=[p.name for p in shard_paths],
    )
    if out_path is not None:
        _write_json_atomic(out_path, payload)
    return payload


def format_ablation_table(payload: dict[str, Any]) -> str:
    arms = payload.get("arms", [])
    summary = payload.get("summary", {})
    pairwise = payload.get("pairwise", {})
    lines = [
        f"### Official AIME 2025 5-Arm Ablation (`{payload.get('model', 'unknown')}`)",
        "",
        "| Arm | Accuracy k/n [95% Wilson CI] | Avg Wall | Avg Calls | Avg Total Tok | Avg Reason Tok | McNemar p vs `vanilla_b1` | McNemar p vs `maj4_prose` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        s = summary.get(arm, {})
        k, n = s.get("correct", 0), s.get("total", 0)
        acc = s.get("accuracy_pct", 0.0)
        ci = s.get("wilson_ci_95_pct", [0.0, 0.0])
        p_van = pairwise.get(f"{arm}_vs_vanilla_b1", {}).get("mcnemar_exact_p")
        p_maj = pairwise.get(f"{arm}_vs_maj4_prose", {}).get("mcnemar_exact_p")
        p_van_str = "—" if p_van is None else format_p_value(p_van)
        p_maj_str = "—" if p_maj is None else format_p_value(p_maj)
        lines.append(
            f"| `{arm}` | {k}/{n} ({acc:5.1f}%) [{ci[0]:4.1f}, {ci[1]:4.1f}] "
            f"| {s.get('avg_wall_s', 0.0):6.1f}s | {s.get('avg_llm_calls', 0.0):5.2f} "
            f"| {s.get('avg_total_tokens', 0.0):8.0f} | {s.get('avg_reasoning_tokens', 0.0):8.0f} "
            f"| {p_van_str} | {p_maj_str} |"
        )
    return "\n".join(lines)


async def run_ablation(
    *,
    base_url: str,
    dataset_path: Path,
    arms: list[str],
    start: int,
    end: int,
    out_path: Path,
    resume: bool = False,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    items = load_aime2025(dataset_path, start=start, end=end)
    if not items:
        raise RuntimeError(f"No AIME 2025 items selected for range [{start}, {end}) from {dataset_path}")

    by_arm_id: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in arms}
    if resume and out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        for arm in arms:
            for row in prev.get("results", {}).get(arm, []):
                by_arm_id[arm][str(row["id"])] = row
        loaded_n = sum(len(v) for v in by_arm_id.values())
        print(f"Resumed {loaded_n} completed arm-problem rows from {out_path}")

    skills = SkillPack()
    t_all = time.perf_counter()

    async with TensorFoldClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_parallel=8,
    ) as client:
        model_id = client.caps.model_id
        print(
            f"=== AIME 2025 ABLATION | model={model_id} | range=[{start},{end}) ({len(items)} problems) "
            f"| arms={','.join(arms)} ==="
        )
        sys.stdout.flush()

        for p_idx, item in enumerate(items, 1):
            for arm in arms:
                if item["id"] in by_arm_id[arm]:
                    continue
                try:
                    row = await run_single_arm(arm, client, skills, item)
                except Exception as exc:
                    row = _make_record(
                        item,
                        predicted="",
                        correct=False,
                        wall_s=0.0,
                        stats=client.stats_dict(),
                        detail=f"EXCEPTION: {type(exc).__name__}: {exc}",
                    )
                by_arm_id[arm][item["id"]] = row
                mark = "PASS" if row["correct"] else "FAIL"
                print(
                    f"[{p_idx:02d}/{len(items):02d}] {item['id']} | {arm:<13} | {mark:4s} "
                    f"| pred={row['predicted'] or '-':<4} exp={row['expected']:<4} "
                    f"| {row['wall_s']:5.1f}s | calls={row['llm_calls']:2d} tok={row['total_tokens']:6d} "
                    f"| {row['detail'][:70]}"
                )
                sys.stdout.flush()

                payload = build_ablation_payload(
                    model=model_id,
                    arms=arms,
                    results_by_arm={a: list(by_arm_id[a].values()) for a in arms},
                    range_info={"start": start, "end": end},
                )
                _write_json_atomic(out_path, payload)

        final_payload = build_ablation_payload(
            model=model_id,
            arms=arms,
            results_by_arm={a: list(by_arm_id[a].values()) for a in arms},
            range_info={"start": start, "end": end},
        )
        _write_json_atomic(out_path, final_payload)

    print(f"\nCompleted in {time.perf_counter() - t_all:.1f}s -> {out_path}\n")
    print(format_ablation_table(final_payload))
    return final_payload


def _parse_arms(raw: str) -> list[str]:
    arms = [a.strip() for a in raw.split(",") if a.strip()]
    if not arms:
        raise argparse.ArgumentTypeError("At least one arm must be specified")
    invalid = [a for a in arms if a not in ALL_ARMS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown arm(s): {invalid}. Valid arms: {', '.join(ALL_ARMS)}"
        )
    # Deduplicate while preserving order
    return list(dict.fromkeys(arms))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="5-arm compute-matched AIME 2025 ablation runner with shard merge support."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8080/v1",
        help="OpenAI-compatible / TensorFold server base URL (default: http://127.0.0.1:8080/v1).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("bench/datasets/aime_2025.json"),
        help="Path to MathArena/aime_2025 JSON (default: bench/datasets/aime_2025.json).",
    )
    parser.add_argument(
        "--arms",
        type=_parse_arms,
        default=list(ALL_ARMS),
        help="Comma-separated list of arms (default: vanilla_b1,maj4_prose,pal_b1,cortheon_w1,cortheon_full).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start problem index inclusive (0-based, default: 0).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=30,
        help="End problem index exclusive (default: 30).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path (for run results or merged shard output).",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        default=None,
        metavar="SHARD_JSON",
        help="Merge 2+ sharded ablation JSON files into --out and print combined statistics.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing --out checkpoint file, skipping completed (arm, id) pairs.",
    )
    parser.add_argument("--model", default=None, help="Optional explicit model ID override.")
    parser.add_argument("--api-key", default=None, help="Optional API bearer key.")
    args = parser.parse_args(argv)

    if args.merge:
        merged = merge_ablation_files(args.merge, out_path=args.out)
        print(f"Merged {len(args.merge)} shard(s) -> {args.out}\n")
        print(format_ablation_table(merged))
        return 0

    asyncio.run(
        run_ablation(
            base_url=args.url,
            dataset_path=args.dataset,
            arms=args.arms,
            start=args.start,
            end=args.end,
            out_path=args.out,
            resume=args.resume,
            model=args.model,
            api_key=args.api_key,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

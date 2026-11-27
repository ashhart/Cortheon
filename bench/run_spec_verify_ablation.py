"""3-Arm Speculative Execution-Verification (`SpecVerify`) Ablation Runner on Official MathArena/aime_2025.

Compares three arms on the exact same problem set:
  1. vanilla_big  : Single prose call on the big verifier model (B=1, temp=0.6, seed=42, tb=1500, max_tokens=8192).
  2. cortheon_big : Full ReasonForge (`solve_reasoning`) on the big verifier model.
  3. spec_verify  : `solve_spec_verify(drafter, verifier, ...)` — parallel sandbox-verified
                    scripts on a tiny fast drafter model, returning immediately on mechanical
                    consensus (0 big-model calls/tokens) and escalating ONLY on falsification.

Sharding & Merging:
    python -m bench.run_spec_verify_ablation \
        --drafter-url http://127.0.0.1:8081/v1 \
        --verifier-url http://127.0.0.1:8080/v1 \
        --start 0 --end 15 --out bench/results/spec_verify_s1.json

    python -m bench.run_spec_verify_ablation \
        --merge bench/results/spec_verify_s1.json bench/results/spec_verify_s2.json \
        --out bench/results/spec_verify_30.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Optional

from flash_forge.client import CallSpec, TensorFoldClient
from flash_forge.memory import EpisodicMemory
from flash_forge.reason_forge import normalize_answer, solve_reasoning
from flash_forge.skill_loader import SkillPack
from flash_forge.spec_verify import solve_spec_verify

from .run_aime_ablation import (
    extract_prose_candidate,
    grade_aime,
    load_aime2025,
    scrub_detail_paths,
)
from .stats import format_p_value, mcnemar_exact_p, paired_contingency, wilson_ci

ALL_ARMS: tuple[str, ...] = (
    "vanilla_big",
    "cortheon_big",
    "spec_verify",
)

_SECRET_RE = re.compile(
    r"(?:Bearer\s+[A-Za-z0-9._\-]{8,}|\b(?:sk-|sk_|tf-|tf_|ghp_|hf_|xoxb-|AIza)[A-Za-z0-9_\-]{8,})"
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def scrub_pii_text(text: str) -> str:
    """Scrub temp paths, host home directories, emails, and bearer/API tokens."""
    if not text:
        return text
    cleaned = scrub_detail_paths(text)
    cleaned = _SECRET_RE.sub("<REDACTED_SECRET>", cleaned)
    cleaned = _EMAIL_RE.sub("<REDACTED_EMAIL>", cleaned)
    return cleaned


def _scrub_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("detail", "escalation_reason"):
        val = out.get(key)
        if isinstance(val, str):
            out[key] = scrub_pii_text(val)
    gv = out.get("drafter_graveyard")
    if isinstance(gv, list):
        out["drafter_graveyard"] = [scrub_pii_text(str(x)) for x in gv]
    return out


def _id_sort_key(pid: str) -> tuple[int, str]:
    m = re.search(r"(\d+)$", pid)
    return (int(m.group(1)), pid) if m else (10**9, pid)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _stats_total_tokens(stats: dict[str, int]) -> int:
    return int(stats.get("prompt_tokens", 0)) + int(stats.get("completion_tokens", 0))


def _make_big_record(
    item: dict[str, Any],
    predicted: str,
    correct: bool,
    wall_s: float,
    stats: dict[str, int],
    detail: str,
    *,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    prompt_tok = int(stats.get("prompt_tokens", 0))
    comp_tok = int(stats.get("completion_tokens", 0))
    reason_tok = int(stats.get("reasoning_tokens", 0))
    cached_tok = int(stats.get("cached_tokens", 0))
    calls = int(stats.get("calls", 0))
    total_tok = prompt_tok + comp_tok
    row: dict[str, Any] = {
        "id": item["id"],
        "expected": item["expected"],
        "predicted": predicted,
        "correct": bool(correct),
        "wall_s": round(float(wall_s), 2),
        "llm_calls": calls,
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp_tok,
        "reasoning_tokens": reason_tok,
        "cached_tokens": cached_tok,
        "total_tokens": total_tok,
        "drafter_calls": 0,
        "drafter_tokens": 0,
        "verifier_calls": calls,
        "verifier_tokens": total_tok,
        "detail": scrub_pii_text(detail),
    }
    if extra:
        row.update(extra)
    return row


async def run_arm_vanilla_big(
    verifier_client: TensorFoldClient,
    item: dict[str, Any],
) -> dict[str, Any]:
    verifier_client.reset_stats()
    t0 = time.perf_counter()
    res = await verifier_client.complete(
        CallSpec(
            messages=[{"role": "user", "content": item["prompt"]}],
            tag=f"sv_vanilla_big_{item['id']}",
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
    err_note = f" | err={scrub_pii_text(res.raw_error)[:100]}" if res.raw_error else ""
    detail = f"vanilla_big | finish={res.finish_reason} | {grade_msg}{err_note}"
    return _make_big_record(item, predicted, correct, wall_s, verifier_client.stats_dict(), detail)


async def run_arm_cortheon_big(
    verifier_client: TensorFoldClient,
    skills: SkillPack,
    item: dict[str, Any],
    *,
    verifier_streams: int = 4,
    sandbox_timeout_s: float = 20.0,
) -> dict[str, Any]:
    verifier_client.reset_stats()
    t0 = time.perf_counter()
    memory = EpisodicMemory(enabled=False)
    trace = await solve_reasoning(
        client=verifier_client,
        skills=skills,
        memory=memory,
        problem=item["prompt"],
        num_wave1_streams=verifier_streams,
        sandbox_timeout_s=sandbox_timeout_s,
        deep=False,
        enable_wave2=True,
    )
    wall_s = time.perf_counter() - t0
    predicted = normalize_answer(trace.final_answer)
    correct, grade_msg = grade_aime(predicted, item["expected"])
    w2_reason = scrub_pii_text(trace.wave2_reason)[:80] if trace.wave2_reason else ""
    w2_note = f" ({w2_reason})" if trace.wave2_triggered and w2_reason else ""
    detail = f"{trace.consensus_type} | wave2={trace.wave2_triggered}{w2_note} | {grade_msg}"
    return _make_big_record(
        item,
        predicted,
        correct,
        wall_s,
        verifier_client.stats_dict(),
        detail,
        extra={"wave2_triggered": bool(trace.wave2_triggered)},
    )


async def run_arm_spec_verify(
    drafter_client: TensorFoldClient,
    verifier_client: TensorFoldClient,
    skills: SkillPack,
    item: dict[str, Any],
    *,
    drafter_streams: int = 3,
    verifier_streams: int = 4,
    min_consensus: int = 2,
    sandbox_timeout_s: float = 15.0,
    baseline_big_tokens: Optional[int] = None,
) -> dict[str, Any]:
    drafter_client.reset_stats()
    verifier_client.reset_stats()
    memory = EpisodicMemory(enabled=False)
    trace = await solve_spec_verify(
        drafter_client=drafter_client,
        verifier_client=verifier_client,
        skills=skills,
        memory=memory,
        problem=item["prompt"],
        drafter_streams=drafter_streams,
        verifier_streams=verifier_streams,
        min_consensus=min_consensus,
        sandbox_timeout_s=sandbox_timeout_s,
    )
    predicted = normalize_answer(trace.answer)
    correct, grade_msg = grade_aime(predicted, item["expected"])

    big_saved_pct: Optional[float]
    if baseline_big_tokens is not None and baseline_big_tokens > 0:
        big_saved_pct = round(100.0 * (1.0 - trace.verifier_tokens / baseline_big_tokens), 2)
    elif trace.accepted_by_drafter and trace.verifier_tokens == 0:
        big_saved_pct = 100.0
    else:
        big_saved_pct = None

    status_tag = (
        f"DRAFT_ACCEPT({trace.drafter_consensus_count}/{trace.drafter_verified_scripts})"
        if trace.accepted_by_drafter
        else f"ESCALATED({scrub_pii_text(trace.escalation_reason)[:70]})"
    )
    detail = f"spec_verify | {status_tag} | {grade_msg}"
    return _scrub_row(
        {
            "id": item["id"],
            "expected": item["expected"],
            "predicted": predicted,
            "correct": bool(correct),
            "accepted_by_drafter": bool(trace.accepted_by_drafter),
            "escalated": bool(trace.escalated),
            "escalation_reason": scrub_pii_text(trace.escalation_reason),
            "drafter_consensus_count": int(trace.drafter_consensus_count),
            "drafter_verified_scripts": int(trace.drafter_verified_scripts),
            "drafter_wall_s": round(float(trace.drafter_wall_s), 2),
            "verifier_wall_s": round(float(trace.verifier_wall_s), 2),
            "wall_s": round(float(trace.total_wall_s), 2),
            "drafter_calls": int(trace.drafter_calls),
            "drafter_tokens": int(trace.drafter_tokens),
            "verifier_calls": int(trace.verifier_calls),
            "verifier_tokens": int(trace.verifier_tokens),
            "llm_calls": int(trace.drafter_calls + trace.verifier_calls),
            "total_tokens": int(trace.drafter_tokens + trace.verifier_tokens),
            "big_model_tokens_saved_pct": big_saved_pct,
            "drafter_graveyard": [scrub_pii_text(g) for g in trace.drafter_graveyard],
            "detail": scrub_pii_text(detail),
        }
    )


def _row_big_calls(row: dict[str, Any]) -> int:
    if "verifier_calls" in row and isinstance(row["verifier_calls"], (int, float)):
        return int(row["verifier_calls"])
    return int(row.get("llm_calls", 0))


def _row_big_tokens(row: dict[str, Any]) -> int:
    if "verifier_tokens" in row and isinstance(row["verifier_tokens"], (int, float)):
        return int(row["verifier_tokens"])
    return int(row.get("total_tokens", 0))


def summarize_spec_verify_arm(arm: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    k = sum(1 for r in rows if r.get("correct"))
    lo, hi = wilson_ci(k, n)
    if n == 0:
        out: dict[str, Any] = {
            "correct": 0,
            "total": 0,
            "accuracy_pct": 0.0,
            "wilson_ci_95_pct": [0.0, 0.0],
            "avg_wall_s": 0.0,
            "total_wall_s": 0.0,
            "avg_llm_calls": 0.0,
            "total_llm_calls": 0,
            "avg_total_tokens": 0.0,
            "total_tokens": 0,
            "avg_drafter_calls": 0.0,
            "total_drafter_calls": 0,
            "avg_drafter_tokens": 0.0,
            "total_drafter_tokens": 0,
            "avg_verifier_calls": 0.0,
            "total_verifier_calls": 0,
            "avg_verifier_tokens": 0.0,
            "total_verifier_tokens": 0,
        }
        if arm == "spec_verify":
            out.update(
                {
                    "drafter_accepted": 0,
                    "drafter_accept_rate_pct": 0.0,
                    "drafter_accepted_correct": 0,
                    "drafter_accepted_accuracy_pct": None,
                    "escalated": 0,
                    "escalation_rate_pct": 0.0,
                    "avg_drafter_wall_s": 0.0,
                    "avg_verifier_wall_s": 0.0,
                }
            )
        return out

    tot_wall = sum(float(r.get("wall_s", 0.0)) for r in rows)
    tot_calls = sum(int(r.get("llm_calls", 0)) for r in rows)
    tot_tok = sum(int(r.get("total_tokens", 0)) for r in rows)
    tot_d_calls = sum(int(r.get("drafter_calls", 0)) for r in rows)
    tot_d_tok = sum(int(r.get("drafter_tokens", 0)) for r in rows)
    tot_v_calls = sum(_row_big_calls(r) for r in rows)
    tot_v_tok = sum(_row_big_tokens(r) for r in rows)

    summary: dict[str, Any] = {
        "correct": k,
        "total": n,
        "accuracy_pct": round(100.0 * k / n, 2),
        "wilson_ci_95_pct": [round(100.0 * lo, 2), round(100.0 * hi, 2)],
        "avg_wall_s": round(tot_wall / n, 2),
        "total_wall_s": round(tot_wall, 2),
        "avg_llm_calls": round(tot_calls / n, 2),
        "total_llm_calls": tot_calls,
        "avg_total_tokens": round(tot_tok / n, 1),
        "total_tokens": tot_tok,
        "avg_drafter_calls": round(tot_d_calls / n, 2),
        "total_drafter_calls": tot_d_calls,
        "avg_drafter_tokens": round(tot_d_tok / n, 1),
        "total_drafter_tokens": tot_d_tok,
        "avg_verifier_calls": round(tot_v_calls / n, 2),
        "total_verifier_calls": tot_v_calls,
        "avg_verifier_tokens": round(tot_v_tok / n, 1),
        "total_verifier_tokens": tot_v_tok,
    }

    if arm == "spec_verify" or any("accepted_by_drafter" in r for r in rows):
        accepted_rows = [r for r in rows if r.get("accepted_by_drafter")]
        acc_n = len(accepted_rows)
        acc_ok = sum(1 for r in accepted_rows if r.get("correct"))
        esc_n = sum(1 for r in rows if r.get("escalated"))
        tot_d_wall = sum(float(r.get("drafter_wall_s", 0.0)) for r in rows)
        tot_v_wall = sum(float(r.get("verifier_wall_s", 0.0)) for r in rows)
        summary.update(
            {
                "drafter_accepted": acc_n,
                "drafter_accept_rate_pct": round(100.0 * acc_n / n, 2),
                "drafter_accepted_correct": acc_ok,
                "drafter_accepted_accuracy_pct": round(100.0 * acc_ok / acc_n, 2) if acc_n > 0 else None,
                "escalated": esc_n,
                "escalation_rate_pct": round(100.0 * esc_n / n, 2),
                "avg_drafter_wall_s": round(tot_d_wall / n, 2),
                "avg_verifier_wall_s": round(tot_v_wall / n, 2),
            }
        )
    return summary


def compute_spec_verify_pairwise(
    results_by_arm: dict[str, list[dict[str, Any]]],
    baselines: tuple[str, ...] = ("vanilla_big", "cortheon_big"),
) -> dict[str, dict[str, Any]]:
    """Compute exact paired McNemar p-values + total and big-model call/token reductions."""
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
            b_wall = sum(float(base_rows[p].get("wall_s", 0.0)) for p in shared)
            t_wall = sum(float(treat_rows[p].get("wall_s", 0.0)) for p in shared)
            b_calls = sum(int(base_rows[p].get("llm_calls", 0)) for p in shared)
            t_calls = sum(int(treat_rows[p].get("llm_calls", 0)) for p in shared)
            b_tok = sum(int(base_rows[p].get("total_tokens", 0)) for p in shared)
            t_tok = sum(int(treat_rows[p].get("total_tokens", 0)) for p in shared)
            b_big_calls = sum(_row_big_calls(base_rows[p]) for p in shared)
            t_big_calls = sum(_row_big_calls(treat_rows[p]) for p in shared)
            b_big_tok = sum(_row_big_tokens(base_rows[p]) for p in shared)
            t_big_tok = sum(_row_big_tokens(treat_rows[p]) for p in shared)

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
                "big_model_call_ratio": round(t_big_calls / b_big_calls, 3) if b_big_calls > 0 else None,
                "big_model_token_ratio": round(t_big_tok / b_big_tok, 3) if b_big_tok > 0 else None,
                "big_model_calls_saved_pct": (
                    round(100.0 * (1.0 - t_big_calls / b_big_calls), 2) if b_big_calls > 0 else None
                ),
                "big_model_tokens_saved_pct": (
                    round(100.0 * (1.0 - t_big_tok / b_big_tok), 2) if b_big_tok > 0 else None
                ),
            }
    return out


def _annotate_spec_verify_savings(ordered_results: dict[str, list[dict[str, Any]]]) -> None:
    """Populate per-row `big_model_tokens_saved_pct` against paired `cortheon_big` (or `vanilla_big`)."""
    sv_rows = ordered_results.get("spec_verify")
    if not sv_rows:
        return
    cortheon_by_id = {r["id"]: r for r in ordered_results.get("cortheon_big", [])}
    vanilla_by_id = {r["id"]: r for r in ordered_results.get("vanilla_big", [])}

    for row in sv_rows:
        pid = row["id"]
        v_tok = _row_big_tokens(row)
        base_row = cortheon_by_id.get(pid) or vanilla_by_id.get(pid)
        if base_row is not None:
            base_tok = _row_big_tokens(base_row)
            if base_tok > 0:
                row["big_model_tokens_saved_pct"] = round(100.0 * (1.0 - v_tok / base_tok), 2)
                continue
        if row.get("accepted_by_drafter") and v_tok == 0:
            row["big_model_tokens_saved_pct"] = 100.0
        elif "big_model_tokens_saved_pct" not in row:
            row["big_model_tokens_saved_pct"] = None


def build_spec_verify_payload(
    *,
    drafter_model: str,
    verifier_model: str,
    arms: list[str],
    results_by_arm: dict[str, list[dict[str, Any]]],
    range_info: dict[str, Any],
    merged_from: Optional[list[str]] = None,
) -> dict[str, Any]:
    ordered_results: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        ordered_results[arm] = [
            _scrub_row(r)
            for r in sorted(results_by_arm.get(arm, []), key=lambda r: _id_sort_key(r["id"]))
        ]

    _annotate_spec_verify_savings(ordered_results)

    summary: dict[str, dict[str, Any]] = {
        arm: summarize_spec_verify_arm(arm, ordered_results[arm]) for arm in arms
    }
    pairwise = compute_spec_verify_pairwise(ordered_results)

    if "spec_verify" in summary:
        sv_sum = summary["spec_verify"]
        vs_c = pairwise.get("spec_verify_vs_cortheon_big")
        vs_v = pairwise.get("spec_verify_vs_vanilla_big")
        sv_sum["big_model_calls_saved_vs_cortheon_big_pct"] = (
            vs_c["big_model_calls_saved_pct"] if vs_c else None
        )
        sv_sum["big_model_tokens_saved_vs_cortheon_big_pct"] = (
            vs_c["big_model_tokens_saved_pct"] if vs_c else None
        )
        sv_sum["big_model_calls_saved_vs_vanilla_big_pct"] = (
            vs_v["big_model_calls_saved_pct"] if vs_v else None
        )
        sv_sum["big_model_tokens_saved_vs_vanilla_big_pct"] = (
            vs_v["big_model_tokens_saved_pct"] if vs_v else None
        )
        if vs_c is not None and vs_c["big_model_tokens_saved_pct"] is not None:
            sv_sum["big_model_tokens_saved_pct"] = vs_c["big_model_tokens_saved_pct"]
            sv_sum["big_model_calls_saved_pct"] = vs_c["big_model_calls_saved_pct"]
        elif vs_v is not None and vs_v["big_model_tokens_saved_pct"] is not None:
            sv_sum["big_model_tokens_saved_pct"] = vs_v["big_model_tokens_saved_pct"]
            sv_sum["big_model_calls_saved_pct"] = vs_v["big_model_calls_saved_pct"]
        elif sv_sum["total"] > 0 and sv_sum["total_verifier_tokens"] == 0:
            sv_sum["big_model_tokens_saved_pct"] = 100.0
            sv_sum["big_model_calls_saved_pct"] = 100.0
        else:
            sv_sum["big_model_tokens_saved_pct"] = None
            sv_sum["big_model_calls_saved_pct"] = None

    payload: dict[str, Any] = {
        "dataset": "MathArena/aime_2025",
        "drafter_model": scrub_pii_text(drafter_model or "unknown"),
        "verifier_model": scrub_pii_text(verifier_model or "unknown"),
        "arms": arms,
        "range": range_info,
        "summary": summary,
        "pairwise": pairwise,
        "results": ordered_results,
    }
    if merged_from is not None:
        payload["merged_from"] = merged_from
    return payload


def merge_spec_verify_files(
    shard_paths: list[Path],
    out_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Merge 2+ SpecVerify ablation JSON files across disjoint problem ranges AND/OR disjoint arms."""
    if not shard_paths:
        raise ValueError("merge_spec_verify_files requires at least one input file")

    merged_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    seen_arms: list[str] = []
    drafter_models: list[str] = []
    verifier_models: list[str] = []
    ranges: list[Any] = []

    for sp in shard_paths:
        data = json.loads(sp.read_text(encoding="utf-8"))
        dm = str(data.get("drafter_model", "")).strip()
        if dm and dm.lower() != "unknown" and dm not in drafter_models:
            drafter_models.append(dm)
        vm = str(data.get("verifier_model", data.get("model", ""))).strip()
        if vm and vm.lower() != "unknown" and vm not in verifier_models:
            verifier_models.append(vm)
        if "range" in data:
            ranges.append(data["range"])
        res_map = data.get("results", {})
        if not isinstance(res_map, dict):
            raise ValueError(f"Shard {sp} missing dict 'results'")
        for arm in data.get("arms", list(res_map.keys())):
            if arm not in seen_arms:
                seen_arms.append(arm)
        for arm, rows in res_map.items():
            if not isinstance(rows, list):
                raise ValueError(f"Shard {sp} arm {arm!r} must map to a list of rows")
            if arm not in seen_arms:
                seen_arms.append(arm)
            arm_dict = merged_by_arm.setdefault(arm, {})
            for row in rows:
                arm_dict[str(row["id"])] = _scrub_row(row)

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

    drafter_str = (
        drafter_models[0]
        if len(drafter_models) == 1
        else (", ".join(drafter_models) if drafter_models else "unknown")
    )
    verifier_str = (
        verifier_models[0]
        if len(verifier_models) == 1
        else (", ".join(verifier_models) if verifier_models else "unknown")
    )

    payload = build_spec_verify_payload(
        drafter_model=drafter_str,
        verifier_model=verifier_str,
        arms=canonical_arms,
        results_by_arm=results_by_arm,
        range_info=range_info,
        merged_from=[p.name for p in shard_paths],
    )
    if out_path is not None:
        _write_json_atomic(out_path, payload)
    return payload


def _fmt_pct_opt(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{val:+.1f}%" if val < 0 else f"{val:.1f}%"


def format_spec_verify_table(payload: dict[str, Any]) -> str:
    arms = payload.get("arms", [])
    summary = payload.get("summary", {})
    pairwise = payload.get("pairwise", {})
    drafter_m = payload.get("drafter_model", "unknown")
    verifier_m = payload.get("verifier_model", "unknown")

    header_cols = [
        "Arm",
        "Accuracy k/n [95% Wilson CI]",
        "Drafter Accept",
        "Avg Wall",
        "Avg Big Calls",
        "Avg Big Tok",
        "Big Call Saved",
        "Big Tok Saved",
        "McNemar p vs `vanilla_big`",
        "McNemar p vs `cortheon_big`",
    ]
    align_cols = ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]
    lines = [
        f"### Official AIME 2025 SpecVerify Ablation (Drafter: `{drafter_m}` -> Verifier: `{verifier_m}`)",
        "",
        "| " + " | ".join(header_cols) + " |",
        "|" + "|".join(align_cols) + "|",
    ]

    for arm in arms:
        s = summary.get(arm, {})
        k, n = s.get("correct", 0), s.get("total", 0)
        acc = s.get("accuracy_pct", 0.0)
        ci = s.get("wilson_ci_95_pct", [0.0, 0.0])

        if arm == "spec_verify":
            acc_cnt = s.get("drafter_accepted", 0)
            acc_rate = s.get("drafter_accept_rate_pct", 0.0)
            accept_str = f"{acc_cnt}/{n} ({acc_rate:5.1f}%)"
            call_saved_str = _fmt_pct_opt(s.get("big_model_calls_saved_pct"))
            tok_saved_str = _fmt_pct_opt(s.get("big_model_tokens_saved_pct"))
        else:
            accept_str = "—"
            call_saved_str = "—"
            tok_saved_str = "—"

        p_van = pairwise.get(f"{arm}_vs_vanilla_big", {}).get("mcnemar_exact_p")
        p_cor = pairwise.get(f"{arm}_vs_cortheon_big", {}).get("mcnemar_exact_p")
        p_van_str = "—" if p_van is None else format_p_value(p_van)
        p_cor_str = "—" if p_cor is None else format_p_value(p_cor)

        lines.append(
            f"| `{arm}` | {k}/{n} ({acc:5.1f}%) [{ci[0]:4.1f}, {ci[1]:4.1f}] "
            f"| {accept_str} "
            f"| {s.get('avg_wall_s', 0.0):6.1f}s "
            f"| {s.get('avg_verifier_calls', 0.0):5.2f} "
            f"| {s.get('avg_verifier_tokens', 0.0):8.0f} "
            f"| {call_saved_str} "
            f"| {tok_saved_str} "
            f"| {p_van_str} "
            f"| {p_cor_str} |"
        )

    sv_s = summary.get("spec_verify")
    if sv_s and sv_s.get("total", 0) > 0:
        acc_rate = sv_s.get("drafter_accept_rate_pct", 0.0)
        acc_prec = sv_s.get("drafter_accepted_accuracy_pct")
        prec_str = "—" if acc_prec is None else f"{acc_prec:.1f}%"
        vs_c = pairwise.get("spec_verify_vs_cortheon_big")
        vs_v = pairwise.get("spec_verify_vs_vanilla_big")
        notes: list[str] = [
            f"Drafter acceptance: **{sv_s.get('drafter_accepted', 0)}/{sv_s['total']} ({acc_rate:.1f}%)**",
            f"Accepted precision: **{prec_str}**",
        ]
        if vs_c:
            notes.append(
                f"vs `cortheon_big`: big calls **{_fmt_pct_opt(vs_c.get('big_model_calls_saved_pct'))}**, "
                f"big tokens **{_fmt_pct_opt(vs_c.get('big_model_tokens_saved_pct'))}**"
            )
        if vs_v:
            notes.append(
                f"vs `vanilla_big`: big calls **{_fmt_pct_opt(vs_v.get('big_model_calls_saved_pct'))}**, "
                f"big tokens **{_fmt_pct_opt(vs_v.get('big_model_tokens_saved_pct'))}**"
            )
        lines.append("")
        lines.append("_SpecVerify Summary: " + " · ".join(notes) + "._")

    return "\n".join(lines)


async def run_spec_verify_ablation(
    *,
    drafter_url: str,
    verifier_url: str,
    dataset_path: Path,
    arms: list[str],
    start: int,
    end: int,
    out_path: Path,
    resume: bool = False,
    drafter_model: Optional[str] = None,
    verifier_model: Optional[str] = None,
    drafter_api_key: Optional[str] = None,
    verifier_api_key: Optional[str] = None,
    drafter_streams: int = 3,
    verifier_streams: int = 4,
    min_consensus: int = 2,
    sandbox_timeout_s: float = 15.0,
) -> dict[str, Any]:
    items = load_aime2025(dataset_path, start=start, end=end)
    if not items:
        raise RuntimeError(f"No AIME 2025 items selected for range [{start}, {end}) from {dataset_path}")

    by_arm_id: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in arms}
    if resume and out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        for arm in arms:
            for row in prev.get("results", {}).get(arm, []):
                by_arm_id[arm][str(row["id"])] = _scrub_row(row)
        loaded_n = sum(len(v) for v in by_arm_id.values())
        print(f"Resumed {loaded_n} completed arm-problem rows from {out_path}")

    skills = SkillPack()
    t_all = time.perf_counter()

    async with AsyncExitStack() as stack:
        verifier_client = await stack.enter_async_context(
            TensorFoldClient(
                base_url=verifier_url,
                model=verifier_model,
                api_key=verifier_api_key,
                max_parallel=8,
            )
        )
        verifier_model_id = verifier_client.caps.model_id

        drafter_client: Optional[TensorFoldClient] = None
        drafter_model_id = drafter_model or "unused"
        if "spec_verify" in arms:
            drafter_client = await stack.enter_async_context(
                TensorFoldClient(
                    base_url=drafter_url,
                    model=drafter_model,
                    api_key=drafter_api_key,
                    max_parallel=8,
                )
            )
            drafter_model_id = drafter_client.caps.model_id

        print(
            f"=== AIME 2025 SPECVERIFY ABLATION | drafter={drafter_model_id} | verifier={verifier_model_id} "
            f"| range=[{start},{end}) ({len(items)} problems) | arms={','.join(arms)} ==="
        )
        sys.stdout.flush()

        for p_idx, item in enumerate(items, 1):
            for arm in arms:
                if item["id"] in by_arm_id[arm]:
                    continue
                try:
                    if arm == "vanilla_big":
                        row = await run_arm_vanilla_big(verifier_client, item)
                    elif arm == "cortheon_big":
                        row = await run_arm_cortheon_big(
                            verifier_client,
                            skills,
                            item,
                            verifier_streams=verifier_streams,
                        )
                    elif arm == "spec_verify":
                        assert drafter_client is not None
                        cb_row = by_arm_id.get("cortheon_big", {}).get(item["id"])
                        base_tok = _row_big_tokens(cb_row) if cb_row else None
                        row = await run_arm_spec_verify(
                            drafter_client,
                            verifier_client,
                            skills,
                            item,
                            drafter_streams=drafter_streams,
                            verifier_streams=verifier_streams,
                            min_consensus=min_consensus,
                            sandbox_timeout_s=sandbox_timeout_s,
                            baseline_big_tokens=base_tok,
                        )
                    else:
                        raise ValueError(f"Unknown arm: {arm!r}")
                except Exception as exc:
                    err_detail = scrub_pii_text(f"EXCEPTION: {type(exc).__name__}: {exc}")
                    if arm == "spec_verify":
                        row = _scrub_row(
                            {
                                "id": item["id"],
                                "expected": item["expected"],
                                "predicted": "",
                                "correct": False,
                                "accepted_by_drafter": False,
                                "escalated": False,
                                "escalation_reason": err_detail,
                                "drafter_consensus_count": 0,
                                "drafter_verified_scripts": 0,
                                "drafter_wall_s": 0.0,
                                "verifier_wall_s": 0.0,
                                "wall_s": 0.0,
                                "drafter_calls": 0,
                                "drafter_tokens": 0,
                                "verifier_calls": 0,
                                "verifier_tokens": 0,
                                "llm_calls": 0,
                                "total_tokens": 0,
                                "big_model_tokens_saved_pct": None,
                                "drafter_graveyard": [],
                                "detail": err_detail,
                            }
                        )
                    else:
                        row = _make_big_record(
                            item,
                            predicted="",
                            correct=False,
                            wall_s=0.0,
                            stats=verifier_client.stats_dict(),
                            detail=err_detail,
                        )

                by_arm_id[arm][item["id"]] = row
                mark = "PASS" if row["correct"] else "FAIL"
                print(
                    f"[{p_idx:02d}/{len(items):02d}] {item['id']} | {arm:<12} | {mark:4s} "
                    f"| pred={row['predicted'] or '-':<4} exp={row['expected']:<4} "
                    f"| {row['wall_s']:5.1f}s | d_tok={row.get('drafter_tokens', 0):5d} "
                    f"v_tok={row.get('verifier_tokens', 0):6d} | {row['detail'][:65]}"
                )
                sys.stdout.flush()

                payload = build_spec_verify_payload(
                    drafter_model=drafter_model_id,
                    verifier_model=verifier_model_id,
                    arms=arms,
                    results_by_arm={a: list(by_arm_id[a].values()) for a in arms},
                    range_info={"start": start, "end": end},
                )
                _write_json_atomic(out_path, payload)

        final_payload = build_spec_verify_payload(
            drafter_model=drafter_model_id,
            verifier_model=verifier_model_id,
            arms=arms,
            results_by_arm={a: list(by_arm_id[a].values()) for a in arms},
            range_info={"start": start, "end": end},
        )
        _write_json_atomic(out_path, final_payload)

    print(f"\nCompleted in {time.perf_counter() - t_all:.1f}s -> {out_path}\n")
    print(format_spec_verify_table(final_payload))
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
    return list(dict.fromkeys(arms))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="3-arm AIME 2025 SpecVerify ablation runner with atomic checkpointing and shard merge support."
    )
    parser.add_argument(
        "--drafter-url",
        default="http://127.0.0.1:8081/v1",
        help="Small drafter OpenAI-compatible / TensorFold server base URL.",
    )
    parser.add_argument(
        "--verifier-url",
        default="http://127.0.0.1:8080/v1",
        help="Big verifier OpenAI-compatible / TensorFold server base URL.",
    )
    parser.add_argument("--drafter-api-key", default=None, help="Optional API bearer key for drafter.")
    parser.add_argument("--verifier-api-key", default=None, help="Optional API bearer key for verifier.")
    parser.add_argument("--drafter-model", default=None, help="Optional explicit drafter model ID override.")
    parser.add_argument("--verifier-model", default=None, help="Optional explicit verifier model ID override.")
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
        help="Comma-separated arms (default: vanilla_big,cortheon_big,spec_verify).",
    )
    parser.add_argument("--start", type=int, default=0, help="Start problem index inclusive (default: 0).")
    parser.add_argument("--end", type=int, default=30, help="End problem index exclusive (default: 30).")
    parser.add_argument("--drafter-streams", type=int, default=3, help="Parallel drafter streams (default: 3).")
    parser.add_argument("--verifier-streams", type=int, default=4, help="Parallel verifier Wave-1 streams (default: 4).")
    parser.add_argument("--min-consensus", type=int, default=2, help="Minimum agreeing drafter scripts (default: 2).")
    parser.add_argument("--sandbox-timeout", type=float, default=15.0, help="Per-script sandbox timeout in seconds (default: 15.0).")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path (for run results or merged shard output).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing --out checkpoint file, skipping completed (arm, id) pairs.",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        default=None,
        metavar="SHARD_JSON",
        help="Merge 2+ SpecVerify ablation JSON shards/arms into --out and print combined statistics.",
    )
    args = parser.parse_args(argv)

    if args.merge:
        merged = merge_spec_verify_files(args.merge, out_path=args.out)
        print(f"Merged {len(args.merge)} shard(s) -> {args.out}\n")
        print(format_spec_verify_table(merged))
        return 0

    asyncio.run(
        run_spec_verify_ablation(
            drafter_url=args.drafter_url,
            verifier_url=args.verifier_url,
            dataset_path=args.dataset,
            arms=args.arms,
            start=args.start,
            end=args.end,
            out_path=args.out,
            resume=args.resume,
            drafter_model=args.drafter_model,
            verifier_model=args.verifier_model,
            drafter_api_key=args.drafter_api_key,
            verifier_api_key=args.verifier_api_key,
            drafter_streams=args.drafter_streams,
            verifier_streams=args.verifier_streams,
            min_consensus=args.min_consensus,
            sandbox_timeout_s=args.sandbox_timeout,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

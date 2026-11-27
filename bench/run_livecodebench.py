"""Official LiveCodeBench v6 Hard Benchmark Runner (All 60/60 AtCoder Competitive Programming).

Publication-grade, compute-matched, sharded benchmark runner supporting 3 arms on
Official `livecodebench/code_generation_lite (test6.jsonl, difficulty=hard)`:

Arms:
  1. vanilla      : Single-pass baseline (B=1, temp=0.6, seed=42, tb=1500, max_tokens=8192)
  2. best4_sample : Compute-matched B=4 parallel baseline (seeds 101/202/303/404,
                    temps 0.25/0.45/0.65/0.85, tb=1000, max_tokens=8192, loop_guard=True)
                    with NO spec oracle, NO red-team, and NO Wave-2 repair. Selects one
                    candidate using ONLY `public_cases` (printed in the prompt; NEVER peeks
                    at `private_cases`): (a) most public cases passed, (b) majority agreement
                    on stdout tuple across public cases among valid-syntax candidates,
                    (c) lowest temperature (earliest slot).
  3. forge        : Frozen Cortheon (`CortheonConfig(num_streams=streams, deep=False,
                    enable_memory=False)`, `force_mode=None`, `learn=False`).

Sharding & Merging:
    # GPU 1 (problems 0..40)
    python -m bench.run_livecodebench --url http://127.0.0.1:8080/v1 \\
        --arms vanilla,best4_sample,forge --start 0 --end 40 --out bench/results/lcb_s1.json --resume
    # GPU 2 (problems 40..60)
    python -m bench.run_livecodebench --url http://127.0.0.1:8081/v1 \\
        --arms vanilla,best4_sample,forge --start 40 --end 60 --out bench/results/lcb_s2.json --resume
    # Merge shards and/or disjoint arm files into one unified JSON + Markdown table
    python -m bench.run_livecodebench --merge bench/results/lcb_s1.json bench/results/lcb_s2.json \\
        --out bench/results/lcb_60_merged.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import pickle
import re
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import urlsplit

from cortheon import Cortheon, CortheonConfig
from flash_forge.client import CallSpec, CompletionResult, TensorFoldClient
from flash_forge.code_forge import CodeForgeTrace
from flash_forge.sandbox import check_syntax, extract_python_code, run_python

try:
    from bench.stats import format_p_value, mcnemar_exact_p, paired_contingency, wilson_ci
except ImportError:  # pragma: no cover
    from .stats import format_p_value, mcnemar_exact_p, paired_contingency, wilson_ci


ALL_ARMS: tuple[str, ...] = ("vanilla", "best4_sample", "forge")

BEST4_CONFIGS: tuple[tuple[int, float], ...] = (
    (101, 0.25),
    (202, 0.45),
    (303, 0.65),
    (404, 0.85),
)

# Canonical paired comparisons as (treatment_arm, baseline_arm)
PAIRWISE_SPECS: tuple[tuple[str, str], ...] = (
    ("forge", "vanilla"),
    ("forge", "best4_sample"),
    ("best4_sample", "vanilla"),
)

_ARM_ALIASES: dict[str, str] = {
    "cortheon": "forge",
}

_TMPDIR_FULL_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:/private)?/var/folders/[^'\"\s/]+/[^'\"\s/]+/T/(?:ff_(?:exec|lcb|wt)_[^'\"\s/]+|tmp[^'\"\s/]+)|"
    r"(?<![A-Za-z0-9_.-])(?:/private)?/tmp/(?:ff_(?:exec|lcb|wt)_[^'\"\s/]+|tmp[A-Za-z0-9_]+)"
)
_TMPDIR_TRUNC_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:/private)?/var/folders(?:/[^'\"\s),;:]*)?|"
    r"(?:[A-Za-z0-9_]+/)?T/(?:ff_(?:exec|lcb|wt)_[^'\"\s/]+|tmp[A-Za-z0-9_]+)"
)
_HOME_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])(?:/Users|/home)/[^'\"\s),;:]+")

_STDIN_SANDBOX_WRAPPER = (
    "import os, runpy, sys\n"
    "fd = os.open('_stdin.txt', os.O_RDONLY)\n"
    "os.dup2(fd, 0)\n"
    "os.close(fd)\n"
    "sys.stdin = open(0, 'r', encoding='utf-8', errors='replace', closefd=False)\n"
    "runpy.run_path('solution.py', run_name='__main__')\n"
)


def _normalize_arm_name(arm: str) -> str:
    cleaned = str(arm).strip()
    return _ARM_ALIASES.get(cleaned, cleaned)


def scrub_detail_paths(text: str, *, redact_url: Optional[str] = None) -> str:
    """Scrub ephemeral sandbox temp paths, user home paths, and optional endpoint URLs."""
    if not text:
        return text
    cleaned = text
    if redact_url:
        raw_url = redact_url.strip().rstrip("/")
        if raw_url:
            cleaned = cleaned.replace(raw_url, "<ENDPOINT>")
            parts = urlsplit(raw_url)
            if parts.scheme and parts.netloc:
                origin = f"{parts.scheme}://{parts.netloc}"
                cleaned = cleaned.replace(origin, "<ENDPOINT>")
    cleaned = _TMPDIR_FULL_RE.sub("<TMPDIR>", cleaned)
    cleaned = _TMPDIR_TRUNC_RE.sub("<TMPDIR>", cleaned)
    cleaned = _HOME_PATH_RE.sub("<HOME>", cleaned)
    return cleaned


def _normalize_stats(stats: Optional[dict[str, Any]]) -> dict[str, int]:
    if not isinstance(stats, dict):
        return {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        }
    calls = int(stats.get("calls", 0) or 0)
    pt = int(stats.get("prompt_tokens", 0) or 0)
    ct = int(stats.get("completion_tokens", 0) or 0)
    rt = int(stats.get("reasoning_tokens", 0) or 0)
    cat = int(stats.get("cached_tokens", 0) or 0)
    raw_tot = stats.get("total_tokens")
    tot = pt + ct if (pt > 0 or ct > 0 or raw_tot is None) else int(raw_tot)
    return {
        "calls": calls,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "reasoning_tokens": rt,
        "cached_tokens": cat,
        "total_tokens": tot,
    }


def _stats_from_results(results: Sequence[CompletionResult]) -> dict[str, int]:
    calls = sum(1 for r in results if r.ok)
    pt = sum(int(r.prompt_tokens or 0) for r in results)
    ct = sum(int(r.completion_tokens or 0) for r in results)
    rt = sum(int(r.reasoning_tokens or 0) for r in results)
    cat = sum(int(r.cached_tokens or 0) for r in results)
    return {
        "calls": calls,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "reasoning_tokens": rt,
        "cached_tokens": cat,
        "total_tokens": pt + ct,
    }


def _extract_row_stats(row: dict[str, Any]) -> Optional[dict[str, int]]:
    stats = row.get("stats")
    if isinstance(stats, dict):
        return _normalize_stats(stats)
    if any(k in row for k in ("llm_calls", "prompt_tokens", "completion_tokens", "total_tokens")):
        pt = int(row.get("prompt_tokens", 0) or 0)
        ct = int(row.get("completion_tokens", 0) or 0)
        rt = int(row.get("reasoning_tokens", 0) or 0)
        cat = int(row.get("cached_tokens", 0) or 0)
        raw_tot = row.get("total_tokens")
        tot = pt + ct if (pt > 0 or ct > 0 or raw_tot is None) else int(raw_tot)
        return {
            "calls": int(row.get("llm_calls", 0) or 0),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "reasoning_tokens": rt,
            "cached_tokens": cat,
            "total_tokens": tot,
        }
    return None


def _scrub_row(row: dict[str, Any], *, redact_url: Optional[str] = None) -> dict[str, Any]:
    out = dict(row)
    if "arm" in out and isinstance(out["arm"], str):
        out["arm"] = _normalize_arm_name(out["arm"])
    detail = out.get("detail")
    if isinstance(detail, str):
        out["detail"] = scrub_detail_paths(detail, redact_url=redact_url)
    if isinstance(out.get("stats"), dict):
        out["stats"] = _normalize_stats(out["stats"])
    return out


def _decode_cases(raw: str) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw))))


def _norm_out(s: str) -> str:
    return "\n".join(ln.rstrip() for ln in s.strip().splitlines())


def grade_livecodebench(
    code: str,
    cases: list[dict[str, Any]],
    per_case_timeout_s: float = 4.0,
) -> tuple[bool, str]:
    syn = check_syntax(code)
    if syn:
        return False, scrub_detail_paths(f"SyntaxError: {syn}")
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
                return False, scrub_detail_paths(f"RE on case {idx+1}/{total} ({passed}/{total}): {msg}")
            got = _norm_out(p.stdout)
            if got != exp:
                return False, f"WA on case {idx+1}/{total} ({passed}/{total}): got={got[:40]!r} exp={exp[:40]!r}"
            passed += 1
        return True, f"AC {passed}/{total} ({(time.perf_counter()-t0)*1000:.0f}ms)"


def evaluate_candidate_public_cases(
    code: str,
    public_cases: list[dict[str, Any]],
    per_case_timeout_s: float = 4.0,
) -> dict[str, Any]:
    """Run one candidate script against ONLY `public_cases` inside the OS sandbox.

    Never touches private test cases. Returns syntax validity, exact public pass
    count, and the normalized stdout tuple across public cases (`None` if syntax
    is invalid or any public case failed to exit cleanly).
    """
    syn = check_syntax(code)
    if syn:
        return {
            "valid_syntax": False,
            "syntax_error": syn,
            "public_passed": 0,
            "n_public": len(public_cases),
            "stdout_tuple": None,
        }

    passed = 0
    outputs: list[Optional[str]] = []
    for tc in public_cases:
        inp = str(tc.get("input", ""))
        exp = _norm_out(str(tc.get("output", "")))
        res = run_python(
            code=_STDIN_SANDBOX_WRAPPER,
            timeout_s=per_case_timeout_s,
            extra_files={"solution.py": code, "_stdin.txt": inp},
        )
        if not res.ok:
            outputs.append(None)
            continue
        got = _norm_out(res.stdout)
        outputs.append(got)
        if got == exp:
            passed += 1

    stdout_tuple: Optional[tuple[str, ...]] = (
        tuple(o for o in outputs if o is not None)
        if all(o is not None for o in outputs)
        else None
    )
    return {
        "valid_syntax": True,
        "syntax_error": None,
        "public_passed": passed,
        "n_public": len(public_cases),
        "stdout_tuple": stdout_tuple,
    }


def select_best4_candidate(
    candidates: list[tuple[float, str]],
    public_cases: list[dict[str, Any]],
    per_case_timeout_s: float = 4.0,
) -> tuple[str, dict[str, Any]]:
    """Select one candidate script using ONLY `public_cases`.

    Selection hierarchy:
      (a) highest number of `public_cases` passed,
      (b) valid syntax over syntax errors, then majority agreement on
          `(stdout_tuple_across_public_cases)` among valid-syntax candidates
          that exited cleanly on all public cases,
      (c) lowest temperature, then earliest slot index.
    """
    if not candidates:
        return "", {
            "public_pass_counts": [],
            "n_public_cases": len(public_cases),
            "selected_slot": -1,
            "selected_temp": None,
            "selected_public_passed": 0,
            "selected_agreement_count": 0,
            "selection_reason": "empty",
        }

    evals: list[dict[str, Any]] = []
    tuple_counts: dict[tuple[str, ...], int] = {}

    for slot, (temp, code) in enumerate(candidates):
        ev = evaluate_candidate_public_cases(code, public_cases, per_case_timeout_s=per_case_timeout_s)
        ev["slot"] = slot
        ev["temp"] = float(temp)
        ev["code"] = code
        evals.append(ev)
        if ev["valid_syntax"] and ev["stdout_tuple"] is not None:
            key = ev["stdout_tuple"]
            tuple_counts[key] = tuple_counts.get(key, 0) + 1

    for ev in evals:
        st = ev["stdout_tuple"]
        ev["agreement_count"] = tuple_counts.get(st, 0) if (ev["valid_syntax"] and st is not None) else 0

    ranked = sorted(
        evals,
        key=lambda ev: (
            -ev["public_passed"],
            -int(ev["valid_syntax"]),
            -ev["agreement_count"],
            ev["temp"],
            ev["slot"],
        ),
    )
    winner = ranked[0]

    tied_pass = [e for e in evals if e["public_passed"] == winner["public_passed"]]
    tied_valid = [e for e in tied_pass if e["valid_syntax"] == winner["valid_syntax"]]
    if len(tied_pass) == 1:
        reason = "public_pass"
    elif len(tied_valid) == 1:
        reason = "valid_syntax_tiebreak"
    else:
        tied_agree = [e for e in tied_valid if e["agreement_count"] == winner["agreement_count"]]
        if len(tied_agree) < len(tied_valid):
            reason = "majority_tiebreak"
        else:
            reason = "low_temp_tiebreak"

    return winner["code"], {
        "public_pass_counts": [ev["public_passed"] for ev in evals],
        "n_public_cases": len(public_cases),
        "selected_slot": winner["slot"],
        "selected_temp": winner["temp"],
        "selected_public_passed": winner["public_passed"],
        "selected_agreement_count": winner["agreement_count"],
        "selection_reason": reason,
    }


def load_lcb_dataset(
    path: Path,
    limit: int = 0,
    start: int = 0,
    end: int = 0,
) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
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
            "public_cases": pub,
            "private_cases": priv,
            "cases": pub + priv,
            "n_pub": len(pub),
            "n_priv": len(priv),
        })
    s = max(0, start)
    sliced = out[s : min(len(out), end)] if end > 0 else out[s:]
    return sliced[:limit] if limit > 0 else sliced


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
    stats = _stats_from_results([res])
    client_url = getattr(client, "base_url", None)
    if not res.ok:
        return {
            "arm": "vanilla",
            "pid": item["pid"],
            "title": item["title"],
            "passed": False,
            "detail": scrub_detail_paths(f"API_ERROR: {res.raw_error or 'unknown'}", redact_url=client_url),
            "wall_s": round(wall, 2),
            "finish_reason": res.finish_reason,
            "stats": stats,
        }
    code = extract_python_code(res.content)
    passed, detail = await asyncio.to_thread(grade_livecodebench, code, item["cases"])
    return {
        "arm": "vanilla",
        "pid": item["pid"],
        "title": item["title"],
        "passed": passed,
        "detail": scrub_detail_paths(detail, redact_url=client_url),
        "wall_s": round(wall, 2),
        "finish_reason": res.finish_reason,
        "stats": stats,
    }


async def run_one_best4_sample(
    client: TensorFoldClient,
    item: dict[str, Any],
    *,
    per_case_timeout_s: float = 4.0,
) -> dict[str, Any]:
    """Compute-matched B=4 parallel sampling baseline with public-sample-only selection."""
    t0 = time.perf_counter()
    specs = [
        CallSpec(
            messages=[{"role": "user", "content": item["prompt"]}],
            tag=f"lcb_b4_s{i}_{item['pid']}",
            max_tokens=8192,
            temperature=temp,
            top_p=0.95,
            top_k=20,
            seed=seed,
            enable_thinking=True,
            thinking_budget=1000,
            loop_guard=True,
        )
        for i, (seed, temp) in enumerate(BEST4_CONFIGS)
    ]
    results = await client.batch(specs)
    stats = _stats_from_results(results)

    codes = [extract_python_code(res.content) if res.ok else "" for res in results]
    candidates = [(temp, code) for (_, temp), code in zip(BEST4_CONFIGS, codes)]

    # Strictly public sample cases ONLY — NEVER read private_cases or cases during selection!
    public_cases = list(item["public_cases"])
    selected_code, sel_meta = await asyncio.to_thread(
        select_best4_candidate,
        candidates,
        public_cases,
        per_case_timeout_s,
    )
    wall = time.perf_counter() - t0

    # Grade ONLY the single selected candidate against ALL official judge cases (public + private)
    passed, detail = await asyncio.to_thread(
        grade_livecodebench,
        selected_code,
        item["cases"],
        per_case_timeout_s,
    )
    client_url = getattr(client, "base_url", None)
    return {
        "arm": "best4_sample",
        "pid": item["pid"],
        "title": item["title"],
        "passed": passed,
        "detail": scrub_detail_paths(detail, redact_url=client_url),
        "wall_s": round(wall, 2),
        "stats": stats,
        "extra": {
            "public_pass_counts": sel_meta["public_pass_counts"],
            "n_public_cases": sel_meta["n_public_cases"],
            "selected_slot": sel_meta["selected_slot"],
            "selected_temp": sel_meta["selected_temp"],
            "selection_reason": sel_meta["selection_reason"],
        },
    }


async def run_one_forge(
    base_url: str,
    item: dict[str, Any],
    streams: int = 4,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    cfg_kwargs: dict[str, Any] = {
        "base_url": base_url,
        "num_streams": streams,
        "deep": False,
        "enable_memory": False,
    }
    if model is not None:
        cfg_kwargs["model"] = model
    if api_key is not None:
        cfg_kwargs["api_key"] = api_key
    cfg = CortheonConfig(**cfg_kwargs)
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
        "arm": "forge",
        "pid": item["pid"],
        "title": item["title"],
        "passed": passed,
        "detail": scrub_detail_paths(detail, redact_url=base_url),
        "wall_s": round(res.wall_s, 2),
        "stats": _normalize_stats(res.token_stats),
        "extra": extra,
    }


def _pid_sort_key(pid: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", str(pid))
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in parts if p)


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    k = sum(1 for r in rows if r.get("passed"))
    lo, hi = wilson_ci(k, n)
    if n == 0:
        return {
            "passed": 0,
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
    stat_rows = [st for r in rows if (st := _extract_row_stats(r)) is not None]
    ns = len(stat_rows)
    if ns > 0:
        tot_calls = sum(st["calls"] for st in stat_rows)
        tot_p = sum(st["prompt_tokens"] for st in stat_rows)
        tot_c = sum(st["completion_tokens"] for st in stat_rows)
        tot_r = sum(st["reasoning_tokens"] for st in stat_rows)
        tot_cache = sum(st["cached_tokens"] for st in stat_rows)
        tot_tok = sum(st["total_tokens"] for st in stat_rows)
        avg_calls: Optional[float] = round(tot_calls / ns, 2)
        avg_p: Optional[float] = round(tot_p / ns, 1)
        avg_c: Optional[float] = round(tot_c / ns, 1)
        avg_r: Optional[float] = round(tot_r / ns, 1)
        avg_cache: Optional[float] = round(tot_cache / ns, 1)
        avg_tok: Optional[float] = round(tot_tok / ns, 1)
        out_tot_calls: Optional[int] = tot_calls
        out_tot_tok: Optional[int] = tot_tok
    else:
        avg_calls = avg_p = avg_c = avg_r = avg_cache = avg_tok = None
        out_tot_calls = out_tot_tok = None

    return {
        "passed": k,
        "total": n,
        "accuracy_pct": round(100.0 * k / n, 2),
        "wilson_ci_95_pct": [round(100.0 * lo, 2), round(100.0 * hi, 2)],
        "avg_wall_s": round(tot_wall / n, 2),
        "avg_llm_calls": avg_calls,
        "avg_prompt_tokens": avg_p,
        "avg_completion_tokens": avg_c,
        "avg_reasoning_tokens": avg_r,
        "avg_cached_tokens": avg_cache,
        "avg_total_tokens": avg_tok,
        "total_wall_s": round(tot_wall, 2),
        "total_llm_calls": out_tot_calls,
        "total_tokens": out_tot_tok,
    }


def compute_pairwise_comparisons(
    results_by_arm: dict[str, list[dict[str, Any]]],
    pairs: Sequence[tuple[str, str]] = PAIRWISE_SPECS,
) -> dict[str, dict[str, Any]]:
    """Compute exact paired two-sided McNemar tests and +fixed / -broken counts."""
    out: dict[str, dict[str, Any]] = {}
    for treat_arm, base_arm in pairs:
        if treat_arm not in results_by_arm or base_arm not in results_by_arm:
            continue
        base_rows = {str(r["pid"]): r for r in results_by_arm[base_arm]}
        treat_rows = {str(r["pid"]): r for r in results_by_arm[treat_arm]}
        shared = sorted(set(base_rows) & set(treat_rows), key=_pid_sort_key)
        if not shared:
            continue

        ct = paired_contingency(
            {pid: bool(base_rows[pid].get("passed")) for pid in shared},
            {pid: bool(treat_rows[pid].get("passed")) for pid in shared},
        )
        n = ct["n"]
        p_val = mcnemar_exact_p(ct["n01"], ct["n10"])

        b_wall = sum(float(base_rows[p].get("wall_s", 0.0)) for p in shared) / n
        t_wall = sum(float(treat_rows[p].get("wall_s", 0.0)) for p in shared) / n

        b_stats = [_extract_row_stats(base_rows[p]) for p in shared]
        t_stats = [_extract_row_stats(treat_rows[p]) for p in shared]
        paired_stats = [(b, t) for b, t in zip(b_stats, t_stats) if b is not None and t is not None]
        if paired_stats:
            b_calls = sum(b["calls"] for b, _ in paired_stats) / len(paired_stats)
            t_calls = sum(t["calls"] for _, t in paired_stats) / len(paired_stats)
            b_tok = sum(b["total_tokens"] for b, _ in paired_stats) / len(paired_stats)
            t_tok = sum(t["total_tokens"] for _, t in paired_stats) / len(paired_stats)
            call_ratio = round(t_calls / b_calls, 3) if b_calls > 0 else None
            token_ratio = round(t_tok / b_tok, 3) if b_tok > 0 else None
        else:
            call_ratio = None
            token_ratio = None

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
            "fixed": ct["n01"],
            "broken": ct["n10"],
            "mcnemar_exact_p": p_val,
            "mcnemar_p_formatted": format_p_value(p_val),
            "wall_ratio": round(t_wall / b_wall, 3) if b_wall > 0 else None,
            "call_ratio": call_ratio,
            "token_ratio": token_ratio,
        }
    return out


def build_lcb_payload(
    *,
    model: str,
    arms: list[str],
    results_by_arm: dict[str, list[dict[str, Any]]],
    range_info: dict[str, Any],
    total_expected: Optional[int] = None,
    pid_order: Optional[dict[str, int]] = None,
    merged_from: Optional[list[str]] = None,
    redact_url: Optional[str] = None,
) -> dict[str, Any]:
    ordered_results: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, dict[str, Any]] = {}

    def _sort_key(row: dict[str, Any]) -> tuple[int, Any]:
        pid = str(row["pid"])
        if pid_order is not None and pid in pid_order:
            return (0, pid_order[pid])
        return (1, _pid_sort_key(pid))

    for arm in arms:
        rows = [
            _scrub_row(r, redact_url=redact_url)
            for r in sorted(results_by_arm.get(arm, []), key=_sort_key)
        ]
        ordered_results[arm] = rows
        summary[arm] = summarize_arm(rows)

    pairwise = compute_pairwise_comparisons(ordered_results)

    if len(arms) == 1:
        only_arm = arms[0]
        rows = ordered_results[only_arm]
        s = summary[only_arm]
        tot = total_expected if total_expected is not None else len(rows)
        payload: dict[str, Any] = {
            "dataset": "livecodebench_v6_hard",
            "model": model,
            "arm": only_arm,
            "arms": [only_arm],
            "range": range_info,
            "completed": len(rows),
            "passed": s["passed"],
            "total": tot,
            "pass_rate_pct": s["accuracy_pct"],
            "avg_wall_s": s["avg_wall_s"],
            "summary": summary,
            "pairwise": pairwise,
            "results": rows,
        }
    else:
        payload = {
            "dataset": "livecodebench_v6_hard",
            "model": model,
            "arms": arms,
            "range": range_info,
            "summary": summary,
            "pairwise": pairwise,
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


def _extract_arms_from_payload(data: dict[str, Any], source_label: str) -> dict[str, list[dict[str, Any]]]:
    """Extract `{arm: [rows]}` from either single-arm list JSON or multi-arm dict JSON."""
    res = data.get("results")
    if isinstance(res, dict):
        out: dict[str, list[dict[str, Any]]] = {}
        for arm_name, rows in res.items():
            if not isinstance(rows, list):
                raise ValueError(f"Shard {source_label} arm {arm_name!r} must map to a list of rows")
            norm_arm = _normalize_arm_name(arm_name)
            out.setdefault(norm_arm, []).extend(rows)
        for arm_name in data.get("arms", []):
            norm_arm = _normalize_arm_name(arm_name)
            out.setdefault(norm_arm, [])
        return out

    if isinstance(res, list):
        raw_arm = data.get("arm")
        if not raw_arm and isinstance(data.get("arms"), list) and len(data["arms"]) == 1:
            raw_arm = data["arms"][0]
        if not raw_arm and res and isinstance(res[0], dict):
            raw_arm = res[0].get("arm")
        if not raw_arm:
            raise ValueError(f"Single-arm shard {source_label} missing 'arm' field")
        norm_arm = _normalize_arm_name(str(raw_arm))
        return {norm_arm: list(res)}

    raise ValueError(f"Shard {source_label} missing list or dict 'results'")


def merge_lcb_files(
    shard_paths: Sequence[Path],
    out_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Merge 1+ LiveCodeBench JSON files across disjoint problem ranges AND/OR disjoint arms."""
    paths = [Path(p) for p in shard_paths]
    if not paths:
        raise ValueError("merge_lcb_files requires at least one input file")

    merged_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    seen_arms: list[str] = []
    models: list[str] = []
    ranges: list[Any] = []

    for sp in paths:
        data = json.loads(sp.read_text(encoding="utf-8"))
        m = str(data.get("model", "")).strip()
        if m and m.lower() != "unknown" and m not in models:
            models.append(m)
        if "range" in data:
            ranges.append(data["range"])
        extracted = _extract_arms_from_payload(data, sp.name)
        for arm, rows in extracted.items():
            if arm not in seen_arms:
                seen_arms.append(arm)
            arm_dict = merged_by_arm.setdefault(arm, {})
            for row in rows:
                scrubbed = _scrub_row(row)
                scrubbed["arm"] = arm
                arm_dict[str(scrubbed["pid"])] = scrubbed

    canonical_arms = [a for a in ALL_ARMS if a in seen_arms] + [a for a in seen_arms if a not in ALL_ARMS]
    results_by_arm = {
        arm: sorted(merged_by_arm.get(arm, {}).values(), key=lambda r: _pid_sort_key(str(r["pid"])))
        for arm in canonical_arms
    }

    starts = [r["start"] for r in ranges if isinstance(r, dict) and isinstance(r.get("start"), int)]
    ends = [r["end"] for r in ranges if isinstance(r, dict) and isinstance(r.get("end"), int)]
    max_arm_len = max((len(v) for v in results_by_arm.values()), default=0)
    range_info: dict[str, Any] = {
        "start": min(starts) if starts else 0,
        "end": max(max(ends, default=0), max_arm_len) if ends else max_arm_len,
        "shards": len(paths),
    }

    model_str = models[0] if len(models) == 1 else (", ".join(models) if models else "unknown")
    payload = build_lcb_payload(
        model=model_str,
        arms=canonical_arms,
        results_by_arm=results_by_arm,
        range_info=range_info,
        total_expected=max_arm_len if len(canonical_arms) == 1 else None,
        # Store basenames only — zero host paths, usernames, or hostnames in merged JSON
        merged_from=[p.name for p in paths],
    )
    if out_path is not None:
        _write_json_atomic(Path(out_path), payload)
    return payload


def _fmt_opt(val: Optional[float], fmt: str) -> str:
    return "—" if val is None else format(val, fmt)


def format_lcb_table(payload: dict[str, Any]) -> str:
    """Render a publication-grade Markdown summary table (per-arm + paired McNemar)."""
    arms: list[str] = list(payload.get("arms") or ([] if not payload.get("arm") else [payload["arm"]]))
    summary: dict[str, dict[str, Any]] = payload.get("summary", {})
    pairwise: dict[str, dict[str, Any]] = payload.get("pairwise", {})
    model = payload.get("model", "unknown")

    lines = [
        f"### Official LiveCodeBench v6 Hard (`{model}`)",
        "",
        "| Arm | Accuracy k/n [95% Wilson CI] | Avg Wall | Avg Calls | Avg Prompt Tok | Avg Comp Tok | Avg Reason Tok | Avg Total Tok |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        s = summary.get(arm, {})
        k = int(s.get("passed", 0))
        n = int(s.get("total", 0))
        acc = float(s.get("accuracy_pct", 0.0))
        ci = s.get("wilson_ci_95_pct", [0.0, 0.0])
        lines.append(
            f"| `{arm}` | {k}/{n} ({acc:5.1f}%) [{ci[0]:4.1f}, {ci[1]:4.1f}] "
            f"| {s.get('avg_wall_s', 0.0):6.1f}s "
            f"| {_fmt_opt(s.get('avg_llm_calls'), '5.2f')} "
            f"| {_fmt_opt(s.get('avg_prompt_tokens'), '7.0f')} "
            f"| {_fmt_opt(s.get('avg_completion_tokens'), '7.0f')} "
            f"| {_fmt_opt(s.get('avg_reasoning_tokens'), '7.0f')} "
            f"| {_fmt_opt(s.get('avg_total_tokens'), '8.0f')} |"
        )

    if pairwise:
        lines.extend([
            "",
            "| Comparison | Paired n | Delta | Both Pass (n11) | +Fixed (n01) | -Broken (n10) | Both Fail (n00) | Exact McNemar p | Wall Ratio | Token Ratio |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        ordered_keys = [f"{t}_vs_{b}" for t, b in PAIRWISE_SPECS if f"{t}_vs_{b}" in pairwise]
        ordered_keys += [k for k in pairwise if k not in ordered_keys]
        for key in ordered_keys:
            pw = pairwise[key]
            sign = "+" if pw["delta_pct"] >= 0 else ""
            w_str = f"{pw['wall_ratio']:.2f}x" if pw.get("wall_ratio") is not None else "—"
            t_str = f"{pw['token_ratio']:.2f}x" if pw.get("token_ratio") is not None else "—"
            p_str = format_p_value(float(pw["mcnemar_exact_p"]))
            lines.append(
                f"| `{pw['treatment']}` vs `{pw['baseline']}` | {pw['n']} "
                f"| {sign}{pw['delta_pct']:.1f} pts "
                f"| {pw['n11']} | +{pw['n01']} | -{pw['n10']} | {pw['n00']} "
                f"| {p_str} | {w_str} | {t_str} |"
            )

    return "\n".join(lines)


async def run_eval(
    dataset_path: Path,
    base_url: str,
    arm: str | Sequence[str],
    limit: int = 0,
    concurrency: int = 0,
    out_path: Optional[Path] = None,
    resume: bool = False,
    *,
    start: int = 0,
    end: int = 0,
    streams: int = 4,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    if isinstance(arm, str):
        arms = [_normalize_arm_name(arm)]
    else:
        arms = list(dict.fromkeys(_normalize_arm_name(a) for a in arm))
    if not arms:
        raise ValueError("At least one arm must be specified")
    invalid = [a for a in arms if a not in ALL_ARMS]
    if invalid:
        raise ValueError(f"Unknown arm(s): {invalid}. Valid arms: {', '.join(ALL_ARMS)}")

    items = load_lcb_dataset(Path(dataset_path), limit=limit, start=start, end=end)
    if not items:
        raise RuntimeError(
            f"No LiveCodeBench items selected for start={start}, end={end}, limit={limit} from {dataset_path}"
        )

    pid_order = {x["pid"]: i for i, x in enumerate(items)}
    effective_end = start + len(items)
    range_info = {"start": start, "end": effective_end}

    by_arm_pid: dict[str, dict[str, dict[str, Any]]] = {a: {} for a in arms}
    if resume and out_path is not None and Path(out_path).exists():
        prev = json.loads(Path(out_path).read_text(encoding="utf-8"))
        prev_arms = _extract_arms_from_payload(prev, Path(out_path).name)
        loaded_n = 0
        for a in arms:
            for row in prev_arms.get(a, []):
                pid = str(row["pid"])
                if pid in pid_order:
                    by_arm_pid[a][pid] = _scrub_row(row, redact_url=base_url)
                    loaded_n += 1
        print(f"Resumed {loaded_n} completed arm-problem rows from {out_path}")

    eff_conc = concurrency or (4 if arms == ["vanilla"] else 1)
    t_all = time.perf_counter()

    async with TensorFoldClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_parallel=max(8, streams),
    ) as client:
        model_id = client.caps.model_id
        total_pairs = len(items) * len(arms)
        done_pairs = sum(len(by_arm_pid[a]) for a in arms)
        print(
            f"=== OFFICIAL LIVECODEBENCH v6 HARD | model={model_id} | arms={','.join(arms)} "
            f"| range=[{start},{effective_end}) ({len(items)} problems) "
            f"| todo={total_pairs - done_pairs}/{total_pairs} | conc={eff_conc} | streams={streams} ==="
        )
        sys.stdout.flush()

        lock = asyncio.Lock()

        async def _run_arm_item(a: str, item: dict[str, Any]) -> dict[str, Any]:
            try:
                if a == "vanilla":
                    return await run_one_vanilla(client, item)
                if a == "best4_sample":
                    return await run_one_best4_sample(client, item)
                if a == "forge":
                    return await run_one_forge(base_url, item, streams=streams, model=model, api_key=api_key)
                raise ValueError(f"Unknown arm: {a!r}")
            except Exception as exc:
                return {
                    "arm": a,
                    "pid": item["pid"],
                    "title": item["title"],
                    "passed": False,
                    "detail": scrub_detail_paths(f"EXC: {type(exc).__name__}: {exc}", redact_url=base_url),
                    "wall_s": 0.0,
                    "stats": _normalize_stats(None),
                }

        async def _record_and_checkpoint(a: str, item: dict[str, Any], row: dict[str, Any]) -> None:
            nonlocal done_pairs
            async with lock:
                done_pairs += 1
                by_arm_pid[a][item["pid"]] = _scrub_row(row, redact_url=base_url)
                mark = "PASS" if row["passed"] else "FAIL"
                st = _extract_row_stats(row) or _normalize_stats(None)
                ex = row.get("extra", {})
                if a == "forge" and ex:
                    note = f" [win={ex.get('winner', '')}, w2={ex.get('wave2')}]"
                elif a == "best4_sample" and ex:
                    note = f" [slot={ex.get('selected_slot')}, pub={ex.get('public_pass_counts')}]"
                else:
                    note = ""
                print(
                    f"[{done_pairs:03d}/{total_pairs:03d}] {mark:4s} | {a:<12} | {item['pid']:<10} "
                    f"| {row['wall_s']:5.1f}s | calls={st['calls']:2d} tok={st['total_tokens']:6d} "
                    f"| {row['detail'][:56]}{note}"
                )
                sys.stdout.flush()
                if out_path is not None:
                    payload = build_lcb_payload(
                        model=model_id,
                        arms=arms,
                        results_by_arm={k: list(by_arm_pid[k].values()) for k in arms},
                        range_info=range_info,
                        total_expected=len(items),
                        pid_order=pid_order,
                        redact_url=base_url,
                    )
                    _write_json_atomic(Path(out_path), payload)

        if eff_conc <= 1:
            for item in items:
                for a in arms:
                    if item["pid"] in by_arm_pid[a]:
                        continue
                    row = await _run_arm_item(a, item)
                    await _record_and_checkpoint(a, item, row)
        else:
            sem = asyncio.Semaphore(eff_conc)

            async def _worker(item: dict[str, Any]) -> None:
                async with sem:
                    for a in arms:
                        if item["pid"] in by_arm_pid[a]:
                            continue
                        row = await _run_arm_item(a, item)
                        await _record_and_checkpoint(a, item, row)

            await asyncio.gather(*[_worker(item) for item in items])

        final_payload = build_lcb_payload(
            model=model_id,
            arms=arms,
            results_by_arm={k: list(by_arm_pid[k].values()) for k in arms},
            range_info=range_info,
            total_expected=len(items),
            pid_order=pid_order,
            redact_url=base_url,
        )
        if out_path is not None:
            _write_json_atomic(Path(out_path), final_payload)

    print(f"\nCompleted in {time.perf_counter() - t_all:.1f}s" + (f" -> {out_path}" if out_path else "") + "\n")
    print(format_lcb_table(final_payload))
    return final_payload


def _parse_arms(raw: str) -> list[str]:
    arms = [_normalize_arm_name(a) for a in raw.split(",") if a.strip()]
    if not arms:
        raise argparse.ArgumentTypeError("At least one arm must be specified")
    invalid = [a for a in arms if a not in ALL_ARMS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown arm(s): {invalid}. Valid arms: {', '.join(ALL_ARMS)}"
        )
    return list(dict.fromkeys(arms))


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Official LiveCodeBench v6 Hard benchmark runner (vanilla, best4_sample, forge) with sharding & merge."
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("bench/datasets/livecodebench_v6_hard.json"),
        help="Path to LiveCodeBench v6 Hard JSON (default: bench/datasets/livecodebench_v6_hard.json).",
    )
    p.add_argument(
        "--url",
        default="http://127.0.0.1:8080/v1",
        help="OpenAI-compatible / TensorFold server base URL (default: http://127.0.0.1:8080/v1).",
    )
    p.add_argument(
        "--arm",
        choices=list(ALL_ARMS),
        default=None,
        help="Run a single arm (vanilla, best4_sample, or forge). Mutually exclusive with --arms.",
    )
    p.add_argument(
        "--arms",
        type=_parse_arms,
        default=None,
        help="Comma-separated list of arms from vanilla,best4_sample,forge (default: all 3 if neither --arm nor --arms is set).",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start problem index inclusive (0-based, default: 0).",
    )
    p.add_argument(
        "--end",
        type=int,
        default=0,
        help="End problem index exclusive (0-based, default: 0 meaning all/limit).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of problems from the selected slice (0 means all; default: 0).",
    )
    p.add_argument(
        "--streams",
        type=int,
        default=4,
        help="Number of parallel Cortheon streams passed to CortheonConfig (default: 4).",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Concurrent problems (0 = auto: 4 for vanilla-only, 1 otherwise).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path (for run results or merged shard output).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing --out checkpoint file, skipping completed (arm, pid) pairs.",
    )
    p.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        default=None,
        metavar="SHARD_JSON",
        help="Merge 1+ LiveCodeBench JSON files (disjoint problem ranges and/or disjoint arms) into --out.",
    )
    p.add_argument("--model", default=None, help="Optional explicit model ID override.")
    p.add_argument("--api-key", default=None, help="Optional API bearer key.")
    args = p.parse_args(argv)

    if args.merge:
        merged = merge_lcb_files(args.merge, out_path=args.out)
        print(f"Merged {len(args.merge)} shard(s) -> {args.out}\n")
        print(format_lcb_table(merged))
        return 0

    if args.arm and args.arms:
        p.error("Specify either --arm or --arms, not both.")
    if args.start < 0:
        p.error("--start must be >= 0")
    if args.end < 0:
        p.error("--end must be >= 0")
    if args.end > 0 and args.end <= args.start:
        p.error("--end must be greater than --start")
    if args.limit < 0:
        p.error("--limit must be >= 0")
    if args.streams < 1:
        p.error("--streams must be >= 1")

    arms = [args.arm] if args.arm else (args.arms if args.arms else list(ALL_ARMS))
    conc = args.concurrency or (4 if arms == ["vanilla"] else 1)
    asyncio.run(
        run_eval(
            args.dataset,
            args.url,
            arms,
            args.limit,
            conc,
            args.out,
            args.resume,
            start=args.start,
            end=args.end,
            streams=args.streams,
            model=args.model,
            api_key=args.api_key,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

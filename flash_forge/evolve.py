"""Cortheon Evolve — Oracle-Free Execution-Verified Self-Distillation & Gated RSI.

Turns Cortheon's mechanically verified test-time rescues into a closed, statistically
gated self-improvement loop for ANY local model family (Qwen, GLM, DeepSeek, Llama,
Nemotron, Inkling) without requiring ground-truth labels or external teacher models.

Core original components:
  1. **Principled Contrastive Curriculum (`build_weighted_curriculum`):**
     Mines execution-verified traces from Cortheon's flywheel/results into weighted
     tiers grounded in calibration telemetry:
       - `disagreement_rescue` (weight 3.0): Wave-1 plausible wrong answer falsified
         mechanically by Wave-2 / diff-arbiter (highest-leverage hallucination cure).
       - `crash_rescue` (weight 2.0): Wave-1 syntax/runtime/TLE crash repaired by
         Wave-2 surgical solver.
       - `fast_distill` (weight 1.0): Wave-1 Pair-First verified consensus distilled
         into single-shot B=1 speed (compressing multi-stream compute into 1 call).
       - Unverified / uncorroborated guesses are strictly discarded (weight 0.0).
  2. **Exact Dual-Regime Statistical Gate (`evaluate_generation_gate`):**
     Gates every candidate checkpoint using exact paired 2x2 contingency tables,
     Wilson 95% CIs, and exact two-sided McNemar p-values across BOTH single-shot
     (`B=1` compression gain) AND multi-stream verified consensus (non-regression),
     plus optional anchor-benchmark non-regression checks.
  3. **Interference-Guarded Adapter Soup (`merge_adapter_weights` / `merge_npz_adapters`):**
     Pure-NumPy shape-verified weighted merging of top-K accepted generations with
     optional sign-consensus trimming so compounding generations do not cancel or drift.
  4. **Generational Ledger & Autonomous Loop (`EvolutionLedger` / `run_evolution_loop`):**
     Pluggable local trainer interface (`mlx-lm`, `TRL`, or custom CLI) with atomic
     JSON checkpointing, automatic rollback on gate rejection, and SHA-256 audits.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

import numpy as np

from .flywheel import (
    DEFAULT_FLYWHEEL_LOG,
    FlywheelPair,
    harvest_from_results_dir,
    scrub_pii,
)

_WILSON_Z_95: float = NormalDist().inv_cdf(0.975)


def _wilson_ci_95(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z2 = _WILSON_Z_95 * _WILSON_Z_95
    p = k / n
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (_WILSON_Z_95 * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))) / denom
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return (lo, hi)


def _mcnemar_exact_p(n01: int, n10: int) -> float:
    m = n01 + n10
    if m == 0:
        return 1.0
    k = min(n01, n10)
    two_tail = 2 * sum(math.comb(m, i) for i in range(k + 1))
    denom = 1 << m
    return 1.0 if two_tail >= denom else two_tail / denom


def _paired_contingency(
    baseline: Mapping[str, bool],
    treatment: Mapping[str, bool],
) -> dict[str, int]:
    shared = sorted(set(baseline.keys()) & set(treatment.keys()))
    n11 = n01 = n10 = n00 = 0
    for pid in shared:
        b = bool(baseline[pid])
        t = bool(treatment[pid])
        if b and t:
            n11 += 1
        elif (not b) and t:
            n01 += 1
        elif b and (not t):
            n10 += 1
        else:
            n00 += 1
    return {
        "n": len(shared),
        "n11": n11,
        "n01": n01,
        "n10": n10,
        "n00": n00,
        "baseline_passed": n11 + n10,
        "treatment_passed": n11 + n01,
    }

# Principled contrastive curriculum weights
TIER_WEIGHTS: dict[str, float] = {
    "disagreement_rescue": 3.0,
    "crash_rescue": 2.0,
    "fast_distill": 1.0,
}

_CRASH_MARKERS = (
    "timeout",
    "traceback",
    "syntaxerror",
    "zerodivisionerror",
    "nameerror",
    "typeerror",
    "valueerror",
    "indexerror",
    "keyerror",
    "memoryerror",
    "runtimeerror",
    "importerror",
    "failed",
    "error",
    "tle",
    "rte",
    "no valid python",
    "no answer",
)


# ---------------------------------------------------------------------------
# 1. Contrastive Curriculum Mining & Weighted Export
# ---------------------------------------------------------------------------


@dataclass
class WeightedExample:
    """A single PII-scrubbed, mechanically verified curriculum record."""

    prompt: str
    chosen: str
    rejected: str
    tier: str
    weight: float
    domain: str = "general"
    source: str = ""
    model: str = ""

    def to_dpo_record(self) -> dict[str, Any]:
        return {
            "prompt": scrub_pii(self.prompt).strip(),
            "chosen": scrub_pii(self.chosen).strip(),
            "rejected": scrub_pii(self.rejected).strip(),
            "tier": self.tier,
            "weight": round(float(self.weight), 3),
            "domain": scrub_pii(self.domain or "general").strip() or "general",
            "source": scrub_pii(self.source).strip(),
        }

    def to_sft_record(self) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": scrub_pii(self.prompt).strip()},
                {"role": "assistant", "content": scrub_pii(self.chosen).strip()},
            ],
            "tier": self.tier,
            "weight": round(float(self.weight), 3),
            "domain": scrub_pii(self.domain or "general").strip() or "general",
            "source": scrub_pii(self.source).strip(),
        }


@dataclass
class CurriculumSummary:
    total_input_pairs: int = 0
    kept_examples: int = 0
    dropped_unverified: int = 0
    deduplicated: int = 0
    by_tier: dict[str, int] = field(default_factory=dict)
    by_domain: dict[str, int] = field(default_factory=dict)
    total_weight: float = 0.0
    dpo_path: Optional[str] = None
    sft_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_pair_tier(pair: FlywheelPair) -> tuple[str, float]:
    """Assign a principled curriculum tier and importance weight to a verified pair.

    Unverified pairs always receive ``("unverified", 0.0)`` and are dropped.
    """
    if not pair.chosen_verified or not (pair.prompt or "").strip() or not (pair.chosen or "").strip():
        return ("unverified", 0.0)

    rejected = (pair.rejected or "").strip()
    reason_low = (pair.rejected_reason or "").lower()
    meta = pair.metadata if isinstance(pair.metadata, dict) else {}

    if not rejected:
        # Pure verified consensus trace without a failing baseline/Wave-1 contrast
        return ("fast_distill", TIER_WEIGHTS["fast_distill"])

    # Distinguish plausible-wrong disagreement falsification from crash/timeout repair
    if meta.get("recovered_from_disagreement") is True or "wrong" in reason_low or "wa " in reason_low or "disagree" in reason_low:
        return ("disagreement_rescue", TIER_WEIGHTS["disagreement_rescue"])

    if any(tok in reason_low for tok in _CRASH_MARKERS) or any(
        tok in rejected.lower()[:240] for tok in ("traceback", "timeout", "error:")
    ):
        return ("crash_rescue", TIER_WEIGHTS["crash_rescue"])

    # Default when a concrete non-empty rejected answer differs from verified chosen answer
    if rejected != pair.chosen.strip():
        return ("disagreement_rescue", TIER_WEIGHTS["disagreement_rescue"])

    return ("fast_distill", TIER_WEIGHTS["fast_distill"])


def build_weighted_curriculum(
    pairs: Sequence[FlywheelPair],
    *,
    include_fast_distill: bool = True,
) -> tuple[list[WeightedExample], CurriculumSummary]:
    """Filter, classify, weight, deduplicate, and PII-scrub verified pairs into a curriculum."""
    summary = CurriculumSummary(total_input_pairs=len(pairs))
    best_by_key: dict[tuple[str, str], WeightedExample] = {}

    for raw_pair in pairs:
        p = raw_pair.scrubbed()
        tier, weight = classify_pair_tier(p)
        if weight <= 0.0 or (tier == "fast_distill" and not include_fast_distill):
            summary.dropped_unverified += 1
            continue

        ex = WeightedExample(
            prompt=p.prompt,
            chosen=p.chosen,
            rejected=p.rejected,
            tier=tier,
            weight=weight,
            domain=p.domain,
            source=p.source,
            model=p.model,
        )
        key = (
            hashlib.sha256(ex.prompt.encode("utf-8")).hexdigest()[:16],
            hashlib.sha256(ex.chosen.encode("utf-8")).hexdigest()[:16],
        )
        prev = best_by_key.get(key)
        if prev is not None:
            summary.deduplicated += 1
            # Keep the higher-leverage contrastive record (e.g. disagreement_rescue > fast_distill)
            if ex.weight > prev.weight or (ex.weight == prev.weight and len(ex.rejected) > len(prev.rejected)):
                best_by_key[key] = ex
        else:
            best_by_key[key] = ex

    examples = sorted(
        best_by_key.values(),
        key=lambda e: (-e.weight, e.domain, e.prompt[:64]),
    )
    tier_counts = Counter(e.tier for e in examples)
    domain_counts = Counter(e.domain for e in examples)

    summary.kept_examples = len(examples)
    summary.by_tier = dict(sorted(tier_counts.items()))
    summary.by_domain = dict(sorted(domain_counts.items()))
    summary.total_weight = round(sum(e.weight for e in examples), 3)
    return examples, summary


def _write_jsonl_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def export_weighted_curriculum(
    *,
    results_dir: Optional[Path] = None,
    flywheel_log: Optional[Path] = None,
    datasets_dir: Optional[Path] = None,
    out_dir: Path,
    include_fast_distill: bool = True,
) -> tuple[list[WeightedExample], CurriculumSummary]:
    """Harvest verified pairs from disk and write weighted DPO + SFT JSONL files atomically."""
    pairs: list[FlywheelPair] = []
    if results_dir is not None and Path(results_dir).exists():
        pairs.extend(harvest_from_results_dir(results_dir, datasets_dir=datasets_dir))
    if flywheel_log is not None and Path(flywheel_log).exists():
        pairs.extend(harvest_from_results_dir(flywheel_log, datasets_dir=datasets_dir))
    examples, summary = build_weighted_curriculum(pairs, include_fast_distill=include_fast_distill)
    out_dir.mkdir(parents=True, exist_ok=True)

    dpo_rows = [e.to_dpo_record() for e in examples if e.rejected]
    sft_rows = [e.to_sft_record() for e in examples]

    dpo_path = out_dir / "curriculum_dpo_weighted.jsonl"
    sft_path = out_dir / "curriculum_sft_weighted.jsonl"
    _write_jsonl_atomic(dpo_path, dpo_rows)
    _write_jsonl_atomic(sft_path, sft_rows)

    summary.dpo_path = str(dpo_path)
    summary.sft_path = str(sft_path)
    return examples, summary


# ---------------------------------------------------------------------------
# 2. Exact Dual-Regime Statistical Regression Gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateConfig:
    """Statistical acceptance thresholds for dual-regime RSI gating."""

    min_single_shot_net_gain: int = 1
    max_single_shot_regressions: int = 2
    min_verified_net_gain: int = 0
    max_verified_regressions: int = 1
    max_anchor_drop_pct: float = 1.0
    require_significant_or_zero_regressions: bool = False
    alpha: float = 0.10


@dataclass
class RegimeGateStats:
    name: str
    n: int
    base_passed: int
    cand_passed: int
    base_pct: float
    cand_pct: float
    delta_pct: float
    base_ci_95: tuple[float, float]
    cand_ci_95: tuple[float, float]
    n11: int
    n01_fixed: int
    n10_broken: int
    n00: int
    mcnemar_exact_p: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "base_passed": self.base_passed,
            "cand_passed": self.cand_passed,
            "base_pct": self.base_pct,
            "cand_pct": self.cand_pct,
            "delta_pct": self.delta_pct,
            "base_ci_95": list(self.base_ci_95),
            "cand_ci_95": list(self.cand_ci_95),
            "n11": self.n11,
            "n01_fixed": self.n01_fixed,
            "n10_broken": self.n10_broken,
            "n00": self.n00,
            "mcnemar_exact_p": self.mcnemar_exact_p,
        }


@dataclass
class GateDecision:
    accepted: bool
    reasons: list[str]
    single_shot: RegimeGateStats
    verified: Optional[RegimeGateStats] = None
    anchor_delta_pct: Optional[float] = None
    composite_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "single_shot": self.single_shot.to_dict(),
            "verified": self.verified.to_dict() if self.verified is not None else None,
            "anchor_delta_pct": self.anchor_delta_pct,
            "composite_score": round(self.composite_score, 4),
        }


def _compute_regime_stats(
    name: str,
    base_outcomes: Mapping[str, bool],
    cand_outcomes: Mapping[str, bool],
) -> RegimeGateStats:
    tab = _paired_contingency(base_outcomes, cand_outcomes)
    n = tab["n"]
    if n == 0:
        raise ValueError(f"Regime {name!r} has zero shared problem IDs between baseline and candidate.")
    bp = tab["baseline_passed"]
    cp = tab["treatment_passed"]
    blo, bhi = _wilson_ci_95(bp, n)
    clo, chi = _wilson_ci_95(cp, n)
    p_val = _mcnemar_exact_p(tab["n01"], tab["n10"])
    return RegimeGateStats(
        name=name,
        n=n,
        base_passed=bp,
        cand_passed=cp,
        base_pct=round(100.0 * bp / n, 2),
        cand_pct=round(100.0 * cp / n, 2),
        delta_pct=round(100.0 * (cp - bp) / n, 2),
        base_ci_95=(round(100.0 * blo, 2), round(100.0 * bhi, 2)),
        cand_ci_95=(round(100.0 * clo, 2), round(100.0 * chi, 2)),
        n11=tab["n11"],
        n01_fixed=tab["n01"],
        n10_broken=tab["n10"],
        n00=tab["n00"],
        mcnemar_exact_p=round(p_val, 6),
    )


def evaluate_generation_gate(
    *,
    base_single_shot: Mapping[str, bool],
    cand_single_shot: Mapping[str, bool],
    base_verified: Optional[Mapping[str, bool]] = None,
    cand_verified: Optional[Mapping[str, bool]] = None,
    base_anchor_pct: Optional[float] = None,
    cand_anchor_pct: Optional[float] = None,
    config: Optional[GateConfig] = None,
) -> GateDecision:
    """Strictly gate a candidate generation using exact paired McNemar & non-regression rules."""
    cfg = config or GateConfig()
    s_stats = _compute_regime_stats("single_shot", base_single_shot, cand_single_shot)
    v_stats: Optional[RegimeGateStats] = None
    if base_verified is not None and cand_verified is not None:
        v_stats = _compute_regime_stats("verified", base_verified, cand_verified)

    reasons: list[str] = []
    accepted = True

    s_net = s_stats.n01_fixed - s_stats.n10_broken
    if s_net < cfg.min_single_shot_net_gain:
        accepted = False
        reasons.append(
            f"single_shot net gain {s_net:+d} < required {cfg.min_single_shot_net_gain:+d} "
            f"(+{s_stats.n01_fixed}/-{s_stats.n10_broken})"
        )
    if s_stats.n10_broken > cfg.max_single_shot_regressions:
        accepted = False
        reasons.append(
            f"single_shot regressions {s_stats.n10_broken} > max {cfg.max_single_shot_regressions}"
        )
    if cfg.require_significant_or_zero_regressions:
        if s_stats.n10_broken > 0 and s_stats.mcnemar_exact_p > cfg.alpha:
            accepted = False
            reasons.append(
                f"single_shot gain not statistically significant (p={s_stats.mcnemar_exact_p:.4f} > {cfg.alpha}) "
                f"with {s_stats.n10_broken} regression(s)"
            )

    if v_stats is not None:
        v_net = v_stats.n01_fixed - v_stats.n10_broken
        if v_net < cfg.min_verified_net_gain:
            accepted = False
            reasons.append(
                f"verified consensus regressed ({v_net:+d} < min {cfg.min_verified_net_gain:+d}, "
                f"+{v_stats.n01_fixed}/-{v_stats.n10_broken})"
            )
        if v_stats.n10_broken > cfg.max_verified_regressions:
            accepted = False
            reasons.append(
                f"verified consensus regressions {v_stats.n10_broken} > max {cfg.max_verified_regressions}"
            )

    anchor_delta: Optional[float] = None
    if base_anchor_pct is not None and cand_anchor_pct is not None:
        anchor_delta = round(float(cand_anchor_pct) - float(base_anchor_pct), 2)
        if anchor_delta < -abs(cfg.max_anchor_drop_pct):
            accepted = False
            reasons.append(
                f"anchor benchmark dropped {anchor_delta:+.2f} pp (limit -{abs(cfg.max_anchor_drop_pct):.2f} pp)"
            )

    # Composite score rewards single-shot compression + verified synergy, penalizing regressions
    composite = s_stats.delta_pct - 2.0 * (100.0 * s_stats.n10_broken / s_stats.n)
    if v_stats is not None:
        composite += 0.75 * v_stats.delta_pct - 2.5 * (100.0 * v_stats.n10_broken / v_stats.n)
    if anchor_delta is not None and anchor_delta < 0:
        composite += anchor_delta

    if accepted:
        ok_note = (
            f"ACCEPTED: single_shot {s_stats.base_pct:.1f}%->{s_stats.cand_pct:.1f}% "
            f"(+{s_stats.n01_fixed}/-{s_stats.n10_broken}, p={s_stats.mcnemar_exact_p:.4f})"
        )
        if v_stats is not None:
            ok_note += (
                f" | verified {v_stats.base_pct:.1f}%->{v_stats.cand_pct:.1f}% "
                f"(+{v_stats.n01_fixed}/-{v_stats.n10_broken})"
            )
        reasons.append(ok_note)

    return GateDecision(
        accepted=accepted,
        reasons=reasons,
        single_shot=s_stats,
        verified=v_stats,
        anchor_delta_pct=anchor_delta,
        composite_score=round(composite, 4),
    )


def load_outcomes_from_eval_json(path: Path, arm: Optional[str] = None) -> dict[str, bool]:
    """Load `{problem_id: passed_bool}` from any Cortheon eval/ablation result JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results")
    rows: list[dict[str, Any]]
    if isinstance(results, dict):
        if arm is not None:
            if arm not in results:
                raise ValueError(f"Arm {arm!r} not found in {path} (available: {sorted(results.keys())})")
            rows = list(results[arm])
        elif len(results) == 1:
            rows = list(next(iter(results.values())))
        else:
            raise ValueError(f"Multi-arm JSON {path} requires explicit arm selection (available: {sorted(results.keys())})")
    elif isinstance(results, list):
        rows = [r for r in results if arm is None or r.get("arm") == arm]
    else:
        raise ValueError(f"Unrecognized result JSON schema in {path}")

    out: dict[str, bool] = {}
    for row in rows:
        pid = row.get("pid") if "pid" in row else row.get("id")
        if pid is None:
            continue
        passed = row["passed"] if "passed" in row else row.get("correct", False)
        out[str(pid)] = bool(passed)
    return out


# ---------------------------------------------------------------------------
# 3. Pure-NumPy Interference-Guarded Adapter Soup Merging
# ---------------------------------------------------------------------------


@dataclass
class AdapterMergeAudit:
    num_adapters: int
    num_tensors: int
    normalized_weights: list[float]
    sign_consensus: bool
    trim_density: float
    sha256: str
    out_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def merge_adapter_weights(
    adapters: Sequence[Mapping[str, np.ndarray]],
    *,
    weights: Optional[Sequence[float]] = None,
    sign_consensus: bool = True,
    trim_density: float = 1.0,
) -> tuple[dict[str, np.ndarray], list[float]]:
    """Merge adapter delta dictionaries with exact shape validation and optional TIES sign-consensus.

    - Strictly validates identical key sets and tensor shapes across all adapters.
    - When `sign_consensus=True` and `len(adapters) >= 2`, resolves per-element sign
      interference by electing the weighted-magnitude majority sign and averaging only
      the aligned deltas (preventing conflicting generations from cancelling out).
    - When `0 < trim_density < 1.0`, zeroes out the smallest `(1 - trim_density)` magnitude
      entries per tensor before sign election to suppress low-signal gradient noise.
    """
    if not adapters:
        raise ValueError("At least one adapter dictionary is required for merging.")
    if not (0.0 < trim_density <= 1.0):
        raise ValueError(f"trim_density must be in (0, 1], got {trim_density}")

    n = len(adapters)
    if weights is None:
        raw_w = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError(f"weights length ({len(weights)}) must match adapters ({n})")
        raw_w = [float(w) for w in weights]
        if any(w < 0.0 or not math.isfinite(w) for w in raw_w) or sum(raw_w) <= 0.0:
            raise ValueError(f"weights must be finite non-negative with positive sum, got {raw_w}")
        w_sum = sum(raw_w)
        raw_w = [w / w_sum for w in raw_w]

    ref_keys = sorted(adapters[0].keys())
    if not ref_keys:
        raise ValueError("Adapter dictionaries must not be empty.")
    for idx, ad in enumerate(adapters[1:], start=1):
        keys_i = sorted(ad.keys())
        if keys_i != ref_keys:
            raise ValueError(f"Adapter #{idx} key mismatch against adapter #0.")

    merged: dict[str, np.ndarray] = {}
    for k in ref_keys:
        ref_arr = np.asarray(adapters[0][k])
        ref_shape = ref_arr.shape
        ref_dtype = ref_arr.dtype
        stack_list: list[np.ndarray] = []
        for idx, ad in enumerate(adapters):
            arr = np.asarray(ad[k])
            if arr.shape != ref_shape:
                raise ValueError(f"Shape mismatch for tensor {k!r} in adapter #{idx}: {arr.shape} vs {ref_shape}")
            f32 = arr.astype(np.float32, copy=True)
            if trim_density < 1.0 and f32.size > 0:
                thresh = float(np.quantile(np.abs(f32), 1.0 - trim_density))
                f32[np.abs(f32) < thresh] = 0.0
            stack_list.append(f32)

        stacked = np.stack(stack_list, axis=0)  # (N, ...)
        w_vec = np.asarray(raw_w, dtype=np.float32).reshape((n,) + (1,) * ref_arr.ndim)

        if not sign_consensus or n == 1:
            comb = np.sum(stacked * w_vec, axis=0)
        else:
            # Elect weighted signed mass per parameter; average only sign-aligned contributors
            elected_sign = np.sign(np.sum(stacked * w_vec, axis=0))
            aligned_mask = (np.sign(stacked) == elected_sign) & (elected_sign != 0)
            eff_w = np.where(aligned_mask, w_vec, 0.0)
            denom = np.sum(eff_w, axis=0)
            comb = np.where(denom > 0.0, np.sum(stacked * eff_w, axis=0) / np.where(denom > 0.0, denom, 1.0), 0.0)

        merged[k] = comb.astype(ref_dtype)

    return merged, [round(w, 6) for w in raw_w]


def merge_npz_adapters(
    adapter_paths: Sequence[Path],
    out_path: Path,
    *,
    weights: Optional[Sequence[float]] = None,
    sign_consensus: bool = True,
    trim_density: float = 1.0,
) -> AdapterMergeAudit:
    """Load `.npz` LoRA/control adapters, merge safely, write atomically, and return SHA-256 audit."""
    loaded: list[dict[str, np.ndarray]] = []
    for p in adapter_paths:
        with np.load(p, allow_pickle=False) as z:
            loaded.append({k: z[k] for k in z.files})

    merged, norm_w = merge_adapter_weights(
        loaded,
        weights=weights,
        sign_consensus=sign_consensus,
        trim_density=trim_density,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(tmp, **merged)
    tmp.replace(out_path)

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return AdapterMergeAudit(
        num_adapters=len(adapter_paths),
        num_tensors=len(merged),
        normalized_weights=norm_w,
        sign_consensus=sign_consensus,
        trim_density=trim_density,
        sha256=digest,
        out_path=str(out_path),
    )


# ---------------------------------------------------------------------------
# 4. Generational Evolution Ledger & Autonomous Pluggable Loop
# ---------------------------------------------------------------------------


@dataclass
class GenerationRecord:
    generation: int
    timestamp_utc: str
    curriculum_kept: int
    curriculum_weight: float
    adapter_path: str
    accepted: bool
    gate: dict[str, Any]
    active_adapter_after: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvolutionLedger:
    """Atomic, PII-scrubbed audit ledger tracking every gated RSI generation."""

    def __init__(self, ledger_path: Path) -> None:
        self.path = ledger_path
        self.records: list[GenerationRecord] = []
        self.merged_audit: Optional[dict[str, Any]] = None
        if self.path.is_file():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for r in raw.get("generations", []):
                self.records.append(GenerationRecord(**r))
            self.merged_audit = raw.get("merged_audit")

    def append(self, rec: GenerationRecord) -> None:
        self.records.append(rec)
        self.save()

    def set_merged_audit(self, audit: AdapterMergeAudit) -> None:
        self.merged_audit = audit.to_dict()
        self.save()

    def accepted_records(self) -> list[GenerationRecord]:
        return [r for r in self.records if r.accepted]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "cortheon.evolve.ledger.v1",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "total_generations": len(self.records),
            "accepted_generations": len(self.accepted_records()),
            "generations": [r.to_dict() for r in self.records],
            "merged_audit": self.merged_audit,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(scrub_pii(json.dumps(payload, indent=2) + "\n"), encoding="utf-8")
        tmp.replace(self.path)


async def run_evolution_loop(
    *,
    work_dir: Path,
    max_generations: int,
    patience: int,
    harvest_fn: Callable[[int, Optional[str]], Awaitable[Sequence[FlywheelPair]]],
    train_fn: Callable[[int, Path, Path, Optional[str]], Awaitable[Path]],
    eval_fn: Callable[[int, Optional[str]], Awaitable[dict[str, Any]]],
    gate_config: Optional[GateConfig] = None,
    merge_top_k: int = 3,
    sign_consensus: bool = True,
) -> EvolutionLedger:
    """Run an autonomous, statistically gated multi-generation Cortheon RSI loop.

    Each generation:
      1. Harvests execution-verified contrastive traces (`harvest_fn`) and writes weighted DPO/SFT JSONL.
      2. Trains a candidate adapter (`train_fn`) from the active accepted adapter.
      3. Evaluates the candidate on locked held-out benchmarks (`eval_fn`) and gates via
         `evaluate_generation_gate`.
      4. Promotes ONLY strictly accepted candidates (rolling back automatically on rejection),
         stops early after `patience` consecutive rejections, and produces a sign-consensus
         adapter soup across the top-K accepted generations when `.npz` adapters are available.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    ledger = EvolutionLedger(work_dir / "evolve_ledger.json")
    cfg = gate_config or GateConfig()

    # Baseline evaluation (generation 0)
    active_adapter: Optional[str] = None
    best_eval = await eval_fn(0, None)
    consecutive_rejects = 0
    accepted_paths: list[tuple[Path, float]] = []

    for gen in range(1, max_generations + 1):
        gen_dir = work_dir / f"gen_{gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        pairs = await harvest_fn(gen, active_adapter)
        examples, cur_summary = build_weighted_curriculum(pairs)
        dpo_path = gen_dir / "curriculum_dpo_weighted.jsonl"
        sft_path = gen_dir / "curriculum_sft_weighted.jsonl"
        _write_jsonl_atomic(dpo_path, [e.to_dpo_record() for e in examples if e.rejected])
        _write_jsonl_atomic(sft_path, [e.to_sft_record() for e in examples])

        if cur_summary.kept_examples == 0:
            consecutive_rejects += 1
            if consecutive_rejects >= patience:
                break
            continue

        cand_adapter = await train_fn(gen, dpo_path, sft_path, active_adapter)
        cand_eval = await eval_fn(gen, str(cand_adapter))

        decision = evaluate_generation_gate(
            base_single_shot=best_eval["single_shot"],
            cand_single_shot=cand_eval["single_shot"],
            base_verified=best_eval.get("verified"),
            cand_verified=cand_eval.get("verified"),
            base_anchor_pct=best_eval.get("anchor_pct"),
            cand_anchor_pct=cand_eval.get("anchor_pct"),
            config=cfg,
        )

        if decision.accepted:
            active_adapter = str(cand_adapter)
            best_eval = cand_eval
            consecutive_rejects = 0
            accepted_paths.append((Path(cand_adapter), max(0.05, float(decision.composite_score))))
        else:
            consecutive_rejects += 1

        def _safe_label(path_str: Optional[str]) -> str:
            if not path_str:
                return "base"
            p = Path(path_str)
            try:
                return scrub_pii(str(p.relative_to(work_dir)))
            except ValueError:
                return scrub_pii(p.name)

        ledger.append(
            GenerationRecord(
                generation=gen,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                curriculum_kept=cur_summary.kept_examples,
                curriculum_weight=cur_summary.total_weight,
                adapter_path=_safe_label(str(cand_adapter)),
                accepted=decision.accepted,
                gate=decision.to_dict(),
                active_adapter_after=_safe_label(active_adapter),
            )
        )

        if consecutive_rejects >= patience:
            break

    # Optional top-K sign-consensus adapter soup across accepted `.npz` adapters
    accepted_npz = [
        (p, score)
        for p, score in accepted_paths
        if p.suffix == ".npz" and p.is_file()
    ]
    if len(accepted_npz) >= 2 and merge_top_k >= 2:
        accepted_npz.sort(key=lambda item: item[1], reverse=True)
        top = accepted_npz[:merge_top_k]
        audit = merge_npz_adapters(
            [p for p, _ in top],
            work_dir / "evolved_top_k_merged.npz",
            weights=[w for _, w in top],
            sign_consensus=sign_consensus,
        )
        ledger.set_merged_audit(audit)

    return ledger


def format_curriculum_plan(summary: CurriculumSummary) -> str:
    """Format a crisp terminal report for `cortheon evolve --dry-run`."""
    lines = [
        "Cortheon Evolve — Oracle-Free Weighted Curriculum",
        f"  Input pairs    : {summary.total_input_pairs}",
        f"  Kept examples  : {summary.kept_examples} (deduped={summary.deduplicated}, dropped_unverified={summary.dropped_unverified})",
        f"  Total weight   : {summary.total_weight:.1f}",
        f"  By tier        : {summary.by_tier}",
        f"  By domain      : {summary.by_domain}",
    ]
    if summary.dpo_path:
        lines.append(f"  Weighted DPO   : {summary.dpo_path}")
    if summary.sft_path:
        lines.append(f"  Weighted SFT   : {summary.sft_path}")
    lines.append(
        "  Next step      : Fine-tune adapter (e.g. mlx_lm.lora / TRL), then gate with "
        "`cortheon evolve --gate-base base.json --gate-cand cand.json`"
    )
    return "\n".join(lines)


def format_gate_decision(decision: GateDecision) -> str:
    """Format a crisp human-readable report for a dual-regime gate decision."""
    verdict = "ACCEPT" if decision.accepted else "REJECT"
    s = decision.single_shot
    lines = [
        f"Cortheon Evolve Gate: [{verdict}] (composite={decision.composite_score:+.2f})",
        f"  single_shot : {s.base_passed}/{s.n} ({s.base_pct:.1f}%) -> {s.cand_passed}/{s.n} ({s.cand_pct:.1f}%) "
        f"| +{s.n01_fixed}/-{s.n10_broken} | exact McNemar p={s.mcnemar_exact_p:.4f}",
    ]
    if decision.verified is not None:
        v = decision.verified
        lines.append(
            f"  verified    : {v.base_passed}/{v.n} ({v.base_pct:.1f}%) -> {v.cand_passed}/{v.n} ({v.cand_pct:.1f}%) "
            f"| +{v.n01_fixed}/-{v.n10_broken} | exact McNemar p={v.mcnemar_exact_p:.4f}"
        )
    if decision.anchor_delta_pct is not None:
        lines.append(f"  anchor      : {decision.anchor_delta_pct:+.2f} pp")
    for r in decision.reasons:
        lines.append(f"  reason      : {r}")
    return "\n".join(lines)

"""Exact paired statistical analysis over benchmark result JSONs in `results/`.

Computes per-suite and pooled:
  - Accuracy k/n with exact Wilson 95% confidence intervals (no scipy required)
  - Paired 2x2 contingency table: n11 (both pass), n01 (Cortheon fixed),
    n10 (Cortheon broke), n00 (both fail)
  - Exact two-sided McNemar p-value via exact binomial tail sum
  - Average wall-clock time and token/call ratios when logged

Usage:
    python -m bench.stats --dir results
    python -m bench.stats --dir results --json
    python -m bench.stats --pair results/official_aime2025_deepseek_vanilla.json results/official_aime2025_deepseek_cortheon.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Optional

WILSON_Z_95: float = NormalDist().inv_cdf(0.975)


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return the exact Wilson score confidence interval (lo, hi) in [0.0, 1.0]."""
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"Invalid binomial counts: k={k}, n={n}")
    if n == 0:
        return (0.0, 0.0)
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    z = WILSON_Z_95 if abs(confidence - 0.95) < 1e-12 else NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z2 = z * z
    p = k / n
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact_p(n01: int, n10: int) -> float:
    """Exact two-sided McNemar p-value via exact Binomial(m, 0.5) tail sum.

    n01: baseline failed, treatment passed (fixed)
    n10: baseline passed, treatment failed (regressed)
    """
    if n01 < 0 or n10 < 0:
        raise ValueError(f"Discordant counts must be non-negative: n01={n01}, n10={n10}")
    m = n01 + n10
    if m == 0:
        return 1.0
    k = min(n01, n10)
    tail_int = sum(math.comb(m, i) for i in range(k + 1))
    # Exact arbitrary-precision ratio: 2 * tail_int / 2^m (safe for arbitrarily large m)
    two_tail = 2 * tail_int
    denom = 1 << m
    if two_tail >= denom:
        return 1.0
    return two_tail / denom


def paired_contingency(
    baseline: dict[str, bool],
    treatment: dict[str, bool],
) -> dict[str, int]:
    """Compute paired 2x2 counts over shared problem IDs."""
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


def _row_pid(row: dict[str, Any]) -> str:
    pid = row.get("pid") if "pid" in row else row.get("id")
    if pid is None:
        raise ValueError(f"Result row missing 'pid'/'id': {row}")
    return str(pid)


def _row_passed(row: dict[str, Any]) -> bool:
    if "passed" in row:
        return bool(row["passed"])
    if "correct" in row:
        return bool(row["correct"])
    raise ValueError(f"Result row missing 'passed'/'correct': {row}")


def _row_wall_s(row: dict[str, Any]) -> Optional[float]:
    val = row.get("wall_s")
    return float(val) if isinstance(val, (int, float)) else None


def _row_calls(row: dict[str, Any]) -> Optional[int]:
    stats = row.get("stats")
    if isinstance(stats, dict) and isinstance(stats.get("calls"), (int, float)):
        return int(stats["calls"])
    if isinstance(row.get("llm_calls"), (int, float)):
        return int(row["llm_calls"])
    return None


def _row_total_tokens(row: dict[str, Any]) -> Optional[int]:
    stats = row.get("stats")
    if isinstance(stats, dict):
        p = stats.get("prompt_tokens")
        c = stats.get("completion_tokens")
        if isinstance(p, (int, float)) and isinstance(c, (int, float)):
            return int(p) + int(c)
    if isinstance(row.get("total_tokens"), (int, float)):
        return int(row["total_tokens"])
    p = row.get("prompt_tokens")
    c = row.get("completion_tokens")
    if isinstance(p, (int, float)) and isinstance(c, (int, float)):
        return int(p) + int(c)
    if isinstance(c, (int, float)):
        return int(c)
    return None


def _mean_or_none(vals: list[Optional[float]]) -> Optional[float]:
    present = [v for v in vals if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


@dataclass
class PairedBenchmarkStats:
    suite_key: str
    dataset: str
    model: str
    n: int
    vanilla_passed: int
    cortheon_passed: int
    vanilla_ci: tuple[float, float]
    cortheon_ci: tuple[float, float]
    n11: int
    n01: int
    n10: int
    n00: int
    mcnemar_p: float
    vanilla_avg_wall_s: Optional[float] = None
    cortheon_avg_wall_s: Optional[float] = None
    vanilla_avg_calls: Optional[float] = None
    cortheon_avg_calls: Optional[float] = None
    vanilla_avg_tokens: Optional[float] = None
    cortheon_avg_tokens: Optional[float] = None
    vanilla_file: str = ""
    cortheon_file: str = ""

    @property
    def vanilla_pct(self) -> float:
        return (100.0 * self.vanilla_passed / self.n) if self.n else 0.0

    @property
    def cortheon_pct(self) -> float:
        return (100.0 * self.cortheon_passed / self.n) if self.n else 0.0

    @property
    def delta_pct(self) -> float:
        return self.cortheon_pct - self.vanilla_pct

    @property
    def wall_ratio(self) -> Optional[float]:
        if (
            self.vanilla_avg_wall_s is not None
            and self.cortheon_avg_wall_s is not None
            and self.vanilla_avg_wall_s > 0
        ):
            return self.cortheon_avg_wall_s / self.vanilla_avg_wall_s
        return None

    @property
    def call_ratio(self) -> Optional[float]:
        if (
            self.vanilla_avg_calls is not None
            and self.cortheon_avg_calls is not None
            and self.vanilla_avg_calls > 0
        ):
            return self.cortheon_avg_calls / self.vanilla_avg_calls
        return None

    @property
    def token_ratio(self) -> Optional[float]:
        if (
            self.vanilla_avg_tokens is not None
            and self.cortheon_avg_tokens is not None
            and self.vanilla_avg_tokens > 0
        ):
            return self.cortheon_avg_tokens / self.vanilla_avg_tokens
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_key": self.suite_key,
            "dataset": self.dataset,
            "model": self.model,
            "n": self.n,
            "vanilla": {
                "passed": self.vanilla_passed,
                "total": self.n,
                "accuracy_pct": round(self.vanilla_pct, 2),
                "wilson_ci_95_pct": [round(100.0 * self.vanilla_ci[0], 2), round(100.0 * self.vanilla_ci[1], 2)],
                "avg_wall_s": round(self.vanilla_avg_wall_s, 2) if self.vanilla_avg_wall_s is not None else None,
                "avg_calls": round(self.vanilla_avg_calls, 2) if self.vanilla_avg_calls is not None else None,
                "avg_tokens": round(self.vanilla_avg_tokens, 1) if self.vanilla_avg_tokens is not None else None,
                "file": self.vanilla_file,
            },
            "cortheon": {
                "passed": self.cortheon_passed,
                "total": self.n,
                "accuracy_pct": round(self.cortheon_pct, 2),
                "wilson_ci_95_pct": [round(100.0 * self.cortheon_ci[0], 2), round(100.0 * self.cortheon_ci[1], 2)],
                "avg_wall_s": round(self.cortheon_avg_wall_s, 2) if self.cortheon_avg_wall_s is not None else None,
                "avg_calls": round(self.cortheon_avg_calls, 2) if self.cortheon_avg_calls is not None else None,
                "avg_tokens": round(self.cortheon_avg_tokens, 1) if self.cortheon_avg_tokens is not None else None,
                "file": self.cortheon_file,
            },
            "delta_pct": round(self.delta_pct, 2),
            "paired": {
                "both_pass_n11": self.n11,
                "cortheon_fixed_n01": self.n01,
                "cortheon_broke_n10": self.n10,
                "both_fail_n00": self.n00,
            },
            "mcnemar_exact_p": self.mcnemar_p,
            "ratios": {
                "wall_c_over_v": round(self.wall_ratio, 3) if self.wall_ratio is not None else None,
                "calls_c_over_v": round(self.call_ratio, 3) if self.call_ratio is not None else None,
                "tokens_c_over_v": round(self.token_ratio, 3) if self.token_ratio is not None else None,
            },
        }


def analyze_pair(
    vanilla_data: dict[str, Any],
    cortheon_data: dict[str, Any],
    *,
    suite_key: str = "",
    vanilla_file: str = "",
    cortheon_file: str = "",
) -> PairedBenchmarkStats:
    """Compute exact paired statistics from loaded Vanilla and Cortheon JSON payloads."""
    v_rows = {_row_pid(r): r for r in vanilla_data.get("results", [])}
    c_rows = {_row_pid(r): r for r in cortheon_data.get("results", [])}
    shared_pids = sorted(set(v_rows.keys()) & set(c_rows.keys()))

    ct = paired_contingency(
        {pid: _row_passed(v_rows[pid]) for pid in shared_pids},
        {pid: _row_passed(c_rows[pid]) for pid in shared_pids},
    )
    n = ct["n"]
    vp = ct["baseline_passed"]
    cp = ct["treatment_passed"]

    v_wall = _mean_or_none([_row_wall_s(v_rows[pid]) for pid in shared_pids])
    c_wall = _mean_or_none([_row_wall_s(c_rows[pid]) for pid in shared_pids])
    v_calls = _mean_or_none([float(x) if x is not None else None for x in (_row_calls(v_rows[pid]) for pid in shared_pids)])
    c_calls = _mean_or_none([float(x) if x is not None else None for x in (_row_calls(c_rows[pid]) for pid in shared_pids)])
    v_tok = _mean_or_none([float(x) if x is not None else None for x in (_row_total_tokens(v_rows[pid]) for pid in shared_pids)])
    c_tok = _mean_or_none([float(x) if x is not None else None for x in (_row_total_tokens(c_rows[pid]) for pid in shared_pids)])

    dataset = str(cortheon_data.get("dataset") or vanilla_data.get("dataset") or suite_key)
    model = str(cortheon_data.get("model") or vanilla_data.get("model") or "unknown")
    return PairedBenchmarkStats(
        suite_key=suite_key or f"{dataset}__{model}",
        dataset=dataset,
        model=model,
        n=n,
        vanilla_passed=vp,
        cortheon_passed=cp,
        vanilla_ci=wilson_ci(vp, n),
        cortheon_ci=wilson_ci(cp, n),
        n11=ct["n11"],
        n01=ct["n01"],
        n10=ct["n10"],
        n00=ct["n00"],
        mcnemar_p=mcnemar_exact_p(ct["n01"], ct["n10"]),
        vanilla_avg_wall_s=v_wall,
        cortheon_avg_wall_s=c_wall,
        vanilla_avg_calls=v_calls,
        cortheon_avg_calls=c_calls,
        vanilla_avg_tokens=v_tok,
        cortheon_avg_tokens=c_tok,
        vanilla_file=vanilla_file,
        cortheon_file=cortheon_file,
    )


def pool_stats(label: str, items: Iterable[PairedBenchmarkStats]) -> PairedBenchmarkStats:
    """Pool exact paired contingency tables and weighted telemetry across suites."""
    seq = list(items)
    n = sum(x.n for x in seq)
    n11 = sum(x.n11 for x in seq)
    n01 = sum(x.n01 for x in seq)
    n10 = sum(x.n10 for x in seq)
    n00 = sum(x.n00 for x in seq)
    vp = n11 + n10
    cp = n11 + n01

    def _wavg(attr: str) -> Optional[float]:
        num = 0.0
        den = 0
        for x in seq:
            v = getattr(x, attr)
            if v is not None and x.n > 0:
                num += v * x.n
                den += x.n
        return (num / den) if den > 0 else None

    models = sorted({x.model for x in seq})
    model_str = models[0] if len(models) == 1 else f"{len(models)} models"
    return PairedBenchmarkStats(
        suite_key=label,
        dataset=label,
        model=model_str,
        n=n,
        vanilla_passed=vp,
        cortheon_passed=cp,
        vanilla_ci=wilson_ci(vp, n),
        cortheon_ci=wilson_ci(cp, n),
        n11=n11,
        n01=n01,
        n10=n10,
        n00=n00,
        mcnemar_p=mcnemar_exact_p(n01, n10),
        vanilla_avg_wall_s=_wavg("vanilla_avg_wall_s"),
        cortheon_avg_wall_s=_wavg("cortheon_avg_wall_s"),
        vanilla_avg_calls=_wavg("vanilla_avg_calls"),
        cortheon_avg_calls=_wavg("cortheon_avg_calls"),
        vanilla_avg_tokens=_wavg("vanilla_avg_tokens"),
        cortheon_avg_tokens=_wavg("cortheon_avg_tokens"),
    )


def _suite_sort_key(suite_key: str) -> tuple[int, str]:
    if suite_key.startswith("official_aime"):
        return (0, suite_key)
    if suite_key.startswith("official_lcb"):
        return (1, suite_key)
    if suite_key.startswith("official_bcb"):
        return (2, suite_key)
    if suite_key.startswith("open_ended"):
        return (3, suite_key)
    if suite_key.startswith("gauntlet"):
        return (4, suite_key)
    return (5, suite_key)


def discover_pairs(results_dir: Path) -> list[tuple[str, Path, Path]]:
    """Find all paired `*_vanilla.json` and `*_cortheon.json` (or `*_forge.json`) files."""
    vanilla_map: dict[str, Path] = {}
    cortheon_map: dict[str, Path] = {}
    for p in sorted(results_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        stem = p.stem
        if stem.endswith("_vanilla"):
            vanilla_map[stem[: -len("_vanilla")]] = p
        elif stem.endswith("_cortheon"):
            cortheon_map[stem[: -len("_cortheon")]] = p
        elif stem.endswith("_forge"):
            cortheon_map.setdefault(stem[: -len("_forge")], p)

    keys = sorted(set(vanilla_map.keys()) & set(cortheon_map.keys()), key=_suite_sort_key)
    return [(k, vanilla_map[k], cortheon_map[k]) for k in keys]


def analyze_directory(results_dir: Path) -> tuple[list[PairedBenchmarkStats], list[PairedBenchmarkStats]]:
    """Analyze all paired result files in `results_dir` and compute standard pooled summaries."""
    pairs = discover_pairs(results_dir)
    per_suite: list[PairedBenchmarkStats] = []
    for key, v_path, c_path in pairs:
        v_data = json.loads(v_path.read_text(encoding="utf-8"))
        c_data = json.loads(c_path.read_text(encoding="utf-8"))
        if not isinstance(v_data.get("results"), list) or not isinstance(c_data.get("results"), list):
            continue
        per_suite.append(
            analyze_pair(
                v_data,
                c_data,
                suite_key=key,
                vanilla_file=v_path.name,
                cortheon_file=c_path.name,
            )
        )

    pooled: list[PairedBenchmarkStats] = []
    aime = [s for s in per_suite if s.suite_key.startswith("official_aime")]
    code_off = [s for s in per_suite if s.suite_key.startswith(("official_lcb", "official_bcb"))]
    official_all = [s for s in per_suite if s.suite_key.startswith("official_")]
    gauntlets = [s for s in per_suite if s.suite_key.startswith("gauntlet")]

    if len(aime) >= 2:
        pooled.append(pool_stats("POOLED: Official AIME 2025", aime))
    if len(code_off) >= 2:
        pooled.append(pool_stats("POOLED: Official Code (LCB + BCB)", code_off))
    if len(official_all) >= 2:
        pooled.append(pool_stats("POOLED: All Official External", official_all))
    if len(gauntlets) >= 2:
        pooled.append(pool_stats("POOLED: Gauntlets (I + II)", gauntlets))
    if len(per_suite) >= 2:
        pooled.append(pool_stats("POOLED: All Benchmarks", per_suite))

    return per_suite, pooled


def format_p_value(p: float) -> str:
    if p < 1e-4:
        base = f"{p:.2e}"
    else:
        base = f"{p:.4f}"
    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = ""
    return f"{base}{sig}"


def _fmt_arm(k: int, n: int, ci: tuple[float, float]) -> str:
    pct = (100.0 * k / n) if n else 0.0
    return f"{k}/{n} ({pct:5.1f}%) [{100*ci[0]:4.1f}, {100*ci[1]:4.1f}]"


def _fmt_wall(s: PairedBenchmarkStats) -> str:
    if s.vanilla_avg_wall_s is None or s.cortheon_avg_wall_s is None:
        return "—"
    ratio = f" ({s.wall_ratio:.2f}x)" if s.wall_ratio is not None else ""
    return f"{s.vanilla_avg_wall_s:.1f}s -> {s.cortheon_avg_wall_s:.1f}s{ratio}"


def _fmt_compute(s: PairedBenchmarkStats) -> str:
    parts: list[str] = []
    if s.call_ratio is not None:
        parts.append(f"calls {s.vanilla_avg_calls:.1f}->{s.cortheon_avg_calls:.1f} ({s.call_ratio:.2f}x)")
    elif s.cortheon_avg_calls is not None:
        parts.append(f"C:{s.cortheon_avg_calls:.1f} calls")
    if s.token_ratio is not None:
        parts.append(f"tok {s.token_ratio:.2f}x")
    elif s.cortheon_avg_tokens is not None:
        parts.append(f"C:{s.cortheon_avg_tokens/1000.0:.1f}k tok")
    return " · ".join(parts) if parts else "—"


def format_markdown_table(
    per_suite: list[PairedBenchmarkStats],
    pooled: list[PairedBenchmarkStats],
) -> str:
    header = (
        "| Suite | Model | Vanilla k/n [95% Wilson CI] | Cortheon k/n [95% Wilson CI] "
        "| Delta | Paired (n11 / +n01 / -n10 / n00) | Exact McNemar p | Avg Wall (V -> C) | Compute (Calls / Tok) |"
    )
    sep = "|---|---|---:|---:|---:|:---:|---:|---:|---:|"
    lines = [header, sep]

    def _row_md(s: PairedBenchmarkStats, bold: bool = False) -> str:
        name = f"**{s.suite_key}**" if bold else s.suite_key
        sign = "+" if s.delta_pct >= 0 else ""
        delta = f"{sign}{s.delta_pct:.1f} pts"
        if bold:
            delta = f"**{delta}**"
        paired_str = f"{s.n11} / +{s.n01} / -{s.n10} / {s.n00}"
        return (
            f"| {name} | {s.model} | {_fmt_arm(s.vanilla_passed, s.n, s.vanilla_ci)} "
            f"| {_fmt_arm(s.cortheon_passed, s.n, s.cortheon_ci)} "
            f"| {delta} | {paired_str} | {format_p_value(s.mcnemar_p)} "
            f"| {_fmt_wall(s)} | {_fmt_compute(s)} |"
        )

    for s in per_suite:
        lines.append(_row_md(s, bold=False))
    for s in pooled:
        lines.append(_row_md(s, bold=True))

    lines.append("")
    lines.append(
        "_Notes: 95% CIs are exact Wilson score intervals. Paired counts are `n11` (both pass), "
        "`+n01` (Cortheon fixed), `-n10` (Cortheon broke), `n00` (both fail). "
        "`Exact McNemar p` is the exact two-sided binomial tail probability "
        "`min(1, 2 * sum_{k=0}^{min(n01,n10)} C(n01+n10, k) * 0.5^(n01+n10))` "
        "(`*` p<0.05, `**` p<0.01, `***` p<0.001)._"
    )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute exact paired McNemar p-values, Wilson 95% CIs, and compute ratios over results/."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("results"),
        help="Directory containing paired *_vanilla.json and *_cortheon.json files (default: results).",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("VANILLA_JSON", "CORTHEON_JSON"),
        default=None,
        help="Analyze a single explicit (vanilla, cortheon) JSON file pair instead of scanning --dir.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit machine-readable JSON instead of a Markdown table.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output file path (otherwise prints to stdout).",
    )
    args = parser.parse_args(argv)

    if args.pair:
        v_path, c_path = Path(args.pair[0]), Path(args.pair[1])
        v_data = json.loads(v_path.read_text(encoding="utf-8"))
        c_data = json.loads(c_path.read_text(encoding="utf-8"))
        per_suite = [
            analyze_pair(
                v_data,
                c_data,
                suite_key=v_path.stem.removesuffix("_vanilla"),
                vanilla_file=v_path.name,
                cortheon_file=c_path.name,
            )
        ]
        pooled: list[PairedBenchmarkStats] = []
    else:
        per_suite, pooled = analyze_directory(args.dir)

    if args.as_json:
        output = json.dumps(
            {
                "suites": [s.to_dict() for s in per_suite],
                "pooled": [s.to_dict() for s in pooled],
            },
            indent=2,
        )
    else:
        output = format_markdown_table(per_suite, pooled)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Adjudicate the hard rule: does the substrate make a small model behave like a
frontier model?

The rule is a claim about capability, and it needs a test that can fail. Two
things make the obvious approach unsound:

**A non-significant difference is not parity.** Superiority tests answer "is
there a detectable difference"; failing to find one is not evidence of
equivalence, only of insufficient evidence. Parity must be argued with a bound:
the substrate is frontier-like if the *lower* bound of its performance clears a
declared threshold.

**Most cases carry no information.** On a case the small model already solves,
substrate and frontier both succeed and nothing is learned about the rule. The
only cases that can discriminate are those where the bare model fails and the
frontier succeeds — the capability gap. A suite is only as informative as the
size of that gap, regardless of how many cases it contains.

So the question this module answers is: *of the cases where the small model
could not succeed alone and the frontier could, how many does the substrate
close, and what is the worst-case-consistent closure rate?*

Intervals are Clopper-Pearson: exact and conservative. On the sample sizes this
project runs, an approximate interval would overstate the evidence in precisely
the direction that flatters the claim.

Dependency-free by product invariant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import comb, log
from typing import Any

__all__ = [
    "GapAnalysis",
    "capability_gap",
    "clopper_pearson",
    "gap_analysis",
    "required_gap_cases",
]

DEFAULT_CONFIDENCE = 0.95
# The bar for "behaves like a frontier model" on gap cases. Declared here so it
# cannot be chosen after seeing a result.
DEFAULT_PARITY_THRESHOLD = 0.80


def _binomial_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""

    return sum(comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(k, n + 1))


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""

    return sum(comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(0, k + 1))


def clopper_pearson(
    successes: int,
    total: int,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial confidence interval.

    Solved by bisection on the exact binomial tails rather than an incomplete
    beta, so no numerical library is required. Conservative by construction: the
    true coverage is at least ``confidence``.
    """

    if total < 0 or successes < 0 or successes > total:
        raise ValueError(f"invalid counts: {successes}/{total}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1): {confidence!r}")
    if total == 0:
        return (0.0, 1.0)

    alpha = 1.0 - confidence
    tail = alpha / 2.0

    def bisect(predicate, low: float, high: float) -> float:
        for _ in range(200):
            mid = (low + high) / 2.0
            if predicate(mid):
                low = mid
            else:
                high = mid
        return (low + high) / 2.0

    if successes == 0:
        lower = 0.0
    else:
        # Largest p with P(X >= k | p) <= tail.
        lower = bisect(lambda p: _binomial_sf(successes, total, p) <= tail, 0.0, 1.0)

    if successes == total:
        upper = 1.0
    else:
        # Smallest p with P(X <= k | p) <= tail.
        upper = bisect(lambda p: _binomial_cdf(successes, total, p) >= tail, 0.0, 1.0)

    return (max(0.0, lower), min(1.0, upper))


def required_gap_cases(
    threshold: float = DEFAULT_PARITY_THRESHOLD,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> int:
    """Smallest capability gap that could ever support the parity claim.

    Even a flawless run cannot prove much from few cases: with every gap case
    closed, the exact lower bound is ``(alpha/2) ** (1/n)``. This returns the
    smallest n for which that bound clears ``threshold``, which is the suite size
    to build *before* running anything.
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1): {threshold!r}")
    tail = (1.0 - confidence) / 2.0
    # (tail) ** (1/n) >= threshold  <=>  n >= ln(tail) / ln(threshold)
    exact = log(tail) / log(threshold)
    n = int(exact)
    while clopper_pearson(n, n, confidence=confidence)[0] < threshold:
        n += 1
        if n > 100_000:  # pragma: no cover - unreachable for sane thresholds
            break
    return n


@dataclass(frozen=True)
class GapAnalysis:
    """The hard rule, adjudicated."""

    total_cases: int
    gap_cases: tuple[str, ...]
    closed: int
    closure_rate: float
    interval: tuple[float, float]
    threshold: float
    confidence: float
    frontier_only: tuple[str, ...]
    substrate_only: tuple[str, ...]
    required_cases: int
    notes: tuple[str, ...]

    @property
    def gap_size(self) -> int:
        return len(self.gap_cases)

    @property
    def informative(self) -> bool:
        """Could this suite support the claim even with a perfect score?"""

        return self.gap_size >= self.required_cases

    @property
    def frontier_like(self) -> bool:
        """True only when the *lower* bound clears the declared threshold."""

        return self.gap_size > 0 and self.interval[0] >= self.threshold

    @property
    def verdict(self) -> str:
        if self.gap_size == 0:
            return (
                "no capability gap in this suite: the bare model already solved "
                "every case the frontier solved, so nothing here can evidence the rule"
            )
        if self.frontier_like:
            return (
                f"frontier-like on the gap: closed {self.closed}/{self.gap_size}, "
                f"lower bound {self.interval[0]:.3f} >= {self.threshold:.2f}"
            )
        if not self.informative:
            return (
                f"inconclusive: {self.gap_size} gap case(s) cannot clear "
                f"{self.threshold:.2f} even at 100% closure; "
                f"{self.required_cases} are required"
            )
        return (
            f"not frontier-like: closed {self.closed}/{self.gap_size}, "
            f"lower bound {self.interval[0]:.3f} < {self.threshold:.2f}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "gap_size": self.gap_size,
            "gap_cases": list(self.gap_cases),
            "closed": self.closed,
            "closure_rate": round(self.closure_rate, 6),
            "interval": [round(self.interval[0], 6), round(self.interval[1], 6)],
            "threshold": self.threshold,
            "confidence": self.confidence,
            "required_cases": self.required_cases,
            "informative": self.informative,
            "frontier_like": self.frontier_like,
            "frontier_only": list(self.frontier_only),
            "substrate_only": list(self.substrate_only),
            "verdict": self.verdict,
            "notes": list(self.notes),
        }


def _index(runs: Sequence[Mapping[str, Any]], condition: str) -> dict[str, Mapping[str, Any]]:
    selected: dict[str, Mapping[str, Any]] = {}
    for run in runs:
        if run.get("condition") != condition:
            continue
        if run.get("process_error") is not None:
            continue
        case_id = run.get("case_id")
        if isinstance(case_id, str):
            selected[case_id] = run
    return selected


def capability_gap(
    runs: Sequence[Mapping[str, Any]],
    *,
    baseline: str = "baseline",
    frontier: str = "frontier",
) -> list[str]:
    """Cases where the bare model failed and the frontier succeeded.

    These are the only cases that can evidence the hard rule. A case the bare
    model already solves proves nothing about the substrate, and a case the
    frontier also fails sets no bar to reach.
    """

    baseline_runs = _index(runs, baseline)
    frontier_runs = _index(runs, frontier)
    return sorted(
        case_id
        for case_id in set(baseline_runs) & set(frontier_runs)
        if baseline_runs[case_id].get("correct") is not True
        and frontier_runs[case_id].get("correct") is True
    )


def gap_analysis(
    runs: Sequence[Mapping[str, Any]],
    *,
    substrate: str = "cortheon",
    baseline: str = "baseline",
    frontier: str = "frontier",
    threshold: float = DEFAULT_PARITY_THRESHOLD,
    confidence: float = DEFAULT_CONFIDENCE,
) -> GapAnalysis:
    """Adjudicate the hard rule against one suite."""

    substrate_runs = _index(runs, substrate)
    baseline_runs = _index(runs, baseline)
    frontier_runs = _index(runs, frontier)
    gap = capability_gap(runs, baseline=baseline, frontier=frontier)

    scored = [case_id for case_id in gap if case_id in substrate_runs]
    closed = sum(substrate_runs[case_id].get("correct") is True for case_id in scored)

    notes: list[str] = []
    missing = [case_id for case_id in gap if case_id not in substrate_runs]
    if missing:
        notes.append(f"{len(missing)} gap case(s) had no usable substrate run and were excluded")
    if not frontier_runs:
        notes.append("no frontier condition present; the rule cannot be assessed")
    if not baseline_runs:
        notes.append("no baseline condition present; the capability gap is undefined")

    all_cases = set(substrate_runs) | set(baseline_runs) | set(frontier_runs)
    ceilinged = [
        case_id
        for case_id in sorted(set(baseline_runs) & set(frontier_runs))
        if baseline_runs[case_id].get("correct") is True
    ]
    if ceilinged and len(ceilinged) > len(gap):
        notes.append(
            f"{len(ceilinged)} of {len(ceilinged) + len(gap)} paired case(s) were "
            "already solved by the bare model and carry no information"
        )

    # Where the substrate and frontier genuinely diverge, for error analysis.
    shared = sorted(set(substrate_runs) & set(frontier_runs))
    frontier_only = tuple(
        case_id
        for case_id in shared
        if frontier_runs[case_id].get("correct") is True
        and substrate_runs[case_id].get("correct") is not True
    )
    substrate_only = tuple(
        case_id
        for case_id in shared
        if substrate_runs[case_id].get("correct") is True
        and frontier_runs[case_id].get("correct") is not True
    )

    return GapAnalysis(
        total_cases=len(all_cases),
        gap_cases=tuple(scored),
        closed=closed,
        closure_rate=closed / len(scored) if scored else 0.0,
        interval=clopper_pearson(closed, len(scored), confidence=confidence),
        threshold=threshold,
        confidence=confidence,
        frontier_only=frontier_only,
        substrate_only=substrate_only,
        required_cases=required_gap_cases(threshold, confidence=confidence),
        notes=tuple(notes),
    )

"""Build suites that can actually settle the hard rule.

The rule — does the substrate make a small model behave like a frontier model —
can only be evidenced on cases where the bare model fails and the frontier
succeeds. Everything else is ceiling or floor. A suite that runs all three
conditions on every case therefore spends most of its budget on runs that
cannot move the verdict: in the published eight-case control, six of eight cases
were already solved by the bare model, so the expensive frontier and substrate
runs on those six taught nothing.

This module plans and screens instead:

1. Run the **cheapest** condition, the bare baseline, across a wide candidate
   pool.
2. Keep the cases it fails. Those are gap candidates.
3. Run the frontier on those only, to confirm there is a bar to clear.
4. Run the substrate on the confirmed gap.

The saving is structural, not incremental: the two expensive conditions execute
only where the verdict can change.

**The selection guard.** Screening must never consult substrate results.
Selecting cases where the substrate happened to do well would manufacture the
result the suite exists to test, and it would do so invisibly. Selecting on
baseline failure and frontier success is legitimate because neither condition
involves the substrate: that defines the population of interest rather than
cherry-picking an outcome within it. `screen` raises if substrate runs are
passed to it.

Dependency-free by product invariant.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import comb
from typing import Any

__all__ = [
    "ScreenResult",
    "ScreeningPlan",
    "SubstrateLeakError",
    "candidates_required",
    "estimate_yield",
    "plan_screening",
    "screen",
]

DEFAULT_PLANNING_CONFIDENCE = 0.90


class SubstrateLeakError(ValueError):
    """Raised when substrate results are offered as a selection input.

    Selecting cases on the substrate's own performance would bias the suite
    towards the conclusion it is meant to test.
    """


def _index(runs: Iterable[Mapping[str, Any]], condition: str) -> dict[str, Mapping[str, Any]]:
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


def candidates_required(
    target_gap: int,
    yield_rate: float,
    *,
    confidence: float = DEFAULT_PLANNING_CONFIDENCE,
) -> int:
    """Candidates to screen so ``target_gap`` gap cases arrive with ``confidence``.

    Exact, via the binomial tail: the smallest N with
    ``P(Binomial(N, yield_rate) >= target_gap) >= confidence``. Planning with the
    mean alone (``target / yield``) succeeds roughly half the time, which means
    half of all runs come back short and have to be repeated.
    """

    if target_gap < 0:
        raise ValueError(f"target_gap must be non-negative: {target_gap}")
    if not 0.0 < yield_rate <= 1.0:
        raise ValueError(f"yield_rate must be in (0, 1]: {yield_rate!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1): {confidence!r}")
    if target_gap == 0:
        return 0

    def probability_at_least(n: int) -> float:
        return sum(
            comb(n, k) * (yield_rate**k) * ((1.0 - yield_rate) ** (n - k))
            for k in range(target_gap, n + 1)
        )

    n = target_gap
    while probability_at_least(n) < confidence:
        n += 1
        if n > 100_000:  # pragma: no cover - unreachable for sane inputs
            raise ValueError("candidate requirement did not converge")
    return n


def estimate_yield(runs: Sequence[Mapping[str, Any]]) -> float | None:
    """Fraction of paired cases that turned out to be gap cases.

    Uses baseline and frontier only. Returns None when there is not enough
    paired evidence to estimate, rather than guessing a rate.
    """

    baseline_runs = _index(runs, "baseline")
    frontier_runs = _index(runs, "frontier")
    shared = set(baseline_runs) & set(frontier_runs)
    if not shared:
        return None
    gap = sum(
        1
        for case_id in shared
        if baseline_runs[case_id].get("correct") is not True
        and frontier_runs[case_id].get("correct") is True
    )
    return gap / len(shared)


@dataclass(frozen=True)
class ScreeningPlan:
    """How wide to cast the net, and what it will cost."""

    target_gap: int
    yield_rate: float
    confidence: float
    candidates: int
    baseline_runs: int
    frontier_runs: int
    substrate_runs: int
    naive_runs: int

    @property
    def total_runs(self) -> int:
        return self.baseline_runs + self.frontier_runs + self.substrate_runs

    @property
    def runs_saved(self) -> int:
        return max(0, self.naive_runs - self.total_runs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_gap": self.target_gap,
            "yield_rate": round(self.yield_rate, 4),
            "confidence": self.confidence,
            "candidates": self.candidates,
            "baseline_runs": self.baseline_runs,
            "frontier_runs": self.frontier_runs,
            "substrate_runs": self.substrate_runs,
            "total_runs": self.total_runs,
            "naive_runs": self.naive_runs,
            "runs_saved": self.runs_saved,
        }


def plan_screening(
    target_gap: int,
    yield_rate: float,
    *,
    confidence: float = DEFAULT_PLANNING_CONFIDENCE,
) -> ScreeningPlan:
    """Plan a screened run and compare it with running everything on everything."""

    candidates = candidates_required(target_gap, yield_rate, confidence=confidence)
    # Every candidate is screened by the baseline. The frontier only sees cases
    # the baseline failed; the substrate only sees confirmed gap cases.
    baseline_failures = round(candidates * (1.0 - _baseline_success_rate(yield_rate)))
    return ScreeningPlan(
        target_gap=target_gap,
        yield_rate=yield_rate,
        confidence=confidence,
        candidates=candidates,
        baseline_runs=candidates,
        frontier_runs=max(target_gap, baseline_failures),
        substrate_runs=target_gap,
        naive_runs=candidates * 3,
    )


def _baseline_success_rate(yield_rate: float) -> float:
    """Baseline pass rate implied by the gap yield.

    Conservative: assumes the frontier solves everything the baseline misses, so
    every baseline failure is a gap candidate. If the frontier also fails some,
    the frontier screening pass costs slightly more than planned, never less.
    """

    return 1.0 - yield_rate


@dataclass(frozen=True)
class ScreenResult:
    """The confirmed gap, and the audit trail for how it was selected."""

    gap_cases: tuple[str, ...]
    screened: int
    baseline_failures: tuple[str, ...]
    rejected_frontier_also_failed: tuple[str, ...]
    observed_yield: float
    notes: tuple[str, ...]

    @property
    def gap_size(self) -> int:
        return len(self.gap_cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_cases": list(self.gap_cases),
            "gap_size": self.gap_size,
            "screened": self.screened,
            "baseline_failures": list(self.baseline_failures),
            "rejected_frontier_also_failed": list(self.rejected_frontier_also_failed),
            "observed_yield": round(self.observed_yield, 4),
            "selection_inputs": ["baseline", "frontier"],
            "notes": list(self.notes),
        }


def screen(
    runs: Sequence[Mapping[str, Any]],
    *,
    substrate_condition: str = "cortheon",
) -> ScreenResult:
    """Select the confirmed capability gap from screening runs.

    Accepts baseline and frontier runs only. Passing substrate runs raises,
    because selecting on them would bias the suite towards the result it exists
    to test.
    """

    leaked = [run.get("case_id") for run in runs if run.get("condition") == substrate_condition]
    if leaked:
        raise SubstrateLeakError(
            f"{len(leaked)} {substrate_condition!r} run(s) were passed to screening; "
            "case selection must not depend on substrate performance"
        )

    baseline_runs = _index(runs, "baseline")
    frontier_runs = _index(runs, "frontier")
    screened = sorted(baseline_runs)

    failures = tuple(
        case_id for case_id in screened if baseline_runs[case_id].get("correct") is not True
    )
    gap: list[str] = []
    rejected: list[str] = []
    unconfirmed: list[str] = []
    for case_id in failures:
        frontier = frontier_runs.get(case_id)
        if frontier is None:
            unconfirmed.append(case_id)
        elif frontier.get("correct") is True:
            gap.append(case_id)
        else:
            rejected.append(case_id)

    notes: list[str] = []
    if unconfirmed:
        notes.append(
            f"{len(unconfirmed)} baseline failure(s) await a frontier run before "
            "they can be confirmed as gap cases"
        )
    if rejected:
        notes.append(
            f"{len(rejected)} case(s) were dropped because the frontier failed them "
            "too, so they set no bar to clear"
        )
    if not screened:
        notes.append("no baseline runs supplied; nothing was screened")

    return ScreenResult(
        gap_cases=tuple(gap),
        screened=len(screened),
        baseline_failures=failures,
        rejected_frontier_also_failed=tuple(rejected),
        observed_yield=len(gap) / len(screened) if screened else 0.0,
        notes=tuple(notes),
    )

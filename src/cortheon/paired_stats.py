"""Exact paired statistics for same-model comparisons.

The scoreboard requires paired wins, losses, and a confidence interval against
the same-model baseline. Those numbers decide whether a release candidate may
claim lift, so they are computed here rather than by hand, and they are exact
rather than approximate.

Three deliberate choices:

* **Exact, not normal-approximation.** Live suites run at n = 8 to 32. The
  normal approximation to McNemar's test is unreliable below roughly 25
  discordant pairs, which is nearly every run this project performs. The exact
  binomial test on discordant pairs is used instead, so a p-value is never
  optimistic because of an approximation.
* **Wilson, not Wald, intervals.** A Wald interval on 16/16 is [1.0, 1.0],
  which asserts certainty from sixteen observations. Wilson degrades sensibly at
  the boundaries.
* **Power reported before significance.** A run with four discordant pairs
  cannot reach two-sided p <= 0.05 no matter which way they fall. Reporting that
  up front stops an underpowered run being read as a null result.

Dependency-free by product invariant: no scipy, no numpy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import comb, sqrt
from typing import Any

__all__ = [
    "PairedOutcome",
    "PairedSummary",
    "exact_binomial_two_sided",
    "mcnemar_exact",
    "minimum_discordant_for_significance",
    "paired_summary",
    "summarize_pairs",
    "wilson_interval",
]

# Two-sided by default: a harness that only tests for improvement will happily
# miss a regression.
DEFAULT_ALPHA = 0.05
DEFAULT_CONFIDENCE = 0.95

# Wilson interval z-scores. A small table avoids pulling in an erf/ppf
# implementation for the two confidence levels this project actually reports.
_Z_SCORES: dict[float, float] = {
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.98: 2.3263478740408408,
    0.99: 2.5758293035489004,
}


def _z_for(confidence: float) -> float:
    try:
        return _Z_SCORES[round(confidence, 2)]
    except KeyError as error:  # pragma: no cover - guarded by validation
        raise ValueError(
            f"unsupported confidence {confidence!r}; supported: {sorted(_Z_SCORES)}"
        ) from error


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Unlike the Wald interval this does not collapse to a point at 0 or 1, which
    matters because perfect scores are common on small suites.
    """

    if total < 0 or successes < 0 or successes > total:
        raise ValueError(f"invalid counts: {successes}/{total}")
    if total == 0:
        return (0.0, 1.0)

    z = _z_for(confidence)
    proportion = successes / total
    denominator = 1.0 + (z * z) / total
    centre = (proportion + (z * z) / (2 * total)) / denominator
    spread = (
        z * sqrt((proportion * (1.0 - proportion) + (z * z) / (4 * total)) / total) / denominator
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def exact_binomial_two_sided(successes: int, trials: int, probability: float = 0.5) -> float:
    """Two-sided exact binomial p-value.

    Uses the total-probability method: sum the probability of every outcome no
    more likely than the observed one. At p = 0.5 this is symmetric and reduces
    to twice the smaller tail, which is what the published results in this
    repository report.
    """

    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError(f"invalid counts: {successes}/{trials}")
    if not 0.0 < probability < 1.0:
        raise ValueError(f"probability must be in (0, 1): {probability!r}")
    if trials == 0:
        return 1.0

    def point(k: int) -> float:
        return comb(trials, k) * (probability**k) * ((1.0 - probability) ** (trials - k))

    observed = point(successes)
    # Floating-point slack so an outcome with mathematically equal probability
    # is not excluded by representation error.
    tolerance = observed * 1e-9
    total = sum(point(k) for k in range(trials + 1) if point(k) <= observed + tolerance)
    return min(1.0, total)


def mcnemar_exact(wins: int, losses: int) -> float:
    """Exact McNemar test on discordant pairs.

    Concordant pairs carry no information about which condition is better and
    are excluded by construction; only the split of the discordant pairs is
    tested against a fair coin.
    """

    if wins < 0 or losses < 0:
        raise ValueError(f"invalid discordant counts: {wins}/{losses}")
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    return exact_binomial_two_sided(wins, discordant)


def minimum_discordant_for_significance(alpha: float = DEFAULT_ALPHA) -> int:
    """Fewest discordant pairs that could ever reach two-sided ``alpha``.

    With every discordant pair falling the same way the p-value is
    ``2 * 0.5**n``. Below the returned n no result is significant regardless of
    outcome, so a run of that size cannot support a lift claim and should not be
    read as evidence of absence either.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1): {alpha!r}")
    if alpha >= 1.0:
        return 0
    # smallest n with 2 * 0.5**n <= alpha
    n = 1
    while 2.0 * (0.5**n) > alpha:
        n += 1
        if n > 1024:  # pragma: no cover - unreachable for sane alpha
            break
    return n


@dataclass(frozen=True)
class PairedOutcome:
    """One case evaluated under both conditions."""

    case_id: str
    treatment_correct: bool
    baseline_correct: bool

    @property
    def is_win(self) -> bool:
        return self.treatment_correct and not self.baseline_correct

    @property
    def is_loss(self) -> bool:
        return self.baseline_correct and not self.treatment_correct

    @property
    def is_tie(self) -> bool:
        return self.treatment_correct == self.baseline_correct


@dataclass(frozen=True)
class PairedSummary:
    """Everything the scoreboard needs about one paired comparison."""

    pairs: int
    wins: int
    losses: int
    ties: int
    treatment_correct: int
    baseline_correct: int
    treatment_accuracy: float
    baseline_accuracy: float
    treatment_interval: tuple[float, float]
    baseline_interval: tuple[float, float]
    p_value: float
    alpha: float
    significant: bool
    powered: bool
    minimum_discordant: int
    confidence: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def discordant(self) -> int:
        return self.wins + self.losses

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairs": self.pairs,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "discordant": self.discordant,
            "treatment_correct": self.treatment_correct,
            "baseline_correct": self.baseline_correct,
            "treatment_accuracy": round(self.treatment_accuracy, 6),
            "baseline_accuracy": round(self.baseline_accuracy, 6),
            "treatment_interval": [
                round(self.treatment_interval[0], 6),
                round(self.treatment_interval[1], 6),
            ],
            "baseline_interval": [
                round(self.baseline_interval[0], 6),
                round(self.baseline_interval[1], 6),
            ],
            "p_value": round(self.p_value, 6),
            "alpha": self.alpha,
            "significant": self.significant,
            "powered": self.powered,
            "minimum_discordant": self.minimum_discordant,
            "confidence": self.confidence,
            "notes": list(self.notes),
            "claim": self.claim,
        }

    @property
    def claim(self) -> str:
        """The strongest statement this evidence supports, and no stronger."""

        if self.pairs == 0:
            return "no paired evidence"
        if not self.powered:
            return (
                f"underpowered: {self.discordant} discordant pair(s); "
                f"{self.minimum_discordant} are required before two-sided "
                f"p<={self.alpha} is reachable at all"
            )
        if self.significant and self.wins > self.losses:
            return f"significant lift (p={self.p_value:.4f})"
        if self.significant and self.losses > self.wins:
            return f"significant regression (p={self.p_value:.4f})"
        return f"no significant difference (p={self.p_value:.4f})"


def summarize_pairs(
    outcomes: Iterable[PairedOutcome],
    *,
    alpha: float = DEFAULT_ALPHA,
    confidence: float = DEFAULT_CONFIDENCE,
) -> PairedSummary:
    """Reduce paired outcomes to wins, losses, intervals, and an exact p-value."""

    collected = list(outcomes)
    seen: set[str] = set()
    duplicates: set[str] = set()
    for outcome in collected:
        if outcome.case_id in seen:
            duplicates.add(outcome.case_id)
        seen.add(outcome.case_id)

    pairs = len(collected)
    wins = sum(outcome.is_win for outcome in collected)
    losses = sum(outcome.is_loss for outcome in collected)
    ties = sum(outcome.is_tie for outcome in collected)
    treatment_correct = sum(outcome.treatment_correct for outcome in collected)
    baseline_correct = sum(outcome.baseline_correct for outcome in collected)

    p_value = mcnemar_exact(wins, losses)
    minimum_discordant = minimum_discordant_for_significance(alpha)
    powered = (wins + losses) >= minimum_discordant

    notes: list[str] = []
    if duplicates:
        notes.append(
            "duplicate case ids collapse independence: " + ", ".join(sorted(duplicates)[:5])
        )
    if pairs and treatment_correct == pairs and baseline_correct == pairs:
        notes.append("both conditions ceilinged; this suite cannot separate them")
    if pairs == 0:
        notes.append("no paired cases supplied")

    return PairedSummary(
        pairs=pairs,
        wins=wins,
        losses=losses,
        ties=ties,
        treatment_correct=treatment_correct,
        baseline_correct=baseline_correct,
        treatment_accuracy=treatment_correct / pairs if pairs else 0.0,
        baseline_accuracy=baseline_correct / pairs if pairs else 0.0,
        treatment_interval=wilson_interval(treatment_correct, pairs, confidence=confidence),
        baseline_interval=wilson_interval(baseline_correct, pairs, confidence=confidence),
        p_value=p_value,
        alpha=alpha,
        significant=powered and p_value <= alpha,
        powered=powered,
        minimum_discordant=minimum_discordant,
        confidence=confidence,
        notes=tuple(notes),
    )


def paired_summary(
    runs: Sequence[Mapping[str, Any]],
    *,
    treatment: str = "cortheon",
    baseline: str = "baseline",
    alpha: float = DEFAULT_ALPHA,
    confidence: float = DEFAULT_CONFIDENCE,
) -> PairedSummary:
    """Pair benchmark runs by case id and summarise them.

    Candidate-caused failures remain scheduled incorrect outcomes. Only a
    separately evaluator-attested external infrastructure failure invalidates
    a case; missing, duplicate, and unstable cells also fail closed.
    """

    def index(condition: str) -> dict[tuple[str, int], list[Mapping[str, Any]]]:
        selected: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
        for run in runs:
            if run.get("condition") != condition:
                continue
            case_id = run.get("case_id")
            if not isinstance(case_id, str):
                continue
            repeat = run.get("repeat", 0)
            if not isinstance(repeat, int) or isinstance(repeat, bool):
                continue
            selected.setdefault((case_id, repeat), []).append(run)
        return selected

    treatment_runs = index(treatment)
    baseline_runs = index(baseline)
    case_ids = sorted({key[0] for key in treatment_runs} | {key[0] for key in baseline_runs})
    outcomes: list[PairedOutcome] = []
    duplicate_cases: set[str] = set()
    incomplete_cases: set[str] = set()
    unstable_cases: set[str] = set()
    for case_id in case_ids:
        repeats = sorted(
            {key[1] for key in treatment_runs if key[0] == case_id}
            | {key[1] for key in baseline_runs if key[0] == case_id}
        )
        treatment_values: list[bool] = []
        baseline_values: list[bool] = []
        for repeat in repeats:
            treatment_cell = treatment_runs.get((case_id, repeat), [])
            baseline_cell = baseline_runs.get((case_id, repeat), [])
            if len(treatment_cell) > 1 or len(baseline_cell) > 1:
                duplicate_cases.add(case_id)
            if (
                len(treatment_cell) != 1
                or len(baseline_cell) != 1
                or treatment_cell[0].get("process_error") is not None
                or baseline_cell[0].get("process_error") is not None
            ):
                incomplete_cases.add(case_id)
                continue
            treatment_values.append(treatment_cell[0].get("correct") is True)
            baseline_values.append(baseline_cell[0].get("correct") is True)
        if case_id in duplicate_cases or case_id in incomplete_cases:
            continue
        if len(set(treatment_values)) != 1 or len(set(baseline_values)) != 1:
            unstable_cases.add(case_id)
            continue
        outcomes.append(PairedOutcome(case_id, treatment_values[0], baseline_values[0]))
    summary = summarize_pairs(outcomes, alpha=alpha, confidence=confidence)
    notes = list(summary.notes)
    if incomplete_cases:
        notes.append(
            f"{len(incomplete_cases)} case(s) present in only one condition or repeat "
            "were excluded from pairing"
        )
    if duplicate_cases:
        notes.append(f"duplicate condition/repeat cells invalidated {len(duplicate_cases)} case(s)")
    if unstable_cases:
        notes.append(f"unstable repetitions invalidated {len(unstable_cases)} case(s)")
    if notes != list(summary.notes):
        summary = replace(summary, notes=tuple(notes))
    return summary

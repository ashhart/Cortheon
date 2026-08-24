"""Select candidate cases that are likely to be informative, before spending a
single model call.

Screening is cheaper than running everything on everything, but it still pays
the baseline once per candidate. At a 0.25 gap yield that is 87 baseline runs to
find 17 informative cases: two thirds of the screening budget is spent on cases
that turn out to be ceiling.

Case objects carry structure that predicts informativeness without executing
anything — how many files must be consulted, how many components the answer
needs, how many distractors are present, how many hops the reasoning requires.
Ordering the candidate pool by that structure puts the likely-informative cases
first, so screening finds its quota earlier and stops.

**Optimise for gap yield, not difficulty.** These are not the same target. A
case the frontier also fails is exactly as useless as one the bare model already
solves: the first sets no bar, the second leaves no headroom. Difficulty is only
a proxy, and it stops being a good one at the top of the range. Calibration
therefore measures observed *gap* yield per difficulty band and will happily
report that the hardest band is worse than the middle one.

**Selection narrows the claim, and says so.** A suite chosen for baseline
failure no longer represents the general case distribution. The resulting claim
is "frontier-like on cases the bare model cannot solve", which is the correct
scope for the hard rule but must be stated rather than assumed. `Calibration`
carries that wording so a report cannot quietly overstate itself.

Dependency-free by product invariant.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

__all__ = [
    "Calibration",
    "CandidateSelection",
    "DifficultyBand",
    "calibrate",
    "difficulty_features",
    "rank_candidates",
    "select_candidates",
]

# Structural signals, in the direction of "more of this is usually harder".
# Extracted by introspection so a new case type gains features automatically
# rather than silently scoring zero.
_COUNT_FIELDS = (
    "files",
    "expected",
    "forbidden_answers",
    "required_any",
    "derived_relations",
    "ordered_steps",
    "required_paths",
    "required_content",
    "protected_paths",
    "paths",
    "symbols",
)


def _sequence_length(value: Any) -> float:
    if isinstance(value, (str, Mapping)):
        return 0.0
    if isinstance(value, Sequence):
        return float(len(value))
    return 0.0


def _nesting(value: Any) -> float:
    """Depth of nested sequences, a proxy for multi-hop structure."""

    if isinstance(value, str) or not isinstance(value, Sequence):
        return 0.0
    return 1.0 + max((_nesting(item) for item in value), default=0.0)


def difficulty_features(case: Any) -> dict[str, float]:
    """Structural features of one case. Deterministic, and free of model calls."""

    features: dict[str, float] = {}
    for field in _COUNT_FIELDS:
        value = getattr(case, field, None)
        if value is None:
            continue
        features[f"{field}_count"] = _sequence_length(value)
        depth = _nesting(value)
        if depth > 1.0:
            features[f"{field}_depth"] = depth

    prompt = getattr(case, "prompt", "")
    if isinstance(prompt, str):
        features["prompt_chars"] = float(len(prompt))

    files = getattr(case, "files", None)
    if isinstance(files, Sequence) and not isinstance(files, str):
        total = 0
        for entry in files:
            if isinstance(entry, Sequence) and not isinstance(entry, str):
                total += sum(len(part) for part in entry if isinstance(part, str))
        features["source_chars"] = float(total)
    return features


def _rank_normalised(values: Sequence[float]) -> list[float]:
    """Map values to [0, 1] by rank.

    Rank rather than min-max because one enormous case would otherwise compress
    every other case towards zero.
    """

    if not values:
        return []
    if len(values) == 1:
        return [0.5]

    order = sorted(range(len(values)), key=lambda index: values[index])
    normalised = [0.0] * len(values)
    span = len(values) - 1

    # Ties must share a rank. Without this a constant feature — every case in a
    # suite having the same prompt, say — would be spread across the full range
    # by original list order, injecting pure noise into the score.
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2 / span
        for index in order[position : end + 1]:
            normalised[index] = average
        position = end + 1
    return normalised


def rank_candidates(cases: Sequence[Any]) -> list[tuple[str, float]]:
    """Score and order candidates by structural difficulty, hardest first.

    Scores are relative to the supplied pool: a case is "hard" compared with its
    peers, not on an absolute scale that would need per-suite tuning.
    """

    if not cases:
        return []
    per_case = [difficulty_features(case) for case in cases]
    names = sorted({name for features in per_case for name in features})
    if not names:
        return [(str(getattr(case, "case_id", index)), 0.0) for index, case in enumerate(cases)]

    columns = {
        name: _rank_normalised([features.get(name, 0.0) for features in per_case]) for name in names
    }
    scored = [
        (
            str(getattr(case, "case_id", index)),
            mean(columns[name][index] for name in names),
        )
        for index, case in enumerate(cases)
    ]
    return sorted(scored, key=lambda item: (-item[1], item[0]))


@dataclass(frozen=True)
class DifficultyBand:
    """Observed informativeness of one slice of the difficulty range."""

    label: str
    lower: float
    upper: float
    cases: int
    gap_cases: int

    @property
    def observed_yield(self) -> float:
        return self.gap_cases / self.cases if self.cases else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score_range": [round(self.lower, 4), round(self.upper, 4)],
            "cases": self.cases,
            "gap_cases": self.gap_cases,
            "observed_yield": round(self.observed_yield, 4),
        }


@dataclass(frozen=True)
class Calibration:
    """What difficulty banding bought, measured rather than assumed."""

    bands: tuple[DifficultyBand, ...]
    overall_yield: float
    best_band: str | None
    lift: float
    scope: str
    notes: tuple[str, ...]

    @property
    def useful(self) -> bool:
        """Did banding beat sampling the pool at random?"""

        return self.lift > 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bands": [band.to_dict() for band in self.bands],
            "overall_yield": round(self.overall_yield, 4),
            "best_band": self.best_band,
            "lift": round(self.lift, 4),
            "useful": self.useful,
            "scope": self.scope,
            "notes": list(self.notes),
        }


SELECTED_SCOPE = (
    "cases selected for baseline failure; any resulting claim is scoped to "
    "'cases the bare model cannot solve', not to the general case distribution"
)


def calibrate(
    observations: Iterable[Mapping[str, Any]],
    *,
    bands: int = 3,
) -> Calibration:
    """Measure gap yield per difficulty band from completed screening.

    Each observation needs ``score`` and ``gap`` (whether the case turned out to
    be a confirmed gap case). Yield, not baseline failure, is the target: the
    hardest band is not automatically the best one, because cases the frontier
    also fails contribute nothing.
    """

    collected = [
        (float(item["score"]), bool(item["gap"]))
        for item in observations
        if "score" in item and "gap" in item
    ]
    if not collected:
        return Calibration(
            bands=(),
            overall_yield=0.0,
            best_band=None,
            lift=0.0,
            scope=SELECTED_SCOPE,
            notes=("no scored observations supplied",),
        )

    collected.sort(key=lambda item: item[0])
    total_gap = sum(1 for _, gap in collected if gap)
    overall = total_gap / len(collected)

    size = max(1, len(collected) // max(1, bands))
    labels = (
        ("low", "medium", "high") if bands == 3 else tuple(f"band{index}" for index in range(bands))
    )
    built: list[DifficultyBand] = []
    for index in range(bands):
        start = index * size
        end = len(collected) if index == bands - 1 else min(len(collected), start + size)
        slice_ = collected[start:end]
        if not slice_:
            continue
        built.append(
            DifficultyBand(
                label=labels[index] if index < len(labels) else f"band{index}",
                lower=slice_[0][0],
                upper=slice_[-1][0],
                cases=len(slice_),
                gap_cases=sum(1 for _, gap in slice_ if gap),
            )
        )

    notes: list[str] = []
    if len(collected) < 20:
        notes.append(
            f"only {len(collected)} observation(s); band yields are indicative, "
            "not a calibrated model"
        )
    best = max(built, key=lambda band: band.observed_yield, default=None)
    if best and built and best.label != built[-1].label:
        notes.append(
            f"the '{best.label}' band out-yielded the hardest band; difficulty is "
            "only a proxy, and past a point the frontier fails too"
        )

    return Calibration(
        bands=tuple(built),
        overall_yield=overall,
        best_band=best.label if best else None,
        lift=(best.observed_yield / overall) if best and overall > 0 else 0.0,
        scope=SELECTED_SCOPE,
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class CandidateSelection:
    """An ordered candidate pool, and what selecting it implies."""

    case_ids: tuple[str, ...]
    pool_size: int
    expected_yield: float
    expected_gap: float
    scope: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_ids": list(self.case_ids),
            "selected": len(self.case_ids),
            "pool_size": self.pool_size,
            "expected_yield": round(self.expected_yield, 4),
            "expected_gap": round(self.expected_gap, 2),
            "scope": self.scope,
            "notes": list(self.notes),
        }


def select_candidates(
    cases: Sequence[Any],
    *,
    take: int,
    calibration: Calibration | None = None,
    fallback_yield: float = 0.25,
) -> CandidateSelection:
    """Take the most promising candidates to screen first.

    Without calibration this orders by structural difficulty and assumes the
    historical yield. With calibration it uses the measured best band, and says
    so, rather than presuming that harder is always better.
    """

    ranked = rank_candidates(cases)
    chosen = tuple(case_id for case_id, _ in ranked[: max(0, take)])

    notes: list[str] = []
    expected = fallback_yield
    if calibration and calibration.bands:
        best = max(calibration.bands, key=lambda band: band.observed_yield)
        expected = best.observed_yield
        if not calibration.useful:
            notes.append(
                "calibration showed no lift from difficulty banding; ordering is "
                "unlikely to beat random sampling on this suite"
            )
        else:
            notes.append(
                f"expected yield taken from the measured '{best.label}' band "
                f"({best.observed_yield:.2f} versus {calibration.overall_yield:.2f} overall)"
            )
    else:
        notes.append(
            f"no calibration supplied; assuming the historical yield of {fallback_yield:.2f}"
        )
    if take > len(ranked):
        notes.append(f"requested {take} candidates but the pool holds {len(ranked)}")

    return CandidateSelection(
        case_ids=chosen,
        pool_size=len(ranked),
        expected_yield=expected,
        expected_gap=len(chosen) * expected,
        scope=SELECTED_SCOPE,
        notes=tuple(notes),
    )

"""Split a goal into sub-goals a small model can hold.

The characteristic failure of a 7B-35B model on a hard task is capacity, not
ignorance: it cannot keep the problem, its constraints, and its partial results
in view at once, so it answers the part it can see and reports completion.
Structure is what a substrate can supply.

Deterministic, never model-mediated: asking the weak model to split its own task
would reintroduce the failure being corrected, and would make the program
non-reproducible, which the audit bundles cannot tolerate. Splits come only from
structure the author already wrote.

Conservative by design. Over-splitting fragments one coherent task into parts
that each look satisfiable alone, which is how an agent declares victory having
done a third of the work. Absent clear structure, the goal stays whole.

Ships stripped of docstrings and comments by setup.py, so this rationale costs
nothing in the deployable artifact. Worked cases: tests/test_decomposition.py.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Decomposition",
    "SubGoal",
    "decompose",
    "max_sub_goals_for",
]

# "quick" never decomposes: the overhead would exceed the task.
_EFFORT_LIMITS = {"quick": 1, "standard": 4, "deep": 8}

# Ordered discourse markers imply a dependency chain rather than independence.
_ORDERED_MARKERS = (
    "first",
    "then",
    "next",
    "after that",
    "afterwards",
    "finally",
    "lastly",
    "once that",
)

_ENUMERATION = re.compile(
    r"(?:^|\n)\s*(?:\(?\d+[.)]|[-*•]|[a-h][.)])\s+(?P<item>[^\n]+)",
    re.IGNORECASE,
)

# " and then ", "; ", " then " — split points that separate imperatives.
_SEQUENTIAL_SPLIT = re.compile(
    r"\s*(?:;|(?:,\s*)?\band then\b|(?:,\s*)?\bthen\b|(?:,\s*)?\bafter that\b)\s*",
    re.IGNORECASE,
)

# Decides whether a conjunction joins two tasks or two objects of one task.
_IMPERATIVES = (
    "add",
    "build",
    "change",
    "check",
    "create",
    "delete",
    "deploy",
    "document",
    "extract",
    "find",
    "fix",
    "implement",
    "install",
    "migrate",
    "move",
    "read",
    "refactor",
    "remove",
    "rename",
    "replace",
    "run",
    "test",
    "update",
    "verify",
    "wire",
    "write",
    "compare",
    "measure",
    "benchmark",
    "audit",
    "review",
)

_MIN_CLAUSE_CHARS = 12


def max_sub_goals_for(effort: str) -> int:
    """How many parts this effort tier is willing to manage."""

    return _EFFORT_LIMITS.get(effort.strip().lower(), _EFFORT_LIMITS["standard"])


@dataclass(frozen=True)
class SubGoal:
    """One part of a decomposed goal, with its own completion obligation."""

    sub_goal_id: str
    statement: str
    proof: str
    ordinal: int
    depends_on: tuple[str, ...]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub_goal_id": self.sub_goal_id,
            "statement": self.statement,
            "proof": self.proof,
            "ordinal": self.ordinal,
            "depends_on": list(self.depends_on),
            "source": self.source,
        }


@dataclass(frozen=True)
class Decomposition:
    """The split, and why it was or was not performed."""

    sub_goals: tuple[SubGoal, ...]
    strategy: str
    effort: str
    limit: int
    notes: tuple[str, ...]

    @property
    def split(self) -> bool:
        return len(self.sub_goals) > 1

    @property
    def sequential(self) -> bool:
        return any(goal.depends_on for goal in self.sub_goals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub_goals": [goal.to_dict() for goal in self.sub_goals],
            "count": len(self.sub_goals),
            "split": self.split,
            "sequential": self.sequential,
            "strategy": self.strategy,
            "effort": self.effort,
            "limit": self.limit,
            "notes": list(self.notes),
        }


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().rstrip(".,;:")


def _starts_with_imperative(clause: str) -> bool:
    first = _clean(clause).lower().split(" ")
    return bool(first) and first[0] in _IMPERATIVES


def _enumerated(goal: str) -> list[str]:
    items = [_clean(match.group("item")) for match in _ENUMERATION.finditer(goal)]
    return [item for item in items if len(item) >= _MIN_CLAUSE_CHARS]


def _sequential(goal: str) -> list[str]:
    parts = [_clean(part) for part in _SEQUENTIAL_SPLIT.split(goal)]
    parts = [part for part in parts if len(part) >= _MIN_CLAUSE_CHARS]
    # Tail clauses must be actions: "read the file then tell me what it does" is
    # one task; "read the file then run the tests" is two.
    if len(parts) < 2:
        return []
    if not all(_starts_with_imperative(part) for part in parts[1:]):
        return []
    return parts


def _conjoined(goal: str) -> list[str]:
    parts = [_clean(part) for part in re.split(r"\s+\band\b\s+", goal, flags=re.I)]
    parts = [part for part in parts if len(part) >= _MIN_CLAUSE_CHARS]
    if len(parts) < 2:
        return []
    # Otherwise "and" joins objects ("the parser and the lexer"), not tasks.
    if not all(_starts_with_imperative(part) for part in parts):
        return []
    return parts


def _implies_order(goal: str) -> bool:
    lowered = goal.lower()
    return any(marker in lowered for marker in _ORDERED_MARKERS)


def decompose(
    goal: str,
    *,
    effort: str = "standard",
    requirements: Iterable[tuple[str, str]] = (),
) -> Decomposition:
    """Split ``goal`` into sub-goals, conservatively.

    Requirements supplied by the host win: they are already an authored
    decomposition with proof obligations attached, and inventing a different
    split alongside them would produce two competing definitions of done.
    """

    limit = max_sub_goals_for(effort)
    supplied = [(str(rid), str(proof)) for rid, proof in requirements]
    notes: list[str] = []

    if limit <= 1:
        return Decomposition(
            sub_goals=(_whole(goal),),
            strategy="none",
            effort=effort,
            limit=limit,
            notes=("effort tier does not decompose",),
        )

    if len(supplied) > 1:
        chosen = supplied[:limit]
        if len(supplied) > limit:
            notes.append(
                f"{len(supplied)} requirements exceeded the {effort} limit of {limit}; "
                "the remainder stay bound to the final sub-goal"
            )
        return Decomposition(
            sub_goals=tuple(
                SubGoal(
                    sub_goal_id=f"sg{index}",
                    statement=requirement_id,
                    proof=proof,
                    ordinal=index,
                    depends_on=(),
                    source="requirements",
                )
                for index, (requirement_id, proof) in enumerate(chosen, 1)
            ),
            strategy="requirements",
            effort=effort,
            limit=limit,
            notes=tuple(notes),
        )

    for strategy, clauses in (
        ("enumeration", _enumerated(goal)),
        ("sequence", _sequential(goal)),
        ("conjunction", _conjoined(goal)),
    ):
        if len(clauses) < 2:
            continue
        if len(clauses) > limit:
            notes.append(
                f"{len(clauses)} parts exceeded the {effort} limit of {limit}; "
                "the tail was kept together"
            )
            clauses = [*clauses[: limit - 1], " and ".join(clauses[limit - 1 :])]
        ordered = strategy == "sequence" or _implies_order(goal)
        sub_goals = []
        for index, clause in enumerate(clauses, 1):
            sub_goals.append(
                SubGoal(
                    sub_goal_id=f"sg{index}",
                    statement=clause,
                    proof=f"evidence that '{clause}' is complete",
                    ordinal=index,
                    depends_on=(f"sg{index - 1}",) if ordered and index > 1 else (),
                    source=strategy,
                )
            )
        if ordered:
            notes.append("ordered markers imply a dependency chain")
        return Decomposition(
            sub_goals=tuple(sub_goals),
            strategy=strategy,
            effort=effort,
            limit=limit,
            notes=tuple(notes),
        )

    return Decomposition(
        sub_goals=(_whole(goal),),
        strategy="none",
        effort=effort,
        limit=limit,
        notes=("no explicit structure found; the goal was left whole",),
    )


def _whole(goal: str) -> SubGoal:
    statement = _clean(goal) or "complete the requested task"
    return SubGoal(
        sub_goal_id="sg1",
        statement=statement,
        proof=f"evidence that '{statement}' is complete",
        ordinal=1,
        depends_on=(),
        source="whole",
    )


def ready_sub_goals(decomposition: Decomposition, completed: Sequence[str]) -> tuple[SubGoal, ...]:
    """Sub-goals whose dependencies are satisfied and which remain open."""

    done = set(completed)
    return tuple(
        goal
        for goal in decomposition.sub_goals
        if goal.sub_goal_id not in done
        and all(dependency in done for dependency in goal.depends_on)
    )

"""Effort and strictness profiles plus capability/cost policy tables."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EffortProfile:
    name: str
    max_turns: int
    max_observations: int
    max_observation_chars: int
    max_total_observation_chars: int
    max_context_chars: int
    max_hypotheses: int
    min_hypotheses: int
    max_calls_per_request: int = 5


EFFORT_PROFILES: dict[str, EffortProfile] = {
    "quick": EffortProfile(
        name="quick",
        max_turns=5,
        max_observations=12,
        max_observation_chars=2_000,
        max_total_observation_chars=10_000,
        max_context_chars=4_000,
        max_hypotheses=3,
        min_hypotheses=1,
        max_calls_per_request=3,
    ),
    "standard": EffortProfile(
        name="standard",
        max_turns=9,
        max_observations=18,
        max_observation_chars=4_000,
        max_total_observation_chars=36_000,
        max_context_chars=10_000,
        max_hypotheses=5,
        min_hypotheses=2,
        max_calls_per_request=5,
    ),
    "deep": EffortProfile(
        name="deep",
        max_turns=15,
        max_observations=32,
        max_observation_chars=6_000,
        max_total_observation_chars=80_000,
        max_context_chars=18_000,
        max_hypotheses=7,
        min_hypotheses=2,
        max_calls_per_request=8,
    ),
}


TASK_KINDS = frozenset({"auto", "code", "research", "documents", "decision", "general"})


MAX_REQUEST_ATTEMPTS = 2


@dataclass(frozen=True, slots=True)
class StrictnessProfile:
    """Evidence strictness policy for each host profile."""

    name: str
    max_request_attempts: int
    corroboration_rounds: int


STRICTNESS_PROFILES: dict[str, StrictnessProfile] = {
    "strict": StrictnessProfile(
        "strict",
        max_request_attempts=4,
        corroboration_rounds=2,
    ),
    "standard": StrictnessProfile(
        "standard",
        max_request_attempts=MAX_REQUEST_ATTEMPTS,
        corroboration_rounds=1,
    ),
    "assist": StrictnessProfile(
        "assist",
        max_request_attempts=1,
        corroboration_rounds=0,
    ),
}


_CODE_HINTS = frozenset(
    {
        "api",
        "bug",
        "build",
        "class",
        "cli",
        "code",
        "commit",
        "dependency",
        "error",
        "exception",
        "feature",
        "file",
        "fix",
        "function",
        "implement",
        "method",
        "package",
        "patch",
        "refactor",
        "repo",
        "repository",
        "runtime",
        "test",
    }
)


_RESEARCH_HINTS = frozenset(
    {
        "current",
        "evidence",
        "latest",
        "market",
        "news",
        "paper",
        "research",
        "sources",
        "study",
        "today",
    }
)


_EXPLICIT_FRESHNESS_HINTS = frozenset(
    {
        "breaking",
        "latest",
        "newest",
        "news",
        "recent",
        "today",
        "tonight",
        "yesterday",
    }
)


_DOCUMENT_HINTS = frozenset(
    {
        "compare",
        "contract",
        "document",
        "documents",
        "memo",
        "policy",
        "proposal",
        "report",
        "requirements",
        "spec",
    }
)


_DECISION_HINTS = frozenset(
    {
        "choose",
        "decision",
        "option",
        "recommend",
        "should",
        "tradeoff",
        "versus",
        "vs",
    }
)


_CHANGE_HINTS = frozenset(
    {
        "add",
        "build",
        "change",
        "copy",
        "correct",
        "correction",
        "create",
        "delete",
        "fix",
        "implement",
        "migrate",
        "move",
        "patch",
        "refactor",
        "remove",
        "rename",
        "repair",
        "replace",
        "update",
    }
)


def _has_hint(value: str, hints: frozenset[str]) -> bool:
    words = {item.casefold() for item in re.findall(r"[A-Za-z]+", value)}
    return bool(words & hints)


def _capability_for_kind(task_kind: str) -> str:
    return {
        "code": "search_or_read",
        "research": "search_or_fetch",
        "documents": "search_or_read",
        "decision": "inspect",
        "general": "inspect",
    }[task_kind]


def _capability_for_falsification(task_kind: str, test: str) -> str:
    lower = test.casefold()
    if re.search(r"\b(?:pytest|run|test)\b", lower):
        return "test"
    if re.search(r"\b(?:diff|patch|what changed)\b", lower):
        return "diff"
    if re.search(r"\b(?:exact|grep|match)\b", lower):
        return "grep"
    if re.search(r"\b(?:current|online|publish|source|web)\b", lower):
        return "search"
    if re.search(r"\b(?:inspect|open|read|trace)\b", lower):
        return "read"
    return _capability_for_kind(task_kind)


def _evidence_action_cost(capability: str) -> float:
    return {
        "diff": 0.5,
        "grep": 0.5,
        "inspect": 1.0,
        "read": 1.0,
        "fetch": 1.5,
        "search_or_read": 1.5,
        "search": 2.0,
        "search_or_fetch": 2.5,
        "test": 3.0,
    }.get(capability, 2.0)


def _evidence_action_reliability(capability: str) -> float:
    return {
        "test": 0.99,
        "diff": 0.98,
        "grep": 0.95,
        "read": 0.9,
        "inspect": 0.85,
        "fetch": 0.85,
        "search_or_read": 0.8,
        "search": 0.7,
        "search_or_fetch": 0.7,
    }.get(capability, 0.7)

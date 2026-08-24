"""Identifier, assurance, and metric validation for experience records."""

from __future__ import annotations

import re
from collections.abc import Iterable

from cortheon.experience_core._compat import facade

EXPERIENCE_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,47}$")
_NAMESPACE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,79}$")
_RESERVED_IDENTIFIERS = frozenset(
    {
        "answer",
        "chain_of_thought",
        "completion",
        "credential",
        "expected_answer",
        "password",
        "prompt",
        "reasoning_trace",
        "secret",
        "system_prompt",
        "token",
        "tool_output",
    }
)
_RESULTS = frozenset({"failure", "recovered"})
_ASSURANCE_RANK = {
    "none": 0,
    "policy": 1,
    "structural": 2,
    "patch_applied": 2,
    "runtime_bind": 3,
    "agent_tools": 3,
    "behavioral": 4,
    "repository_tests": 5,
    "independent_grader": 6,
}
_VERIFIABLE_ASSURANCE = frozenset(
    {"agent_tools", "behavioral", "repository_tests", "independent_grader"}
)


def _identifier(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string identifier")
    normalized = value.strip().lower()
    api = facade()
    if not api._IDENTIFIER.fullmatch(normalized):
        raise ValueError(
            f"{field} must match {api._IDENTIFIER.pattern}; arbitrary text is forbidden"
        )
    pieces = frozenset(re.split(r"[_.:-]", normalized))
    if pieces.intersection(api._RESERVED_IDENTIFIERS):
        raise ValueError(f"{field} may not identify retained sensitive content")
    if api._looks_secret(normalized):
        raise ValueError(f"{field} resembles a secret and will not be retained")
    return normalized


def _namespace(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("namespace must be a string identifier")
    normalized = value.strip().lower()
    pattern = facade()._NAMESPACE
    if not pattern.fullmatch(normalized):
        raise ValueError(f"namespace must match {pattern.pattern}; arbitrary text is forbidden")
    return normalized


def _identifiers(
    values: Iterable[str],
    field: str,
    *,
    maximum: int,
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field} must be an iterable of identifiers, not text")
    normalized = tuple(dict.fromkeys(_identifier(value, field) for value in values))
    if len(normalized) > maximum:
        raise ValueError(f"{field} may contain at most {maximum} identifiers")
    return normalized


def _looks_secret(value: str) -> bool:
    if len(value) < 24:
        return False
    if value.startswith(("sk-", "ghp_", "github_pat_", "xox")):
        return True
    alphabet = set(value)
    entropy = -sum(
        (value.count(character) / len(value))
        * facade().math.log2(value.count(character) / len(value))
        for character in alphabet
    )
    return entropy >= 4.25


def _latency_bucket(value: float | int | None) -> str:
    if value is None:
        return "unknown"
    try:
        latency = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("latency_ms must be numeric or None") from exc
    if not facade().math.isfinite(latency) or latency < 0:
        raise ValueError("latency_ms must be finite and non-negative")
    if latency < 100:
        return "lt_100ms"
    if latency < 1_000:
        return "100ms_1s"
    if latency < 5_000:
        return "1s_5s"
    if latency < 30_000:
        return "5s_30s"
    return "gte_30s"


def _limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("limit must be an integer")
    return max(1, min(value, 100))


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _recovery_rate(recoveries: int, failures: int) -> float:
    return _rate(min(recoveries, failures), failures)


def _assurance_for_rank(rank: int) -> str:
    return {
        6: "independent_grader",
        5: "repository_tests",
        4: "behavioral",
        3: "agent_tools",
        2: "structural",
        1: "policy",
        0: "none",
    }.get(rank, "none")

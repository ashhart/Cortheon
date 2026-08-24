"""Policy for deciding when a task needs knowledge beyond the local project."""

from __future__ import annotations

import re

_EXPLICIT_FRONTIER_RE = re.compile(
    r"\b(?:"
    r"current|latest|newest|state[- ]of[- ]the[- ]art|frontier|"
    r"official\s+(?:documentation|docs|guidance|release notes?)|"
    r"scientific\s+papers?|primary\s+research|recent\s+(?:papers?|research)|"
    r"maintained\s+(?:repositories|repos?|implementations?)|"
    r"reference\s+(?:repositories|repos?|implementations?)|"
    r"exact\s+(?:installed\s+)?(?:runtime|dependency|package|library)\s+versions?|"
    r"installed\s+(?:runtime|dependency|package|library)\s+versions?|"
    r"search\s+(?:the\s+)?web|online\s+research"
    r")\b",
    flags=re.IGNORECASE,
)


_QUALITY_FRONTIER_RE = re.compile(
    r"\b(?:production[- ]ready|best\s+practices?|best\s+available|"
    r"industry[- ]standard|well[- ]maintained|battle[- ]tested)\b",
    flags=re.IGNORECASE,
)


_EXTERNAL_TECHNOLOGY_RE = re.compile(
    r"\b(?:api|dependency|framework|library|package|protocol|runtime|sdk|"
    r"standard|algorithm|architecture|client|server|database|model)\b",
    flags=re.IGNORECASE,
)


_EXTERNAL_IMPLEMENTATION_RE = re.compile(
    r"\b(?:add|adopt|build|choose|create|design|implement|integrate|migrate|modernize|"
    r"optimize|replace|secure|upgrade)\b",
    flags=re.IGNORECASE,
)


_LOCAL_CODE_PATH_RE = re.compile(r"\b(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]+\b")


_RESEARCH_SOURCE_RE = re.compile(
    r"\b(?:paper|papers|study|studies|research|benchmark|dataset|methodology|"
    r"scientific|clinical|academic|arxiv|doi)\b",
    flags=re.IGNORECASE,
)


def needs_frontier_grounding(goal: str, task_kind: str) -> bool:
    """Return whether a non-research task explicitly needs current external knowledge."""

    if task_kind != "code":
        return False
    if _EXPLICIT_FRONTIER_RE.search(goal):
        return True
    if _QUALITY_FRONTIER_RE.search(goal) and _EXTERNAL_TECHNOLOGY_RE.search(goal):
        return True
    return bool(
        not _LOCAL_CODE_PATH_RE.search(goal)
        and _EXTERNAL_IMPLEMENTATION_RE.search(goal)
        and _EXTERNAL_TECHNOLOGY_RE.search(goal)
    )


def source_classes(goal: str) -> list[str]:
    """Return the smallest useful source portfolio for the requested task."""

    classes = [
        "official_documentation",
        "official_release_or_standard",
        "maintained_reference_repository",
    ]
    if _RESEARCH_SOURCE_RE.search(goal):
        classes.extend(("primary_research", "independent_replication_or_review"))
    classes.append("credible_counterevidence")
    return classes


def needs_scholarly_sources(goal: str) -> bool:
    """Return whether the task would benefit from primary research."""

    return bool(_RESEARCH_SOURCE_RE.search(goal))


SOURCE_QUALITY_SIGNALS = [
    "direct_relevance",
    "authority",
    "freshness",
    "independence",
    "maintenance_activity",
    "version_compatibility",
    "reproducibility",
]

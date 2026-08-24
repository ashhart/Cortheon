"""Content-free task classification for experience lookup."""

from __future__ import annotations

import re

from cortheon.experience_core._compat import facade

_PACKAGE_TASK = re.compile(
    r"\b(?:api|dependency|import|library|module|package|pip|sdk|version)\b",
    re.IGNORECASE,
)
_REPOSITORY_TASK = re.compile(
    r"\b(?:debug|diff|implement|patch|repo(?:sitory)?|test)\b|"
    r"\b(?:write|review|execute|fix|repair)\b.{0,40}\bcode\b",
    re.IGNORECASE,
)
_DOCUMENT_TASK = re.compile(
    r"\b(?:document|documents|file|paper|passage|policy|report)\b",
    re.IGNORECASE,
)
_CROSS_DOCUMENT_TASK = re.compile(
    r"\b(?:across|compare|connect|contradiction|join|relate|synthesi[sz]e)\b",
    re.IGNORECASE,
)
_RESEARCH_TASK = re.compile(
    r"\b(?:citation|current|evidence|latest|news|recent|research|search|"
    r"source|today|web)\b|https?://",
    re.IGNORECASE,
)
_QUANTITATIVE_TASK = re.compile(
    r"\b(?:average|calculate|equation|numeric|percentage|probability|rate|"
    r"statistics?|sum)\b|\d+\s*[-+*/%]\s*\d+",
    re.IGNORECASE,
)


def classify_experience_task(task: str) -> tuple[str, str, tuple[str, ...]]:
    """Map a mission to a small content-free experience taxonomy."""

    api = facade()
    text = " ".join(str(task).split())[:12_000]
    tags: list[str] = []
    if api._PACKAGE_TASK.search(text):
        tags.append("package_source")
    if api._REPOSITORY_TASK.search(text):
        tags.append("repository")
    if api._DOCUMENT_TASK.search(text):
        tags.append("documents")
    if api._CROSS_DOCUMENT_TASK.search(text):
        tags.append("cross_source")
    if api._RESEARCH_TASK.search(text):
        tags.append("current_information")
    if api._QUANTITATIVE_TASK.search(text):
        tags.append("quantitative")

    if "repository" in tags:
        return "repository_code", "repository_patch", tuple(tags)
    if "package_source" in tags:
        return "package_api", "current_api", tuple(tags)
    if "documents" in tags:
        family = "cross_document_synthesis" if "cross_source" in tags else "document_question"
        return "document_reasoning", family, tuple(tags)
    if "current_information" in tags:
        return "live_research", "source_grounding", tuple(tags)
    if "quantitative" in tags:
        return "quantitative_reasoning", "calculation", tuple(tags)
    return "general_reasoning", "general_question", tuple(tags)

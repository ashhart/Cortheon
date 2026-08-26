"""Goal parsing, task classification, and deliverable inference."""

from __future__ import annotations

import re
from collections.abc import Iterable

from cortheon.cognitive_core.models import Observation
from cortheon.cognitive_core.profiles import (
    _CHANGE_HINTS,
    _CODE_HINTS,
    _DECISION_HINTS,
    _DOCUMENT_HINTS,
    _RESEARCH_HINTS,
    _has_hint,
)
from cortheon.cognitive_core.semantic_graph import _keywords, _semantic_terms
from cortheon.cognitive_core.text import _SPACE_RE, _normalized

_CODE_PATH_RE = re.compile(
    r"\b[A-Za-z0-9_./-]+\."
    r"(?:c|cc|cpp|cs|css|go|h|hpp|html|java|js|jsx|json|kt|php|py|rb|rs|sh|sql|"
    r"swift|toml|ts|tsx|vue|xml|yaml|yml)\b",
    flags=re.IGNORECASE,
)


_DOCUMENT_PATH_RE = re.compile(
    r"\b[A-Za-z0-9_./-]+\."
    r"(?:adoc|docx|log|markdown|md|pdf|rst|text|txt)\b",
    flags=re.IGNORECASE,
)


_TECHNOLOGY_NAMES_THAT_LOOK_LIKE_PATHS = frozenset(
    {
        "d3.js",
        "next.js",
        "node.js",
        "nuxt.js",
        "react.js",
        "three.js",
        "vue.js",
    }
)


_CODE_SYMBOL_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")


_QUALIFIED_CODE_SYMBOL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.([A-Za-z_][A-Za-z0-9_]*)\b")


_CODE_EXTENSION_NAMES = frozenset(
    {
        "c",
        "cc",
        "cpp",
        "cs",
        "css",
        "go",
        "h",
        "hpp",
        "html",
        "java",
        "js",
        "jsx",
        "json",
        "kt",
        "log",
        "markdown",
        "md",
        "php",
        "pdf",
        "py",
        "rb",
        "rs",
        "rst",
        "sh",
        "sql",
        "swift",
        "text",
        "toml",
        "ts",
        "tsx",
        "txt",
        "vue",
        "xml",
        "yaml",
        "yml",
    }
)


_INTEGER_TOKEN = r"[-+]?(?:0[xX][0-9A-Fa-f][0-9A-Fa-f_]*|[0-9][0-9_,]*)"


_CROSS_SOURCE_HINTS = frozenset(
    {
        "across",
        "both",
        "compare",
        "connect",
        "document",
        "documents",
        "join",
        "joined",
        "joining",
        "relationship",
        "sources",
    }
)


_ABDUCTIVE_GOAL_RE = re.compile(
    r"\b(?:abductive|ambiguous|ambiguity|competing|hypotheses|hypothesis|"
    r"causal|deriv(?:e|ation)|diagnos(?:e|is)|disprov(?:e|ing)|"
    r"explanation|falsif(?:y|ication)|infer|inference|synthesi[sz]e)\b",
    flags=re.IGNORECASE,
)


_AMBIGUITY_GOAL_RE = re.compile(
    r"\b(?:ambiguous|ambiguity|cannot determine|insufficient|preserve ambiguity|"
    r"rather than guessing|do not (?:guess|invent)|tie-break|clarif(?:y|ication)|"
    r"competing interpretations|viable interpretations)\b",
    flags=re.IGNORECASE,
)

_DISCRIMINATING_TEST_DESIGN_RE = re.compile(
    r"\b(?:choose|select|identify)\b.{0,100}\b(?:probe|test|intervention)\b"
    r".{0,120}\b(?:separate|distinguish|discriminate)\w*\b",
    flags=re.IGNORECASE | re.DOTALL,
)


_EXTERNAL_RESEARCH_REPORT_RE = re.compile(
    r"\b(?:report|research brief|survey|white paper)\b",
    flags=re.IGNORECASE,
)


_EXTERNAL_SOURCE_RE = re.compile(
    r"\b(?:arxiv|doi|github|online|papers?|pubmed|research|sources?|web)\b",
    flags=re.IGNORECASE,
)


def _is_discriminating_test_design_goal(goal: str) -> bool:
    """Whether the task asks for a test, not for a decided hypothesis."""

    return _DISCRIMINATING_TEST_DESIGN_RE.search(goal) is not None


def _is_cross_source_derivation_goal(goal: str) -> bool:
    """Whether the task requests an explicit source-bound relational chain."""

    return bool(
        re.search(r"\bderive\w*\b", goal, re.I)
        and re.search(r"\b(?:separately sourced|source-bound|premise path)\b", goal, re.I)
    )


def _is_adaptive_stopping_goal(goal: str) -> bool:
    """Whether the task exposes sequential probes and an explicit stop condition."""

    return bool(
        re.search(r"\b(?:probe|action)s?\b", goal, re.I)
        and re.search(r"\bstop\b", goal, re.I)
        and re.search(r"\b(?:decision|sufficient)\b", goal, re.I)
    )


def _is_hypothesis_design_goal(goal: str) -> bool:
    """Whether the requested output is a hypothesis set and a test design."""

    return (
        bool(re.search(r"\b(?:frame|formulate|generate|propose)\w*\b", goal, re.I))
        and bool(re.search(r"\b(?:hypothes(?:is|es)|rival|alternative)\w*\b", goal, re.I))
        and bool(re.search(r"\b(?:falsif\w*|intervention|test)\b", goal, re.I))
    )


def _is_contradiction_revision_goal(goal: str) -> bool:
    """Whether the requested output explicitly binds a prior to a revision."""

    return bool(
        re.search(r"\b(?:original|prior) hypothesis\b", goal, re.I)
        and re.search(
            r"\b(?:revised(?: or retained)? hypothesis|retained hypothesis|revision|new status)\b",
            goal,
            re.I,
        )
        and re.search(r"\b(?:decisive source|forced the change|counterevidence)\b", goal, re.I)
    )


def _requested_hypothesis_count(goal: str, minimum: int) -> int:
    """Honor an explicit request for competing hypotheses even in quick mode."""

    if re.search(r"\b(?:rival|alternative|competing|hypotheses)\b", goal, re.I):
        return max(2, minimum)
    return minimum


def _infer_join_operation(goal: str) -> str | None:
    normalized = _normalized(goal)
    if re.search(r"\b(?:sum|total|add(?:ed|ing)?|plus)\b", normalized) or "+" in goal:
        return "sum"
    return None


def _goal_code_paths(goal: str) -> list[str]:
    """Extract explicit local paths without treating framework names as files."""

    return list(
        dict.fromkeys(
            path
            for path in _CODE_PATH_RE.findall(_without_url_literals(goal))
            if path.casefold() not in _TECHNOLOGY_NAMES_THAT_LOOK_LIKE_PATHS
        )
    )


def _goal_document_paths(goal: str) -> list[str]:
    """Extract explicit local document paths while ignoring URL path segments."""

    return list(dict.fromkeys(_DOCUMENT_PATH_RE.findall(_without_url_literals(goal))))


def _without_url_literals(value: str) -> str:
    return re.sub(r"https?://[^\s<>\"'`]+", " ", value, flags=re.IGNORECASE)


def _goal_code_symbols(goal: str) -> list[str]:
    symbols = list(_CODE_SYMBOL_RE.findall(goal))
    for match in _QUALIFIED_CODE_SYMBOL_RE.finditer(goal):
        symbol = match.group(1)
        if symbol.casefold() in _CODE_EXTENSION_NAMES or symbol in symbols:
            continue
        symbols.append(symbol)
    return symbols


def _parse_integer(token: str) -> int:
    normalized = token.replace("_", "").replace(",", "")
    return int(normalized, 0)


def _answer_integer_assertions(answer: str) -> set[int]:
    tokens: list[str] = []
    equations = re.findall(rf"=\s*({_INTEGER_TOKEN})(?!\s*[+*/-])", answer)
    if equations:
        tokens.append(equations[-1])
    tokens.extend(
        re.findall(
            rf"\b(?:sum|total|answer|result)\s+(?:is|equals?|:)\s*"
            rf"({_INTEGER_TOKEN})(?!\s*[+*/-])",
            answer,
            flags=re.IGNORECASE,
        )
    )
    results: set[int] = set()
    for token in tokens:
        try:
            results.add(_parse_integer(token))
        except ValueError:
            continue
    return results


def _is_test_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].casefold()
    return bool(
        re.search(
            r"(?:^test[_-]|[_-]test\.|(?:^|[._-])tests?(?:[._/-]|$)|"
            r"\.(?:spec|test)\.(?:js|jsx|ts|tsx)$)",
            name,
        )
    )


def _discovered_project_paths(
    goal: str,
    observations: Iterable[Observation],
    *,
    pattern: re.Pattern[str],
    maximum: int,
) -> list[str]:
    """Rank safe project-relative paths from host-receipted search output."""

    goal_terms = _keywords(goal)
    scores: dict[str, tuple[int, int]] = {}
    allowed_tools = {
        "bash",
        "find",
        "glob",
        "grep",
        "ls",
        "read",
        "search",
        "shell",
    }
    for observation in observations:
        receipt = observation.host_receipt
        if receipt is None or str(receipt.get("tool", "")).casefold() not in allowed_tools:
            continue
        for line in observation.content.splitlines():
            line_terms = _keywords(line)
            for match in pattern.finditer(line):
                prefix = line[max(0, match.start() - 3) : match.start()]
                candidate = match.group(0)
                path = candidate.replace("\\", "/").strip(".,:;()[]{}'\"")
                parts = [part for part in path.split("/") if part]
                if (
                    not path
                    or prefix.endswith(("/", "\\"))
                    or ".." in prefix
                    or path.startswith("/")
                    or "://" in path
                    or ".." in parts
                    or any(part in {".git", ".cortheon", "node_modules"} for part in parts)
                    or len(path) > 240
                ):
                    continue
                path_terms = _keywords(path.rsplit("/", 1)[-1])
                relevance = len(goal_terms & line_terms) * 4 + len(goal_terms & path_terms) * 2
                previous_relevance, appearances = scores.get(path, (0, 0))
                scores[path] = (max(previous_relevance, relevance), appearances + 1)
    ranked = sorted(
        scores,
        key=lambda path: (
            -scores[path][0],
            -scores[path][1],
            len(path),
            path.casefold(),
        ),
    )
    return ranked[:maximum]


def _abductive_proposition(content: str, goal: str) -> str:
    """Select one compact public clue without promoting it to a conclusion."""

    goal_terms = _semantic_terms(goal)
    candidates = [
        _SPACE_RE.sub(" ", item).strip(" -\t")
        for item in re.split(r"(?<=[.!?])\s+|\n+", content)
        if 12 <= len(item.strip()) <= 500
    ]
    if not candidates:
        return ""
    selected = max(
        candidates,
        key=lambda item: (
            len(_semantic_terms(item) & goal_terms),
            len(_semantic_terms(item)),
            -len(item),
        ),
    )
    return selected[:240]


def _observation_score(observation: Observation, keywords: set[str]) -> float:
    overlap = len(_keywords(observation.content) & keywords)
    status_bonus = 5 if observation.status == "verified" else 0
    kind_bonus = 3 if observation.kind in {"test", "code", "diff"} else 0
    failure_penalty = -5 if observation.status == "failed" else 0
    return float(overlap + status_bonus + kind_bonus + failure_penalty)


def _infer_task_kind(goal: str) -> str:
    if _goal_code_paths(goal):
        return "code"
    if _goal_document_paths(goal):
        return "documents"
    if (
        (_ABDUCTIVE_GOAL_RE.search(goal) or _AMBIGUITY_GOAL_RE.search(goal))
        and re.search(
            r"\b(?:do\s+not|don't|without)\s+(?:modify|edit|change)(?:ing)?\s+files?\b",
            goal,
            flags=re.IGNORECASE,
        )
        and not re.search(
            r"\b(?:api|bug|class|cli|code|exception|function|method|package|"
            r"parser|repository|runtime|stack trace)\b",
            goal,
            flags=re.IGNORECASE,
        )
    ):
        return "documents"
    if (
        _EXTERNAL_RESEARCH_REPORT_RE.search(goal)
        and _EXTERNAL_SOURCE_RE.search(goal)
        and not _requests_change(goal)
    ):
        return "research"
    if _has_hint(goal, _CODE_HINTS):
        return "code"
    if _has_hint(goal, _DOCUMENT_HINTS) and _has_hint(goal, _CROSS_SOURCE_HINTS):
        return "documents"
    if _has_hint(goal, _RESEARCH_HINTS):
        return "research"
    if _has_hint(goal, _DOCUMENT_HINTS):
        return "documents"
    if _has_hint(goal, _DECISION_HINTS):
        return "decision"
    return "general"


def _requests_change(goal: str) -> bool:
    positive = re.sub(
        r"\b(?:do\s+not|don't|must\s+not|without)\s+"
        r"(?:chang(?:e|ing)|modif(?:y|ying)|edit(?:ing)?)\b",
        "",
        goal,
        flags=re.IGNORECASE,
    )
    # File paths are locations, not intents: a read-only goal naming
    # patch_runner.py or fix_imports.py must not classify as a change.
    for path in (*_goal_code_paths(goal), *_goal_document_paths(goal)):
        positive = positive.replace(path, " ")
    return _has_hint(positive, _CHANGE_HINTS)


def _infer_deliverable(goal: str, task_kind: str) -> str:
    if task_kind == "code":
        return "code_change" if _requests_change(goal) else "code_understanding"
    if task_kind == "research":
        return "research_answer"
    if task_kind == "documents":
        return "document_synthesis"
    if task_kind == "decision":
        return "decision"
    return "answer"

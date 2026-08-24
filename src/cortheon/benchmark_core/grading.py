"""Semantic answer grading for every benchmark case family."""

from __future__ import annotations

import re

from cortheon.benchmark_core.models import (
    WITHHELD_PREFIX,
    BenchmarkCase,
    DiagnosticCase,
    JoinCase,
    LongHorizonCase,
    PatchCase,
    PlanningCase,
    ReasoningCase,
    ResearchCase,
    SemanticCase,
)


def _semantic_text(text: str) -> str:
    """Normalize presentation markup without changing semantic word order."""

    unmarked = re.sub(
        r"[*_`~#\[\]()\-\u2010-\u2015\u2192\u21d2]+",
        " ",
        text.casefold(),
    )
    return re.sub(r"\s+", " ", unmarked).strip()


def _reasoning_semantic_text(text: str, mode: str) -> str:
    semantic = _semantic_text(text)
    if mode != "ambiguity":
        return semantic
    if any(
        marker in semantic
        for marker in (
            "ambiguity",
            "conflict",
            "incompatible",
            "underspecified",
            "not actionable",
            "cannot determine",
            "does not specify",
            "no basis",
        )
    ):
        semantic += " ambiguous"
    if any(
        marker in semantic
        for marker in (
            "alternative",
            "interpretation",
            "either",
            "both",
            "two different",
            "maps to two",
            " while ",
        )
    ):
        semantic += " alternative"
    return semantic


def _reasoning_expected_present(text: str, expected: str) -> bool:
    """Accept narrow lexical variants without weakening the required concept."""

    semantic = _semantic_text(text)
    term = _semantic_text(expected)
    if term in semantic:
        return True
    if expected == "legacy token broker":
        return "legacy broker" in semantic
    if expected == "invoice line":
        return "invoice" in semantic and "line" in semantic
    if expected == "visitor-to-account":
        return "visitor" in semantic and any(
            marker in semantic for marker in ("account", "signup", "sign up")
        )
    if expected == "checkout-to-paid-order":
        return "checkout" in semantic and any(
            marker in semantic for marker in ("paid", "purchase", "order")
        )
    if expected == "hourly reconciliation":
        return "reconciliation" in semantic and any(
            marker in semantic for marker in ("hourly", "top of hour", "minute 00")
        )
    if expected == "hot partition":
        return "partition" in semantic and any(
            marker in semantic for marker in ("hot", "saturat", "throttl")
        )
    return False


def _derived_relation_present(
    text: str,
    relation: tuple[tuple[str, ...], ...],
) -> bool:
    """Require cross-source concepts to be joined in an explanatory passage."""

    bridged = re.sub(r"[\u2192\u21d2]", " therefore ", text)
    semantic = _semantic_text(bridged)
    causal_markers = (
        "because",
        "caus",
        "consequently",
        "drop",
        "drives",
        "due to",
        "explains",
        "expires",
        "leads to",
        "makes",
        "means that",
        "omit",
        "opens",
        "prevents",
        "produces",
        "returns",
        "result",
        "so that",
        "therefore",
        "thereby",
        "triggers",
        "unauthorized access",
    )
    windows = [
        semantic[max(0, match.start() - 700) : match.end() + 700]
        for marker in causal_markers
        for match in re.finditer(re.escape(marker), semantic)
    ]
    return any(
        all(
            any(_semantic_text(alternative) in passage for alternative in concept)
            for concept in relation
        )
        for passage in windows
    )


def _semantic_forbidden_asserted(text: str, forbidden: str) -> bool:
    """Return whether a distractor is asserted rather than explicitly rejected."""

    rejected_markers = (
        "archived",
        "discard",
        "distractor",
        "former",
        "not current",
        "old directory",
        "previous",
        "contradict",
        "does not explain",
        "doesn't explain",
        "doesn t explain",
        "fails to explain",
        "less likely",
        "unlikely",
        "weaker",
        "alternative",
        "competing",
        "separate",
        "stale",
        "supersed",
        "unrelated",
    )
    normalized_forbidden = _semantic_text(forbidden)
    normalized_text = _semantic_text(text)
    cursor = 0
    while True:
        index = normalized_text.find(normalized_forbidden, cursor)
        if index < 0:
            return False
        window = normalized_text[max(0, index - 260) : index + len(normalized_forbidden) + 420]
        selection_prefix = normalized_text[max(0, index - 160) : index]
        selection_suffix = normalized_text[
            index + len(normalized_forbidden) : index + len(normalized_forbidden) + 120
        ]
        selected_markers = (
            "best supported",
            "leading hypothesis",
            "most likely",
            "root cause",
            "selected explanation",
            "strongest explanation",
            "therefore caused by",
        )
        if any(marker in selection_prefix for marker in selected_markers):
            return True
        for marker in selected_markers:
            marker_index = selection_suffix.find(marker)
            if marker_index >= 0 and not any(
                rejected in selection_suffix[:marker_index] for rejected in rejected_markers
            ):
                return True
        if not any(marker in window for marker in rejected_markers):
            return True
        cursor = index + len(normalized_forbidden)


def _ambiguity_forbidden_asserted(text: str, forbidden: str) -> bool:
    """Distinguish enumerated interpretations from an unjustified chosen action."""

    normalized_text = _semantic_text(text)
    normalized_forbidden = _semantic_text(forbidden)
    cursor = 0
    while True:
        index = normalized_text.find(normalized_forbidden, cursor)
        if index < 0:
            return False
        window = normalized_text[max(0, index - 240) : index + len(normalized_forbidden) + 160]
        ambiguity_markers = (
            "ambiguous",
            "could mean",
            "either",
            "interpretation",
            "neither",
            "which",
            "without additional context",
        )
        action_markers = (
            "therefore deploy",
            "proceed with",
            "recommended action",
            "should deploy",
            "will deploy",
        )
        if any(marker in window for marker in action_markers):
            return True
        if not any(marker in window for marker in ambiguity_markers):
            return True
        cursor = index + len(normalized_forbidden)


def _grade(case: BenchmarkCase, text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized or normalized.startswith(WITHHELD_PREFIX.lower()):
        return False
    if isinstance(case, (PatchCase, LongHorizonCase)):
        return bool(normalized)
    if isinstance(case, (SemanticCase, DiagnosticCase)):
        semantic = _semantic_text(text)
        return all(_semantic_text(term) in semantic for term in case.expected) and not any(
            _semantic_forbidden_asserted(text, term) for term in case.forbidden_answers
        )
    if isinstance(case, ReasoningCase):
        semantic = _reasoning_semantic_text(text, case.mode)
        return bool(
            all(_reasoning_expected_present(text, term) for term in case.expected)
            and all(
                any(_semantic_text(term) in semantic for term in alternatives)
                for alternatives in case.required_any
            )
            and not any(
                (
                    _ambiguity_forbidden_asserted(text, term)
                    if case.mode == "ambiguity"
                    else _semantic_forbidden_asserted(text, term)
                )
                for term in case.forbidden_answers
            )
            and all(
                _derived_relation_present(text, relation) for relation in case.derived_relations
            )
        )
    if isinstance(case, PlanningCase):
        semantic = _semantic_text(text)
        positions = [semantic.find(_semantic_text(step)) for step in case.ordered_steps]
        return bool(
            all(position >= 0 for position in positions)
            and positions == sorted(positions)
            and len(set(positions)) == len(positions)
            and all(_semantic_text(term) in semantic for term in case.expected)
            and not any(_semantic_forbidden_asserted(text, term) for term in case.forbidden_answers)
        )
    if isinstance(case, ResearchCase):
        version = re.search(
            rf"(?<![0-9])v?{re.escape(case.expected)}(?![0-9]|\.[0-9])",
            normalized,
        )
        return bool(
            version
            and "github.com/" in normalized
            and "pypi.org/" in normalized
            and re.search(
                r"\b(?:agree|conflict|contradict|differ|consistent)\w*\b",
                normalized,
            )
        )
    if isinstance(case, JoinCase):
        integer = r"[-+]?(?:0x[0-9a-f][0-9a-f_]*|[0-9][0-9_,]*)"
        numeric_text = re.sub(r"[`*_]", "", normalized)
        observed_values: set[int] = set()
        for token in re.findall(integer, numeric_text):
            try:
                observed_values.add(int(token.replace("_", "").replace(",", ""), 0))
            except ValueError:
                continue
        if not set(case.values).issubset(observed_values):
            return False
        tokens: list[str] = []
        equations = re.findall(rf"=\s*({integer})(?!\s*[+*/-])", numeric_text)
        if equations:
            tokens.append(equations[-1])
        tokens.extend(
            re.findall(
                rf"\b(?:sum|total|answer|result)\s+(?:is|equals?|:)\s*"
                rf"({integer})(?!\s*[+*/-])",
                numeric_text,
            )
        )
        asserted: set[int] = set()
        for token in tokens:
            token = token.replace("_", "").replace(",", "")
            try:
                asserted.add(int(token, 0))
            except ValueError:
                continue
        return asserted == {case.expected}
    polarity_text = re.sub(r"[`*_]", "", normalized)
    explicit_yes = (
        re.search(r"^(?:answer:\s*)?yes\b", polarity_text) is not None
        or re.search(r"\banswer\s*:\s*yes\b", polarity_text) is not None
    )
    explicit_no = (
        re.search(r"^(?:answer:\s*)?no\b", polarity_text) is not None
        or re.search(r"\banswer\s*:\s*no\b", polarity_text) is not None
    )
    negative = (
        explicit_no
        or re.search(
            r"\b(?:does not|doesn't|not import|no import|without)\b",
            normalized,
        )
        is not None
    )
    static_import = (
        re.search(
            rf"(?:\bfrom\s+{re.escape(case.module.lower())}\s+import\b|"
            rf"\bimport\s+{re.escape(case.module.lower())}\b)",
            normalized,
        )
        is not None
    )
    described_import = (
        re.search(
            rf"\bimports?\s+(?:the\s+)?(?:module\s+)?"
            rf"{re.escape(case.module.lower())}\b",
            polarity_text,
        )
        is not None
    )
    if case.expected:
        return (
            not negative
            and (explicit_yes or static_import or described_import)
            and case.module.lower() in normalized
        )
    return negative and not explicit_yes

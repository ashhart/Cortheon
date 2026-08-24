from __future__ import annotations

import re
from typing import Any

_RELATION = re.compile(
    r"(?:\bis\b|\bare\b|\bwas\b|\bwere\b|\bequals?\b|"
    r"\bidentif(?:y|ies|ied)\b|\bnames?\b|\blists?\b|"
    r"\bassign(?:s|ed)?\b|\bserves?\s+as\b|\bas\b|[:=])",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:not|never|no|isn'?t|wasn'?t|doesn'?t|can(?:not|'t)|"
    r"denies|rejects|unsupported|incorrect|false)\b",
    re.IGNORECASE,
)


def grade_document_relations(grader: dict[str, Any], answer: str) -> list[str]:
    """Grade structured identity claims with relation, polarity, and source binding."""

    claims = list(grader.get("claims") or [])
    all_sources = [alias for claim in claims for alias in _string_list(claim.get("source_aliases"))]
    units = _evidence_units(answer)
    shared_source_text = answer if grader.get("shared_source") is True else None
    failures: list[str] = []
    for claim in claims:
        failure = _grade_claim(
            claim,
            units,
            all_sources=all_sources,
            shared_source_text=shared_source_text,
        )
        if failure is not None:
            failures.append(f"{failure}:{claim['id']}")
    return failures


def grade_pypi_metadata(grader: dict[str, Any], answer: str) -> list[str]:
    """Bind live PyPI values to their fields and to the named live source."""

    key = grader["answer_key"]
    package = str(grader.get("package") or "package")
    claims = [
        {
            "id": "package_version",
            "relation": "identity",
            "subject_aliases": [package, f"{package} version", "package version"],
            "object_aliases": [str(key["version"])],
            "source_aliases": ["PyPI", "pypi.org"],
        }
    ]
    requirement = str(key.get("requires_python") or "")
    if requirement:
        claims.append(
            {
                "id": "python_requirement",
                "relation": "identity",
                "subject_aliases": [
                    "Requires-Python",
                    "Requires Python",
                    "Python requirement",
                    "Python constraint",
                ],
                "object_aliases": [requirement],
                "source_aliases": ["PyPI", "pypi.org"],
            }
        )
    return grade_document_relations({"claims": claims, "shared_source": True}, answer)


def _grade_claim(
    claim: dict[str, Any],
    units: list[str],
    *,
    all_sources: list[str],
    shared_source_text: str | None,
) -> str | None:
    subjects = _string_list(claim.get("subject_aliases"))
    objects = _string_list(claim.get("object_aliases"))
    sources = _string_list(claim.get("source_aliases"))
    relation_seen = False
    wrong_polarity = False
    for unit in units:
        for relation_span in _relation_spans(unit, subjects, objects):
            relation_seen = True
            polarity_text = unit.replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
            if _NEGATION.search(polarity_text):
                wrong_polarity = True
                continue
            if _source_bound(unit, relation_span, sources, all_sources):
                return None
            if shared_source_text is not None and _source_is_cited(shared_source_text, sources):
                return None
    if wrong_polarity:
        return "wrong_polarity"
    if relation_seen:
        return "missing_source_binding"
    return "missing_relation"


def _relation_spans(
    text: str,
    subjects: list[str],
    objects: list[str],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for subject in _literal_spans(text, subjects):
        for object_ in _literal_spans(text, objects):
            first, second = sorted((subject, object_))
            connector = text[first[1] : second[0]]
            if len(connector) <= 100 and _RELATION.search(connector):
                spans.append((first[0], second[1]))
    return spans


def _source_bound(
    text: str,
    claim_span: tuple[int, int],
    desired_aliases: list[str],
    all_aliases: list[str],
) -> bool:
    desired = _literal_spans(text, desired_aliases)
    if not desired:
        return False
    desired_distance = min(_span_distance(claim_span, span) for span in desired)
    competing_aliases = [
        alias
        for alias in all_aliases
        if alias.casefold() not in {v.casefold() for v in desired_aliases}
    ]
    competitors = _literal_spans(text, competing_aliases)
    if not competitors:
        return True
    competing_distance = min(_span_distance(claim_span, span) for span in competitors)
    return desired_distance < competing_distance


def _literal_spans(text: str, aliases: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for alias in aliases:
        parts = [re.escape(part) for part in alias.split()]
        pattern = r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)"
        spans.extend(match.span() for match in re.finditer(pattern, text, re.IGNORECASE))
    return spans


def _source_is_cited(text: str, aliases: list[str]) -> bool:
    for start, end in _literal_spans(text, aliases):
        context = text[max(0, start - 40) : min(len(text), end + 40)]
        if re.search(
            r"\b(?:according to|checked|from|metadata|reports?|source|via)\b",
            context,
            re.IGNORECASE,
        ):
            return True
    return False


def _span_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    if right[1] < left[0]:
        return left[0] - right[1]
    if left[1] < right[0]:
        return right[0] - left[1]
    return 0


def _evidence_units(answer: str) -> list[str]:
    return [
        unit.strip()
        for unit in re.split(
            r"(?<=[.!?])\s+|[\r\n]+|;\s*|,\s*(?=(?:while|whereas|and)\b)",
            answer,
        )
        if unit.strip()
    ]


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value or [] if str(item)]

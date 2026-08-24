"""Keyword and semantic edge extraction from evidence text."""

from __future__ import annotations

import re

from cortheon.cognitive_core.models import SemanticEdge, SemanticRule
from cortheon.cognitive_core.text import _WORD_RE


def _keywords(value: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(value) if len(token) >= 3}


_SEMANTIC_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "also",
        "answer",
        "before",
        "between",
        "could",
        "current",
        "catalog",
        "directory",
        "document",
        "documents",
        "every",
        "from",
        "have",
        "into",
        "named",
        "only",
        "question",
        "read",
        "register",
        "roster",
        "should",
        "source",
        "that",
        "their",
        "then",
        "these",
        "they",
        "this",
        "through",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
    }
)


def _semantic_phrase(value: str) -> str:
    cleaned = re.sub(r"^[\s#>*-]+", "", value).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(
        r"^(?:the|a|an)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(?:change\s+class|classified\s+as)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+changes?$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" \t:;,.`'\"")[:120]


def _semantic_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    return re.sub(
        r"^(?:the\s+)?(?:service|system|project|certificate|team|role|group|"
        r"component|application|app|queue|database|db|job)\s+",
        "",
        key,
    )


def _semantic_edge(
    source: str,
    target: str,
    *,
    document: str,
    relation: str,
    priority: int = 1,
) -> SemanticEdge | None:
    source_display = _semantic_phrase(source)
    target_display = _semantic_phrase(target)
    source_key = _semantic_key(source_display)
    target_key = _semantic_key(target_display)
    if (
        not source_key
        or not target_key
        or source_key == target_key
        or len(source_key.split()) > 12
        or len(target_key.split()) > 12
    ):
        return None
    return SemanticEdge(
        source_key=source_key,
        source=source_display,
        target_key=target_key,
        target=target_display,
        document=document,
        relation=relation,
        priority=priority,
    )


def _semantic_edges(line: str, document: str) -> list[SemanticEdge]:
    text = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", line).strip()
    if not text or text.startswith("#"):
        return []
    patterns: tuple[tuple[str, str, int], ...] = (
        (
            r"^(?:as\s+of\s+[^,.;]+,\s*)?(?:the\s+)?current\s+"
            r"(?:owner|approver|responder|on-call|maintainer)\s+"
            r"(?:for|of)\s+(.+?)\s+is\s+(.+?)(?:[.;]|$)",
            "ownership",
            2,
        ),
        (
            r"^(?:as\s+of\s+[^,.;]+,\s*)?(.+?)\s+is\s+currently\s+"
            r"(?:owned|approved|managed|maintained|operated)\s+by\s+"
            r"(.+?)(?:[.;]|$)",
            "ownership",
            2,
        ),
        (
            r"^(.+?)\s+\((?:also\s+known\s+as|internally\s+(?:called|known\s+as)|"
            r"internal\s+codename|codename|alias|formerly)\s+(.+?)\)"
            r"(?:[.;]|$)",
            "alias",
            1,
        ),
        (
            r"^(.+?)\s+is\s+(?:also\s+known\s+as|internally\s+(?:called|known\s+as)|"
            r"codenamed|called|referred\s+to\s+as)\s+(.+?)(?:[.;]|$)",
            "alias",
            1,
        ),
        (
            r"^(.+?)\s+cannot\s+resume\s+until\s+(.+?)\s+is\s+"
            r"(?:renewed|restored|resolved|repaired|available)(?:[.;]|$)",
            "blocking_dependency",
            1,
        ),
        (
            r"^(.+?)\s+is\s+waiting\s+on\s+(.+?)(?:[.;]|$)",
            "blocking_dependency",
            1,
        ),
        (
            r"^(.+?)\s+(?:depends|relies)\s+on\s+(.+?)(?:[.;]|$)",
            "dependency",
            1,
        ),
        (
            r"^(.+?)\s+belongs\s+to\s+(.+?)(?:[.;]|$)",
            "ownership",
            1,
        ),
        (
            r"^(.+?)\s+is\s+(?:owned|managed|maintained|operated)\s+by\s+"
            r"(.+?)(?:[.;]|$)",
            "ownership",
            1,
        ),
        (
            r"^(?:the\s+)?(?:owner|approver|responder|on-call|maintainer)\s+"
            r"(?:for|of)\s+(.+?)\s+is\s+(.+?)(?:[.;]|$)",
            "ownership",
            1,
        ),
        (
            r"^(.+?)\s+(?:routes|escalates|maps)\s+to\s+(.+?)(?:[.;]|$)",
            "mapping",
            1,
        ),
        (
            r"^(.+?)\s+requires?\s+.+?\s+for\s+risk\s+band\s+"
            r"(.+?)(?:[.;]|$)",
            "classification",
            1,
        ),
        (
            r"^(.+?)\s+releases?\s+(?:are\s+approved\s+by|uses?|need|needs)\s+"
            r"(?:the\s+)?(.+?)(?:[.;]|$)",
            "requirement",
            1,
        ),
        (
            r"^systems?\s+storing\s+(.+?)\s+(?:require|requires|need|needs)"
            r"\s+(?:(?:approval|sign-off)\s+(?:from|by)\s+)"
            r"(?:the\s+)?(.+?)(?:\s+before\b|[.;]|$)",
            "requirement",
            1,
        ),
        (
            r"^(.+?)\s+stores\s+(.+?)(?:\s+for\b|[.;]|$)",
            "attribute",
            1,
        ),
        (
            r"^(.+?)\s+(?:handles|processes|uses)\s+(.+?)(?:[.;]|$)",
            "attribute",
            1,
        ),
        (
            r"^(.+?)\s+(?:serves|supports)\s+(.+?)(?:[.;]|$)",
            "scope",
            1,
        ),
        (
            r"^(.+?)\s+operates\s+in\s+(.+?)(?:[.;]|$)",
            "scope",
            1,
        ),
        (
            r"^(.+?)\s+is\s+(?:(?:classified\s+as|an?|the)\s+)?"
            r"(?:(?:change\s+class)\s+)?(.+?)(?:\s+and\b|[.;]|$)",
            "classification",
            1,
        ),
        (
            r"^(.+?)\s+(?:require|requires|need|needs)\s+"
            r"(?:(?:approval|sign-off)\s+(?:from|by)\s+)?"
            r"(?:(?:an?|the)\s+)?(.+?)(?:\s+before\b|[.;]|$)",
            "requirement",
            1,
        ),
        (
            r"^([^:\n]{2,100}):\s*([^:\n]{2,120})$",
            "mapping",
            1,
        ),
    )
    for pattern, relation, priority in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        edge = _semantic_edge(
            match.group(1),
            match.group(2),
            document=document,
            relation=relation,
            priority=priority,
        )
        return [edge] if edge is not None else []
    return []


def _semantic_table_cells(line: str) -> list[str]:
    if "|" not in line:
        return []
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    cleaned = [
        cell if re.fullmatch(r":?-{3,}:?", cell) else _semantic_phrase(cell) for cell in cells
    ]
    return cleaned if len(cleaned) >= 2 and all(cleaned) else []


def _semantic_table_relation(header: str) -> tuple[str, int]:
    key = _semantic_key(header)
    priority = 2 if re.search(r"\b(?:current|effective|active|primary)\b", key) else 1
    if re.search(r"\b(?:alias|codename|internal name|also known)\b", key):
        return "alias", priority
    if re.search(
        r"\b(?:depend|upstream|blocker|prerequisite|resource|dataset|queue|"
        r"certificate|database)\b",
        key,
    ):
        return "dependency", priority
    if re.search(
        r"\b(?:owner|steward|maintainer|responder|on call|approver|operator)\b",
        key,
    ):
        return "ownership", priority
    if re.search(r"\b(?:required approver|required role|approval|sign off)\b", key):
        return "requirement", priority
    if re.search(r"\b(?:class|risk band|category|tier|type)\b", key):
        return "classification", priority
    if re.search(r"\b(?:region|market|jurisdiction|scope|residents)\b", key):
        return "scope", priority
    if re.search(r"\b(?:data|handles|processes|stores)\b", key):
        return "attribute", priority
    return "mapping", priority


def _semantic_table_edges(content: str, document: str) -> list[SemanticEdge]:
    """Extract provenance-bound relations from ordinary Markdown tables."""

    lines = content.splitlines()
    edges: list[SemanticEdge] = []
    index = 0
    while index + 1 < len(lines):
        headers = _semantic_table_cells(lines[index])
        separators = _semantic_table_cells(lines[index + 1])
        if (
            not headers
            or len(headers) != len(separators)
            or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separators)
        ):
            index += 1
            continue
        row_index = index + 2
        while row_index < len(lines):
            cells = _semantic_table_cells(lines[row_index])
            if len(cells) != len(headers):
                break
            subject = cells[0]
            for header, target in zip(headers[1:], cells[1:], strict=True):
                if target.casefold() in {"n/a", "none", "unknown", "-"}:
                    continue
                relation, priority = _semantic_table_relation(header)
                edge = _semantic_edge(
                    subject,
                    target,
                    document=document,
                    relation=relation,
                    priority=priority,
                )
                if edge is not None:
                    edges.append(edge)
            row_index += 1
        index = max(index + 1, row_index)
    return edges


def _semantic_rules(line: str, document: str) -> list[SemanticRule]:
    text = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", line).strip()
    if not text or text.startswith("#"):
        return []
    pattern = (
        r"^(?:systems|services|applications|workloads)\s+(?:that\s+)?"
        r"(?:handle|process|store|use)\s+(.+?)\s+and\s+"
        r"(?:serve|support|operate\s+in)\s+(.+?)\s+"
        r"(?:require|need)\s+"
        r"(?:(?:approval|sign-off)\s+(?:from|by)\s+)?"
        r"(?:the\s+)?(.+?)(?:[.;]|$)"
    )
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return []
    first = _semantic_phrase(match.group(1))
    second = _semantic_phrase(match.group(2))
    target = _semantic_phrase(match.group(3))
    first_key = _semantic_key(first)
    second_key = _semantic_key(second)
    target_key = _semantic_key(target)
    if not first_key or not second_key or not target_key:
        return []
    return [
        SemanticRule(
            conditions=(
                ("attribute", first_key, first),
                ("scope", second_key, second),
            ),
            target_key=target_key,
            target=target,
            document=document,
        )
    ]


def _phrase_mentioned(answer: str, phrase: str) -> bool:
    answer_key = _semantic_key(answer)
    phrase_key = _semantic_key(phrase)
    if not phrase_key:
        return False
    return (
        re.search(
            rf"(?:^|\s){re.escape(phrase_key)}(?:\s|$)",
            answer_key,
        )
        is not None
    )


def _affirmatively_mentions(answer: str, phrase: str) -> bool:
    answer_key = _semantic_key(answer)
    phrase_key = _semantic_key(phrase)
    if not phrase_key:
        return False
    matches = list(
        re.finditer(
            rf"(?:^|\s){re.escape(phrase_key)}(?:\s|$)",
            answer_key,
        )
    )
    for match in matches:
        prefix = answer_key[max(0, match.start() - 50) : match.start()]
        if re.search(
            r"\b(?:not|never|neither|unlike|exclude|excludes|excluded|"
            r"rather than|instead of)\b[^.]{0,45}$",
            prefix,
        ):
            continue
        return True
    return False


def _semantic_terms(value: str) -> set[str]:
    """Return bounded lexical anchors for a public cross-source claim."""

    terms: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", value):
        normalized = token.casefold().replace("_", "-").strip("-")
        if len(normalized) < 4 or normalized in _SEMANTIC_STOPWORDS:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 4:
                normalized = normalized[: -len(suffix)]
                break
        terms.add(normalized)
    return terms

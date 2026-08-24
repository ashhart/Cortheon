"""Lexical and vector ranking plus cross-document join presentation."""

from __future__ import annotations

from collections import Counter
from typing import Any


def rank_chunks(
    query: str,
    query_terms: set[str],
    rows: list[dict[str, Any]],
    *,
    query_embedding: list[float] | None = None,
    vector_weight: float = 0.35,
    retrieval_terms: Any,
    cosine: Any,
    logarithm: Any,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    document_frequency: Counter[str] = Counter()
    for row in rows:
        document_frequency.update(set(retrieval_terms(row)))
    average_length = sum(max(1, len(retrieval_terms(row))) for row in rows) / max(1, len(rows))
    query_lower = query.casefold().strip()
    scored: list[dict[str, Any]] = []
    for row in rows:
        row_terms = retrieval_terms(row)
        term_counts = Counter(row_terms)
        length = max(1, len(row_terms))
        lexical_score = 0.0
        for term in query_terms:
            frequency = term_counts[term]
            if not frequency:
                continue
            frequency_in_documents = document_frequency[term]
            inverse = logarithm(
                1 + (len(rows) - frequency_in_documents + 0.5) / (frequency_in_documents + 0.5)
            )
            lexical_score += inverse * (
                frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / average_length))
            )
        if query_lower and query_lower in row["text"].casefold():
            lexical_score += 2.0
        vector_score = (
            max(0.0, cosine(query_embedding, row["_embedding"]))
            if query_embedding is not None and isinstance(row.get("_embedding"), list)
            else 0.0
        )
        enriched = dict(row)
        enriched["_lexical_score"] = lexical_score
        enriched["_vector_score"] = vector_score
        scored.append(enriched)
    max_lexical = max(
        (float(item["_lexical_score"]) for item in scored),
        default=0.0,
    )
    ranked: list[dict[str, Any]] = []
    for item in scored:
        lexical_score = float(item["_lexical_score"])
        vector_score = float(item["_vector_score"])
        if lexical_score <= 0 and vector_score < 0.08:
            continue
        if query_embedding is None:
            score = lexical_score
            retrieval_mode = "lexical"
        else:
            normalized_lexical = lexical_score / max_lexical if max_lexical > 0 else 0.0
            score = (1.0 - vector_weight) * normalized_lexical + vector_weight * vector_score
            retrieval_mode = "hybrid"
        item["score"] = round(score, 4)
        item["lexical_score"] = round(lexical_score, 4)
        item["vector_score"] = round(vector_score, 4)
        item["retrieval_mode"] = retrieval_mode
        ranked.append(item)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def retrieval_terms(row: dict[str, Any], *, content_terms: Any) -> list[str]:
    source_terms = content_terms(f"{row.get('title') or ''} {row.get('uri') or ''}")
    return [str(item) for item in row["_terms"]] + sorted(source_terms)


def public_chunk(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": item["chunk_id"],
        "document_id": item["document_id"],
        "uri": item["uri"],
        "title": item["title"],
        "source_type": item["source_type"],
        "char_start": item["char_start"],
        "char_end": item["char_end"],
        "score": item.get("score"),
        "lexical_score": item.get("lexical_score"),
        "vector_score": item.get("vector_score"),
        "retrieval_mode": item.get("retrieval_mode", "lexical"),
        "excerpt": item["text"][:1_200],
        "updated_at": item["updated_at"],
    }


def join_reason(bridges: list[str], complementary: int) -> str:
    parts: list[str] = []
    if bridges:
        parts.append("shared bridge terms: " + ", ".join(bridges))
    if complementary:
        parts.append(f"the chunks cover {complementary} complementary question term(s)")
    return "; ".join(parts) or "both chunks independently match the question"

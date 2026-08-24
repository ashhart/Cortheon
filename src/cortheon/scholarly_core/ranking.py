"""Deduplication, relevance, recency, and combined ranking."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from cortheon.models import ScholarlyWork


def dedupe_works(works: list[ScholarlyWork], *, normalize_title: Any) -> list[ScholarlyWork]:
    seen: set[str] = set()
    result: list[ScholarlyWork] = []
    for work in works:
        key = (
            work.identifiers.get("doi")
            or work.identifiers.get("arxiv")
            or work.url
            or normalize_title(work.title)
        )
        normalized = key.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(work)
    return result


def score_work_relevance(
    work: ScholarlyWork,
    query: str,
    *,
    query_terms: Any,
    normalize: Any,
) -> ScholarlyWork:
    terms = query_terms(query)
    if not terms:
        work.relevance_score = 0.0
        return work
    title = normalize(work.title)
    abstract = normalize(work.abstract or "")
    venue = normalize(work.venue or "")
    haystack = f"{title} {abstract} {venue}"
    title_hits = sum(1 for term in terms if term in title)
    body_hits = sum(1 for term in terms if term in haystack)
    phrase = normalize(query)
    phrase_boost = 0.2 if phrase and phrase in haystack else 0.0
    title_boost = min(0.25, title_hits / len(terms) * 0.35)
    overlap = body_hits / len(terms)
    work.relevance_score = round(min(1.0, overlap * 0.72 + title_boost + phrase_boost), 3)
    return work


def work_recency_score(
    published_at: datetime | None,
    now: datetime | None,
    *,
    current_time: Any,
    recency_steps: tuple[tuple[int, float], ...],
    recency_floor: float,
    undated_recency: float,
    utc: Any,
) -> float:
    if published_at is None:
        return undated_recency
    current = now or current_time()
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=utc)
    days = max((current - published_at.astimezone(utc)).days, 0)
    for horizon, score in recency_steps:
        if days <= horizon:
            return score
    return recency_floor


def scholarly_rank_key(work: ScholarlyWork) -> float:
    return round(
        work.relevance_score * 0.55 + work.authority_score * 0.25 + work.recency_score * 0.2,
        6,
    )


def minimum_relevance(query: str, *, query_terms: Any) -> float:
    term_count = len(query_terms(query))
    if term_count <= 2:
        return 0.25
    if term_count <= 5:
        return 0.32
    return 0.38

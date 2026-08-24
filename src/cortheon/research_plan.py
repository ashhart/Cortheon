from __future__ import annotations

import re

from cortheon.models import ResearchQuery

BIOMEDICAL_TERMS = {
    "biomarker",
    "biology",
    "cancer",
    "clinical",
    "cure",
    "disease",
    "drug",
    "genome",
    "medical",
    "medicine",
    "protein",
    "therapy",
    "trial",
}

ALIFE_TERMS = {
    "alife",
    "alife-style",
    "artificial",
    "evolution",
    "evolutionary",
    "life",
    "novelty",
    "open-ended",
    "openended",
}

SOFTWARE_TERMS = {
    "api",
    "code",
    "compiler",
    "framework",
    "library",
    "package",
    "repo",
    "sdk",
}


def plan_research_queries(topic: str, max_follow_up_queries: int = 2) -> list[ResearchQuery]:
    cleaned = normalize_query(topic)
    if not cleaned:
        return []
    mission_queries = source_friendly_queries(cleaned)
    primary = mission_queries[0] if mission_queries else cleaned
    queries = [
        ResearchQuery(
            query=primary,
            purpose="primary mission query",
            source="user_topic",
        )
    ]
    if max_follow_up_queries <= 0:
        return queries

    candidates = [*mission_queries[1:], *follow_up_candidates(cleaned)]
    seen = {primary.lower()}
    for query in candidates:
        normalized = normalize_query(query)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        queries.append(
            ResearchQuery(
                query=normalized,
                purpose=purpose_for_query(cleaned, normalized),
                source="mission_planner",
            )
        )
        if len(queries) >= max_follow_up_queries + 1:
            break
    return queries


def plan_gap_follow_up_queries(
    topic: str,
    evidence_gaps: list[str],
    existing_queries: list[ResearchQuery],
    max_adaptive_queries: int = 1,
) -> list[ResearchQuery]:
    if max_adaptive_queries <= 0:
        return []
    seen = {item.query.lower() for item in existing_queries}
    queries: list[ResearchQuery] = []
    for gap in evidence_gaps:
        for candidate in gap_follow_up_candidates(topic, gap):
            normalized = normalize_query(candidate)
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            queries.append(
                ResearchQuery(
                    query=normalized,
                    purpose=purpose_for_gap(gap),
                    source="evidence_gap",
                    target_gap=gap,
                )
            )
            if len(queries) >= max_adaptive_queries:
                return queries
    return queries


def gap_follow_up_candidates(topic: str, gap: str) -> list[str]:
    lower = gap.lower()
    if "too few extracted claims" in lower:
        return [
            f"{topic} survey review evidence",
            f"{topic} recent empirical results",
        ]
    if "too few independent sources" in lower:
        return [
            f"{topic} independent replication comparison",
            f"{topic} multiple studies evidence",
        ]
    if "benchmark" in lower or "evaluation" in lower:
        return [
            f"{topic} benchmark evaluation metrics dataset",
            f"{topic} leaderboard benchmark results",
        ]
    if "under-covered" in lower:
        missing = missing_terms_from_gap(gap)
        if missing:
            return [
                f"{topic} {' '.join(missing[:4])} evidence",
                f"{topic} {' '.join(missing[:4])} benchmark",
            ]
    if "claim clusters" in lower:
        return [
            f"{topic} systematic review state of the art",
            f"{topic} taxonomy survey",
        ]
    return [f"{topic} limitations failure modes"]


def purpose_for_gap(gap: str) -> str:
    lower = gap.lower()
    if "benchmark" in lower or "evaluation" in lower:
        return "close synthesis gap: find benchmark or evaluation evidence"
    if "too few independent" in lower:
        return "close synthesis gap: find independent sources"
    if "too few extracted" in lower:
        return "close synthesis gap: find more extractable claims"
    if "under-covered" in lower:
        return "close synthesis gap: cover missing topic terms"
    if "claim clusters" in lower:
        return "close synthesis gap: find organizing survey or taxonomy"
    return "close synthesis gap"


def missing_terms_from_gap(gap: str) -> list[str]:
    if ":" not in gap:
        return []
    tail = gap.split(":", 1)[1].strip().rstrip(".")
    return [term.strip() for term in tail.split(",") if term.strip()]


def follow_up_candidates(topic: str) -> list[str]:
    terms = query_terms(topic)
    candidates: list[str] = []

    compact = compact_query(topic)
    candidates.append(f"{compact} benchmark dataset evaluation")
    candidates.append(f"{compact} implementation source code")

    if terms & BIOMEDICAL_TERMS:
        candidates.extend(
            [
                f"{compact} systematic review mechanism",
                f"{compact} clinical trial evidence",
                f"{compact} dataset biomarker target validation",
            ]
        )
    elif terms & ALIFE_TERMS:
        candidates.extend(
            [
                f"{compact} quality diversity novelty search",
                f"{compact} artificial life survey open-ended evolution",
            ]
        )
    elif terms & SOFTWARE_TERMS:
        candidates.extend(
            [
                f"{compact} official documentation examples",
                f"{compact} migration guide changelog",
            ]
        )
    else:
        candidates.append(f"{compact} survey review state of the art")

    candidates.append(f"{compact} limitations failure modes")
    candidates.append(f"{compact} leaderboard state of the art")
    return candidates


def source_friendly_queries(topic: str) -> list[str]:
    terms = query_terms(topic)
    queries: list[str] = []
    architecture_mission = bool(
        {"architecture", "architectural", "build", "implementation"} & terms
    )

    if architecture_mission and terms & ALIFE_TERMS:
        queries.extend(
            [
                "open-ended evolution artificial life architecture benchmark",
                "quality diversity novelty search artificial life benchmark",
            ]
        )

    if {"senolytic", "senolytics", "senescence"} & terms:
        queries.extend(
            [
                "senolytics cellular senescence clinical trial",
                "senolytic therapy cancer dasatinib quercetin",
                "senolytic drug discovery machine learning",
            ]
        )
    elif {"cure", "therapy", "therapies", "disease", "clinical"} & terms:
        queries.extend(
            [
                "therapeutic discovery clinical trial evidence",
                "drug discovery target validation systematic review",
            ]
        )

    if terms & ALIFE_TERMS and not architecture_mission:
        queries.extend(
            [
                "open-ended evolution artificial life",
                "quality diversity novelty search artificial life",
                "open-ended evolution benchmark",
            ]
        )

    if terms & SOFTWARE_TERMS and not queries:
        queries.append(compact_query(topic))

    if not queries:
        queries.append(compact_query(topic))
    return unique_queries(queries)


def compact_query(topic: str, limit: int = 8) -> str:
    terms = ordered_query_terms(topic)
    if not terms:
        return topic
    return " ".join(terms[:limit])


def unique_queries(queries: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = normalize_query(query)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output


def purpose_for_query(topic: str, query: str) -> str:
    lower = query.lower()
    if "implementation" in lower or "source code" in lower or "github" in lower:
        return "find implementable code and source artifacts"
    if "benchmark" in lower or "leaderboard" in lower or "evaluation" in lower:
        return "find benchmarks, datasets, and evaluation signals"
    if "clinical" in lower or "trial" in lower:
        return "find clinical or intervention evidence"
    if "systematic review" in lower or "survey" in lower or "state of the art" in lower:
        return "find review and state-of-the-art sources"
    if "limitations" in lower or "failure" in lower:
        return "find limitations and negative evidence"
    if topic.lower() != lower:
        return "expand mission coverage"
    return "primary mission query"


def query_terms(query: str) -> set[str]:
    return set(ordered_query_terms(query))


def ordered_query_terms(query: str) -> list[str]:
    normalized = query.lower().replace("open ended", "open-ended")
    stopwords = {
        "and",
        "can",
        "choose",
        "commit",
        "current",
        "first",
        "lab",
        "strongest",
        "tell",
        "that",
        "the",
        "want",
        "what",
    }
    return list(
        dict.fromkeys(
            term
            for term in re.findall(r"[a-z0-9][a-z0-9-]{2,}", normalized)
            if term not in stopwords
        )
    )


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()

from __future__ import annotations

import re
from collections import Counter

from cortheon.claims import classify_claim_stance, content_terms
from cortheon.models import (
    ClaimCluster,
    ContradictionGroup,
    ResearchClaim,
    ResearchSynthesis,
    utc_now,
)

CORROBORATION_MIN_SOURCES = 3


POSITIVE_CUES = {
    "demonstrate",
    "effective",
    "evidence",
    "improve",
    "introduce",
    "outperform",
    "present",
    "propose",
    "show",
    "support",
}
NEGATIVE_CUES = {
    "cannot",
    "challenge",
    "difficult",
    "fail",
    "fails",
    "lack",
    "limited",
    "limitation",
    "not",
    "open question",
    "problem",
    "unclear",
    "unknown",
}
STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "been",
    "being",
    "between",
    "coherent",
    "could",
    "demonstrate",
    "demonstrates",
    "foster",
    "fosters",
    "from",
    "have",
    "introduce",
    "introduces",
    "into",
    "method",
    "more",
    "most",
    "present",
    "presents",
    "propose",
    "proposes",
    "show",
    "shows",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "using",
    "were",
    "where",
    "which",
    "while",
    "with",
    "would",
}


def synthesize_research(
    topic: str,
    claims: list[ResearchClaim],
) -> ResearchSynthesis:
    clusters = cluster_claims(topic, claims)
    contradictions = build_contradiction_groups(clusters, claims)
    key_findings = build_key_findings(clusters)
    contested_points = build_contested_points(contradictions)
    evidence_gaps = build_evidence_gaps(topic, claims, clusters)
    confidence = synthesis_confidence(claims, clusters)
    status = synthesis_status(claims, clusters, contradictions, confidence, evidence_gaps)
    current_best_direction = best_direction(topic, clusters, status, evidence_gaps)
    return ResearchSynthesis(
        topic=topic,
        generated_at=utc_now(),
        status=status,
        confidence=confidence,
        current_best_direction=current_best_direction,
        key_findings=key_findings,
        contested_points=contested_points,
        evidence_gaps=evidence_gaps,
        clusters=clusters,
        contradictions=contradictions,
    )


def cluster_claims(topic: str, claims: list[ResearchClaim]) -> list[ClaimCluster]:
    clusters: list[list[int]] = []
    cluster_terms: list[set[str]] = []
    for index, claim in enumerate(claims):
        terms = claim_terms(topic, claim_text_for_terms(claim))
        best_index = None
        best_overlap = 0.0
        for cluster_index, terms_for_cluster in enumerate(cluster_terms):
            overlap = jaccard(terms, terms_for_cluster)
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = cluster_index
        if best_index is not None and best_overlap >= 0.24:
            clusters[best_index].append(index)
            cluster_terms[best_index] |= terms
        else:
            clusters.append([index])
            cluster_terms.append(set(terms))

    built: list[ClaimCluster] = []
    for cluster_index, indexes in enumerate(clusters, start=1):
        cluster_claims_ = [claims[index] for index in indexes]
        representative_index = max(indexes, key=lambda item: claims[item].confidence)
        representative = claims[representative_index]
        terms = top_terms([claim_text_for_terms(claims[index]) for index in indexes], topic)
        stance_counts = Counter(normalized_stance(claim) for claim in cluster_claims_)
        conflict = conflict_score(stance_counts)
        support = round(
            sum(claim.confidence for claim in cluster_claims_) / len(cluster_claims_), 3
        )
        source_urls = dedupe([claim.source_url for claim in cluster_claims_])
        independent_source_count = len(source_urls)
        corroboration = round(min(1.0, (independent_source_count - 1) / 3.0), 3)
        # Strength scales with independent corroborating sources, not raw claim
        # count: five claims from one source must not outweigh three from three.
        breadth = min(1.0, 0.4 + 0.2 * independent_source_count)
        built.append(
            ClaimCluster(
                id=f"cluster-{cluster_index}",
                label=", ".join(terms[:4]) or f"cluster {cluster_index}",
                representative_claim=representative.text,
                claim_indexes=indexes,
                source_urls=source_urls,
                average_confidence=support,
                stance_counts=dict(stance_counts),
                conflict_score=conflict,
                support_score=round(support * breadth, 3),
                terms=terms,
                independent_source_count=independent_source_count,
                corroboration=corroboration,
            )
        )
    built.sort(key=lambda cluster: (cluster.support_score, -cluster.conflict_score), reverse=True)
    return built


def build_key_findings(clusters: list[ClaimCluster]) -> list[str]:
    findings: list[str] = []
    for cluster in clusters[:6]:
        prefix = finding_prefix(cluster)
        findings.append(
            f"{prefix}: {cluster.representative_claim} "
            f"({cluster.independent_source_count} independent source(s), confidence {cluster.average_confidence:.2f})."
        )
    return findings


def finding_prefix(cluster: ClaimCluster) -> str:
    if cluster.conflict_score >= 0.35:
        return "Contested"
    if cluster.independent_source_count >= CORROBORATION_MIN_SOURCES:
        return "Corroborated"
    return "Supported"


def build_contested_points(contradictions: list[ContradictionGroup]) -> list[str]:
    return [item.summary for item in contradictions[:6]]


def build_contradiction_groups(
    clusters: list[ClaimCluster],
    claims: list[ResearchClaim],
) -> list[ContradictionGroup]:
    groups: list[ContradictionGroup] = []
    for index, cluster in enumerate(clusters, start=1):
        support_indexes: list[int] = []
        challenge_indexes: list[int] = []
        neutral_indexes: list[int] = []
        for claim_index in cluster.claim_indexes:
            stance = normalized_stance(claims[claim_index])
            if stance == "support":
                support_indexes.append(claim_index)
            elif stance == "challenge":
                challenge_indexes.append(claim_index)
            else:
                neutral_indexes.append(claim_index)
        if not support_indexes or not challenge_indexes:
            continue
        severity = contradiction_severity(
            support_indexes,
            challenge_indexes,
            claims,
            cluster.conflict_score,
        )
        groups.append(
            ContradictionGroup(
                id=f"contradiction-{index}",
                axis=cluster.label,
                support_claim_indexes=support_indexes,
                challenge_claim_indexes=challenge_indexes,
                neutral_claim_indexes=neutral_indexes,
                support_sources=dedupe([claims[item].source_url for item in support_indexes]),
                challenge_sources=dedupe([claims[item].source_url for item in challenge_indexes]),
                summary=(
                    f"{cluster.label}: {len(support_indexes)} supporting claim(s) and "
                    f"{len(challenge_indexes)} challenging claim(s) across "
                    f"{len(set(cluster.source_urls))} source(s)."
                ),
                severity=severity,
            )
        )
    groups.sort(key=lambda item: item.severity, reverse=True)
    return groups


def build_evidence_gaps(
    topic: str,
    claims: list[ResearchClaim],
    clusters: list[ClaimCluster],
) -> list[str]:
    gaps: list[str] = []
    if len(claims) < 5:
        gaps.append("Too few extracted claims for a strong synthesis.")
    if len({claim.source_url for claim in claims}) < 3:
        gaps.append("Too few independent sources.")
    claim_and_title_text = " ".join(
        f"{claim.text} {claim.source_title or ''}".lower() for claim in claims
    )
    if not any(
        cue in claim_and_title_text
        for cue in ("benchmark", "evaluate", "evaluation", "empirical", "metric", "test ")
    ):
        gaps.append("No clear benchmark or evaluation claim was extracted.")
    topic_terms = content_terms(topic)
    missing_terms = sorted(
        term for term in topic_terms if not any(term in claim.text.lower() for claim in claims)
    )
    if missing_terms:
        gaps.append(
            f"Topic terms under-covered in extracted claims: {', '.join(missing_terms[:6])}."
        )
    if not clusters:
        gaps.append("No claim clusters could be formed.")
    return gaps[:6]


def synthesis_confidence(claims: list[ResearchClaim], clusters: list[ClaimCluster]) -> float:
    if not claims or not clusters:
        return 0.0
    source_count = len({claim.source_url for claim in claims})
    avg_claim = sum(claim.confidence for claim in claims) / len(claims)
    cluster_strength = sum(cluster.support_score for cluster in clusters[:3]) / min(
        len(clusters), 3
    )
    source_factor = min(1.0, source_count / 6)
    return round(min(0.98, avg_claim * 0.45 + cluster_strength * 0.35 + source_factor * 0.2), 3)


def synthesis_status(
    claims: list[ResearchClaim],
    clusters: list[ClaimCluster],
    contradictions: list[ContradictionGroup],
    confidence: float,
    evidence_gaps: list[str],
) -> str:
    if contradictions and contradictions[0].severity >= 0.42:
        return "contested"
    if len(claims) < 3 or confidence < 0.45:
        return "insufficient_evidence"
    if any("benchmark" in gap.lower() or "too few" in gap.lower() for gap in evidence_gaps):
        return "promising_but_incomplete"
    corroborated = max((cluster.independent_source_count for cluster in clusters), default=0)
    if (
        confidence >= 0.72
        and len({claim.source_url for claim in claims}) >= 4
        and corroborated >= CORROBORATION_MIN_SOURCES
    ):
        return "emerging_consensus"
    return "promising_but_incomplete"


def best_direction(
    topic: str,
    clusters: list[ClaimCluster],
    status: str,
    evidence_gaps: list[str],
) -> str:
    if not clusters:
        return f"Do not choose a build direction for {topic!r} yet; discovery produced no clustered claims."
    if senolytic_cure_engine_topic(topic):
        return senolytic_cure_engine_direction(topic, status, evidence_gaps)
    top = select_direction_cluster(clusters)
    if status == "insufficient_evidence":
        return (
            f"Treat {topic!r} as a research-planning task first. "
            f"The strongest weak signal is: {top.representative_claim}"
        )
    if status == "contested":
        return (
            f"For {topic!r}, build around the contested axis '{top.label}' and design falsifiers before committing. "
            f"Representative claim: {top.representative_claim}"
        )
    if evidence_gaps:
        return (
            f"Use '{top.label}' as the first design axis for {topic!r}, but treat it as a benchmark-building "
            f"hypothesis until the gaps are closed. Representative claim: {top.representative_claim}"
        )
    return (
        f"Use '{top.label}' as the first design axis for {topic!r}, then verify it with an executable benchmark. "
        f"Representative claim: {top.representative_claim}"
    )


def senolytic_cure_engine_topic(topic: str) -> bool:
    lower = topic.lower()
    return ("senolytic" in lower or "senescence" in lower) and (
        "cure" in lower or "therap" in lower or "alife" in lower
    )


def senolytic_cure_engine_direction(
    topic: str,
    status: str,
    evidence_gaps: list[str],
) -> str:
    base = (
        "Build an evidence-first discovery substrate before committing to a cure-engine architecture: "
        "ingest PubMed and ClinicalTrials.gov senolytic evidence, map compounds, mechanisms, assays, "
        "targets, trial status, and endpoints, then benchmark ML candidate discovery against known "
        "senolytics such as dasatinib, quercetin, fisetin, and navitoclax."
    )
    if status == "insufficient_evidence":
        return (
            f"Treat {topic!r} as a research acquisition task first. "
            "Do not choose a lab build direction until biomedical literature and trial evidence are collected."
        )
    if any("under-covered" in gap.lower() for gap in evidence_gaps):
        return (
            f"{base} Do not claim the ALIFE/architecture layer is selected yet; the current evidence "
            "supports senolytic discovery and validation surfaces more strongly than a specific ALIFE architecture."
        )
    return base


def select_direction_cluster(clusters: list[ClaimCluster]) -> ClaimCluster:
    def score(cluster: ClaimCluster) -> float:
        text = f"{cluster.label} {cluster.representative_claim}".lower()
        action_bonus = 0.0
        for cue in (
            "benchmark",
            "evaluate",
            "empirical",
            "method",
            "quality",
            "diversity",
            "metric",
            "design",
        ):
            if cue in text:
                action_bonus += 0.08
        history_penalty = 0.18 if "workshop" in text or "editorial" in text else 0.0
        return (
            cluster.support_score + action_bonus - history_penalty - cluster.conflict_score * 0.25
        )

    return max(clusters, key=score)


def classify_stance(text: str) -> str:
    return classify_claim_stance(text)


def normalized_stance(claim: ResearchClaim) -> str:
    if claim.stance in {"support", "challenge", "neutral"}:
        return claim.stance
    return classify_claim_stance(claim.text)


def claim_terms(topic: str, text: str) -> set[str]:
    topic_words = content_terms(topic)
    words = {
        word
        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower())
        if word not in STOPWORDS
    }
    return words | {word for word in words if word in topic_words}


def claim_text_for_terms(claim: ResearchClaim) -> str:
    return f"{claim.source_title or ''} {claim.text}"


def top_terms(texts: list[str], topic: str) -> list[str]:
    topic_words = content_terms(topic)
    counts: Counter[str] = Counter()
    for text in texts:
        for word in claim_terms(topic, text):
            weight = 3 if word in topic_words else 1
            counts[word] += weight
    return [word for word, _ in counts.most_common(8)]


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def conflict_score(stance_counts: Counter[str]) -> float:
    support = stance_counts.get("support", 0)
    challenge = stance_counts.get("challenge", 0)
    total = sum(stance_counts.values()) or 1
    if support and challenge:
        return round(min(1.0, (support + challenge) / total * 0.65), 3)
    if challenge >= 2:
        return round(min(0.45, challenge / total * 0.45), 3)
    return 0.0


def contradiction_severity(
    support_indexes: list[int],
    challenge_indexes: list[int],
    claims: list[ResearchClaim],
    base_conflict: float,
) -> float:
    support_conf = sum(claims[index].confidence for index in support_indexes) / len(support_indexes)
    challenge_conf = sum(claims[index].confidence for index in challenge_indexes) / len(
        challenge_indexes
    )
    balance = min(len(support_indexes), len(challenge_indexes)) / max(
        len(support_indexes), len(challenge_indexes)
    )
    return round(
        min(1.0, base_conflict * 0.4 + ((support_conf + challenge_conf) / 2) * 0.4 + balance * 0.2),
        3,
    )


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result

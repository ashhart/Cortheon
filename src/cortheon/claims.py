from __future__ import annotations

import re
from dataclasses import dataclass

from cortheon.models import CrawledPage, ResearchClaim, ScholarlyWork, SupportLevel, utc_now
from cortheon.sanitize import scan_text

CLAIM_CUES = {
    "argue",
    "benchmark",
    "demonstrate",
    "describe",
    "design",
    "develop",
    "evaluate",
    "find",
    "introduce",
    "outperform",
    "present",
    "propose",
    "report",
    "show",
}
POSITIVE_STANCE_CUES = {
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
CHALLENGE_STANCE_CUES = {
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


@dataclass(slots=True)
class SentenceMatch:
    text: str
    start: int
    end: int
    source_text: str


def extract_claims(
    topic: str,
    works: list[ScholarlyWork],
    pages: list[CrawledPage],
    limit: int = 30,
) -> list[ResearchClaim]:
    topic_terms = content_terms(topic)
    claims: list[ResearchClaim] = []
    for work in works:
        # Fetched content is data, never instructions: quarantine injection-like
        # segments before any sentence can become a claim, and dent the trust of
        # sources that carried them.
        scan = scan_text(work.abstract or work.title)
        match = best_sentence_match(scan.clean_text, topic_terms)
        if match:
            authority = work.authority_score * (0.55 if scan.flags else 1.0)
            claims.append(
                ResearchClaim(
                    text=match.text,
                    source_url=work.url,
                    source_title=work.title,
                    source_type=f"scholarly:{work.source}",
                    support=SupportLevel.OBSERVED,
                    confidence=claim_confidence(match.text, topic_terms, authority),
                    stance=classify_claim_stance(match.text),
                    source_excerpt=excerpt_around(match.source_text, match.start, match.end),
                    source_char_start=match.start,
                    source_char_end=match.end,
                    extracted_at=utc_now(),
                )
            )
    for page in pages:
        if page.error or not page.text:
            continue
        scan = scan_text(page.text)
        match = best_sentence_match(scan.clean_text, topic_terms)
        if match:
            authority = page.authority_score * (0.55 if scan.flags else 1.0)
            claims.append(
                ResearchClaim(
                    text=match.text,
                    source_url=page.final_url,
                    source_title=page.title,
                    source_type=page.source_type,
                    support=SupportLevel.OBSERVED,
                    confidence=claim_confidence(match.text, topic_terms, authority),
                    stance=classify_claim_stance(match.text),
                    source_excerpt=excerpt_around(match.source_text, match.start, match.end),
                    source_char_start=match.start,
                    source_char_end=match.end,
                    extracted_at=utc_now(),
                )
            )
    claims.sort(key=lambda claim: claim.confidence, reverse=True)
    return dedupe_claims(claims)[:limit]


def best_sentence(text: str, topic_terms: set[str]) -> str | None:
    match = best_sentence_match(text, topic_terms)
    return match.text if match else None


def best_sentence_match(text: str, topic_terms: set[str]) -> SentenceMatch | None:
    candidates = [
        match for match in split_sentence_matches(text) if 60 <= len(match.text.strip()) <= 420
    ]
    if not candidates:
        return None
    scored = sorted(
        ((sentence_score(match.text, topic_terms), match) for match in candidates),
        reverse=True,
        key=lambda item: item[0],
    )
    score, match = scored[0]
    return match if score > 0 else None


def split_sentences(text: str) -> list[str]:
    return [match.text for match in split_sentence_matches(text)]


def split_sentence_matches(text: str) -> list[SentenceMatch]:
    cleaned = normalize_source_text(text)
    if not cleaned:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    matches: list[SentenceMatch] = []
    cursor = 0
    for piece in pieces:
        sentence = piece.strip()
        if not sentence:
            continue
        start = cleaned.find(sentence, cursor)
        if start < 0:
            start = cleaned.find(sentence)
        if start < 0:
            continue
        end = start + len(sentence)
        matches.append(SentenceMatch(text=sentence, start=start, end=end, source_text=cleaned))
        cursor = end
    return matches


def sentence_score(sentence: str, topic_terms: set[str]) -> float:
    lower = sentence.lower()
    overlap = sum(1 for term in topic_terms if term in lower)
    cue_hits = sum(1 for cue in CLAIM_CUES if re.search(rf"\b{cue}\w*\b", lower))
    return overlap * 1.0 + cue_hits * 1.5


def claim_confidence(sentence: str, topic_terms: set[str], authority_score: float) -> float:
    raw = sentence_score(sentence, topic_terms)
    return round(min(0.98, 0.25 + authority_score * 0.45 + min(raw, 8) * 0.04), 3)


def classify_claim_stance(text: str) -> str:
    lower = text.lower()
    if any(cue in lower for cue in CHALLENGE_STANCE_CUES):
        return "challenge"
    if any(cue in lower for cue in POSITIVE_STANCE_CUES):
        return "support"
    return "neutral"


def excerpt_around(source_text: str, start: int, end: int, context_chars: int = 180) -> str:
    left = max(0, start - context_chars)
    right = min(len(source_text), end + context_chars)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(source_text) else ""
    return f"{prefix}{source_text[left:right].strip()}{suffix}"


def normalize_source_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def content_terms(topic: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", topic.lower())
        if word not in {"with", "from", "into", "that", "this", "what", "best", "engine"}
    }


def dedupe_claims(claims: list[ResearchClaim]) -> list[ResearchClaim]:
    seen: set[str] = set()
    result: list[ResearchClaim] = []
    for claim in claims:
        key = re.sub(r"[^a-z0-9]+", " ", claim.text.lower()).strip()[:180]
        if key in seen:
            continue
        seen.add(key)
        result.append(claim)
    return result

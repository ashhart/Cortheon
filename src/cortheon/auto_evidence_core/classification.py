"""Evidence classification and prompt-target extraction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def unique_tags(tags: list[str]) -> list[str]:
    cleaned = [tag.strip().lower() for tag in tags if tag and tag.strip()]
    return list(dict.fromkeys(cleaned))


def package_evidence_tags(
    report: Any,
    proposed_action: str | None,
    *,
    named_technology_candidates: Callable[[str], set[str]],
    normalize_name: Callable[[str], str],
) -> list[str]:
    if not report.winner or not report.candidates:
        return []
    proposed = named_technology_candidates(proposed_action or "")
    if proposed and normalize_name(report.winner) not in proposed:
        return []
    tags = ["recommendation_report"]
    winner = report.candidates[0]
    if winner.metadata and not winner.errors:
        tags.append("package_verified")
    return tags


def api_evidence_tags(report: Any) -> list[str]:
    if not report.matches:
        return []
    return ["api_evidence", "source_symbol_evidence"]


def research_evidence_tags(
    report: Any,
    *,
    technology_choice: bool,
    research_report_has_substance: Callable[[Any], bool],
    grounded_claim_count: Callable[[Any], int],
    technology_report_has_substance: Callable[[Any], bool],
    architecture_report_has_substance: Callable[[Any], bool],
    unique_tags: Callable[[list[str]], list[str]],
) -> list[str]:
    tags: list[str] = []
    if research_report_has_substance(report):
        tags.append("research_report")
    if any(item.status == "covered" for item in report.source_coverage):
        tags.append("source_coverage")
    if grounded_claim_count(report) > 0:
        tags.append("grounded_claims")
    if technology_choice and technology_report_has_substance(report):
        tags.append("technology_research_report")
    if technology_choice and architecture_report_has_substance(report):
        tags.append("architecture_research_report")
    return unique_tags(tags)


def research_agent_satisfied(missing: str, tags: list[str]) -> bool:
    if missing == "research_report":
        return "research_report" in tags
    if missing == "current_package_evidence":
        return "technology_research_report" in tags or "research_report" in tags
    if missing == "architecture_evidence":
        return "architecture_research_report" in tags or "architecture_evidence" in tags
    return bool(tags)


def research_report_has_substance(report: Any) -> bool:
    if report.synthesis is None or report.synthesis.status == "insufficient_evidence":
        return False
    return bool(
        report.claims or report.scholarly_works or report.artifacts or report.search_results
    )


def technology_report_has_substance(report: Any) -> bool:
    if report.synthesis is None or report.synthesis.status == "insufficient_evidence":
        return False
    return bool(
        report.claims or report.search_results or report.artifact_assessments or report.artifacts
    )


def architecture_report_has_substance(report: Any) -> bool:
    if report.synthesis is None or report.synthesis.status == "insufficient_evidence":
        return False
    combined = " ".join(
        [
            report.synthesis.current_best_direction,
            *report.synthesis.key_findings,
            *[claim.text for claim in report.claims],
            *[artifact.title or "" for artifact in report.artifacts],
        ]
    ).lower()
    if not any(
        term in combined for term in {"architecture", "benchmark", "implementation", "evaluation"}
    ):
        return False
    blocking_gaps = [
        gap.lower() for gap in report.synthesis.evidence_gaps if "under-covered" in gap.lower()
    ]
    return not any(
        any(term in gap for term in {"architecture", "alife", "build"}) for gap in blocking_gaps
    )


def grounded_claim_count(report: Any) -> int:
    return sum(
        1
        for claim in report.claims
        if claim.source_excerpt
        and claim.source_char_start is not None
        and claim.source_char_end is not None
    )


def extract_api_target(
    text: str,
    *,
    findall: Callable[[str, str], list[str]],
) -> tuple[str, str] | None:
    matches = findall(r"\b([a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*){1,3})\b", text)
    for match in matches:
        parts = match.split(".")
        if len(parts) < 2:
            continue
        package = parts[0]
        query = ".".join(parts[1:])
        if package.lower() in {"self", "this", "client"}:
            continue
        return package, query
    return None


def named_technology_candidates(
    text: str,
    *,
    extract_api_target: Callable[[str], tuple[str, str] | None],
    normalize_name: Callable[[str], str],
    findall: Callable[[str, str], list[str]],
) -> set[str]:
    candidates: set[str] = set()
    for package, _query in [target for target in [extract_api_target(text)] if target]:
        candidates.add(normalize_name(package))
    patterns = (
        r"\b(?:use|install|select|choose|pick|recommend|commit to)\s+(?:the\s+|a\s+|an\s+)?([A-Za-z][A-Za-z0-9_-]{2,})",
        r"\bcalled\s+([A-Za-z][A-Za-z0-9_-]{2,})",
        r"\b([A-Z][A-Za-z0-9_-]{2,})(?:\s+package|\s+framework|\s+database|\s+db)\b",
    )
    stopwords = {
        "current",
        "best",
        "new",
        "production",
        "strongest",
        "first",
        "basic",
        "project",
        "package",
        "framework",
        "database",
    }
    for pattern in patterns:
        for match in findall(pattern, text):
            normalized = normalize_name(match)
            if normalized and normalized not in stopwords:
                candidates.add(normalized)
    return candidates


def normalize_name(
    value: str,
    *,
    substitute: Callable[[str, str, str], str],
) -> str:
    return substitute(r"[^a-z0-9]+", "", value.lower())


def research_topic(task: str, proposed_action: str | None, context: str | None) -> str:
    return " ".join(part for part in [task, proposed_action or "", context or ""] if part).strip()

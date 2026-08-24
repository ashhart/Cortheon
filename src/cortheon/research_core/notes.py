from __future__ import annotations

from collections.abc import Sequence

from cortheon.models import (
    ResearchArtifact,
    ResearchCoverageItem,
    ResearchDiscoveryPass,
    ResearchQuery,
    ScholarlyWork,
    SearchResult,
)
from cortheon.research_core._compat import facade


def source_mix(pages: Sequence[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for page in pages:
        source_type = getattr(page, "source_type", "unknown")
        counts[source_type] = counts.get(source_type, 0) + 1
    return counts


def artifact_mix(artifacts: list[ResearchArtifact]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        counts[artifact.kind] = counts.get(artifact.kind, 0) + 1
    return counts


def artifact_notes(artifacts: list[ResearchArtifact]) -> list[str]:
    if not artifacts:
        return [
            "No reusable code, dataset, benchmark, DOI, or paper-source artifacts were identified."
        ]
    mix = ", ".join(
        f"{kind}={count}" for kind, count in sorted(facade().artifact_mix(artifacts).items())
    )
    return [f"Artifacts are extracted from discovered sources and ranked by confidence: {mix}."]


def coverage_notes(source_coverage: list[ResearchCoverageItem]) -> list[str]:
    missing = [item.name for item in source_coverage if item.status == "missing"]
    if not missing:
        return []
    return [f"Source coverage gaps remain: {', '.join(missing)}."]


def mission_plan_notes(
    mission_queries: list[ResearchQuery],
    discovery_passes: list[ResearchDiscoveryPass],
) -> list[str]:
    notes: list[str] = []
    if len(mission_queries) > 1:
        notes.append(f"Mission planner ran {len(mission_queries)} bounded discovery query passes.")
    adaptive_count = sum(1 for item in mission_queries if item.source == "evidence_gap")
    if adaptive_count:
        notes.append(
            f"Adaptive planner added {adaptive_count} gap-driven discovery query pass(es)."
        )
    zero_result_passes = [
        item.query
        for item in discovery_passes
        if item.scholarly_work_count == 0
        and item.search_result_count == 0
        and item.github_artifact_count == 0
    ]
    if zero_result_passes:
        notes.append(
            f"{len(zero_result_passes)} planned query pass(es) returned no direct sources or artifacts."
        )
    return notes


def research_notes(
    search_results: list[SearchResult],
    scholarly_works: list[ScholarlyWork],
    pages: Sequence[object],
) -> list[str]:
    notes: list[str] = []
    if not search_results:
        notes.append(
            "Generic web search was unavailable or returned no results; discovery used scholarly, GitHub, and seed sources."
        )
    if not pages:
        notes.append("No pages were crawled successfully.")
    if pages:
        notes.append("Pages are ranked by authority heuristic, not by final truth.")
    if scholarly_works:
        notes.append("Scholarly works are ranked by relevance plus authority, not authority alone.")
    if not scholarly_works and not any(
        getattr(page, "source_type", "") in {"paper", "official_health_authority"} for page in pages
    ):
        notes.append(
            "No scholarly paper or official authority source was found in the current discovery pass."
        )
    return notes

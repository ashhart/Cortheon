"""Stable audit-detail and summary rendering for acquired evidence."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def api_source_details(report: Any) -> list[dict[str, Any]]:
    sources = [
        {
            "title": f"PyPI metadata: {report.package} {report.version}",
            "url": f"https://pypi.org/pypi/{report.package}/json",
            "source_type": "pypi_package_metadata",
            "authority_score": 0.74,
            "summary": f"Package metadata used before API symbol lookup for {report.package}:{report.query}.",
        }
    ]
    if report.artifact_url:
        match_summary = (
            f"Source artifact lookup found {len(report.matches)} match(es) for {report.query}."
            if report.matches
            else f"Source artifact lookup found no matches for {report.query}."
        )
        sources.append(
            {
                "title": f"Source artifact: {report.artifact_filename or report.package}",
                "url": report.artifact_url,
                "source_type": "source_artifact_ast",
                "authority_score": 0.9,
                "summary": match_summary,
            }
        )
    return sources


def recommendation_details(
    report: Any,
    *,
    package_source_details: Callable[[Any], list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "winner": report.winner,
        "profile": report.profile,
        "candidate_count": len(report.candidates),
        "candidates": [candidate.package for candidate in report.candidates[:10]],
        "sources": package_source_details(report),
        "notes": report.notes,
    }


def package_source_details(report: Any) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for index, candidate in enumerate(report.candidates[:4]):
        score = candidate.score.overall if candidate.score else None
        if candidate.metadata:
            sources.append(
                {
                    "title": f"PyPI metadata: {candidate.package} {candidate.version or 'unknown'}",
                    "url": candidate.metadata.source_url,
                    "source_type": "pypi_package_metadata",
                    "authority_score": score or 0.72,
                    "summary": candidate.metadata.summary,
                }
            )
        if index != 0:
            continue
        if candidate.documentation and candidate.documentation.docs_url:
            sources.append(
                {
                    "title": f"Documentation: {candidate.package}",
                    "url": candidate.documentation.docs_url,
                    "source_type": "official_docs",
                    "authority_score": score or 0.7,
                    "summary": "Documentation link discovered from package project metadata.",
                }
            )
        if candidate.github:
            sources.append(
                {
                    "title": f"GitHub repository: {candidate.github.repo}",
                    "url": candidate.github.html_url,
                    "source_type": "github_repository",
                    "authority_score": score or 0.65,
                    "summary": candidate.github.description,
                }
            )
        if candidate.vulnerabilities:
            sources.append(
                {
                    "title": f"OSV vulnerabilities: {candidate.package}",
                    "url": candidate.vulnerabilities.source_url,
                    "source_type": "osv_vulnerability_report",
                    "authority_score": 0.7,
                    "summary": f"OSV returned {candidate.vulnerabilities.count} known vulnerability record(s) for the selected version.",
                }
            )
    return sources


def research_details(
    report: Any,
    *,
    grounded_claim_count: Callable[[Any], int],
) -> dict[str, Any]:
    return {
        "topic": report.topic,
        "search_provider": report.search_provider,
        "source_plan": [
            {
                "name": item.name,
                "selected": item.selected,
                "source_type": item.source_type,
                "planner": item.planner,
                "budget": item.budget,
            }
            for item in report.source_plan
        ],
        "search_results": len(report.search_results),
        "scholarly_works": len(report.scholarly_works),
        "artifacts": len(report.artifacts),
        "claims": len(report.claims),
        "grounded_claims": grounded_claim_count(report),
        "synthesis_status": report.synthesis.status if report.synthesis else None,
        "covered_sources": [
            item.name for item in report.source_coverage if item.status == "covered"
        ],
    }


def research_summary(report: Any, tags: list[str]) -> str:
    status = report.synthesis.status if report.synthesis else "missing"
    if tags:
        return (
            f"Live research produced {len(report.claims)} claim(s), {len(report.artifacts)} artifact(s), "
            f"and synthesis status {status}."
        )
    return (
        f"Live research ran but did not meet the evidence threshold "
        f"(claims={len(report.claims)}, artifacts={len(report.artifacts)}, status={status})."
    )


def auto_notes(initial: Any, final: Any, runs: list[Any]) -> list[str]:
    if initial.verdict != "needs_evidence":
        return ["Auto-evidence did not run because the initial decision did not need evidence."]
    if final.verdict == "allow":
        return ["Auto-evidence satisfied the decision gate."]
    if any(run.status == "manual_required" for run in runs):
        return [
            "Auto-evidence gathered what it could, but at least one missing item requires manual/local work."
        ]
    if any(run.status == "failed" for run in runs):
        return [
            "Auto-evidence encountered provider or connector failures; inspect agent_runs.errors."
        ]
    return ["Auto-evidence ran, but the decision still needs more evidence."]

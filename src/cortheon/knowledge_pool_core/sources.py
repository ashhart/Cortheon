"""Source selection, de-duplication, and compact rendering."""

from __future__ import annotations

from types import ModuleType
from typing import Any


def top_sources(bindings: ModuleType, report: Any, limit: int) -> list[Any]:
    sources: list[Any] = []
    claim_map = bindings.claims_by_source(report)
    for page in report.crawled_pages:
        url = page.final_url or page.url
        sources.append(
            bindings.PooledSource(
                title=page.title,
                url=url,
                source_type=page.source_type,
                authority_score=page.authority_score,
                relevance_score=None,
                summary=bindings.compact_text(bindings.scan_text(page.text).clean_text),
                derived_claims=claim_map.get(url, [])[:3],
            )
        )
    sources.extend(
        bindings.PooledSource(
            title=work.title,
            url=work.url,
            source_type=f"scholarly:{work.source}",
            authority_score=work.authority_score,
            relevance_score=work.relevance_score,
            summary=bindings.compact_text(bindings.scan_text(work.abstract or "").clean_text),
            derived_claims=claim_map.get(work.url, [])[:3],
        )
        for work in report.scholarly_works
    )
    sources.extend(
        bindings.PooledSource(
            title=artifact.title,
            url=artifact.url,
            source_type=artifact.kind,
            authority_score=artifact.confidence,
            relevance_score=None,
            summary=bindings.compact_text(artifact.evidence or ""),
            derived_claims=claim_map.get(artifact.url, [])[:3],
        )
        for artifact in report.artifacts
    )
    sources.extend(
        bindings.PooledSource(
            title=result.title,
            url=result.url,
            source_type=f"search:{result.provider}",
            authority_score=1.0 / max(result.rank, 1),
            relevance_score=None,
            summary=bindings.compact_text(result.snippet or ""),
            derived_claims=claim_map.get(result.url, [])[:3],
        )
        for result in report.search_results
    )
    return bindings.dedupe_sources(sources)[:limit]


def agent_pooled_sources(bindings: ModuleType, decision: Any) -> list[Any]:
    sources: list[Any] = []
    for agent_run in decision.agent_runs:
        for item in agent_run.details.get("sources", []):
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not isinstance(url, str) or not url:
                continue
            sources.append(
                bindings.PooledSource(
                    title=bindings.string_or_none(item.get("title")),
                    url=url,
                    source_type=(
                        bindings.string_or_none(item.get("source_type")) or agent_run.agent
                    ),
                    authority_score=bindings.float_or_none(item.get("authority_score")),
                    relevance_score=bindings.float_or_none(item.get("relevance_score")),
                    summary=bindings.compact_text(
                        bindings.string_or_none(item.get("summary")) or ""
                    ),
                    derived_claims=[],
                )
            )
    return sources


def agent_winner(decision: Any) -> str | None:
    for agent_run in decision.agent_runs:
        winner = agent_run.details.get("winner")
        if isinstance(winner, str) and winner.strip():
            return winner.strip()
    return None


def dedupe_sources(sources: list[Any]) -> list[Any]:
    ranked = sorted(
        sources,
        key=lambda item: (
            item.authority_score or 0.0,
            item.relevance_score or 0.0,
            len(item.derived_claims),
        ),
        reverse=True,
    )
    seen: set[str] = set()
    output: list[Any] = []
    for source in ranked:
        if source.url in seen:
            continue
        seen.add(source.url)
        output.append(source)
    return output


def claims_by_source(report: Any) -> dict[str, list[str]]:
    claims: dict[str, list[str]] = {}
    for claim in report.claims:
        claims.setdefault(claim.source_url, []).append(claim.text)
    return claims


def string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def float_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def source_plan_summary(bindings: ModuleType, report: Any) -> list[Any]:
    coverage_by_source = {
        source_name: item.observed_count
        for item in report.source_coverage
        for source_name in item.source_names
    }
    return [
        bindings.SourcePlanSummary(
            name=item.name,
            source_type=item.source_type,
            selected=item.selected,
            reason=item.reason,
            trust_tier=item.trust_tier,
            budget=item.budget,
            observed_count=coverage_by_source.get(item.name),
        )
        for item in report.source_plan
    ]


def discovery_counts(report: Any) -> dict[str, int]:
    return {
        "search_results": len(report.search_results),
        "scholarly_works": len(report.scholarly_works),
        "crawled_pages": len(report.crawled_pages),
        "artifacts": len(report.artifacts),
        "claims": len(report.claims),
        "covered_sources": sum(1 for item in report.source_coverage if item.status == "covered"),
    }


def compact_text(text: str, limit: int) -> str | None:
    cleaned = " ".join(text.split())
    if not cleaned:
        return None
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def unique_notes(items: list[str]) -> list[str]:
    cleaned = [item.strip() for item in items if item and item.strip()]
    return list(dict.fromkeys(cleaned))

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from cortheon.models import Evidence, ResearchSourceDecision, SupportLevel
from cortheon.source_planner_core._compat import facade
from cortheon.source_planner_core.types import SourceProfile


def build_research_source_profiles(
    scholarly_profiles: Iterable[dict[str, Any]],
    trial_registry_profiles: Iterable[dict[str, Any]] | None = None,
    *,
    search_provider_name: str | None,
    seed_url_count: int,
) -> list[SourceProfile]:
    api = facade()
    profiles = [api.source_profile_from_mapping(item) for item in scholarly_profiles]
    profiles.extend(api.source_profile_from_mapping(item) for item in trial_registry_profiles or [])
    profiles.extend(
        [
            api.SourceProfile(
                name="web_search",
                source_type="web_search",
                capabilities=(
                    "current_web",
                    "official_docs",
                    "standards",
                    "news",
                    "benchmarks",
                ),
                domains=("general", "software", "science", "policy", "medicine"),
                trust_tier="mixed_current_web",
                default_priority=0.58,
                available=bool(search_provider_name and search_provider_name != "none"),
            ),
            api.SourceProfile(
                name="github_repositories",
                source_type="code_search",
                capabilities=(
                    "code_artifacts",
                    "reference_implementations",
                    "repository_health",
                    "implementation_signals",
                ),
                domains=("software", "ai", "science", "benchmarks", "engineering"),
                trust_tier="implementation_artifact",
                default_priority=0.66,
                available=True,
            ),
            api.SourceProfile(
                name="seed_urls",
                source_type="crawl_seed",
                capabilities=("user_supplied_sources", "bounded_crawl", "known_references"),
                domains=("general", "software", "science", "medicine", "policy"),
                trust_tier="caller_supplied",
                default_priority=0.72,
                available=seed_url_count > 0,
            ),
        ]
    )
    return profiles


def source_profile_from_mapping(item: dict[str, Any]) -> SourceProfile:
    return facade().SourceProfile(
        name=str(item.get("name") or "unknown"),
        source_type=str(item.get("source_type") or "unknown"),
        capabilities=tuple(str(value) for value in item.get("capabilities") or ()),
        domains=tuple(str(value) for value in item.get("domains") or ()),
        trust_tier=str(item.get("trust_tier") or "unknown"),
        default_priority=float(item.get("default_priority", 0.5)),
        available=bool(item.get("available", True)),
    )


def selected_source_names(
    source_plan: list[ResearchSourceDecision],
    source_type: str,
) -> list[str]:
    return [item.name for item in source_plan if item.selected and item.source_type == source_type]


def is_source_selected(source_plan: list[ResearchSourceDecision], name: str) -> bool:
    return any(item.name == name and item.selected for item in source_plan)


def source_plan_evidence(topic: str, source_plan: list[ResearchSourceDecision]) -> Evidence:
    selected = [item.name for item in source_plan if item.selected]
    skipped = [item.name for item in source_plan if not item.selected]
    return Evidence(
        claim=(
            f"Research source planner selected {len(selected)} source(s) "
            f"and skipped {len(skipped)} source(s) for topic: {topic}"
        ),
        source_type="research_source_plan",
        source_url=None,
        support=SupportLevel.INFERRED,
        details={
            "topic": topic,
            "selected_sources": selected,
            "skipped_sources": skipped,
            "decisions": [
                {
                    "name": item.name,
                    "source_type": item.source_type,
                    "selected": item.selected,
                    "available": item.available,
                    "priority": item.priority,
                    "budget": item.budget,
                    "planner": item.planner,
                    "reason": item.reason,
                    "capabilities": item.capabilities,
                    "trust_tier": item.trust_tier,
                }
                for item in source_plan
            ],
        },
    )


def source_plan_notes(source_plan: list[ResearchSourceDecision]) -> list[str]:
    selected = [item.name for item in source_plan if item.selected]
    if not selected:
        return [
            "Source planner selected no live discovery sources; research depends on explicit seeds or existing inputs."
        ]
    return [f"Source planner selected: {', '.join(selected)}."]

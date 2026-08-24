from __future__ import annotations

import re
from collections.abc import Iterable

from cortheon.models import ResearchSourceDecision
from cortheon.source_planner_core._compat import facade
from cortheon.source_planner_core.types import SourcePlanningConstraints, SourceProfile


class SourcePlanner:
    """Select sources from declared capabilities with deterministic policy."""

    def plan(
        self,
        topic: str,
        profiles: Iterable[SourceProfile],
        constraints: SourcePlanningConstraints,
    ) -> list[ResearchSourceDecision]:
        categories = facade().classify_topic(topic)
        decisions = [
            self._decision(topic, categories, profile, constraints) for profile in profiles
        ]
        return sorted(
            decisions,
            key=lambda item: (item.selected, item.priority, item.name),
            reverse=True,
        )

    def _decision(
        self,
        topic: str,
        categories: set[str],
        profile: SourceProfile,
        constraints: SourcePlanningConstraints,
    ) -> ResearchSourceDecision:
        api = facade()
        budget = api.budget_for(profile, constraints)
        available = profile.available and api.availability_for(profile, constraints)
        priority = api.source_priority(topic, categories, profile)
        selected = (
            available and budget != 0 and priority >= api.selection_threshold(profile.source_type)
        )
        reason = api.decision_reason(
            profile,
            categories,
            selected,
            available,
            budget,
            priority,
        )
        return ResearchSourceDecision(
            name=profile.name,
            source_type=profile.source_type,
            selected=selected,
            available=available,
            reason=reason,
            capabilities=list(profile.capabilities),
            trust_tier=profile.trust_tier,
            priority=round(priority, 3),
            budget=budget,
            planner="heuristic",
            domains=list(profile.domains),
        )


def default_source_planner(strategy: str | None = None) -> SourcePlanner:
    normalized = (strategy or "auto").strip().lower()
    if normalized not in {"auto", "heuristic"}:
        raise ValueError("source planner strategy must be 'auto' or 'heuristic'")
    return SourcePlanner()


def source_priority(topic: str, categories: set[str], profile: SourceProfile) -> float:
    priority = profile.default_priority
    domain_overlap = categories.intersection(profile.domains)
    if domain_overlap:
        priority += min(0.22, 0.08 * len(domain_overlap))

    caps = set(profile.capabilities)
    if "software" in categories and {"code_artifacts", "official_docs"}.intersection(caps):
        priority += 0.18
    if "science" in categories and {
        "scholarly_metadata",
        "preprints",
        "citation_metadata",
    }.intersection(caps):
        priority += 0.16
    if "medicine" in categories and {
        "biomedical_metadata",
        "citation_metadata",
        "official_docs",
    }.intersection(caps):
        priority += 0.16
    if categories.intersection({"biology", "health"}) and {
        "biomedical_metadata",
        "clinical_literature",
    }.intersection(caps):
        priority += 0.16
    if categories.intersection({"medicine", "health"}) and {
        "clinical_trials",
        "registered_studies",
    }.intersection(caps):
        priority += 0.22
    if {"clinical_trials", "registered_studies"}.intersection(caps) and not categories.intersection(
        {"medicine", "health"}
    ):
        priority -= 0.24
    if {"biomedical_metadata", "clinical_literature"}.intersection(
        caps
    ) and not categories.intersection({"medicine", "biology", "health"}):
        priority -= 0.22
    if "current" in categories and {"current_web", "news", "standards"}.intersection(caps):
        priority += 0.14
    if "artifact" in categories and {"code_artifacts", "datasets", "benchmarks"}.intersection(caps):
        priority += 0.12
    if "research" in categories and {
        "scholarly_metadata",
        "preprints",
        "citation_metadata",
    }.intersection(caps):
        priority += 0.08

    terms = facade().topic_terms(topic)
    if profile.name in terms:
        priority += 0.15
    return min(1.0, max(0.0, priority))


def classify_topic(topic: str) -> set[str]:
    api = facade()
    terms = set(api.topic_terms(topic))
    categories = {"general"}
    if terms.intersection(api.SOFTWARE_TERMS):
        categories.add("software")
    if terms.intersection(api.SCIENCE_TERMS):
        categories.add("science")
    if terms.intersection(api.BIOLOGY_TERMS):
        categories.update({"biology", "science"})
    if terms.intersection(api.MEDICINE_TERMS):
        categories.update({"medicine", "health", "science"})
    if terms.intersection(api.CURRENT_TERMS):
        categories.add("current")
    if terms.intersection(api.ARTIFACT_TERMS):
        categories.add("artifact")
    if terms.intersection(api.RESEARCH_TERMS):
        categories.add("research")
    return categories


def budget_for(profile: SourceProfile, constraints: SourcePlanningConstraints) -> int:
    if profile.source_type == "scholarly":
        return constraints.max_scholarly_results
    if profile.source_type == "web_search":
        return constraints.max_search_results
    if profile.source_type == "code_search":
        return constraints.max_github_results
    if profile.source_type == "trial_registry":
        return constraints.max_trial_results
    if profile.source_type == "crawl_seed":
        return constraints.seed_url_count
    return 0


def availability_for(profile: SourceProfile, constraints: SourcePlanningConstraints) -> bool:
    if profile.source_type == "web_search":
        return bool(constraints.search_provider_name and constraints.search_provider_name != "none")
    if profile.source_type == "crawl_seed":
        return constraints.seed_url_count > 0
    return profile.available


def selection_threshold(source_type: str) -> float:
    if source_type == "scholarly":
        return 0.5
    if source_type == "code_search":
        return 0.48
    if source_type == "trial_registry":
        return 0.5
    if source_type == "web_search":
        return 0.5
    if source_type == "crawl_seed":
        return 0.0
    return 0.55


def decision_reason(
    profile: SourceProfile,
    categories: set[str],
    selected: bool,
    available: bool,
    budget: int,
    priority: float,
) -> str:
    if not available:
        if profile.source_type == "web_search":
            return "Skipped because no web search provider credentials are configured."
        if profile.source_type == "crawl_seed":
            return "Skipped because no seed URLs were supplied."
        return "Skipped because the source is unavailable."
    if budget == 0:
        return f"Skipped because the configured {profile.source_type} budget is zero."
    matched = ", ".join(sorted(categories.intersection(profile.domains)))
    capability = facade().strongest_capability(profile, categories)
    if selected:
        if matched and capability:
            return f"Selected for {matched} mission fit and {capability} capability."
        if matched:
            return f"Selected for {matched} mission fit."
        if capability:
            return f"Selected for {capability} capability."
        return "Selected as a broadly useful discovery source."
    return (
        f"Skipped because priority {priority:.2f} is below the "
        f"{facade().selection_threshold(profile.source_type):.2f} threshold for this mission."
    )


def strongest_capability(profile: SourceProfile, categories: set[str]) -> str | None:
    caps = set(profile.capabilities)
    if "software" in categories and "code_artifacts" in caps:
        return "code artifact"
    if "software" in categories and "official_docs" in caps:
        return "official/current documentation"
    if "medicine" in categories and "biomedical_metadata" in caps:
        return "biomedical metadata"
    if categories.intersection({"biology", "health"}) and "biomedical_metadata" in caps:
        return "biomedical metadata"
    if categories.intersection({"medicine", "health"}) and "clinical_trials" in caps:
        return "clinical trial registry"
    if "science" in categories and "scholarly_metadata" in caps:
        return "scholarly metadata"
    if "current" in categories and "current_web" in caps:
        return "current web"
    if "artifact" in categories and "benchmarks" in caps:
        return "benchmark"
    return None


def topic_terms(topic: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"[a-z][a-z0-9]+", topic.lower())))


SOFTWARE_TERMS = {
    "api",
    "app",
    "build",
    "client",
    "code",
    "coding",
    "framework",
    "implementation",
    "library",
    "package",
    "plugin",
    "python",
    "repo",
    "rest",
    "sdk",
    "service",
    "software",
}
SCIENCE_TERMS = {
    "alife",
    "artificial",
    "benchmark",
    "biology",
    "climate",
    "discovery",
    "evolution",
    "experiment",
    "life",
    "materials",
    "paper",
    "physics",
    "robotics",
    "science",
    "scientific",
}
BIOLOGY_TERMS = {
    "bio",
    "biology",
    "biological",
    "cell",
    "genome",
    "genomics",
    "molecular",
    "protein",
}
MEDICINE_TERMS = {
    "biomedical",
    "cancer",
    "clinical",
    "cure",
    "disease",
    "drug",
    "genomics",
    "health",
    "medical",
    "medicine",
    "protein",
    "therapy",
    "trial",
}
CURRENT_TERMS = {"2025", "2026", "current", "frontier", "latest", "live", "new", "recent", "today"}
ARTIFACT_TERMS = {
    "artifact",
    "benchmark",
    "code",
    "dataset",
    "implementation",
    "repo",
    "repository",
    "source",
}
RESEARCH_TERMS = {"evidence", "literature", "paper", "research", "review", "study", "survey"}

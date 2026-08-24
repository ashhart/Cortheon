from __future__ import annotations

import re

from cortheon.models import (
    CrawledPage,
    ResearchArtifact,
    ResearchClaim,
    ResearchCoverageItem,
    ResearchDiscoveryPass,
    ResearchSourceDecision,
    ScholarlyWork,
    SearchResult,
)


def analyze_source_coverage(
    topic: str,
    *,
    source_plan: list[ResearchSourceDecision],
    discovery_passes: list[ResearchDiscoveryPass],
    scholarly_works: list[ScholarlyWork],
    search_results: list[SearchResult],
    crawled_pages: list[CrawledPage],
    artifacts: list[ResearchArtifact],
    claims: list[ResearchClaim],
) -> list[ResearchCoverageItem]:
    categories = topic_categories(topic)
    selected = {item.name for item in source_plan if item.selected}
    selected_types = {item.source_type for item in source_plan if item.selected}
    return [
        coverage_item(
            name="scholarly_literature",
            expected=bool(selected_types.intersection({"scholarly"}))
            or bool(categories.intersection({"science", "medicine", "research"})),
            observed_count=len(scholarly_works),
            source_names=selected_sources(source_plan, "scholarly"),
            covered_reason=f"Found {len(scholarly_works)} scholarly work(s).",
            missing_reason="Scholarly literature was expected but no work passed relevance filtering.",
            next_action="Run or add domain-appropriate scholarly connectors, then inspect abstracts and full text.",
        ),
        coverage_item(
            name="clinical_trial_registry",
            expected=("clinicaltrials_gov" in selected)
            or bool(categories.intersection({"medicine", "clinical_trials"})),
            observed_count=clinical_trial_count(artifacts, discovery_passes),
            source_names=["clinicaltrials_gov"] if "clinicaltrials_gov" in selected else [],
            covered_reason=f"Found {clinical_trial_count(artifacts, discovery_passes)} registered clinical-trial artifact(s).",
            missing_reason="Clinical-trial registry coverage was expected but no trial artifact was found.",
            next_action="Query ClinicalTrials.gov or a domain registry and link records to publications and results.",
        ),
        coverage_item(
            name="code_artifacts",
            expected=("github_repositories" in selected) or "software" in categories,
            observed_count=sum(1 for artifact in artifacts if artifact.kind == "code_repository"),
            source_names=["github_repositories"] if "github_repositories" in selected else [],
            covered_reason="Found implementation/code artifact(s).",
            missing_reason="Implementation artifacts were expected but no code repository was found.",
            next_action="Run repository search and inspect build, license, tests, and implementation health.",
        ),
        coverage_item(
            name="current_web",
            expected=("web_search" in selected) or "current" in categories,
            observed_count=len(search_results),
            source_names=["web_search"] if "web_search" in selected else [],
            covered_reason=f"Found {len(search_results)} current web search result(s).",
            missing_reason="Current web search was expected but no search result was available.",
            next_action="Configure Brave, Tavily, or SerpAPI, or provide trusted seed URLs.",
        ),
        coverage_item(
            name="crawl_pages",
            expected=bool(search_results or crawled_pages),
            observed_count=len(crawled_pages),
            source_names=["web_crawler"] if crawled_pages else [],
            covered_reason=f"Crawled {len(crawled_pages)} page(s).",
            missing_reason="Crawlable seeds existed but no page was crawled successfully.",
            next_action="Increase crawl budget, inspect fetch errors, or provide an allowlisted seed domain.",
        ),
        coverage_item(
            name="grounded_claims",
            expected=bool(scholarly_works or crawled_pages),
            observed_count=grounded_claim_count(claims),
            source_names=["claim_extraction"] if claims else [],
            covered_reason=f"Extracted {grounded_claim_count(claims)} grounded claim(s).",
            missing_reason="Sources were available but no grounded claim was extracted.",
            next_action="Fetch richer abstracts/full text and improve extraction patterns before synthesizing strong claims.",
        ),
    ]


def coverage_item(
    *,
    name: str,
    expected: bool,
    observed_count: int,
    source_names: list[str],
    covered_reason: str,
    missing_reason: str,
    next_action: str,
) -> ResearchCoverageItem:
    if observed_count > 0:
        status = "covered"
        reason = covered_reason
    elif expected:
        status = "missing"
        reason = missing_reason
    else:
        status = "not_expected"
        reason = "This evidence surface was not expected for the current mission and source plan."
    return ResearchCoverageItem(
        name=name,
        status=status,
        expected=expected,
        observed_count=observed_count,
        reason=reason,
        next_action=next_action,
        source_names=source_names,
    )


def clinical_trial_count(
    artifacts: list[ResearchArtifact],
    discovery_passes: list[ResearchDiscoveryPass],
) -> int:
    artifact_count = sum(1 for artifact in artifacts if artifact.kind == "clinical_trial")
    pass_count = sum(item.registry_artifact_count for item in discovery_passes)
    return max(artifact_count, pass_count)


def grounded_claim_count(claims: list[ResearchClaim]) -> int:
    return sum(
        1
        for claim in claims
        if claim.source_excerpt
        and claim.source_char_start is not None
        and claim.source_char_end is not None
    )


def selected_sources(source_plan: list[ResearchSourceDecision], source_type: str) -> list[str]:
    return [item.name for item in source_plan if item.selected and item.source_type == source_type]


def topic_categories(topic: str) -> set[str]:
    terms = set(re.findall(r"[a-z][a-z0-9]+", topic.lower()))
    categories: set[str] = set()
    if terms.intersection(
        {
            "api",
            "build",
            "code",
            "framework",
            "implementation",
            "package",
            "python",
            "repo",
            "software",
        }
    ):
        categories.add("software")
    if terms.intersection(
        {
            "alife",
            "artificial",
            "benchmark",
            "biology",
            "evolution",
            "paper",
            "research",
            "science",
            "scientific",
        }
    ):
        categories.add("science")
    if terms.intersection(
        {
            "cancer",
            "clinical",
            "cure",
            "disease",
            "drug",
            "health",
            "medical",
            "medicine",
            "therapy",
            "trial",
        }
    ):
        categories.update({"medicine", "science"})
    if terms.intersection({"clinical", "intervention", "therapy", "trial", "trials"}):
        categories.add("clinical_trials")
    if terms.intersection(
        {"2025", "2026", "current", "frontier", "latest", "live", "new", "recent", "today"}
    ):
        categories.add("current")
    if terms.intersection(
        {"evidence", "literature", "paper", "research", "review", "study", "survey"}
    ):
        categories.add("research")
    return categories

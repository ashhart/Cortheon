from __future__ import annotations

from typing import Any

from cortheon.models import (
    Evidence,
    ResearchArtifact,
    ResearchDiscoveryPass,
    ResearchQuery,
    ScholarlyWork,
    SearchResult,
)
from cortheon.research_core._compat import facade


def scholarly_source_profiles(discovery: object) -> list[dict[str, object]]:
    source_profiles = getattr(discovery, "source_profiles", None)
    if callable(source_profiles):
        profiles = source_profiles()
        if isinstance(profiles, list):
            return profiles
    return [
        {
            "name": "scholarly",
            "source_type": "scholarly",
            "capabilities": ["scholarly_metadata", "papers"],
            "domains": ["science", "research"],
            "trust_tier": "custom_scholarly_discovery",
            "default_priority": 0.58,
            "available": True,
        }
    ]


def trial_registry_source_profiles(discovery: object) -> list[dict[str, object]]:
    source_profiles = getattr(discovery, "source_profiles", None)
    if callable(source_profiles):
        profiles = source_profiles()
        if isinstance(profiles, list):
            return profiles
    return []


def per_query_limit(total_limit: int, query_count: int) -> int:
    if total_limit <= 0 or query_count <= 0:
        return 0
    return max(1, facade().math.ceil(total_limit / query_count))


def merge_scholarly_works(
    topic: str,
    works: list[ScholarlyWork],
    limit: int,
) -> list[ScholarlyWork]:
    if limit <= 0:
        return []
    api = facade()
    rescored = [
        api.score_work_recency(api.score_work_relevance(work, topic))
        for work in api.dedupe_works(works)
    ]
    return sorted(rescored, key=api.scholarly_rank_key, reverse=True)[:limit]


def merge_search_results(results: list[SearchResult], limit: int) -> list[SearchResult]:
    if limit <= 0:
        return []
    seen: set[str] = set()
    merged: list[SearchResult] = []
    for result in results:
        if result.url in seen:
            continue
        seen.add(result.url)
        merged.append(result)
        if len(merged) >= limit:
            break
    return merged


def limit_discovered_artifacts(
    artifacts: list[ResearchArtifact],
    *,
    max_github_results: int,
    max_trial_results: int,
) -> list[ResearchArtifact]:
    code_count = 0
    trial_count = 0
    others: list[ResearchArtifact] = []
    for artifact in facade().dedupe_artifacts(artifacts):
        if artifact.kind == "code_repository":
            if code_count >= max(0, max_github_results):
                continue
            code_count += 1
        elif artifact.kind == "clinical_trial":
            if trial_count >= max(0, max_trial_results):
                continue
            trial_count += 1
        others.append(artifact)
    return others


def count_pass_seeds(
    scholarly_works: list[ScholarlyWork],
    search_results: list[SearchResult],
) -> int:
    urls = [work.url for work in scholarly_works if work.url.startswith(("http://", "https://"))]
    urls.extend(result.url for result in search_results)
    return len(facade().dedupe(urls))


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def run_discovery_queries(
    engine: Any,
    planned_queries: list[ResearchQuery],
    *,
    scholarly_limit: int,
    search_limit: int,
    github_limit: int,
    trial_limit: int,
    scholarly_connectors: list[str],
) -> tuple[
    list[ScholarlyWork],
    list[SearchResult],
    list[ResearchArtifact],
    list[Evidence],
    list[str],
    list[ResearchDiscoveryPass],
]:
    all_scholarly_works: list[ScholarlyWork] = []
    all_search_results: list[SearchResult] = []
    all_discovered_artifacts: list[ResearchArtifact] = []
    evidence: list[Evidence] = []
    errors: list[str] = []
    discovery_passes: list[ResearchDiscoveryPass] = []
    api = facade()

    for planned_query in planned_queries:
        pass_errors: list[str] = []
        pass_scholarly_works: list[ScholarlyWork] = []
        pass_search_results: list[SearchResult] = []
        pass_github_artifacts: list[ResearchArtifact] = []
        pass_trial_artifacts: list[ResearchArtifact] = []

        if scholarly_limit > 0:
            scholarly = engine.scholarly_discovery.search(
                planned_query.query,
                scholarly_limit,
                connector_names=scholarly_connectors,
            )
            pass_scholarly_works = scholarly.works
            all_scholarly_works.extend(pass_scholarly_works)
            evidence.extend(scholarly.evidence)
            errors.extend(scholarly.errors)
            pass_errors.extend(scholarly.errors)

        if search_limit > 0:
            search_results, search_evidence, search_errors = api.search_with_errors(
                engine.search_provider,
                planned_query.query,
                search_limit,
            )
            pass_search_results = search_results
            all_search_results.extend(search_results)
            evidence.extend(search_evidence)
            errors.extend(search_errors)
            pass_errors.extend(search_errors)

        if github_limit > 0:
            github_artifacts, github_evidence, github_errors = engine.github_discovery.search(
                planned_query.query,
                github_limit,
            )
            pass_github_artifacts = github_artifacts
            all_discovered_artifacts.extend(github_artifacts)
            evidence.extend(github_evidence)
            errors.extend(github_errors)
            pass_errors.extend(github_errors)

        if trial_limit > 0:
            trial_artifacts, trial_evidence, trial_errors = engine.trial_discovery.search(
                planned_query.query,
                trial_limit,
            )
            pass_trial_artifacts = trial_artifacts
            all_discovered_artifacts.extend(trial_artifacts)
            evidence.extend(trial_evidence)
            errors.extend(trial_errors)
            pass_errors.extend(trial_errors)

        discovery_passes.append(
            api.ResearchDiscoveryPass(
                query=planned_query.query,
                purpose=planned_query.purpose,
                source=planned_query.source,
                scholarly_work_count=len(pass_scholarly_works),
                search_result_count=len(pass_search_results),
                github_artifact_count=len(pass_github_artifacts),
                registry_artifact_count=len(pass_trial_artifacts),
                seed_count=api.count_pass_seeds(pass_scholarly_works, pass_search_results),
                errors=pass_errors,
                target_gap=planned_query.target_gap,
            )
        )
    return (
        all_scholarly_works,
        all_search_results,
        all_discovered_artifacts,
        evidence,
        errors,
        discovery_passes,
    )

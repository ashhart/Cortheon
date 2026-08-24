"""Stable facade for current-knowledge research orchestration."""

import math
from urllib.parse import urlparse

from cortheon.artifact_assessment import assess_artifacts
from cortheon.artifacts import dedupe_artifacts, derive_research_artifacts
from cortheon.claims import extract_claims
from cortheon.clinical_trials import ClinicalTrialsGovDiscovery
from cortheon.connectors.github import GitHubRepositorySearch
from cortheon.coverage import analyze_source_coverage
from cortheon.ledger import EvidenceLedger
from cortheon.lineage import build_source_lineage
from cortheon.models import (
    Evidence,
    ResearchArtifact,
    ResearchArtifactAssessment,
    ResearchCoverageItem,
    ResearchDiscoveryPass,
    ResearchGapClosure,
    ResearchQuery,
    ResearchReport,
    ScholarlyWork,
    SearchResult,
    SupportLevel,
    utc_now,
)
from cortheon.research_core.discovery import (
    count_pass_seeds,
    dedupe,
    limit_discovered_artifacts,
    merge_scholarly_works,
    merge_search_results,
    per_query_limit,
    run_discovery_queries,
    scholarly_source_profiles,
    trial_registry_source_profiles,
)
from cortheon.research_core.engine import ResearchEngine as _ResearchEngine
from cortheon.research_core.evidence import (
    artifact_assessment_evidence,
    artifact_evidence,
    grounding_evidence,
    lineage_evidence,
    mission_plan_evidence,
    source_coverage_evidence,
    synthesis_evidence,
)
from cortheon.research_core.gaps import (
    build_gap_closures,
    gap_closure_evidence,
    gap_kind,
    gap_metric_improved,
)
from cortheon.research_core.notes import (
    artifact_mix,
    artifact_notes,
    coverage_notes,
    mission_plan_notes,
    research_notes,
    source_mix,
)
from cortheon.research_plan import plan_gap_follow_up_queries, plan_research_queries
from cortheon.sanitize import quarantine_notes
from cortheon.scholarly import (
    CompositeScholarlyDiscovery,
    dedupe_works,
    scholarly_rank_key,
    score_work_recency,
    score_work_relevance,
)
from cortheon.search import ConfiguredSearchProvider, SearchProvider, search_with_errors
from cortheon.source_planner import (
    SourcePlanner,
    SourcePlanningConstraints,
    build_research_source_profiles,
    default_source_planner,
    is_source_selected,
    selected_source_names,
    source_plan_evidence,
    source_plan_notes,
)
from cortheon.synthesis import synthesize_research
from cortheon.web_crawler import CrawlBudget, WebCrawler


class ResearchEngine(_ResearchEngine):
    pass


__all__ = [
    "ClinicalTrialsGovDiscovery",
    "CompositeScholarlyDiscovery",
    "ConfiguredSearchProvider",
    "CrawlBudget",
    "Evidence",
    "EvidenceLedger",
    "GitHubRepositorySearch",
    "ResearchArtifact",
    "ResearchArtifactAssessment",
    "ResearchCoverageItem",
    "ResearchDiscoveryPass",
    "ResearchEngine",
    "ResearchGapClosure",
    "ResearchQuery",
    "ResearchReport",
    "ScholarlyWork",
    "SearchProvider",
    "SearchResult",
    "SourcePlanner",
    "SourcePlanningConstraints",
    "SupportLevel",
    "WebCrawler",
    "analyze_source_coverage",
    "artifact_assessment_evidence",
    "artifact_evidence",
    "artifact_mix",
    "artifact_notes",
    "assess_artifacts",
    "build_gap_closures",
    "build_research_source_profiles",
    "build_source_lineage",
    "count_pass_seeds",
    "coverage_notes",
    "dedupe",
    "dedupe_artifacts",
    "dedupe_works",
    "default_source_planner",
    "derive_research_artifacts",
    "extract_claims",
    "gap_closure_evidence",
    "gap_kind",
    "gap_metric_improved",
    "grounding_evidence",
    "is_source_selected",
    "limit_discovered_artifacts",
    "lineage_evidence",
    "math",
    "merge_scholarly_works",
    "merge_search_results",
    "mission_plan_evidence",
    "mission_plan_notes",
    "per_query_limit",
    "plan_gap_follow_up_queries",
    "plan_research_queries",
    "quarantine_notes",
    "research_notes",
    "run_discovery_queries",
    "scholarly_rank_key",
    "scholarly_source_profiles",
    "score_work_recency",
    "score_work_relevance",
    "search_with_errors",
    "selected_source_names",
    "source_coverage_evidence",
    "source_mix",
    "source_plan_evidence",
    "source_plan_notes",
    "synthesis_evidence",
    "synthesize_research",
    "trial_registry_source_profiles",
    "urlparse",
    "utc_now",
]

_DEFINED_NAMES = {
    "ResearchEngine",
    "artifact_assessment_evidence",
    "artifact_evidence",
    "artifact_mix",
    "artifact_notes",
    "build_gap_closures",
    "count_pass_seeds",
    "coverage_notes",
    "dedupe",
    "gap_closure_evidence",
    "gap_kind",
    "gap_metric_improved",
    "grounding_evidence",
    "limit_discovered_artifacts",
    "lineage_evidence",
    "merge_scholarly_works",
    "merge_search_results",
    "mission_plan_evidence",
    "mission_plan_notes",
    "per_query_limit",
    "research_notes",
    "scholarly_source_profiles",
    "source_coverage_evidence",
    "source_mix",
    "synthesis_evidence",
    "trial_registry_source_profiles",
}

for _public_name in _DEFINED_NAMES:
    _public_object = globals()[_public_name]
    if callable(_public_object) and hasattr(_public_object, "__module__"):
        _public_object.__module__ = __name__

ResearchEngine.research.__module__ = __name__
ResearchEngine._run_discovery_queries.__module__ = __name__
del _public_name, _public_object

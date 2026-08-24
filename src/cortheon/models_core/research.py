from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from cortheon.models_core.base import Evidence, SupportLevel, utc_now
from cortheon.models_core.discovery import (
    CrawledPage,
    ResearchArtifact,
    ResearchArtifactAssessment,
    ScholarlyWork,
    SearchResult,
)


@dataclass(slots=True)
class DecisionCheck:
    name: str
    status: str
    reason: str
    next_action: str | None = None


@dataclass(slots=True)
class DecisionReport:
    task: str
    proposed_action: str | None
    verdict: str
    confidence: float
    checks: list[DecisionCheck]
    required_evidence: list[str]
    recommended_tools: list[str]
    notes: list[str]
    cortheon: dict[str, Any] | None = None


@dataclass(slots=True)
class ResearchQuery:
    query: str
    purpose: str
    source: str
    target_gap: str | None = None


@dataclass(slots=True)
class ResearchSourceDecision:
    name: str
    source_type: str
    selected: bool
    available: bool
    reason: str
    capabilities: list[str]
    trust_tier: str
    priority: float
    budget: int | None = None
    planner: str = "heuristic"
    domains: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResearchDiscoveryPass:
    query: str
    purpose: str
    source: str
    scholarly_work_count: int
    search_result_count: int
    github_artifact_count: int
    seed_count: int
    registry_artifact_count: int = 0
    errors: list[str] = field(default_factory=list)
    target_gap: str | None = None


@dataclass(slots=True)
class ResearchGapClosure:
    target_gap: str
    query: str
    status: str
    before_claim_count: int
    after_claim_count: int
    before_source_count: int
    after_source_count: int
    before_gap_present: bool
    after_gap_present: bool
    after_related_gap: str | None = None


@dataclass(slots=True)
class ResearchCoverageItem:
    name: str
    status: str
    expected: bool
    observed_count: int
    reason: str
    next_action: str
    source_names: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResearchClaim:
    text: str
    source_url: str
    source_title: str | None
    source_type: str
    support: SupportLevel
    confidence: float
    stance: str = "neutral"
    source_excerpt: str | None = None
    source_char_start: int | None = None
    source_char_end: int | None = None
    extracted_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class SourceLineage:
    source_url: str
    source_title: str | None
    source_type: str
    authority_score: float | None
    relevance_score: float | None
    derived_claim_indexes: list[int]


@dataclass(slots=True)
class ClaimCluster:
    id: str
    label: str
    representative_claim: str
    claim_indexes: list[int]
    source_urls: list[str]
    average_confidence: float
    stance_counts: dict[str, int]
    conflict_score: float
    support_score: float
    terms: list[str]
    independent_source_count: int = 0
    corroboration: float = 0.0


@dataclass(slots=True)
class ContradictionGroup:
    id: str
    axis: str
    support_claim_indexes: list[int]
    challenge_claim_indexes: list[int]
    neutral_claim_indexes: list[int]
    support_sources: list[str]
    challenge_sources: list[str]
    summary: str
    severity: float


@dataclass(slots=True)
class ResearchSynthesis:
    topic: str
    generated_at: datetime
    status: str
    confidence: float
    current_best_direction: str
    key_findings: list[str]
    contested_points: list[str]
    evidence_gaps: list[str]
    clusters: list[ClaimCluster]
    contradictions: list[ContradictionGroup]


@dataclass(slots=True)
class ResearchReport:
    topic: str
    generated_at: datetime
    search_provider: str | None
    seed_urls: list[str]
    search_results: list[SearchResult]
    scholarly_works: list[ScholarlyWork]
    crawled_pages: list[CrawledPage]
    artifacts: list[ResearchArtifact]
    claims: list[ResearchClaim]
    source_lineage: list[SourceLineage]
    synthesis: ResearchSynthesis | None
    evidence: list[Evidence]
    notes: list[str]
    errors: list[str] = field(default_factory=list)
    mission_queries: list[ResearchQuery] = field(default_factory=list)
    source_plan: list[ResearchSourceDecision] = field(default_factory=list)
    discovery_passes: list[ResearchDiscoveryPass] = field(default_factory=list)
    source_coverage: list[ResearchCoverageItem] = field(default_factory=list)
    artifact_assessments: list[ResearchArtifactAssessment] = field(default_factory=list)
    gap_closures: list[ResearchGapClosure] = field(default_factory=list)

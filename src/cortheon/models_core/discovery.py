from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from cortheon.models_core.base import Evidence


@dataclass(slots=True)
class DocsPage:
    url: str
    final_url: str
    title: str | None
    fetched_at: datetime
    text: str
    headings: list[str]
    code_block_count: int
    code_blocks: list[str] = field(default_factory=list)
    quarantined_segments: int = 0
    is_changelog: bool = False
    error: str | None = None


@dataclass(slots=True)
class DocsExample:
    page_url: str
    code: str


@dataclass(slots=True)
class DocsSiteReport:
    package: str
    version: str
    generated_at: datetime
    docs_url: str | None
    changelog_url: str | None
    pages: list[DocsPage]
    examples: list[DocsExample]
    changelog_head: str | None
    evidence: list[Evidence]
    errors: list[str] = field(default_factory=list)
    requested_version: str | None = None
    docs_version_match: str | None = None


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str | None
    provider: str
    rank: int


@dataclass(slots=True)
class CrawledPage:
    url: str
    final_url: str
    status: int | None
    title: str | None
    text: str
    links: list[str]
    source_type: str
    authority_score: float
    fetched_at: datetime
    error: str | None = None


@dataclass(slots=True)
class ScholarlyWork:
    title: str
    url: str
    abstract: str | None
    authors: list[str]
    published_at: datetime | None
    source: str
    venue: str | None
    identifiers: dict[str, str]
    cited_by_count: int | None
    authority_score: float
    relevance_score: float = 0.0
    recency_score: float = 0.0


@dataclass(slots=True)
class ResearchArtifact:
    kind: str
    title: str | None
    url: str
    source_url: str | None
    provider: str
    evidence: str | None
    confidence: float
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ResearchArtifactAssessment:
    artifact_url: str
    artifact_kind: str
    title: str | None
    score: float
    decision: str
    reasons: list[str]
    risks: list[str]
    next_actions: list[str]

"""Repository-only report value types and serialization helpers.

Implementation ownership lives in :mod:`cortheon.models_core`. This facade
keeps the original import path, object identities, and patch points stable.
"""

# Former module globals remain observable patch points for moved code.
# ruff: noqa: F401

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum, StrEnum
from typing import Any

from cortheon.models_core.base import (
    Evidence,
    EvidenceStatus,
    SupportLevel,
    parse_datetime,
    to_jsonable,
    utc_now,
)
from cortheon.models_core.discovery import (
    CrawledPage,
    DocsExample,
    DocsPage,
    DocsSiteReport,
    ResearchArtifact,
    ResearchArtifactAssessment,
    ScholarlyWork,
    SearchResult,
)
from cortheon.models_core.package import (
    ApiDiffReport,
    ApiEvidenceReport,
    ApiSymbol,
    ApiSymbolChange,
    DistributionArtifact,
    DocumentationReport,
    ExampleRunResult,
    GitHubRepoReport,
    PackageMetadata,
    VerificationResult,
    VulnerabilityReport,
)
from cortheon.models_core.repository import (
    CodeUsageFinding,
    CodeUsageReport,
    PackageReport,
    PatchReport,
    PatchTestRun,
    RecommendationReport,
    RepoDependency,
    RepoFitReport,
    RepoReport,
    ScoreBreakdown,
)
from cortheon.models_core.research import (
    ClaimCluster,
    ContradictionGroup,
    DecisionCheck,
    DecisionReport,
    ResearchClaim,
    ResearchCoverageItem,
    ResearchDiscoveryPass,
    ResearchGapClosure,
    ResearchQuery,
    ResearchReport,
    ResearchSourceDecision,
    ResearchSynthesis,
    SourceLineage,
)

_MODEL_TYPES = (
    ApiDiffReport,
    ApiEvidenceReport,
    ApiSymbol,
    ApiSymbolChange,
    ClaimCluster,
    CodeUsageFinding,
    CodeUsageReport,
    ContradictionGroup,
    CrawledPage,
    DecisionCheck,
    DecisionReport,
    DistributionArtifact,
    DocsExample,
    DocsPage,
    DocsSiteReport,
    DocumentationReport,
    Evidence,
    EvidenceStatus,
    ExampleRunResult,
    GitHubRepoReport,
    PackageMetadata,
    PackageReport,
    PatchReport,
    PatchTestRun,
    RecommendationReport,
    RepoDependency,
    RepoFitReport,
    RepoReport,
    ResearchArtifact,
    ResearchArtifactAssessment,
    ResearchClaim,
    ResearchCoverageItem,
    ResearchDiscoveryPass,
    ResearchGapClosure,
    ResearchQuery,
    ResearchReport,
    ResearchSourceDecision,
    ResearchSynthesis,
    ScholarlyWork,
    ScoreBreakdown,
    SearchResult,
    SourceLineage,
    SupportLevel,
    VerificationResult,
    VulnerabilityReport,
)

for _definition in (parse_datetime, to_jsonable, utc_now, *_MODEL_TYPES):
    _definition.__module__ = __name__

for _model in _MODEL_TYPES:
    for _member in vars(_model).values():
        if isinstance(_member, (classmethod, staticmethod)):
            _member = _member.__func__
        accessors = (
            (_member.fget, _member.fset, _member.fdel)
            if isinstance(_member, property)
            else (_member,)
        )
        for _accessor in accessors:
            if (
                callable(_accessor)
                and hasattr(_accessor, "__module__")
                and _accessor.__module__.startswith("cortheon.models_core")
            ):
                _accessor.__module__ = __name__

del _accessor, _definition, _member, _model

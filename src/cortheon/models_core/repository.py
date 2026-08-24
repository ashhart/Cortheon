from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from cortheon.models_core.base import Evidence
from cortheon.models_core.package import (
    DocumentationReport,
    GitHubRepoReport,
    PackageMetadata,
    VerificationResult,
    VulnerabilityReport,
)


@dataclass(slots=True)
class RepoDependency:
    name: str
    constraint: str | None
    source: str


@dataclass(slots=True)
class RepoReport:
    root: str
    generated_at: datetime
    dependency_managers: list[str]
    python_requirement: str | None
    declared_dependencies: list[RepoDependency]
    lockfiles: list[str]
    test_commands: list[str]
    src_layout: bool
    framework_signals: list[str]
    python_file_count: int
    imported_third_party: list[str]
    imported_local: list[str]
    undeclared_imports: list[str]
    unused_declared: list[str]
    evidence: list[Evidence]
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RepoFitReport:
    package: str
    version: str | None
    repo_root: str
    generated_at: datetime
    already_declared: bool
    declared_constraint: str | None
    already_imported: bool
    python_compatible: bool | None
    python_note: str
    notes: list[str]
    risks: list[str]
    evidence: list[Evidence]
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CodeUsageFinding:
    package: str
    attribute: str
    line: int
    reason: str
    kind: str = ""


@dataclass(slots=True)
class CodeUsageReport:
    package: str
    version: str | None
    generated_at: datetime
    parsed: bool
    checked_calls: int
    verdict: str
    findings: list[CodeUsageFinding]
    notes: list[str]
    evidence: list[Evidence]
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PatchTestRun:
    command: str
    ran: bool
    passed: bool | None
    returncode: int | None
    duration_seconds: float
    output_tail: str


@dataclass(slots=True)
class PatchReport:
    repo_root: str
    generated_at: datetime
    applied: bool
    files_changed: list[str]
    insertions: int
    deletions: int
    baseline: PatchTestRun | None
    after: PatchTestRun | None
    verdict: str
    earned_evidence_tags: list[str]
    rollback_plan: list[str]
    notes: list[str]
    evidence: list[Evidence]
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ScoreBreakdown:
    fit: float
    docs: float
    maintenance: float
    security: float
    license: float
    dependency_risk: float
    execution: float
    adoption: float
    overall: float
    decision: str
    reasons: list[str]
    risks: list[str]


@dataclass(slots=True)
class PackageReport:
    package: str
    version: str | None
    fetched_at: datetime
    metadata: PackageMetadata | None
    vulnerabilities: VulnerabilityReport | None
    github: GitHubRepoReport | None
    documentation: DocumentationReport | None
    verification: VerificationResult | None
    evidence: list[Evidence]
    score: ScoreBreakdown | None
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RecommendationReport:
    task: str
    profile: str | None
    generated_at: datetime
    winner: str | None
    candidates: list[PackageReport]
    evidence: list[Evidence]
    notes: list[str]

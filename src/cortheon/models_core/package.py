from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from cortheon.models_core.base import Evidence


@dataclass(slots=True)
class DistributionArtifact:
    filename: str
    package_type: str
    url: str
    size: int | None
    upload_time: datetime | None
    digests: dict[str, str]


@dataclass(slots=True)
class PackageMetadata:
    name: str
    version: str
    summary: str | None
    requires_python: str | None
    license: str | None
    project_urls: dict[str, str]
    classifiers: list[str]
    requires_dist: list[str]
    release_upload_time: datetime | None
    release_count: int
    artifacts: list[DistributionArtifact]
    source_url: str
    description: str | None = None


@dataclass(slots=True)
class VulnerabilityReport:
    package: str
    version: str
    vulnerabilities: list[dict[str, Any]]
    source_url: str

    @property
    def count(self) -> int:
        return len(self.vulnerabilities)


@dataclass(slots=True)
class GitHubRepoReport:
    repo: str
    html_url: str
    description: str | None
    stars: int | None
    forks: int | None
    open_issues: int | None
    default_branch: str | None
    pushed_at: datetime | None
    archived: bool
    license_spdx: str | None
    source_url: str


@dataclass(slots=True)
class DocumentationReport:
    docs_url: str | None
    homepage_url: str | None
    reachable_urls: list[str]
    source_url: str | None


@dataclass(slots=True)
class ExampleRunResult:
    index: int
    ok: bool
    returncode: int | None
    duration_seconds: float
    code: str
    stdout_tail: str
    stderr_tail: str


@dataclass(slots=True)
class VerificationResult:
    package: str
    version: str
    install_ran: bool
    install_ok: bool | None
    import_name: str | None
    import_ok: bool | None
    command: list[str]
    stdout_tail: str
    stderr_tail: str
    duration_seconds: float
    source: str = "local_ephemeral_venv"
    example_results: list[ExampleRunResult] = field(default_factory=list)


@dataclass(slots=True)
class ApiSymbol:
    name: str
    kind: str
    module: str
    qualname: str
    signature: str | None
    file_path: str
    line: int
    docstring: str | None
    deprecated: bool = False


@dataclass(slots=True)
class ApiSymbolChange:
    qualname: str
    kind: str
    old_signature: str | None
    new_signature: str | None


@dataclass(slots=True)
class ApiDiffReport:
    package: str
    old_version: str
    new_version: str
    generated_at: datetime
    old_total_symbols: int
    new_total_symbols: int
    added_count: int
    removed_count: int
    changed_count: int
    deprecated_count: int
    added: list[ApiSymbol]
    removed: list[ApiSymbol]
    changed: list[ApiSymbolChange]
    deprecated_in_new: list[ApiSymbol]
    evidence: list[Evidence]
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ApiEvidenceReport:
    package: str
    version: str
    query: str
    artifact_filename: str | None
    artifact_url: str | None
    extracted_at: datetime
    total_symbols: int
    matches: list[ApiSymbol]
    evidence: list[Evidence]
    errors: list[str] = field(default_factory=list)

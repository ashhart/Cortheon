from __future__ import annotations

import json
import re
from pathlib import Path

from cortheon.models import (
    ApiDiffReport,
    ApiEvidenceReport,
    CodeUsageReport,
    DocsSiteReport,
    Evidence,
    PackageReport,
    PatchReport,
    RecommendationReport,
    RepoFitReport,
    RepoReport,
    ResearchReport,
    to_jsonable,
    utc_now,
)


class EvidenceLedger:
    def __init__(self, base_dir: Path | str = ".cortheon") -> None:
        self.base_dir = Path(base_dir)
        self.reports_dir = self.base_dir / "reports"
        self.evidence_path = self.base_dir / "evidence.jsonl"

    def ensure(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def append_evidence(self, evidence: list[Evidence]) -> None:
        self.ensure()
        with self.evidence_path.open("a", encoding="utf-8") as handle:
            for item in evidence:
                item.refresh_status()
                handle.write(json.dumps(to_jsonable(item), sort_keys=True) + "\n")

    def write_package_report(self, report: PackageReport) -> Path:
        self.ensure()
        package = _slug(report.package)
        version = _slug(report.version or "unknown")
        timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        path = self.reports_dir / f"{timestamp}-{package}-{version}.json"
        path.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
        self.append_evidence(report.evidence)
        return path

    def write_recommendation_report(self, report: RecommendationReport) -> Path:
        self.ensure()
        profile = _slug(report.profile or "adhoc")
        timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        path = self.reports_dir / f"{timestamp}-recommend-{profile}.json"
        path.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
        self.append_evidence(report.evidence)
        for candidate in report.candidates:
            self.append_evidence(candidate.evidence)
        return path

    def write_api_evidence_report(self, report: ApiEvidenceReport) -> Path:
        self.ensure()
        package = _slug(report.package)
        version = _slug(report.version)
        query = _slug(report.query)
        timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        path = self.reports_dir / f"{timestamp}-api-{package}-{version}-{query}.json"
        path.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
        self.append_evidence(report.evidence)
        return path

    def write_api_diff_report(self, report: ApiDiffReport) -> Path:
        self.ensure()
        package = _slug(report.package)
        old_version = _slug(report.old_version)
        new_version = _slug(report.new_version)
        timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        path = self.reports_dir / f"{timestamp}-apidiff-{package}-{old_version}-{new_version}.json"
        path.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
        self.append_evidence(report.evidence)
        return path

    def write_docs_report(self, report: DocsSiteReport) -> Path:
        self.ensure()
        package = _slug(report.package)
        version = _slug(report.version)
        timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        path = self.reports_dir / f"{timestamp}-docs-{package}-{version}.json"
        path.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
        self.append_evidence(report.evidence)
        return path

    def write_repo_report(self, report: RepoReport) -> Path:
        self.ensure()
        name = _slug(Path(report.root).name or "repo")
        timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        path = self.reports_dir / f"{timestamp}-repo-{name}.json"
        path.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
        self.append_evidence(report.evidence)
        return path

    def write_repo_fit_report(self, report: RepoFitReport) -> Path:
        self.ensure()
        package = _slug(report.package)
        timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        path = self.reports_dir / f"{timestamp}-repofit-{package}.json"
        path.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
        self.append_evidence(report.evidence)
        return path

    def write_code_usage_report(self, report: CodeUsageReport) -> Path:
        self.ensure()
        package = _slug(report.package)
        timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        path = self.reports_dir / f"{timestamp}-codecheck-{package}.json"
        path.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
        self.append_evidence(report.evidence)
        return path

    def write_patch_report(self, report: PatchReport) -> Path:
        self.ensure()
        name = _slug(Path(report.repo_root).name or "repo")
        timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        path = self.reports_dir / f"{timestamp}-patch-{name}.json"
        path.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
        self.append_evidence(report.evidence)
        return path

    def write_research_report(self, report: ResearchReport) -> Path:
        self.ensure()
        topic = _slug(report.topic)
        timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        path = self.reports_dir / f"{timestamp}-research-{topic}.json"
        path.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
        self.append_evidence(report.evidence)
        return path


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-")
    return cleaned[:80] or "item"

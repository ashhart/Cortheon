"""Repository evidence workflows exposed through the stable engine API."""

from __future__ import annotations

import re
import sys
from types import ModuleType
from typing import TYPE_CHECKING

from cortheon.api_diff import build_api_diff_report
from cortheon.api_indexer import ApiEvidenceExtractor
from cortheon.cache import FactCache
from cortheon.code_check import check_api_usage, extract_code_blocks
from cortheon.connectors.docs import DocumentationConnector
from cortheon.connectors.github import GitHubConnector
from cortheon.connectors.http import ConnectorError
from cortheon.connectors.osv import OSVConnector
from cortheon.connectors.pypi import PyPIConnector
from cortheon.docs_reader import API_GUIDE_KEYWORDS, DocsSiteReader, find_symbol_mention
from cortheon.engine_core import api, recommendations, repository
from cortheon.engine_core import packages as package_workflows
from cortheon.examples import extract_runnable_examples
from cortheon.ledger import EvidenceLedger
from cortheon.models import (
    ApiDiffReport,
    ApiEvidenceReport,
    DocsSiteReport,
    Evidence,
    PackageReport,
    PatchReport,
    RecommendationReport,
    RepoFitReport,
    RepoReport,
    SupportLevel,
    utc_now,
)
from cortheon.option_ranker import OptionRankingReport, rank_options
from cortheon.patch_runner import run_patch_verification
from cortheon.repo_scanner import normalize_dep_name, python_compatibility, scan_repo
from cortheon.sandbox import run_sandboxed_install_import_test
from cortheon.scoring import score_package
from cortheon.tasks import Candidate, find_profile
from cortheon.verifier import guess_import_name, run_install_import_test

if TYPE_CHECKING:
    from cortheon.models import CodeUsageReport

_LATE_BOUND_DEPENDENCIES = (
    API_GUIDE_KEYWORDS,
    ConnectorError,
    Evidence,
    FactCache,
    PackageReport,
    RecommendationReport,
    RepoFitReport,
    SupportLevel,
    build_api_diff_report,
    check_api_usage,
    extract_code_blocks,
    extract_runnable_examples,
    find_profile,
    find_symbol_mention,
    guess_import_name,
    normalize_dep_name,
    python_compatibility,
    rank_options,
    run_install_import_test,
    run_patch_verification,
    run_sandboxed_install_import_test,
    scan_repo,
    score_package,
    utc_now,
)


def _bindings() -> ModuleType:
    return sys.modules[__name__]


def _code_usage_models() -> tuple[type, type]:
    from cortheon.models import CodeUsageFinding, CodeUsageReport

    return CodeUsageFinding, CodeUsageReport


class CortheonEngine:
    def __init__(
        self,
        pypi: PyPIConnector | None = None,
        osv: OSVConnector | None = None,
        github: GitHubConnector | None = None,
        docs: DocumentationConnector | None = None,
        api_extractor: ApiEvidenceExtractor | None = None,
        docs_reader: DocsSiteReader | None = None,
        ledger: EvidenceLedger | None = None,
    ) -> None:
        self.pypi = pypi or PyPIConnector()
        self.osv = osv or OSVConnector()
        self.github = github or GitHubConnector()
        self.docs = docs or DocumentationConnector()
        self.ledger = ledger or EvidenceLedger()
        self.api_extractor = api_extractor or ApiEvidenceExtractor(
            cache=FactCache(self.ledger.base_dir / "cache")
        )
        self.docs_reader = docs_reader or DocsSiteReader()

    def inspect_package(
        self,
        package: str,
        *,
        task_text: str | None = None,
        candidate: Candidate | None = None,
        run_install: bool = False,
        run_examples: bool = False,
        sandbox: bool = False,
        write_report: bool = True,
    ) -> PackageReport:
        return package_workflows.inspect_package(
            self,
            _bindings(),
            package,
            task_text=task_text,
            candidate=candidate,
            run_install=run_install,
            run_examples=run_examples,
            sandbox=sandbox,
            write_report=write_report,
        )

    def compare(
        self,
        packages: list[str],
        *,
        task_text: str | None = None,
        run_install: bool = False,
        write_report: bool = True,
    ) -> RecommendationReport:
        return package_workflows.compare(
            self,
            packages,
            task_text=task_text,
            run_install=run_install,
            write_report=write_report,
        )

    def recommend(
        self,
        task: str,
        *,
        run_install: bool = False,
        write_report: bool = True,
    ) -> RecommendationReport:
        return package_workflows.recommend(
            self,
            _bindings(),
            task,
            run_install=run_install,
            write_report=write_report,
        )

    def retrieve_api_evidence(
        self,
        package: str,
        query: str,
        *,
        include_docs: bool = False,
        write_report: bool = True,
    ) -> ApiEvidenceReport:
        return api.retrieve_api_evidence(
            self,
            _bindings(),
            package,
            query,
            include_docs=include_docs,
            write_report=write_report,
        )

    def check_generated_code(
        self,
        package: str,
        code: str,
        *,
        write_report: bool = True,
    ) -> CodeUsageReport:
        return api.check_generated_code(
            self,
            _bindings(),
            package,
            code,
            write_report=write_report,
        )

    def fetch_docs(
        self,
        package: str,
        *,
        version: str | None = None,
        max_pages: int = 4,
        write_report: bool = True,
    ) -> DocsSiteReport:
        return api.fetch_docs(
            self,
            package,
            version=version,
            max_pages=max_pages,
            write_report=write_report,
        )

    def diff_api(
        self,
        package: str,
        old_version: str,
        new_version: str,
        *,
        write_report: bool = True,
    ) -> ApiDiffReport:
        return api.diff_api(
            self,
            _bindings(),
            package,
            old_version,
            new_version,
            write_report=write_report,
        )

    def verify_patch(
        self,
        repo_path: str,
        patch_text: str,
        *,
        test_command: str | None = None,
        run_baseline: bool = True,
        test_isolation: str = "host",
        sandbox_image: str = "python:3.12-slim-bookworm",
        write_report: bool = True,
    ) -> PatchReport:
        return repository.verify_patch(
            self,
            _bindings(),
            repo_path,
            patch_text,
            test_command=test_command,
            run_baseline=run_baseline,
            test_isolation=test_isolation,
            sandbox_image=sandbox_image,
            write_report=write_report,
        )

    def scan_repo(self, path: str = ".", *, write_report: bool = True) -> RepoReport:
        return repository.scan_repository(
            self,
            _bindings(),
            path,
            write_report=write_report,
        )

    def check_repo_fit(
        self,
        package: str,
        repo_path: str = ".",
        *,
        write_report: bool = True,
    ) -> RepoFitReport:
        return repository.check_repo_fit(
            self,
            _bindings(),
            package,
            repo_path,
            write_report=write_report,
        )

    def _recommendation(
        self,
        *,
        task: str,
        profile: str | None,
        candidates: list[PackageReport],
        notes: list[str],
    ) -> RecommendationReport:
        return recommendations.build_recommendation(
            _bindings(),
            task=task,
            profile=profile,
            candidates=candidates,
            notes=notes,
        )


def ranking_to_recommendation(
    task: str,
    ranking: OptionRankingReport,
) -> RecommendationReport:
    """Convert an OptionRankingReport into the existing RecommendationReport shape."""
    return recommendations.ranking_to_recommendation(_bindings(), task, ranking)


def merge_examples(*groups: list[str], limit: int = 4) -> list[str]:
    return recommendations.merge_examples(*groups, limit=limit, substitute=re.sub)

import unittest
from datetime import timedelta

from cortheon.models import (
    DocumentationReport,
    ExampleRunResult,
    GitHubRepoReport,
    PackageMetadata,
    PackageReport,
    VerificationResult,
    VulnerabilityReport,
    utc_now,
)
from cortheon.scoring import score_package
from cortheon.tasks import Candidate


class ScoringTests(unittest.TestCase):
    def test_strong_package_scores_recommend(self) -> None:
        now = utc_now()
        report = PackageReport(
            package="fastapi",
            version="1.0.0",
            fetched_at=now,
            metadata=PackageMetadata(
                name="fastapi",
                version="1.0.0",
                summary="Fast API framework with OpenAPI and typed validation",
                requires_python=">=3.8",
                license="MIT",
                project_urls={"Documentation": "https://fastapi.tiangolo.com"},
                classifiers=["License :: OSI Approved :: MIT License"],
                requires_dist=["starlette", "pydantic"],
                release_upload_time=now - timedelta(days=20),
                release_count=100,
                artifacts=[],
                source_url="https://pypi.org/pypi/fastapi/json",
            ),
            vulnerabilities=VulnerabilityReport(
                package="fastapi",
                version="1.0.0",
                vulnerabilities=[],
                source_url="https://api.osv.dev/v1/query",
            ),
            github=GitHubRepoReport(
                repo="fastapi/fastapi",
                html_url="https://github.com/fastapi/fastapi",
                description="FastAPI framework",
                stars=90000,
                forks=7000,
                open_issues=100,
                default_branch="master",
                pushed_at=now - timedelta(days=2),
                archived=False,
                license_spdx="MIT",
                source_url="https://api.github.com/repos/fastapi/fastapi",
            ),
            documentation=DocumentationReport(
                docs_url="https://fastapi.tiangolo.com",
                homepage_url="https://fastapi.tiangolo.com",
                reachable_urls=["https://fastapi.tiangolo.com"],
                source_url="https://fastapi.tiangolo.com",
            ),
            verification=None,
            evidence=[],
            score=None,
            errors=[],
        )

        score = score_package(
            report,
            candidate=Candidate("fastapi", 0.18, "Strong typed API fit."),
            task_text="build a REST API with OpenAPI",
            now=now,
        )

        self.assertEqual(score.decision, "recommend")
        self.assertGreater(score.overall, 0.78)
        self.assertIn("Install/import smoke test was not run.", score.risks)

    def test_failed_readme_examples_dent_execution_score(self) -> None:
        passing = verification_with_examples(example_ok=True)
        failing = verification_with_examples(example_ok=False)

        passing_score = score_package(package_report_with(passing))
        failing_score = score_package(package_report_with(failing))

        self.assertEqual(passing_score.execution, 1.0)
        self.assertEqual(failing_score.execution, 0.7)
        self.assertIn("1 official example(s) executed successfully.", passing_score.reasons)
        self.assertIn("1 of 1 official example(s) failed to execute.", failing_score.risks)


def verification_with_examples(example_ok: bool) -> VerificationResult:
    return VerificationResult(
        package="examplepkg",
        version="1.0.0",
        install_ran=True,
        install_ok=True,
        import_name="examplepkg",
        import_ok=True,
        command=["python", "-m", "pip", "install", "examplepkg==1.0.0"],
        stdout_tail="",
        stderr_tail="",
        duration_seconds=1.0,
        example_results=[
            ExampleRunResult(
                index=1,
                ok=example_ok,
                returncode=0 if example_ok else 1,
                duration_seconds=0.1,
                code="import examplepkg",
                stdout_tail="",
                stderr_tail="",
            )
        ],
    )


def package_report_with(verification: VerificationResult) -> PackageReport:
    return PackageReport(
        package="examplepkg",
        version="1.0.0",
        fetched_at=utc_now(),
        metadata=None,
        vulnerabilities=None,
        github=None,
        documentation=None,
        verification=verification,
        evidence=[],
        score=None,
        errors=[],
    )


if __name__ == "__main__":
    unittest.main()


def test_renamed_repository_is_a_deprecation_signal() -> None:
    """Regression: PyPDF2's repo resolves to py-pdf/pypdf and the verify
    report never said so, although the redirect was in the fetched evidence.
    """

    from datetime import datetime

    from cortheon.models import UTC, GitHubRepoReport, PackageReport
    from cortheon.scoring import _deprecation_signal, score_package

    github = GitHubRepoReport(
        repo="py-pdf/pypdf",
        html_url="https://github.com/py-pdf/pypdf",
        description=None,
        stars=1000,
        forks=10,
        open_issues=5,
        default_branch="main",
        pushed_at=datetime(2026, 1, 1, tzinfo=UTC),
        archived=False,
        license_spdx="BSD-3-Clause",
        source_url="https://api.github.com/repos/py-pdf/PyPDF2",
    )
    report = PackageReport(
        package="PyPDF2",
        version="3.0.1",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata=None,
        vulnerabilities=None,
        github=github,
        documentation=None,
        verification=None,
        evidence=[],
        score=None,
    )
    signal = _deprecation_signal(report)
    assert signal is not None and "py-pdf/pypdf" in signal
    score = score_package(report)
    assert "deprecated or superseded" in score.decision
    assert any("py-pdf/pypdf" in risk for risk in score.risks)


def test_verdict_follows_thin_evidence() -> None:
    """Regression: allow at 0.95 beside thin_evidence at 0.5 in one report."""

    from cortheon.knowledge_pool import couple_verdict_to_answer_status

    verdict, notes = couple_verdict_to_answer_status("allow", "thin_evidence")
    assert verdict == "needs_evidence" and notes
    verdict, notes = couple_verdict_to_answer_status("allow", "answered")
    assert verdict == "allow" and not notes
    verdict, notes = couple_verdict_to_answer_status("block", "thin_evidence")
    assert verdict == "block" and not notes


def test_supersession_requires_an_actual_rename_redirect() -> None:
    """Regression: package-vs-repo name mismatch flagged healthy SDKs.

    openai (repo openai/openai-python), redis (redis/redis-py), and
    python-dateutil (dateutil/dateutil) are current packages whose repos are
    named differently. Only a GitHub rename redirect, the requested URL
    resolving to a different canonical repo, is supersession evidence.
    """

    from datetime import datetime

    from cortheon.models import UTC, GitHubRepoReport, PackageReport
    from cortheon.scoring import _deprecation_signal

    def report_for(package: str, repo: str, source_url: str | None) -> PackageReport:
        github = GitHubRepoReport(
            repo=repo,
            html_url=f"https://github.com/{repo}",
            description=None,
            stars=1000,
            forks=10,
            open_issues=5,
            default_branch="main",
            pushed_at=None,
            archived=False,
            license_spdx="MIT",
            source_url=source_url,
        )
        return PackageReport(
            package=package,
            version=None,
            fetched_at=datetime(2026, 8, 20, tzinfo=UTC),
            metadata=None,
            vulnerabilities=None,
            github=github,
            documentation=None,
            verification=None,
            evidence=[],
            score=None,
        )

    healthy = [
        ("openai", "openai/openai-python"),
        ("anthropic", "anthropics/anthropic-sdk-python"),
        ("redis", "redis/redis-py"),
        ("python-dateutil", "dateutil/dateutil"),
    ]
    for package, repo in healthy:
        direct = report_for(package, repo, f"https://api.github.com/repos/{repo}")
        assert _deprecation_signal(direct) is None, (package, repo)

    # No requested URL on record: the rename cannot be established, so the
    # mismatch between package and repo names must stay silent.
    anonymous = report_for("grpcio", "grpc/grpc", None)
    assert _deprecation_signal(anonymous) is None

    # PyPDF2 links its old repo name; GitHub redirects to the successor.
    renamed = report_for("PyPDF2", "py-pdf/pypdf", "https://api.github.com/repos/py-pdf/PyPDF2")
    signal = _deprecation_signal(renamed)
    assert signal is not None and "py-pdf/pypdf" in signal

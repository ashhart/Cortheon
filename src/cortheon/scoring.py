from __future__ import annotations

import math
import re
from datetime import datetime

from cortheon.models import UTC, PackageReport, ScoreBreakdown, utc_now
from cortheon.tasks import Candidate


def score_package(
    report: PackageReport,
    candidate: Candidate | None = None,
    task_text: str | None = None,
    now: datetime | None = None,
) -> ScoreBreakdown:
    current_time = now or utc_now()
    metadata = report.metadata
    github = report.github
    vulnerabilities = report.vulnerabilities
    verification = report.verification

    fit = 0.45
    if candidate:
        fit += candidate.fit_boost
    if metadata and task_text:
        fit += _keyword_fit(metadata.summary or "", metadata.classifiers, task_text)
    fit = _clamp(fit)

    docs = 0.25
    if report.documentation:
        if report.documentation.docs_url:
            docs += 0.35
        if report.documentation.homepage_url:
            docs += 0.15
        if report.documentation.reachable_urls:
            docs += 0.2
    if metadata and metadata.summary and len(metadata.summary) > 40:
        docs += 0.05
    docs = _clamp(docs)

    maintenance = _maintenance_score(report, current_time)

    if vulnerabilities is None:
        security = 0.55
    elif vulnerabilities.count == 0:
        security = 0.92
    else:
        security = max(0.1, 0.72 - vulnerabilities.count * 0.18)

    license_score = _license_score(
        metadata.license if metadata else None, metadata.classifiers if metadata else []
    )
    dependency_risk = _dependency_score(len(metadata.requires_dist) if metadata else None)

    if verification is None or not verification.install_ran:
        execution = 0.5
    elif verification.install_ok and verification.import_ok:
        execution = 1.0
        # A package whose own README examples fail is weaker execution evidence
        # than install/import alone would suggest.
        if verification.example_results and any(
            not item.ok for item in verification.example_results
        ):
            execution = 0.7
    elif verification.install_ok:
        execution = 0.45
    else:
        execution = 0.1

    adoption = _adoption_score(
        github.stars if github else None, metadata.release_count if metadata else 0
    )

    overall = (
        fit * 0.22
        + docs * 0.13
        + maintenance * 0.18
        + security * 0.18
        + license_score * 0.08
        + dependency_risk * 0.08
        + execution * 0.08
        + adoption * 0.05
    )
    decision = _decision(
        overall,
        vulnerabilities.count if vulnerabilities else 0,
        github.archived if github else False,
    )
    reasons, risks = _reasons_and_risks(report, candidate, overall)
    deprecation = _deprecation_signal(report)
    if deprecation:
        risks.insert(0, deprecation)
        if decision == "allow":
            decision = "inspect: deprecated or superseded"
        elif decision.startswith("inspect"):
            decision = decision + "; deprecated or superseded"

    return ScoreBreakdown(
        fit=round(fit, 3),
        docs=round(docs, 3),
        maintenance=round(maintenance, 3),
        security=round(security, 3),
        license=round(license_score, 3),
        dependency_risk=round(dependency_risk, 3),
        execution=round(execution, 3),
        adoption=round(adoption, 3),
        overall=round(overall, 3),
        decision=decision,
        reasons=reasons,
        risks=risks,
    )


def _keyword_fit(summary: str, classifiers: list[str], task_text: str) -> float:
    haystack = " ".join([summary, *classifiers]).lower()
    words = {
        word.strip(".,:;()[]{}")
        for word in task_text.lower().split()
        if len(word.strip(".,:;()[]{}")) > 4
    }
    if not words:
        return 0.0
    hits = sum(1 for word in words if word in haystack)
    return min(0.15, hits * 0.035)


def _maintenance_score(report: PackageReport, now: datetime) -> float:
    metadata = report.metadata
    github = report.github
    if github and github.archived:
        return 0.05

    recency_scores: list[float] = []
    if metadata and metadata.release_upload_time:
        recency_scores.append(_recency(metadata.release_upload_time, now))
    if github and github.pushed_at:
        recency_scores.append(_recency(github.pushed_at, now))
    if not recency_scores:
        return 0.5
    return sum(recency_scores) / len(recency_scores)


def _recency(value: datetime, now: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    days = max((now - value.astimezone(UTC)).days, 0)
    if days <= 45:
        return 0.98
    if days <= 180:
        return 0.86
    if days <= 365:
        return 0.72
    if days <= 730:
        return 0.52
    return 0.32


def _license_score(license_text: str | None, classifiers: list[str]) -> float:
    combined = " ".join([license_text or "", *classifiers]).lower()
    if not combined.strip():
        return 0.42
    permissive = ("mit", "apache", "bsd", "isc", "mpl")
    restrictive = ("gpl", "agpl")
    if any(token in combined for token in permissive):
        return 0.92
    if any(token in combined for token in restrictive):
        return 0.58
    return 0.72


def _dependency_score(count: int | None) -> float:
    if count is None:
        return 0.5
    if count <= 2:
        return 0.92
    if count <= 8:
        return 0.8
    if count <= 20:
        return 0.62
    if count <= 40:
        return 0.45
    return 0.28


def _adoption_score(stars: int | None, release_count: int) -> float:
    star_score = 0.45
    if stars is not None:
        star_score = min(1.0, math.log10(max(stars, 1)) / 5)
    release_score = min(1.0, release_count / 120) if release_count else 0.3
    return (star_score * 0.65) + (release_score * 0.35)


def _requested_repo(source_url: str | None) -> str | None:
    """The owner/repo actually requested before GitHub's rename redirect.

    Only the redirect is supersession evidence: a healthy package whose
    repo is merely named differently (openai resolves to
    openai/openai-python) must not be flagged.
    """
    if not source_url:
        return None
    matched = re.search(r"github\.com/repos/([\w.-]+/[\w.-]+)", source_url)
    return matched.group(1) if matched else None


def _repo_slug(name: str) -> str:
    return re.sub(r"[-_]", "", name.split("/")[-1].lower())


def _deprecation_signal(report: PackageReport) -> str | None:
    """Derive an explicit deprecation finding from evidence already fetched.

    PyPI self-descriptions, the Inactive trove classifier, and a project
    repository whose requested URL redirects to a renamed successor are
    each a stronger conclusion than their raw fields suggest, and a small
    model told to use the package will not draw it unaided.
    """

    metadata = report.metadata
    haystack = " ".join(
        [
            (metadata.summary or "") if metadata else "",
            (metadata.description or "")[:2000] if metadata else "",
            *(metadata.classifiers if metadata else ""),
        ]
    ).lower()
    if "development status :: 7 - inactive" in haystack:
        return "PyPI marks this package Development Status 7 (Inactive)."
    matched = re.search(r"(?:is\s+)?deprecated[^.\n]{0,90}", haystack)
    if matched:
        return f'The package describes itself as deprecated: "{matched.group(0).strip()}".'
    github = report.github
    if github and github.repo:
        requested_repo = _requested_repo(github.source_url)
        if requested_repo and _repo_slug(requested_repo) != _repo_slug(github.repo):
            return (
                f"The project repository {requested_repo} was renamed; it "
                f"now resolves to {github.repo}, so the package appears "
                "superseded by that project."
            )
    return None


def _decision(overall: float, vulnerability_count: int, archived: bool) -> str:
    if archived:
        return "avoid: repository archived"
    if vulnerability_count:
        return "inspect: known vulnerabilities"
    if overall >= 0.78:
        return "recommend"
    if overall >= 0.64:
        return "consider"
    return "inspect before use"


def _reasons_and_risks(
    report: PackageReport,
    candidate: Candidate | None,
    overall: float,
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    risks: list[str] = []
    if candidate:
        reasons.append(candidate.rationale)
    if report.metadata:
        reasons.append(f"PyPI latest version is {report.metadata.version}.")
    if report.vulnerabilities and report.vulnerabilities.count == 0:
        reasons.append("OSV returned no known vulnerabilities for the selected version.")
    if report.documentation and report.documentation.docs_url:
        reasons.append("Package metadata includes a documentation link.")
    if report.github and not report.github.archived:
        reasons.append("Linked GitHub repository is not archived.")

    if report.verification is None:
        risks.append("Install/import smoke test was not run.")
    elif not (report.verification.install_ok and report.verification.import_ok):
        risks.append("Install/import smoke test did not fully pass.")
    if (
        report.verification
        and report.verification.source == "docker_sandbox"
        and report.verification.install_ok
        and report.verification.import_ok
    ):
        reasons.append("Execution evidence came from a network-isolated Docker sandbox.")
    if report.verification and report.verification.example_results:
        example_results = report.verification.example_results
        failed = sum(1 for item in example_results if not item.ok)
        if failed:
            risks.append(
                f"{failed} of {len(example_results)} official example(s) failed to execute."
            )
        else:
            reasons.append(f"{len(example_results)} official example(s) executed successfully.")
    if report.vulnerabilities and report.vulnerabilities.count:
        risks.append(f"OSV returned {report.vulnerabilities.count} known vulnerabilities.")
    if report.github and report.github.archived:
        risks.append("Linked GitHub repository is archived.")
    if report.documentation and not report.documentation.reachable_urls:
        risks.append("Documentation/homepage links were not confirmed reachable.")
    if overall < 0.64:
        risks.append("Overall score is below the normal recommendation threshold.")
    return reasons[:6], risks[:6]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))

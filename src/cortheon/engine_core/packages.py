"""Package inspection, comparison, and recommendation workflows."""

from __future__ import annotations

from types import ModuleType
from typing import Any


def inspect_package(
    engine: Any,
    bindings: ModuleType,
    package: str,
    *,
    task_text: str | None,
    candidate: Any,
    run_install: bool,
    run_examples: bool,
    sandbox: bool,
    write_report: bool,
) -> Any:
    evidence: list[Any] = []
    errors: list[str] = []
    metadata = None
    vulnerabilities = None
    github = None
    documentation = None
    verification = None

    try:
        metadata, items = engine.pypi.fetch(package)
        evidence.extend(items)
    except bindings.ConnectorError as exc:
        errors.append(str(exc))

    if metadata:
        try:
            vulnerabilities, items = engine.osv.fetch(metadata.name, metadata.version)
            evidence.extend(items)
        except bindings.ConnectorError as exc:
            errors.append(str(exc))

        try:
            github, items = engine.github.fetch_from_project_urls(metadata.project_urls)
            evidence.extend(items)
        except bindings.ConnectorError as exc:
            errors.append(str(exc))

        try:
            documentation, items = engine.docs.inspect(metadata.project_urls)
            evidence.extend(items)
        except bindings.ConnectorError as exc:
            errors.append(str(exc))

        if run_install or run_examples:
            examples = None
            if run_examples:
                readme_examples = bindings.extract_runnable_examples(
                    metadata.description,
                    [bindings.guess_import_name(metadata.name)],
                )
                docs_report = engine.docs_reader.read(metadata, max_pages=3)
                evidence.extend(docs_report.evidence)
                examples = bindings.merge_examples(
                    readme_examples,
                    [example.code for example in docs_report.examples],
                )
            verifier = (
                bindings.run_sandboxed_install_import_test
                if sandbox
                else bindings.run_install_import_test
            )
            verification, items = verifier(
                metadata.name,
                metadata.version,
                examples=examples,
            )
            evidence.extend(items)

    report = bindings.PackageReport(
        package=metadata.name if metadata else package,
        version=metadata.version if metadata else None,
        fetched_at=bindings.utc_now(),
        metadata=metadata,
        vulnerabilities=vulnerabilities,
        github=github,
        documentation=documentation,
        verification=verification,
        evidence=evidence,
        score=None,
        errors=errors,
    )
    report.score = bindings.score_package(report, candidate=candidate, task_text=task_text)
    if write_report:
        engine.ledger.write_package_report(report)
    return report


def compare(
    engine: Any,
    packages: list[str],
    *,
    task_text: str | None,
    run_install: bool,
    write_report: bool,
) -> Any:
    candidates = [
        engine.inspect_package(
            package,
            task_text=task_text,
            run_install=run_install,
            write_report=False,
        )
        for package in packages
    ]
    report = engine._recommendation(
        task=task_text or "Compare packages",
        profile=None,
        candidates=candidates,
        notes=["Candidate packages were provided explicitly by the user."],
    )
    if write_report:
        engine.ledger.write_recommendation_report(report)
    return report


def recommend(
    engine: Any,
    bindings: ModuleType,
    task: str,
    *,
    run_install: bool,
    write_report: bool,
) -> Any:
    profile = bindings.find_profile(task)
    if not profile:
        ranking = bindings.rank_options(engine, task, run_install=run_install)
        report = bindings.ranking_to_recommendation(task, ranking)
        if write_report:
            engine.ledger.write_recommendation_report(report)
        return report

    reports = [
        engine.inspect_package(
            item.package,
            task_text=task,
            candidate=item,
            run_install=run_install,
            write_report=False,
        )
        for item in profile.candidates
    ]
    notes = list(profile.notes)
    notes.append("Live evidence was fetched for each candidate before scoring.")
    report = engine._recommendation(
        task=task,
        profile=profile.name,
        candidates=reports,
        notes=notes,
    )
    if write_report:
        engine.ledger.write_recommendation_report(report)
    return report

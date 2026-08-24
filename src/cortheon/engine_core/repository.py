"""Repository scanning, fit analysis, and patch verification workflows."""

from __future__ import annotations

from types import ModuleType
from typing import Any


def verify_patch(
    engine: Any,
    bindings: ModuleType,
    repo_path: str,
    patch_text: str,
    *,
    test_command: str | None,
    run_baseline: bool,
    test_isolation: str,
    sandbox_image: str,
    write_report: bool,
) -> Any:
    report = bindings.run_patch_verification(
        repo_path,
        patch_text,
        test_command=test_command,
        run_baseline=run_baseline,
        test_isolation=test_isolation,
        sandbox_image=sandbox_image,
    )
    if write_report:
        engine.ledger.write_patch_report(report)
    return report


def scan_repository(
    engine: Any,
    bindings: ModuleType,
    path: str,
    *,
    write_report: bool,
) -> Any:
    report = bindings.scan_repo(path)
    if write_report:
        engine.ledger.write_repo_report(report)
    return report


def check_repo_fit(
    engine: Any,
    bindings: ModuleType,
    package: str,
    repo_path: str,
    *,
    write_report: bool,
) -> Any:
    repo = bindings.scan_repo(repo_path)
    errors = list(repo.errors)
    metadata = None
    pypi_evidence: list[Any] = []
    try:
        metadata, pypi_evidence = engine.pypi.fetch(package)
    except bindings.ConnectorError as exc:
        errors.append(str(exc))

    normalized = bindings.normalize_dep_name(package)
    declared = next(
        (
            item
            for item in repo.declared_dependencies
            if bindings.normalize_dep_name(item.name) == normalized
        ),
        None,
    )
    import_name = bindings.guess_import_name(metadata.name if metadata else package).lower()
    already_imported = import_name in {name.lower() for name in repo.imported_third_party}
    python_compatible, python_note = bindings.python_compatibility(
        repo.python_requirement,
        metadata.requires_python if metadata else None,
    )

    notes: list[str] = [python_note]
    risks: list[str] = []
    if declared:
        notes.append(
            f"{package} is already declared in {declared.source}"
            + (
                f" with constraint {declared.constraint!r}."
                if declared.constraint
                else " without a version constraint."
            )
        )
    if already_imported and not declared:
        risks.append(f"{package} is imported by the code but not declared as a dependency.")
    if python_compatible is False:
        risks.append("Adding this package would raise the repository's minimum Python version.")
    if not declared and not already_imported:
        notes.append(f"{package} would be a new dependency for this repository.")

    report = bindings.RepoFitReport(
        package=metadata.name if metadata else package,
        version=metadata.version if metadata else None,
        repo_root=repo.root,
        generated_at=bindings.utc_now(),
        already_declared=declared is not None,
        declared_constraint=declared.constraint if declared else None,
        already_imported=already_imported,
        python_compatible=python_compatible,
        python_note=python_note,
        notes=notes,
        risks=risks,
        evidence=repo.evidence
        + pypi_evidence
        + [
            bindings.Evidence(
                claim=(
                    f"Repo fit for {package} in {repo.root}: "
                    f"declared={declared is not None}, imported={already_imported}, "
                    f"python_compatible={python_compatible}."
                ),
                source_type="repo_fit",
                source_url=None,
                package=metadata.name if metadata else package,
                version=metadata.version if metadata else None,
                support=bindings.SupportLevel.INFERRED,
                details={
                    "repo_root": repo.root,
                    "already_declared": declared is not None,
                    "declared_constraint": declared.constraint if declared else None,
                    "already_imported": already_imported,
                    "python_compatible": python_compatible,
                },
            )
        ],
        errors=errors,
    )
    if write_report:
        engine.ledger.write_repo_fit_report(report)
    return report

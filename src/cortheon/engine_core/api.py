"""Source API, documentation, and generated-code evidence workflows."""

from __future__ import annotations

from types import ModuleType
from typing import Any


def retrieve_api_evidence(
    engine: Any,
    bindings: ModuleType,
    package: str,
    query: str,
    *,
    include_docs: bool,
    write_report: bool,
) -> Any:
    errors: list[str] = []
    try:
        metadata, pypi_evidence = engine.pypi.fetch(package)
    except bindings.ConnectorError as exc:
        return bindings.ApiEvidenceReport(
            package=package,
            version="unknown",
            query=query,
            artifact_filename=None,
            artifact_url=None,
            extracted_at=bindings.utc_now(),
            total_symbols=0,
            matches=[],
            evidence=[],
            errors=[str(exc)],
        )

    report = engine.api_extractor.retrieve(metadata, query)
    report.evidence = pypi_evidence + report.evidence
    report.errors = errors + report.errors
    if include_docs and report.matches:
        docs_report = engine.docs_reader.read(
            metadata,
            max_pages=3,
            max_examples=0,
            guide_keywords=bindings.API_GUIDE_KEYWORDS,
        )
        mention = bindings.find_symbol_mention(docs_report, query)
        if mention:
            docs_page_url, snippet = mention
            report.evidence.append(
                bindings.Evidence(
                    claim=(
                        f"Official documentation for {metadata.name} {metadata.version} "
                        f"mentions {query!r} at {docs_page_url}."
                    ),
                    source_type="official_docs_symbol",
                    source_url=docs_page_url,
                    package=metadata.name,
                    version=metadata.version,
                    support=bindings.SupportLevel.OBSERVED,
                    details={"query": query, "snippet": snippet},
                )
            )
        else:
            report.errors.append(
                "Symbol was not found on the fetched official documentation pages; "
                "it exists in source but may be undocumented or documented elsewhere."
            )
    if write_report:
        engine.ledger.write_api_evidence_report(report)
    return report


def check_generated_code(
    engine: Any,
    bindings: ModuleType,
    package: str,
    code: str,
    *,
    write_report: bool,
) -> Any:
    code_usage_finding, code_usage_report = bindings._code_usage_models()
    errors: list[str] = []
    version: str | None = None
    metadata_name = package
    symbols = []
    try:
        metadata, _ = engine.pypi.fetch(package)
        metadata_name = metadata.name
        version = metadata.version
        _, symbols, load_errors = engine.api_extractor.load_symbols(metadata)
        errors.extend(load_errors)
    except bindings.ConnectorError as exc:
        errors.append(str(exc))

    blocks = bindings.extract_code_blocks(code)
    if not blocks:
        errors.append("No Python code block could be extracted from the input.")
    all_findings: list[Any] = []
    notes: list[str] = []
    parsed_any = False
    checked = 0
    for block in blocks:
        report = bindings.check_api_usage(block, package, symbols)
        parsed_any = parsed_any or report.parsed
        checked += report.checked_calls
        notes.extend(report.notes)
        all_findings.extend(
            code_usage_finding(
                package=finding.package,
                attribute=finding.attribute,
                line=finding.line,
                reason=finding.reason,
                kind=finding.kind,
            )
            for finding in report.findings
        )

    verdict = "block" if (all_findings or (blocks and not parsed_any)) else "allow"
    support = bindings.SupportLevel.FAILED if verdict == "block" else bindings.SupportLevel.VERIFIED
    claim = (
        f"Generated code for {package} {version or ''}: {len(all_findings)} hallucinated API call(s) "
        f"across {checked} checked call(s); verdict {verdict}."
    )
    report = code_usage_report(
        package=metadata_name,
        version=version,
        generated_at=bindings.utc_now(),
        parsed=parsed_any,
        checked_calls=checked,
        verdict=verdict,
        findings=all_findings,
        notes=notes,
        evidence=[
            bindings.Evidence(
                claim=claim,
                source_type="code_api_usage_check",
                source_url=None,
                package=package,
                version=version,
                support=support,
                details={
                    "verdict": verdict,
                    "checked_calls": checked,
                    "findings": [
                        {"attribute": finding.attribute, "line": finding.line}
                        for finding in all_findings
                    ],
                },
            )
        ],
        errors=errors,
    )
    if write_report:
        engine.ledger.write_code_usage_report(report)
    return report


def fetch_docs(
    engine: Any,
    package: str,
    *,
    version: str | None,
    max_pages: int,
    write_report: bool,
) -> Any:
    metadata, pypi_evidence = engine.pypi.fetch(package, version)
    report = engine.docs_reader.read(metadata, max_pages=max_pages, target_version=version)
    report.evidence = pypi_evidence + report.evidence
    if write_report:
        engine.ledger.write_docs_report(report)
    return report


def diff_api(
    engine: Any,
    bindings: ModuleType,
    package: str,
    old_version: str,
    new_version: str,
    *,
    write_report: bool,
) -> Any:
    errors: list[str] = []
    pypi_evidence: list[Any] = []
    old_symbols: list[Any] = []
    new_symbols: list[Any] = []
    old_artifact = None
    new_artifact = None
    resolved_name = package

    try:
        old_metadata, items = engine.pypi.fetch(package, old_version)
        pypi_evidence.extend(items)
        resolved_name = old_metadata.name
        old_artifact, old_symbols, old_errors = engine.api_extractor.load_symbols(old_metadata)
        errors.extend(f"{old_version}: {item}" for item in old_errors)
    except bindings.ConnectorError as exc:
        errors.append(f"{old_version}: {exc}")

    try:
        new_metadata, items = engine.pypi.fetch(package, new_version)
        pypi_evidence.extend(items)
        resolved_name = new_metadata.name
        new_artifact, new_symbols, new_errors = engine.api_extractor.load_symbols(new_metadata)
        errors.extend(f"{new_version}: {item}" for item in new_errors)
    except bindings.ConnectorError as exc:
        errors.append(f"{new_version}: {exc}")

    report = bindings.build_api_diff_report(
        resolved_name,
        old_version,
        new_version,
        old_symbols,
        new_symbols,
        old_artifact=old_artifact,
        new_artifact=new_artifact,
        errors=errors,
    )
    report.evidence = pypi_evidence + report.evidence
    if write_report:
        engine.ledger.write_api_diff_report(report)
    return report

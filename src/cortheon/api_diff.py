from __future__ import annotations

from cortheon.models import (
    ApiDiffReport,
    ApiSymbol,
    ApiSymbolChange,
    DistributionArtifact,
    Evidence,
    SupportLevel,
    utc_now,
)

# Reports cap listed symbols so a v1 -> v2 rewrite (thousands of moved symbols)
# stays readable; the *_count fields always carry the full totals.
MAX_LISTED_SYMBOLS = 200


def diff_symbols(
    old_symbols: list[ApiSymbol],
    new_symbols: list[ApiSymbol],
) -> tuple[list[ApiSymbol], list[ApiSymbol], list[ApiSymbolChange], list[ApiSymbol]]:
    old_map = _by_qualname(old_symbols)
    new_map = _by_qualname(new_symbols)
    added = sorted(
        (symbol for qualname, symbol in new_map.items() if qualname not in old_map),
        key=lambda symbol: symbol.qualname,
    )
    removed = sorted(
        (symbol for qualname, symbol in old_map.items() if qualname not in new_map),
        key=lambda symbol: symbol.qualname,
    )
    changed = sorted(
        (
            ApiSymbolChange(
                qualname=qualname,
                kind=new_map[qualname].kind,
                old_signature=old_map[qualname].signature,
                new_signature=new_map[qualname].signature,
            )
            for qualname in new_map
            if qualname in old_map
            and old_map[qualname].signature
            and new_map[qualname].signature
            and old_map[qualname].signature != new_map[qualname].signature
        ),
        key=lambda change: change.qualname,
    )
    deprecated_in_new = sorted(
        (symbol for symbol in new_map.values() if symbol.deprecated),
        key=lambda symbol: symbol.qualname,
    )
    return added, removed, changed, deprecated_in_new


def build_api_diff_report(
    package: str,
    old_version: str,
    new_version: str,
    old_symbols: list[ApiSymbol],
    new_symbols: list[ApiSymbol],
    *,
    old_artifact: DistributionArtifact | None = None,
    new_artifact: DistributionArtifact | None = None,
    errors: list[str] | None = None,
) -> ApiDiffReport:
    added, removed, changed, deprecated_in_new = diff_symbols(old_symbols, new_symbols)
    parsed_both = bool(old_symbols) and bool(new_symbols)
    evidence = [
        Evidence(
            claim=(
                f"{package} {old_version} -> {new_version}: {len(added)} added, {len(removed)} removed, "
                f"{len(changed)} signature-changed, and {len(deprecated_in_new)} deprecated public symbol(s), "
                "derived from source artifacts without importing the package."
            ),
            source_type="source_artifact_api_diff",
            source_url=new_artifact.url if new_artifact else None,
            package=package,
            version=new_version,
            support=SupportLevel.VERIFIED if parsed_both else SupportLevel.FAILED,
            details={
                "old_version": old_version,
                "new_version": new_version,
                "old_artifact": old_artifact.filename if old_artifact else None,
                "new_artifact": new_artifact.filename if new_artifact else None,
                "added": len(added),
                "removed": len(removed),
                "changed": len(changed),
                "deprecated": len(deprecated_in_new),
            },
        )
    ]
    return ApiDiffReport(
        package=package,
        old_version=old_version,
        new_version=new_version,
        generated_at=utc_now(),
        old_total_symbols=len(old_symbols),
        new_total_symbols=len(new_symbols),
        added_count=len(added),
        removed_count=len(removed),
        changed_count=len(changed),
        deprecated_count=len(deprecated_in_new),
        added=added[:MAX_LISTED_SYMBOLS],
        removed=removed[:MAX_LISTED_SYMBOLS],
        changed=changed[:MAX_LISTED_SYMBOLS],
        deprecated_in_new=deprecated_in_new[:MAX_LISTED_SYMBOLS],
        evidence=evidence,
        errors=list(errors or []),
    )


def _by_qualname(symbols: list[ApiSymbol]) -> dict[str, ApiSymbol]:
    mapped: dict[str, ApiSymbol] = {}
    for symbol in symbols:
        mapped.setdefault(symbol.qualname, symbol)
    return mapped

"""Construction of bounded current-source facts for a model prompt."""

from __future__ import annotations

from typing import Any


def build_evidence(
    engine: Any,
    text: str,
    packages: list[str],
    *,
    token_pattern: Any,
    comparison_base_version: Any,
    public_added_names: Any,
    is_legacy_path: Any,
    constructor_params: Any,
    official_recovery_facts: Any,
    bound_names: Any,
    suggest_replacements: Any,
    callable_param_index: Any,
    max_evidence_chars: int,
) -> tuple[str, dict[str, Any]]:
    mentioned: list[str] = []
    for token in token_pattern.findall(text):
        if token not in mentioned:
            mentioned.append(token)

    lines: list[str] = []
    versions: dict[str, str] = {}
    comparison_base = comparison_base_version(text)
    for package in packages:
        try:
            metadata, _ = engine.pypi.fetch(package)
            _, symbols, _ = engine.api_extractor.load_symbols(metadata)
        except Exception:
            continue
        versions[metadata.name] = metadata.version
        lines.append(
            f"VERIFIED: the current version of {metadata.name} is {metadata.version}. "
            "Your training memory has an older version. Use this one."
        )
        if comparison_base and comparison_base != metadata.version:
            try:
                diff = engine.diff_api(
                    package,
                    comparison_base,
                    metadata.version,
                    write_report=False,
                )
                added = public_added_names(diff.added)
                if added:
                    lines.append(
                        f"VERIFIED SOURCE DIFF: public API additions in {metadata.name} "
                        f"since {comparison_base}: {', '.join(added[:16])}. "
                        "These names come from comparing the published source artifacts."
                    )
            except Exception:
                pass
        if not symbols:
            continue

        live: dict[str, Any] = {}
        public_shorts: set[str] = set()
        legacy: dict[str, str] = {}
        all_parts: set[str] = set()

        def rank(qualname: str) -> tuple[bool, int]:
            parts = qualname.split(".")
            return (not any(part.startswith("_") for part in parts[:-1]), -len(parts))

        for symbol in symbols:
            parts = symbol.qualname.split(".")
            all_parts.update(parts)
            short = parts[-1]
            if short.startswith("_"):
                continue
            if symbol.deprecated or is_legacy_path(symbol.qualname):
                legacy.setdefault(short, symbol.qualname)
                continue
            if rank(symbol.qualname)[0]:
                public_shorts.add(short)
            if short not in live or rank(symbol.qualname) > rank(live[short].qualname):
                live[short] = symbol
        legacy = {short: qualname for short, qualname in legacy.items() if short not in live}

        shown = 0
        usage_docs_added = False
        for token in mentioned:
            if shown >= 4:
                break
            live_name = token
            symbol = live.get(live_name)
            if symbol is None:
                lowered = token.lower()
                live_name = next(
                    (
                        name
                        for name in live
                        if len(name) >= 4
                        and lowered
                        in {
                            f"{name.lower()}s",
                            f"{name.lower()}es",
                            f"{name.lower()}ed",
                            f"{name.lower()}ing",
                        }
                    ),
                    token,
                )
                symbol = live.get(live_name)
            if symbol is None:
                continue
            if symbol.kind == "class":
                params = constructor_params(symbol.qualname, symbols)
                if params:
                    lines.append(
                        f"VERIFIED: {symbol.qualname}{params} — these are the exact "
                        "constructor parameters. Use them. Do not invent others."
                    )
                else:
                    lines.append(f"VERIFIED: {symbol.qualname} exists (class).")
            elif symbol.signature:
                lines.append(
                    f"VERIFIED: {symbol.qualname} — exact signature: {symbol.signature}. "
                    "Use this. Do not invent a different one."
                )
                if not usage_docs_added:
                    recovery = official_recovery_facts(engine, package, [live_name])
                    if recovery:
                        preferred = next(
                            (fact for fact in recovery if ".iter_bytes()" in fact and "\n" in fact),
                            recovery[-1],
                        )
                        lines.append(preferred)
                        usage_docs_added = True
            else:
                lines.append(f"VERIFIED: {symbol.qualname} exists ({symbol.kind}).")
            shown += 1

        fates = 0
        for token in mentioned:
            if fates >= 3:
                break
            if token not in legacy:
                continue
            replacements = suggest_replacements(token, public_shorts or set(live))
            hint = f" — use instead: {', '.join(replacements)}" if replacements else ""
            lines.append(
                f"VERIFIED: {token!r} is DEPRECATED in the current version. "
                f"It still exists but will be removed{hint}."
            )
            fates += 1

        ghosts = 0
        recovery_docs_added = False
        for token in bound_names(text, package):
            if ghosts >= 3:
                break
            if token in all_parts:
                continue
            replacements = suggest_replacements(token, public_shorts or set(live))
            alternative = ""
            if replacements:
                descriptions: list[str] = []
                for replacement in replacements[:2]:
                    candidate = live.get(replacement)
                    if candidate is not None and candidate.signature:
                        descriptions.append(
                            f"{candidate.qualname} with signature {candidate.signature}"
                        )
                    elif candidate is not None:
                        descriptions.append(candidate.qualname)
                    else:
                        descriptions.append(replacement)
                alternative = (
                    " Closest verified current alternative"
                    + ("s are: " if len(descriptions) > 1 else " is: ")
                    + "; ".join(descriptions)
                    + "."
                )
            lines.append(
                f"VERIFIED: {token!r} does NOT exist in the current source. "
                f"Do not import or call it.{alternative}"
            )
            if replacements and not recovery_docs_added:
                recovery = official_recovery_facts(engine, package, replacements)
                if recovery:
                    lines.extend(recovery)
                    recovery_docs_added = True
            ghosts += 1

        param_index = callable_param_index(symbols)
        keywords = 0
        for token in mentioned:
            if keywords >= 2:
                break
            if not token.islower() or len(token) < 4 or token in live or token in legacy:
                continue
            owners = sorted(
                short
                for short, candidates in param_index.items()
                if not short.startswith("_") and any(token in params for params, _ in candidates)
            )
            if owners and len(owners) <= 8:
                lines.append(
                    f"VERIFIED: {token!r} is accepted by "
                    f"{', '.join(owners[:4])}; no other verified callable takes it."
                )
                keywords += 1

    kept: list[str] = []
    total = 0
    for line in lines:
        if total + len(line) > max_evidence_chars:
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept), {"packages": versions, "facts": len(kept)}

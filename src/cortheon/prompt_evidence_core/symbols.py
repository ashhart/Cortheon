"""Symbol helpers and bounded official-documentation recovery facts."""

from __future__ import annotations

from typing import Any


def constructor_params(qualname: str, symbols: list[Any]) -> str | None:
    for symbol in symbols:
        if symbol.qualname == f"{qualname}.__init__" and symbol.signature:
            params = symbol.signature.removeprefix("__init__")
            if params.startswith("(self, "):
                return "(" + params[len("(self, ") :]
            if params.startswith("(self)"):
                return "()" + params[len("(self)") :]
            return params
    return None


def comparison_base_version(text: str, *, pattern: Any) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def public_added_names(symbols: list[Any]) -> list[str]:
    names: list[str] = []
    for symbol in symbols:
        parts = symbol.qualname.split(".")
        if any(part.startswith("_") for part in parts):
            continue
        short = parts[-1]
        if short not in names:
            names.append(short)
    return names


def official_recovery_facts(
    engine: Any,
    package: str,
    replacements: list[str],
    *,
    guide_keywords: tuple[str, ...],
) -> list[str]:
    if not replacements:
        return []
    try:
        metadata, _ = engine.pypi.fetch(package)
        report = engine.docs_reader.read(
            metadata,
            max_pages=12,
            guide_keywords=(
                *replacements,
                "stream",
                "client",
                "advanced",
                *guide_keywords,
            ),
        )
    except Exception:
        try:
            report = engine.fetch_docs(package, max_pages=12, write_report=False)
        except Exception:
            return []
    lowered = [name.lower() for name in replacements]
    facts: list[str] = []
    for page in report.pages:
        for block in page.code_blocks:
            block_lower = block.lower()
            if not any(
                f".{name}(" in block_lower or f"{package.lower()}.{name}(" in block_lower
                for name in lowered
            ):
                continue
            compact = block.strip()
            if compact and len(compact) <= 900:
                facts.append(f"VERIFIED OFFICIAL DOCS USAGE SHAPE ({page.final_url}):\n{compact}")
                break
        text_lower = page.text.lower()
        marker_positions = [
            text_lower.find(f".{name}()") for name in lowered if text_lower.find(f".{name}()") >= 0
        ]
        if marker_positions:
            iter_bytes = text_lower.find(".iter_bytes()")
            anchor = (
                iter_bytes if "stream" in lowered and iter_bytes >= 0 else min(marker_positions)
            )
            start = max(0, anchor - 180)
            end = min(len(page.text), anchor + 300)
            excerpt = " ".join(page.text[start:end].split())[:420]
            if excerpt:
                facts.append(f"VERIFIED OFFICIAL DOCS GUIDANCE ({page.final_url}): {excerpt}")
        if facts:
            return facts[:2]
    return []

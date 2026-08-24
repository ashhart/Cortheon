from __future__ import annotations

from cortheon.docs_reader_core._compat import facade
from cortheon.models import DocsExample, DocsPage, DocsSiteReport


def runnable_examples(
    page: DocsPage,
    code_blocks: list[str],
    import_names: list[str],
    limit: int,
) -> list[DocsExample]:
    if limit <= 0 or page.error:
        return []
    api = facade()
    examples: list[DocsExample] = []
    for block in code_blocks:
        code = block.strip()
        if ">>>" in code:
            code = api.doctest_to_script(code) or ""
        if not code or len(code) > api.MAX_EXAMPLE_CHARS:
            continue
        if not api.is_runnable_candidate(code, import_names):
            continue
        examples.append(api.DocsExample(page_url=page.final_url, code=code))
        if len(examples) >= limit:
            break
    return examples


def find_symbol_mention(report: DocsSiteReport, query: str) -> tuple[str, str] | None:
    """Locate the docs page (and snippet) that documents a symbol query.

    This is the first docs-to-symbol link: source artifacts prove the symbol
    exists; the docs mention proves it is documented, current, and shows where
    to read about it.
    """
    needle = query.strip()
    if not needle:
        return None
    tail = needle.rsplit(".", 1)[-1]
    # Docs renderers tokenize signatures ("httpx. request ( method , url ...")
    # so matching must tolerate whitespace around dots and parens. The full
    # dotted query is preferred across all pages before the tail fallback.
    api = facade()
    patterns = [
        api.re.compile(
            r"\s*\.\s*".join(api.re.escape(part) for part in needle.split(".")), api.re.IGNORECASE
        )
    ]
    if len(tail) >= 4:
        patterns.append(api.re.compile(rf"\b{api.re.escape(tail)}\s*\(", api.re.IGNORECASE))
    for pattern in patterns:
        for page in report.pages:
            if page.error:
                continue
            # Reference pages render signatures inside <pre>, so code blocks
            # are first-class search targets alongside prose and headings.
            for segment in (page.text, " ".join(page.headings), *page.code_blocks):
                if not segment:
                    continue
                found = pattern.search(segment)
                if not found:
                    continue
                start = max(0, found.start() - 120)
                end = min(len(segment), found.start() + 160)
                return page.final_url, api.normalize_space(segment[start:end])
    return None


def error_docs_page(url: str, error: str, is_changelog: bool) -> DocsPage:
    api = facade()
    return api.DocsPage(
        url=url,
        final_url=url,
        title=None,
        fetched_at=api.utc_now(),
        text="",
        headings=[],
        code_block_count=0,
        is_changelog=is_changelog,
        error=error,
    )

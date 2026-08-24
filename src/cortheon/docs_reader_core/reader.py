from __future__ import annotations

import urllib.robotparser

from cortheon.connectors.http import JsonHttpClient
from cortheon.docs_reader_core._compat import facade
from cortheon.docs_reader_core.constants import GUIDE_KEYWORDS
from cortheon.models import DocsExample, DocsPage, DocsSiteReport, PackageMetadata


class DocsSiteReader:
    """Fetch and parse a bounded set of official documentation pages.

    The harness stays online; this reader is how the substrate learns what the
    docs *say right now* — quickstart examples, API prose, and the changelog
    head — instead of relying on whatever the model remembers.
    """

    def __init__(self, client: JsonHttpClient | None = None, obey_robots: bool = True) -> None:
        api = facade()
        self.client = client or api.JsonHttpClient(
            timeout_seconds=20,
            url_validator=api.validate_public_http_url,
        )
        self.obey_robots = obey_robots
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}

    def read(
        self,
        metadata: PackageMetadata,
        *,
        max_pages: int = 4,
        max_examples: int = 3,
        guide_keywords: tuple[str, ...] = GUIDE_KEYWORDS,
        target_version: str | None = None,
    ) -> DocsSiteReport:
        api = facade()
        docs_url = api.resolve_docs_url(metadata.project_urls)
        changelog_url = api.pick_url(metadata.project_urls, api.CHANGELOG_LABELS)
        pages: list[DocsPage] = []
        examples: list[DocsExample] = []
        errors: list[str] = []
        import_names = [api.guess_import_name(metadata.name)]
        budget = max(1, max_pages)

        docs_version_match: str | None = None
        if target_version and docs_url:
            docs_url, docs_version_match = self._resolve_versioned_docs(docs_url, target_version)
        elif target_version:
            docs_version_match = "unversioned"

        if not docs_url and not changelog_url:
            errors.append("Package metadata declares no documentation or changelog URL.")

        root_links: list[tuple[str, str]] = []
        if docs_url:
            page, links, blocks = self._fetch_page(docs_url)
            pages.append(page)
            root_links = links
            examples.extend(
                api.runnable_examples(page, blocks, import_names, max_examples - len(examples))
            )

        if docs_url and not pages[-1].error:
            guide_budget = budget - len(pages) - (1 if changelog_url else 0)
            for guide_url in api.select_guide_links(
                root_links, docs_url, max(0, guide_budget), keywords=guide_keywords
            ):
                if docs_version_match in {"exact", "major_minor"}:
                    # Absolute-path nav links escape the versioned subtree and
                    # land on latest docs — the exact hazard we resolved away.
                    # Deliberately no fallback: a 404 error entry beats
                    # silently reading wrong-version content.
                    guide_url = api.prefer_versioned_link(guide_url, docs_url)
                page, _, blocks = self._fetch_page(guide_url)
                pages.append(page)
                examples.extend(
                    api.runnable_examples(page, blocks, import_names, max_examples - len(examples))
                )

        changelog_head: str | None = None
        if changelog_url and len(pages) < budget:
            page, _, _ = self._fetch_page(changelog_url, is_changelog=True)
            pages.append(page)
            if not page.error and page.text:
                changelog_head = page.text[: api.MAX_CHANGELOG_HEAD_CHARS]

        errors.extend(f"{page.url}: {page.error}" for page in pages if page.error)
        fetched = [page for page in pages if not page.error]
        quarantined = sum(page.quarantined_segments for page in pages)
        evidence = [
            api.Evidence(
                claim=(
                    f"Fetched {len(fetched)} official documentation page(s) for {metadata.name} "
                    f"{metadata.version} with {len(examples)} runnable example(s)."
                ),
                source_type="official_docs",
                source_url=docs_url or changelog_url,
                package=metadata.name,
                version=metadata.version,
                support=api.SupportLevel.OBSERVED if fetched else api.SupportLevel.FAILED,
                details={
                    "docs_url": docs_url,
                    "changelog_url": changelog_url,
                    "page_urls": [page.final_url for page in fetched],
                    "example_count": len(examples),
                    "quarantined_segments": quarantined,
                },
            )
        ]
        if changelog_head:
            evidence.append(
                api.Evidence(
                    claim=f"Changelog head for {metadata.name} {metadata.version} was fetched from {changelog_url}.",
                    source_type="official_changelog",
                    source_url=changelog_url,
                    package=metadata.name,
                    version=metadata.version,
                    support=api.SupportLevel.OBSERVED,
                    details={"head": changelog_head[:240]},
                )
            )
        if target_version and docs_url:
            if docs_version_match in {"exact", "major_minor"}:
                evidence.append(
                    api.Evidence(
                        claim=(
                            f"Documentation for {metadata.name} was resolved to a version-matched URL "
                            f"{docs_url} ({docs_version_match}) for requested version {target_version}."
                        ),
                        source_type="official_docs_version",
                        source_url=docs_url,
                        package=metadata.name,
                        version=target_version,
                        support=api.SupportLevel.OBSERVED,
                        details={"match": docs_version_match, "requested_version": target_version},
                    )
                )
            else:
                # Serving latest docs for a pinned version is a real hazard for
                # a small model; the mismatch must be explicit evidence, not a
                # silent substitution.
                evidence.append(
                    api.Evidence(
                        claim=(
                            f"Requested documentation for {metadata.name} {target_version}, but only "
                            f"unversioned/current docs at {docs_url} were reachable; content may "
                            "describe a different version."
                        ),
                        source_type="official_docs_version",
                        source_url=docs_url,
                        package=metadata.name,
                        version=target_version,
                        support=api.SupportLevel.INFERRED,
                        details={"match": "unversioned", "requested_version": target_version},
                    )
                )
        return api.DocsSiteReport(
            package=metadata.name,
            version=metadata.version,
            generated_at=api.utc_now(),
            docs_url=docs_url,
            changelog_url=changelog_url,
            pages=pages,
            examples=examples,
            changelog_head=changelog_head,
            evidence=evidence,
            errors=errors,
            requested_version=target_version,
            docs_version_match=docs_version_match,
        )

    def _resolve_versioned_docs(self, docs_url: str, target_version: str) -> tuple[str, str]:
        api = facade()
        detected = api.detect_version_in_url(docs_url, target_version)
        if detected:
            return docs_url, detected
        for candidate_url, kind in api.versioned_docs_candidates(docs_url, target_version):
            status = self.client.head_or_get_status(candidate_url)
            if status and 200 <= status < 400:
                return candidate_url, kind
        return docs_url, "unversioned"

    def _fetch_page(
        self, url: str, is_changelog: bool = False
    ) -> tuple[DocsPage, list[tuple[str, str]], list[str]]:
        api = facade()
        fetch_url = api.raw_github_url(url)
        if self.obey_robots and not self._allowed_by_robots(fetch_url):
            return api.error_docs_page(url, "Blocked by robots.txt", is_changelog), [], []
        try:
            response = self.client.get(
                fetch_url,
                headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.8,*/*;q=0.5"
                },
            )
        except api.ConnectorError as exc:
            return api.error_docs_page(url, str(exc), is_changelog), [], []
        charset = api.charset_from_content_type(response.headers.get("Content-Type", ""))
        raw = response.body.decode(charset, errors="replace")
        content_type = response.headers.get("Content-Type", "").lower()
        if "html" not in content_type and not raw.lstrip().startswith("<"):
            # Raw markdown/plain files (e.g. a GitHub raw CHANGELOG.md) skip the
            # HTML parser entirely.
            scan = api.scan_text(api.normalize_space(raw))
            page = api.DocsPage(
                url=url,
                final_url=response.url,
                title=None,
                fetched_at=api.utc_now(),
                text=scan.clean_text[: api.MAX_PAGE_TEXT_CHARS],
                headings=[],
                code_block_count=0,
                quarantined_segments=len(scan.flags),
                is_changelog=is_changelog,
            )
            return page, [], []
        parser = api.DocsHtmlParser(base_url=response.url)
        parser.feed(raw)
        parser.close()
        # Fetched docs are data, never instructions: quarantine before the text
        # can reach claims, summaries, or evidence details.
        scan = api.scan_text(api.normalize_space(" ".join(parser.text_parts)))
        kept_blocks: list[str] = []
        quarantined_blocks = 0
        for block in parser.code_blocks[: api.MAX_CODE_BLOCKS_PER_PAGE]:
            # Injection can hide in code comments too; drop whole blocks that
            # carry instruction-shaped text rather than mangling the code.
            if api.injection_flags(block):
                quarantined_blocks += 1
                continue
            kept_blocks.append(block[: api.MAX_CODE_BLOCK_CHARS])
        page = api.DocsPage(
            url=url,
            final_url=response.url,
            title=parser.title,
            fetched_at=api.utc_now(),
            text=scan.clean_text[: api.MAX_PAGE_TEXT_CHARS],
            headings=parser.headings[:20],
            code_block_count=len(parser.code_blocks),
            code_blocks=kept_blocks,
            quarantined_segments=len(scan.flags) + quarantined_blocks,
            is_changelog=is_changelog,
        )
        return page, parser.links, parser.code_blocks

    def _allowed_by_robots(self, url: str) -> bool:
        api = facade()
        parsed = api.urllib.parse.urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        robot = self._robots.get(root)
        if robot is None:
            robot = api.fetch_robots(self.client, root)
            self._robots[root] = robot
        return robot.can_fetch("cortheon/0.1", url)

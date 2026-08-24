from __future__ import annotations

import html
import re
import urllib.parse
import urllib.robotparser
from collections import deque
from dataclasses import dataclass
from html.parser import HTMLParser

from cortheon.connectors.http import (
    ConnectorError,
    JsonHttpClient,
    validate_public_http_url,
)
from cortheon.models import CrawledPage, Evidence, SupportLevel, utc_now

ASSET_EXTENSIONS = {
    ".7z",
    ".avi",
    ".css",
    ".csv",
    ".dmg",
    ".doc",
    ".docx",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".svg",
    ".tar",
    ".tgz",
    ".wav",
    ".xls",
    ".xlsx",
    ".zip",
}


@dataclass(slots=True)
class CrawlBudget:
    max_pages: int = 20
    max_depth: int = 1
    max_links_per_page: int = 25
    max_text_chars: int = 12000
    obey_robots: bool = True


class WebCrawler:
    def __init__(self, client: JsonHttpClient | None = None) -> None:
        self.client = client or JsonHttpClient(
            timeout_seconds=20,
            url_validator=validate_public_http_url,
        )
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}

    def crawl(
        self,
        seed_urls: list[str],
        *,
        allowed_domains: list[str] | None = None,
        budget: CrawlBudget | None = None,
    ) -> tuple[list[CrawledPage], list[Evidence]]:
        budget = budget or CrawlBudget()
        allowed = {normalize_domain(domain) for domain in allowed_domains or [] if domain}
        queue: deque[tuple[str, int, str]] = deque()
        seen: set[str] = set()
        for seed in seed_urls:
            normalized = normalize_url(seed)
            if normalized:
                queue.append((normalized, 0, domain_of(normalized)))
        pages: list[CrawledPage] = []

        while queue and len(pages) < budget.max_pages:
            url, depth, origin_domain = queue.popleft()
            if url in seen:
                continue
            seen.add(url)
            if allowed and domain_of(url) not in allowed:
                continue
            if budget.obey_robots and not self._allowed_by_robots(url):
                pages.append(error_page(url, "Blocked by robots.txt"))
                continue

            page = self.fetch_page(url, budget.max_text_chars)
            pages.append(page)
            if page.error or depth >= budget.max_depth:
                continue
            scheduled = 0
            for link in page.links:
                if scheduled >= budget.max_links_per_page:
                    break
                normalized = normalize_url(link, base=page.final_url)
                if not normalized or normalized in seen:
                    continue
                if allowed and domain_of(normalized) not in allowed:
                    continue
                if domain_of(normalized) != origin_domain:
                    continue
                queue.append((normalized, depth + 1, origin_domain))
                scheduled += 1

        evidence = [
            Evidence(
                claim=f"Crawled {len([page for page in pages if not page.error])} page(s) from {len(seed_urls)} seed URL(s).",
                source_type="web_crawl",
                source_url=None,
                support=SupportLevel.OBSERVED,
                details={
                    "seed_count": len(seed_urls),
                    "page_count": len(pages),
                    "max_pages": budget.max_pages,
                    "max_depth": budget.max_depth,
                },
            )
        ]
        return pages, evidence

    def fetch_page(self, url: str, max_text_chars: int) -> CrawledPage:
        try:
            response = self.client.get(
                url,
                headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.8,*/*;q=0.5",
                },
            )
            charset = charset_from_content_type(response.headers.get("Content-Type", ""))
            text = response.body.decode(charset, errors="replace")
            parsed = HtmlTextExtractor(base_url=response.url)
            parsed.feed(text)
            parsed.close()
            clean_text = normalize_space("\n".join(parsed.text_parts))[:max_text_chars]
            return CrawledPage(
                url=url,
                final_url=response.url,
                status=response.status,
                title=parsed.title,
                text=clean_text,
                links=dedupe(parsed.links),
                source_type=classify_source(response.url, parsed.title, clean_text),
                authority_score=authority_score(response.url, parsed.title, clean_text),
                fetched_at=utc_now(),
            )
        except ConnectorError as exc:
            return error_page(url, str(exc))

    def _allowed_by_robots(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        robot = self._robots.get(root)
        if robot is None:
            # RobotFileParser.read() uses the Python-urllib UA (often 403'd by
            # CDNs) and silently turns 401/403 into disallow-all; RFC 9309 says
            # unreachable robots.txt means allowed. Fetch with our own client.
            robot = urllib.robotparser.RobotFileParser()
            try:
                response = self.client.get(
                    f"{root}/robots.txt", headers={"Accept": "text/plain,*/*;q=0.8"}
                )
                robot.parse(response.body.decode("utf-8", errors="replace").splitlines())
            except ConnectorError:
                robot.parse([])
            self._robots[root] = robot
        return robot.can_fetch("cortheon/0.1", url)


class HtmlTextExtractor(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title: str | None = None
        self.links: list[str] = []
        self.text_parts: list[str] = []
        self._tag_stack: list[str] = []
        self._in_title = False
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self._tag_stack.append(tag)
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                normalized = normalize_url(href, base=self.base_url)
                if normalized:
                    self.links.append(normalized)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
            self.title = normalize_space(" ".join(self._title_parts))[:240] or None
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if not data.strip():
            return
        if self._tag_stack and self._tag_stack[-1] in {"script", "style", "noscript", "svg"}:
            return
        cleaned = html.unescape(data).strip()
        if self._in_title:
            self._title_parts.append(cleaned)
        elif cleaned:
            self.text_parts.append(cleaned)


def normalize_url(url: str, base: str | None = None) -> str | None:
    resolved = urllib.parse.urljoin(base, url) if base else url
    parsed = urllib.parse.urlparse(resolved)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    path = parsed.path or "/"
    if any(path.lower().endswith(ext) for ext in ASSET_EXTENSIONS):
        return None
    return urllib.parse.urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            "",
            parsed.query,
            "",
        )
    )


def domain_of(url: str) -> str:
    return normalize_domain(urllib.parse.urlparse(url).netloc)


def normalize_domain(domain: str) -> str:
    return domain.lower().removeprefix("www.")


def classify_source(url: str, title: str | None, text: str) -> str:
    lower_url = url.lower()
    lower = f"{title or ''} {text[:1000]}".lower()
    if "github.com" in lower_url:
        return "source_repository"
    if "arxiv.org" in lower_url or "doi.org" in lower_url or "pubmed.ncbi.nlm.nih.gov" in lower_url:
        return "paper"
    if any(
        domain in lower_url
        for domain in ("nih.gov", "who.int", "cdc.gov", "ema.europa.eu", "fda.gov")
    ):
        return "official_health_authority"
    if "docs." in lower_url or "/docs" in lower_url or "documentation" in lower:
        return "official_docs"
    if "benchmark" in lower or "leaderboard" in lower:
        return "benchmark"
    if "blog" in lower_url or "medium.com" in lower_url or "substack.com" in lower_url:
        return "blog"
    return "web_page"


def authority_score(url: str, title: str | None, text: str) -> float:
    source_type = classify_source(url, title, text)
    base = {
        "official_health_authority": 0.96,
        "paper": 0.9,
        "official_docs": 0.86,
        "source_repository": 0.82,
        "benchmark": 0.78,
        "web_page": 0.5,
        "blog": 0.38,
    }.get(source_type, 0.45)
    if len(text) > 2000:
        base += 0.04
    if "citation" in text.lower() or "references" in text.lower():
        base += 0.03
    return round(min(base, 1.0), 3)


def error_page(url: str, error: str) -> CrawledPage:
    return CrawledPage(
        url=url,
        final_url=url,
        status=None,
        title=None,
        text="",
        links=[],
        source_type="error",
        authority_score=0.0,
        fetched_at=utc_now(),
        error=error,
    )


def charset_from_content_type(content_type: str) -> str:
    match = re.search(r"charset=([^\s;]+)", content_type, flags=re.IGNORECASE)
    return match.group(1).strip("\"'") if match else "utf-8"


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result

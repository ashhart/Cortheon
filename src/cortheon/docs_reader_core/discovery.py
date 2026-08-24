from __future__ import annotations

import urllib.robotparser

from cortheon.connectors.http import JsonHttpClient
from cortheon.docs_reader_core._compat import facade
from cortheon.docs_reader_core.constants import GUIDE_KEYWORDS


def version_variants(version: str) -> list[tuple[str, str]]:
    ver = version.strip()
    variants: list[tuple[str, str]] = [(ver, "exact"), (f"v{ver}", "exact")]
    if ver.count(".") >= 2:
        major_minor = ".".join(ver.split(".")[:2])
        if major_minor != ver:
            variants.extend([(major_minor, "major_minor"), (f"v{major_minor}", "major_minor")])
    return variants


def detect_version_in_url(docs_url: str, version: str) -> str | None:
    segments = {
        segment for segment in facade().urllib.parse.urlparse(docs_url).path.split("/") if segment
    }
    for segment_value, kind in facade().version_variants(version):
        if segment_value in segments:
            return kind
    return None


def versioned_docs_candidates(docs_url: str, version: str) -> list[tuple[str, str]]:
    """Derive candidate version-matched docs URLs from an unversioned root.

    Covers the two dominant hosting conventions: ReadTheDocs (/en/{ver}/) and
    mike/path-segment sites ({root}/{ver}/ or {root}/{major.minor}/). Exact
    version candidates are preferred over major.minor ones.
    """
    parsed = facade().urllib.parse.urlparse(docs_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path
    candidates: list[tuple[str, str]] = []
    rtd_style = parsed.netloc.lower().endswith("readthedocs.io") or "/en/" in path
    for segment_value, kind in facade().version_variants(version):
        if rtd_style:
            prefix = path[: path.index("/en/")] if "/en/" in path else ""
            candidates.append((f"{root}{prefix}/en/{segment_value}/", kind))
        candidates.append((f"{root}/{segment_value}/", kind))
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for url, kind in candidates:
        if url in seen:
            continue
        seen.add(url)
        unique.append((url, kind))
    unique.sort(key=lambda item: 0 if item[1] == "exact" else 1)
    return unique[: facade().MAX_VERSION_PROBES]


def fetch_robots(client: JsonHttpClient, root: str) -> urllib.robotparser.RobotFileParser:
    """Fetch robots.txt with our own client and User-Agent.

    RobotFileParser.read() uses the default Python-urllib UA, which CDNs often
    403 — and robotparser silently converts 401/403 into disallow-all. RFC 9309
    says an unreachable/4xx robots.txt means crawling is ALLOWED, so fetch
    failures parse as empty rules.
    """
    api = facade()
    robot = api.urllib.robotparser.RobotFileParser()
    try:
        response = client.get(f"{root}/robots.txt", headers={"Accept": "text/plain,*/*;q=0.8"})
        robot.parse(response.body.decode("utf-8", errors="replace").splitlines())
    except api.ConnectorError:
        robot.parse([])
    return robot


def prefer_versioned_link(link: str, versioned_root: str) -> str:
    parsed_root = facade().urllib.parse.urlparse(versioned_root)
    parsed_link = facade().urllib.parse.urlparse(link)
    if parsed_link.netloc.lower() != parsed_root.netloc.lower():
        return link
    root_path = parsed_root.path if parsed_root.path.endswith("/") else f"{parsed_root.path}/"
    if not root_path.strip("/") or parsed_link.path.startswith(root_path):
        return link
    return f"{parsed_root.scheme}://{parsed_root.netloc}{root_path.rstrip('/')}{parsed_link.path}"


def resolve_docs_url(project_urls: dict[str, str]) -> str | None:
    api = facade()
    docs_url = api.pick_url(project_urls, api.DOCS_LABELS)
    if docs_url:
        return docs_url
    homepage = api.pick_url(project_urls, ("Homepage", "Home", "Website"))
    if homepage and api.looks_like_docs(homepage):
        return homepage
    return None


def looks_like_docs(url: str) -> bool:
    lower = url.lower()
    return "docs" in lower or "readthedocs" in lower or "documentation" in lower


def select_guide_links(
    links: list[tuple[str, str]],
    docs_url: str,
    limit: int,
    keywords: tuple[str, ...] = GUIDE_KEYWORDS,
) -> list[str]:
    if limit <= 0:
        return []
    api = facade()
    docs_domain = api.domain_of(docs_url)
    scored: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for order, (href, text) in enumerate(links):
        if href in seen or api.domain_of(href) != docs_domain:
            continue
        haystack = f"{api.urllib.parse.urlparse(href).path} {text}".lower()
        for priority, keyword in enumerate(keywords):
            if keyword in haystack:
                seen.add(href)
                scored.append((priority, order, href))
                break
    scored.sort()
    return [href for _, _, href in scored[:limit]]


def raw_github_url(url: str) -> str:
    """Rewrite a github.com blob URL to its raw file so we read the actual
    document instead of the GitHub web-app chrome around it."""
    parsed = facade().urllib.parse.urlparse(url)
    if parsed.netloc.lower().removeprefix("www.") != "github.com":
        return url
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 5 and parts[2] == "blob":
        owner, repo, _, ref = parts[:4]
        file_path = "/".join(parts[4:])
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{file_path}"
    return url

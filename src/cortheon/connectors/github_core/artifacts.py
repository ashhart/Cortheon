from __future__ import annotations

from cortheon.connectors.github_core._compat import facade
from cortheon.models import ResearchArtifact


def repository_item_to_artifact(
    item: object,
    source_url: str,
    query: str = "",
) -> ResearchArtifact | None:
    api = facade()
    if not isinstance(item, dict):
        return None
    html_url = item.get("html_url")
    full_name = item.get("full_name")
    if not isinstance(html_url, str) or not isinstance(full_name, str):
        return None
    description = item.get("description") if isinstance(item.get("description"), str) else None
    relevance = api.repository_relevance(query, item)
    if query and relevance < 0.34:
        return None
    stars = api._int_or_none(item.get("stargazers_count")) or 0
    forks = api._int_or_none(item.get("forks_count")) or 0
    archived = bool(item.get("archived"))
    language = item.get("language") if isinstance(item.get("language"), str) else None
    pushed_at = (
        api.parse_datetime(item.get("pushed_at"))
        if isinstance(item.get("pushed_at"), str)
        else None
    )
    evidence = (
        f"GitHub search found repository {full_name}"
        f" with {stars} star(s){' archived' if archived else ''}."
    )
    metadata = {
        "repo": full_name,
        "stars": str(stars),
        "forks": str(forks),
        "archived": str(archived).lower(),
        "relevance": f"{relevance:.3f}",
    }
    if language:
        metadata["language"] = language
    if pushed_at:
        metadata["pushed_at"] = pushed_at.isoformat()
    if description:
        metadata["description"] = description[:240]
    return api.ResearchArtifact(
        kind="code_repository",
        title=full_name,
        url=api.normalize_github_url(html_url),
        source_url=source_url,
        provider="github_search",
        evidence=evidence,
        confidence=api.repository_artifact_confidence(stars, archived, description, relevance),
        metadata=metadata,
    )


def repository_relevance(query: str, item: dict[object, object]) -> float:
    api = facade()
    terms = api.repository_query_terms(query)
    if not terms:
        return 1.0
    full_name = item.get("full_name") if isinstance(item.get("full_name"), str) else ""
    description = item.get("description") if isinstance(item.get("description"), str) else ""
    raw_topics = item.get("topics")
    topics = (
        " ".join(topic for topic in raw_topics if isinstance(topic, str))
        if isinstance(raw_topics, list)
        else ""
    )
    haystack = f"{full_name} {description} {topics}".lower().replace("_", "-")
    tokens = set(api.re.findall(r"[a-z0-9][a-z0-9-]{2,}", haystack))
    phrase_text = haystack.replace("-", " ")
    hits = 0
    for term in terms:
        if term in tokens:
            hits += 1
            continue
        if "-" in term and term.replace("-", " ") in phrase_text:
            hits += 1
    return hits / len(terms)


def repository_query_terms(query: str) -> set[str]:
    api = facade()
    return {
        term
        for term in api.re.findall(r"[a-z0-9][a-z0-9-]{2,}", query.lower())
        if term not in api.REPOSITORY_SEARCH_STOPWORDS
    }


def repository_artifact_confidence(
    stars: int,
    archived: bool,
    description: str | None,
    relevance: float,
) -> float:
    score = 0.66
    if stars >= 10_000:
        score += 0.18
    elif stars >= 1_000:
        score += 0.14
    elif stars >= 100:
        score += 0.1
    elif stars >= 10:
        score += 0.05
    if description:
        score += 0.04
    score += min(0.12, relevance * 0.12)
    if archived:
        score -= 0.22
    return round(max(0.2, min(score, 0.92)), 3)


def find_github_url(project_urls: dict[str, str]) -> str | None:
    preferred_names = ("Source", "Source Code", "Repository", "Homepage", "Code")
    for name in preferred_names:
        value = project_urls.get(name)
        if value and "github.com" in value:
            return facade().normalize_github_url(value)
    for value in project_urls.values():
        if "github.com" in value:
            return facade().normalize_github_url(value)
    return None


def parse_owner_repo(url: str) -> tuple[str, str] | None:
    api = facade()
    parsed = api.urlparse(url)
    target = parsed.netloc + parsed.path
    match = api.GITHUB_RE.search(target)
    if not match:
        return None
    repo = match.group("repo").removesuffix(".git")
    return match.group("owner"), repo


def normalize_github_url(url: str) -> str:
    return url.split("#", 1)[0].split("?", 1)[0].rstrip("/")


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None

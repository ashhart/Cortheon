from __future__ import annotations

from cortheon.connectors.github_core._compat import facade
from cortheon.connectors.http import JsonHttpClient


def repository_search_url(query: str, limit: int) -> str:
    q = f"{query.strip()} in:name,description,readme"
    params = facade().urllib.parse.urlencode(
        {
            "q": q,
            "sort": "stars",
            "order": "desc",
            "per_page": str(min(max(limit, 1), 25)),
        }
    )
    return f"https://api.github.com/search/repositories?{params}"


def github_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = facade().os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def safe_get_json(client: JsonHttpClient, url: str, headers: dict[str, str]) -> object:
    api = facade()
    try:
        return client.get_json(url, headers=headers)
    except api.ConnectorError:
        return None

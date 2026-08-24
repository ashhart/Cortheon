from __future__ import annotations

import os
import urllib.parse
from typing import Any

from cortheon.connectors.http import ConnectorError, JsonHttpClient
from cortheon.models import Evidence, SearchResult, SupportLevel


class SearchProvider:
    name = "none"

    def search(self, query: str, limit: int) -> tuple[list[SearchResult], list[Evidence]]:
        return [], []


class ConfiguredSearchProvider(SearchProvider):
    def __init__(self, client: JsonHttpClient | None = None) -> None:
        self.client = client or JsonHttpClient(timeout_seconds=20)
        self.provider = self._select_provider()
        self.name = self.provider or "none"

    def search(self, query: str, limit: int) -> tuple[list[SearchResult], list[Evidence]]:
        if self.provider == "brave":
            return self._brave(query, limit)
        if self.provider == "tavily":
            return self._tavily(query, limit)
        if self.provider == "serpapi":
            return self._serpapi(query, limit)
        return [], [
            Evidence(
                claim="No web search provider is configured; research used seed URLs only.",
                source_type="search_provider_config",
                source_url=None,
                support=SupportLevel.FAILED,
                details={
                    "supported_env": ["BRAVE_SEARCH_API_KEY", "TAVILY_API_KEY", "SERPAPI_API_KEY"]
                },
            )
        ]

    def _select_provider(self) -> str | None:
        if os.environ.get("BRAVE_SEARCH_API_KEY") or os.environ.get("BRAVE_API_KEY"):
            return "brave"
        if os.environ.get("TAVILY_API_KEY"):
            return "tavily"
        if os.environ.get("SERPAPI_API_KEY"):
            return "serpapi"
        return None

    def _brave(self, query: str, limit: int) -> tuple[list[SearchResult], list[Evidence]]:
        token = os.environ.get("BRAVE_SEARCH_API_KEY") or os.environ.get("BRAVE_API_KEY")
        url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
            {"q": query, "count": min(limit, 20)}
        )
        payload = self.client.get_json(
            url,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": token or "",
            },
        )
        items = ((payload.get("web") or {}).get("results") or [])[:limit]
        return self._results_from_items(
            items,
            provider="brave",
            title_key="title",
            url_key="url",
            snippet_key="description",
            source_url=url,
            query=query,
        )

    def _tavily(self, query: str, limit: int) -> tuple[list[SearchResult], list[Evidence]]:
        url = "https://api.tavily.com/search"
        payload = self.client.post_json(
            url,
            {
                "api_key": os.environ.get("TAVILY_API_KEY"),
                "query": query,
                "max_results": limit,
                "search_depth": "advanced",
            },
        )
        items = list(payload.get("results") or [])[:limit]
        return self._results_from_items(
            items,
            provider="tavily",
            title_key="title",
            url_key="url",
            snippet_key="content",
            source_url=url,
            query=query,
        )

    def _serpapi(self, query: str, limit: int) -> tuple[list[SearchResult], list[Evidence]]:
        url = "https://serpapi.com/search.json?" + urllib.parse.urlencode(
            {"q": query, "engine": "google", "api_key": os.environ.get("SERPAPI_API_KEY")}
        )
        payload = self.client.get_json(url)
        items = list(payload.get("organic_results") or [])[:limit]
        return self._results_from_items(
            items,
            provider="serpapi",
            title_key="title",
            url_key="link",
            snippet_key="snippet",
            source_url="https://serpapi.com/search.json",
            query=query,
        )

    def _results_from_items(
        self,
        items: list[dict[str, Any]],
        *,
        provider: str,
        title_key: str,
        url_key: str,
        snippet_key: str,
        source_url: str,
        query: str,
    ) -> tuple[list[SearchResult], list[Evidence]]:
        results: list[SearchResult] = []
        for index, item in enumerate(items, start=1):
            title = item.get(title_key)
            url = item.get(url_key)
            if not isinstance(title, str) or not isinstance(url, str):
                continue
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=item.get(snippet_key)
                    if isinstance(item.get(snippet_key), str)
                    else None,
                    provider=provider,
                    rank=index,
                )
            )
        evidence = [
            Evidence(
                claim=f"{provider} search returned {len(results)} result(s) for research query.",
                source_type="web_search",
                source_url=source_url,
                support=SupportLevel.OBSERVED,
                details={"query": query, "result_count": len(results)},
            )
        ]
        return results, evidence


def search_with_errors(
    provider: SearchProvider,
    query: str,
    limit: int,
) -> tuple[list[SearchResult], list[Evidence], list[str]]:
    try:
        results, evidence = provider.search(query, limit)
        return results, evidence, []
    except ConnectorError as exc:
        return [], [], [str(exc)]

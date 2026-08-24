from __future__ import annotations

from cortheon.connectors.github_core._compat import facade
from cortheon.connectors.http import JsonHttpClient
from cortheon.models import Evidence, GitHubRepoReport, ResearchArtifact


class GitHubConnector:
    def __init__(self, client: JsonHttpClient | None = None) -> None:
        self.client = client or facade().JsonHttpClient()

    def fetch_from_project_urls(
        self,
        project_urls: dict[str, str],
    ) -> tuple[GitHubRepoReport | None, list[Evidence]]:
        api = facade()
        github_url = api.find_github_url(project_urls)
        if not github_url:
            return None, []
        owner_repo = api.parse_owner_repo(github_url)
        if not owner_repo:
            return None, []
        owner, repo = owner_repo
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        headers = {"Accept": "application/vnd.github+json"}
        token = api.os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = self.client.get_json(api_url, headers=headers)
        report = api.GitHubRepoReport(
            # The API follows rename redirects; its canonical full_name is
            # the live identity (PyPDF2 resolves to py-pdf/pypdf) and a
            # rename is itself evidence of a superseded package.
            repo=payload.get("full_name")
            if isinstance(payload.get("full_name"), str)
            else f"{owner}/{repo}",
            html_url=payload.get("html_url") or github_url,
            description=payload.get("description"),
            stars=api._int_or_none(payload.get("stargazers_count")),
            forks=api._int_or_none(payload.get("forks_count")),
            open_issues=api._int_or_none(payload.get("open_issues_count")),
            default_branch=payload.get("default_branch"),
            pushed_at=api.parse_datetime(payload.get("pushed_at")),
            archived=bool(payload.get("archived")),
            license_spdx=(payload.get("license") or {}).get("spdx_id")
            if isinstance(payload.get("license"), dict)
            else None,
            source_url=api_url,
        )
        evidence = [
            api.Evidence(
                claim=f"GitHub repository {report.repo} is {'archived' if report.archived else 'active'} and has {report.stars or 0} stars.",
                source_type="github_repo_metadata",
                source_url=api_url,
                support=api.SupportLevel.OBSERVED,
                details={
                    "repo": report.repo,
                    "pushed_at": report.pushed_at.isoformat() if report.pushed_at else None,
                    "open_issues": report.open_issues,
                },
            )
        ]
        return report, evidence


class GitHubRepositorySearch:
    def __init__(self, client: JsonHttpClient | None = None) -> None:
        self.client = client or facade().JsonHttpClient(timeout_seconds=20)

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> tuple[list[ResearchArtifact], list[Evidence], list[str]]:
        api = facade()
        if limit <= 0:
            return [], [], []
        fetch_limit = min(max(limit * 5, limit), 25)
        api_url = api.repository_search_url(query, fetch_limit)
        headers = {"Accept": "application/vnd.github+json"}
        token = api.os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            payload = self.client.get_json(api_url, headers=headers)
        except api.ConnectorError as exc:
            return [], [], [f"GitHub repository search unavailable: {exc}"]
        items = payload.get("items")
        if not isinstance(items, list):
            return [], [], ["GitHub repository search returned an unexpected response shape."]

        artifacts: list[ResearchArtifact] = []
        for item in items:
            artifact = api.repository_item_to_artifact(item, api_url, query)
            if artifact is None:
                continue
            artifacts.append(artifact)
            if len(artifacts) >= limit:
                break
        evidence = [
            api.Evidence(
                claim=f"GitHub repository search returned {len(artifacts)} code artifact(s) for: {query}",
                source_type="github_repository_search",
                source_url=api_url,
                support=api.SupportLevel.OBSERVED,
                details={
                    "query": query,
                    "artifact_count": len(artifacts),
                    "incomplete_results": bool(payload.get("incomplete_results")),
                },
            )
        ]
        return artifacts, evidence, []

    def inspect_artifacts(
        self,
        artifacts: list[ResearchArtifact],
        limit: int = 3,
    ) -> tuple[list[ResearchArtifact], list[Evidence], list[str]]:
        api = facade()
        if limit <= 0:
            return artifacts, [], []
        inspected: list[ResearchArtifact] = []
        evidence: list[Evidence] = []
        errors: list[str] = []
        remaining = limit
        for artifact in artifacts:
            if artifact.kind != "code_repository" or remaining <= 0:
                inspected.append(artifact)
                continue
            owner_repo = api.parse_owner_repo(artifact.url)
            if not owner_repo:
                inspected.append(artifact)
                continue
            owner, repo = owner_repo
            try:
                enriched, item_evidence = self.inspect_repository_artifact(artifact, owner, repo)
            except api.ConnectorError as exc:
                errors.append(f"GitHub artifact inspection unavailable for {owner}/{repo}: {exc}")
                inspected.append(artifact)
                continue
            inspected.append(enriched)
            evidence.extend(item_evidence)
            remaining -= 1
        return inspected, evidence, errors

    def inspect_repository_artifact(
        self,
        artifact: ResearchArtifact,
        owner: str,
        repo: str,
    ) -> tuple[ResearchArtifact, list[Evidence]]:
        api = facade()
        headers = api.github_headers()
        api_base = f"https://api.github.com/repos/{owner}/{repo}"
        repo_payload = self.client.get_json(api_base, headers=headers)
        languages_payload = api.safe_get_json(self.client, f"{api_base}/languages", headers)
        contents_payload = api.safe_get_json(self.client, f"{api_base}/contents", headers)
        readme_payload = api.safe_get_json(self.client, f"{api_base}/readme", headers)
        metadata = dict(artifact.metadata)
        metadata.update(api.repository_metadata(repo_payload))
        metadata.update(api.language_metadata(languages_payload))
        metadata.update(api.root_content_metadata(contents_payload))
        readme = api.readme_text(readme_payload)
        if readme:
            metadata.update(api.readme_metadata(readme))
        signals = api.implementation_signals(metadata)
        metadata["implementation_signals"] = ",".join(signals)
        metadata["repository_health_score"] = f"{api.repository_health_score(metadata):.3f}"
        metadata["inspected_at"] = api.utc_now().isoformat()
        enriched = api.ResearchArtifact(
            kind=artifact.kind,
            title=artifact.title,
            url=artifact.url,
            source_url=artifact.source_url,
            provider=artifact.provider,
            evidence=artifact.evidence,
            confidence=api.adjusted_repository_confidence(artifact.confidence, metadata),
            metadata=metadata,
        )
        repo_name = metadata.get("repo", f"{owner}/{repo}")
        evidence = [
            api.Evidence(
                claim=(
                    f"GitHub inspection enriched repository artifact {repo_name} "
                    f"with health score {metadata['repository_health_score']} and signals: "
                    f"{metadata['implementation_signals'] or 'none'}."
                ),
                source_type="github_artifact_inspection",
                source_url=api_base,
                support=api.SupportLevel.OBSERVED,
                details={
                    "repo": repo_name,
                    "health_score": metadata["repository_health_score"],
                    "signals": signals,
                    "root_entries": metadata.get("root_entries", ""),
                    "primary_language": metadata.get("primary_language"),
                    "license_spdx": metadata.get("license_spdx"),
                },
            )
        ]
        return enriched, evidence

from __future__ import annotations

from cortheon.connectors.http import JsonHttpClient
from cortheon.models import DocumentationReport, Evidence, SupportLevel

DOC_LABELS = (
    "Documentation",
    "Docs",
    "Homepage",
    "Home",
    "Website",
    "Read the Docs",
)


class DocumentationConnector:
    def __init__(self, client: JsonHttpClient | None = None) -> None:
        self.client = client or JsonHttpClient()

    def inspect(self, project_urls: dict[str, str]) -> tuple[DocumentationReport, list[Evidence]]:
        docs_url = pick_url(project_urls, ("Documentation", "Docs", "Read the Docs"))
        homepage_url = pick_url(project_urls, ("Homepage", "Home", "Website"))
        reachable: list[str] = []
        for url in {docs_url, homepage_url}:
            if not url:
                continue
            status = self.client.head_or_get_status(url)
            if status and 200 <= status < 400:
                reachable.append(url)
        source_url = docs_url or homepage_url
        report = DocumentationReport(
            docs_url=docs_url,
            homepage_url=homepage_url,
            reachable_urls=reachable,
            source_url=source_url,
        )
        evidence: list[Evidence] = []
        if docs_url:
            evidence.append(
                Evidence(
                    claim=f"Package metadata links to documentation at {docs_url}.",
                    source_type="package_documentation_link",
                    source_url=docs_url,
                    support=SupportLevel.OBSERVED,
                    details={"reachable": docs_url in reachable},
                )
            )
        if homepage_url and homepage_url != docs_url:
            evidence.append(
                Evidence(
                    claim=f"Package metadata links to homepage at {homepage_url}.",
                    source_type="package_homepage_link",
                    source_url=homepage_url,
                    support=SupportLevel.OBSERVED,
                    details={"reachable": homepage_url in reachable},
                )
            )
        return report, evidence


def pick_url(project_urls: dict[str, str], labels: tuple[str, ...]) -> str | None:
    lowered = {key.lower(): value for key, value in project_urls.items()}
    for label in labels:
        value = project_urls.get(label) or lowered.get(label.lower())
        if value:
            return value
    for key, value in project_urls.items():
        if any(label.lower() in key.lower() for label in labels):
            return value
    return None

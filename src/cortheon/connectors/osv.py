from __future__ import annotations

from cortheon.connectors.http import JsonHttpClient
from cortheon.models import Evidence, SupportLevel, VulnerabilityReport


class OSVConnector:
    def __init__(self, client: JsonHttpClient | None = None) -> None:
        self.client = client or JsonHttpClient()

    def fetch(self, package: str, version: str) -> tuple[VulnerabilityReport, list[Evidence]]:
        url = "https://api.osv.dev/v1/query"
        payload = {
            "package": {
                "name": package,
                "ecosystem": "PyPI",
            },
            "version": version,
        }
        response = self.client.post_json(url, payload)
        vulns = list(response.get("vulns") or [])
        report = VulnerabilityReport(
            package=package,
            version=version,
            vulnerabilities=vulns,
            source_url=url,
        )
        if vulns:
            claim = f"OSV returned {len(vulns)} known vulnerabilities for {package} {version}."
        else:
            claim = f"OSV returned no known vulnerabilities for {package} {version}."
        return report, [
            Evidence(
                claim=claim,
                source_type="osv_vulnerability_query",
                source_url=url,
                package=package,
                version=version,
                support=SupportLevel.OBSERVED,
                details={"vulnerability_count": len(vulns)},
            )
        ]

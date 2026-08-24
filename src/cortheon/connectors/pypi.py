from __future__ import annotations

from datetime import datetime
from typing import Any

from cortheon.connectors.http import JsonHttpClient, normalize_package_name
from cortheon.models import (
    DistributionArtifact,
    Evidence,
    PackageMetadata,
    SupportLevel,
    parse_datetime,
)

MAX_DESCRIPTION_CHARS = 20_000


class PyPIConnector:
    def __init__(self, client: JsonHttpClient | None = None) -> None:
        self.client = client or JsonHttpClient()

    def fetch(
        self, package: str, version: str | None = None
    ) -> tuple[PackageMetadata, list[Evidence]]:
        name_part = normalize_package_name(package)
        if version:
            url = f"https://pypi.org/pypi/{name_part}/{version}/json"
        else:
            url = f"https://pypi.org/pypi/{name_part}/json"
        payload = self.client.get_json(url)
        info = payload.get("info") or {}
        version = str(info.get("version") or version or "")
        release_files = payload.get("urls") or []
        upload_time = _latest_upload_time(release_files)
        project_urls = _clean_urls(info.get("project_urls") or {})
        for label, key in (
            ("Homepage", "home_page"),
            ("Documentation", "docs_url"),
            ("Issue Tracker", "bugtrack_url"),
            ("PyPI", "package_url"),
        ):
            value = info.get(key)
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                project_urls.setdefault(label, value)
        releases = payload.get("releases") or {}
        metadata = PackageMetadata(
            name=info.get("name") or package,
            version=version,
            summary=info.get("summary"),
            requires_python=info.get("requires_python"),
            license=_first_text(info.get("license"), info.get("license_expression")),
            project_urls=project_urls,
            classifiers=list(info.get("classifiers") or []),
            requires_dist=list(info.get("requires_dist") or []),
            release_upload_time=upload_time,
            release_count=len(releases),
            artifacts=_artifacts(release_files),
            source_url=url,
            description=_capped_description(info.get("description")),
        )
        evidence = [
            Evidence(
                claim=f"PyPI reports {metadata.name} version {metadata.version} metadata."
                if version
                else f"PyPI reports latest {metadata.name} version as {metadata.version}.",
                source_type="pypi_metadata",
                source_url=url,
                package=metadata.name,
                version=metadata.version,
                support=SupportLevel.OBSERVED,
                details={
                    "requires_python": metadata.requires_python,
                    "release_upload_time": metadata.release_upload_time.isoformat()
                    if metadata.release_upload_time
                    else None,
                },
            )
        ]
        return metadata, evidence


def _latest_upload_time(files: list[dict[str, Any]]) -> datetime | None:
    parsed = [
        parse_datetime(item.get("upload_time_iso_8601") or item.get("upload_time"))
        for item in files
    ]
    values = [item for item in parsed if item is not None]
    if not values:
        return None
    return max(values)


def _artifacts(files: list[dict[str, Any]]) -> list[DistributionArtifact]:
    artifacts: list[DistributionArtifact] = []
    for item in files:
        filename = item.get("filename")
        url = item.get("url")
        if not isinstance(filename, str) or not isinstance(url, str):
            continue
        artifacts.append(
            DistributionArtifact(
                filename=filename,
                package_type=str(item.get("packagetype") or "unknown"),
                url=url,
                size=item.get("size") if isinstance(item.get("size"), int) else None,
                upload_time=parse_datetime(
                    item.get("upload_time_iso_8601") or item.get("upload_time")
                ),
                digests=dict(item.get("digests") or {}),
            )
        )
    return artifacts


def _clean_urls(raw: dict[str, Any]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            cleaned[str(key)] = value
    return cleaned


def _first_text(*values: str | None) -> str | None:
    for value in values:
        if value and value.strip() and value.strip().upper() != "UNKNOWN":
            return value.strip()
    return None


def _capped_description(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value[:MAX_DESCRIPTION_CHARS]

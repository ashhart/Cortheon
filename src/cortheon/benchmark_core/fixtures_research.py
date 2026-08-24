"""Held-out live-research fixtures resolved from public package indexes."""

from __future__ import annotations

import hashlib
import random
import re
import urllib.error
import urllib.request

from cortheon.benchmark_core._compat import facade
from cortheon.benchmark_core.models import ResearchCase
from cortheon.scholarly import bounded_xml_root


def _latest_pypi_release(project: str) -> str:
    url = f"https://pypi.org/rss/project/{project}/releases.xml"
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/rss+xml", "User-Agent": "cortheon-bench/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = response.read(1_000_001)
    except (OSError, urllib.error.URLError) as exc:
        raise ValueError(f"could not resolve current PyPI release for {project}") from exc
    if len(payload) > 1_000_000:
        raise ValueError(f"PyPI release feed for {project} exceeded 1 MB")
    try:
        root = bounded_xml_root(payload, max_bytes=1_000_000)
    except ValueError as exc:
        raise ValueError(f"PyPI release feed for {project} was invalid XML") from exc
    title = next(
        (
            value.strip()
            for value in (item.text for item in root.findall("./channel/item/title"))
            if value
            and re.fullmatch(
                r"\d+\.\d+(?:\.\d+){0,3}",
                value.strip(),
            )
        ),
        "",
    )
    if not title:
        raise ValueError(f"PyPI release feed for {project} lacked a stable version")
    return title


def discover_research_cases(*, count: int, seed: int) -> list[ResearchCase]:
    """Resolve current release facts privately before model execution."""

    definitions = [
        ("uv", "astral-sh/uv"),
        ("ruff", "astral-sh/ruff"),
        ("pytest", "pytest-dev/pytest"),
        ("httpx", "encode/httpx"),
    ]
    if count > len(definitions):
        raise ValueError(f"research suite has {len(definitions)} held-out cases; requested {count}")
    random.Random(seed ^ 0xC0FFEE).shuffle(definitions)
    cases: list[ResearchCase] = []
    for project, repository in definitions[:count]:
        expected = facade()._latest_pypi_release(project)
        github_url = f"https://github.com/{repository}/releases/latest"
        pypi_url = f"https://pypi.org/project/{project}/"
        raw = f"{seed}\0{project}\0{expected}".encode()
        cases.append(
            ResearchCase(
                case_id="research_" + hashlib.sha256(raw).hexdigest()[:12],
                project=project,
                expected=expected,
                github_url=github_url,
                pypi_url=pypi_url,
                prompt=(
                    f"Research the latest released {project} version from the current "
                    f"web. Fetch {github_url} and independently corroborate it with "
                    f"{pypi_url}. Check freshness and contradictions. State the exact "
                    "version and include clickable URLs from both origins. Do not "
                    "modify files."
                ),
            )
        )
    return cases

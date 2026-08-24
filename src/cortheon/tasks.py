from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Candidate:
    package: str
    fit_boost: float
    rationale: str


@dataclass(frozen=True, slots=True)
class TaskProfile:
    name: str
    keywords: tuple[str, ...]
    candidates: tuple[Candidate, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)


PROFILES: tuple[TaskProfile, ...] = (
    TaskProfile(
        name="python_rest_api",
        keywords=("rest api", "restful", "api service", "openapi", "web api", "http api"),
        candidates=(
            Candidate(
                "fastapi",
                0.18,
                "Strong default for typed Python APIs, OpenAPI, async handlers, and low ceremony.",
            ),
            Candidate(
                "litestar",
                0.15,
                "Modern ASGI API framework with typed handlers and batteries-included options.",
            ),
            Candidate(
                "django-ninja",
                0.08,
                "Good fit when the repo is already Django-based and wants OpenAPI-style APIs.",
            ),
            Candidate(
                "flask",
                0.03,
                "Mature minimal framework, but typed validation/OpenAPI usually requires extra packages.",
            ),
        ),
        notes=(
            "Prototype candidate discovery used a built-in REST API profile.",
            "Repo constraints should override the winner when an existing stack is detected.",
        ),
    ),
    TaskProfile(
        name="async_http_client",
        keywords=(
            "async http",
            "async client",
            "asyncclient",
            "http client",
            "httpx",
            "requests async",
            "retrying http",
            "stream bytes",
            "web requests",
        ),
        candidates=(
            Candidate(
                "httpx", 0.17, "Modern sync/async HTTP client with requests-like ergonomics."
            ),
            Candidate("aiohttp", 0.11, "Established async HTTP client/server library."),
            Candidate("requests", 0.02, "Very mature sync HTTP client, but not async-first."),
        ),
    ),
    TaskProfile(
        name="cli_app",
        keywords=("cli", "command line", "terminal app", "console app"),
        candidates=(
            Candidate("typer", 0.15, "Typed CLI framework with good developer ergonomics."),
            Candidate("click", 0.11, "Mature and widely used CLI framework."),
            Candidate("argparse", 0.07, "Standard library option with zero dependency risk."),
        ),
    ),
    TaskProfile(
        name="structured_logging",
        keywords=("structured logging", "json logging", "log json", "logging"),
        candidates=(
            Candidate("structlog", 0.14, "Purpose-built structured logging library."),
            Candidate(
                "loguru",
                0.08,
                "Ergonomic logging library, but structured production logging may need care.",
            ),
            Candidate("python-json-logger", 0.07, "Focused JSON formatter for standard logging."),
        ),
    ),
)


def find_profile(task: str) -> TaskProfile | None:
    normalized = " ".join(task.lower().split())
    for profile in PROFILES:
        if any(keyword_matches(normalized, keyword) for keyword in profile.keywords):
            return profile
    return None


def keyword_matches(text: str, keyword: str) -> bool:
    normalized_keyword = " ".join(keyword.lower().split())
    if not normalized_keyword:
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])", text))

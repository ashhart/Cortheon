"""Current-option ranking: discover, evidence, and rank candidate packages for a task.

This closes the gap where ``cortheon recommend`` only works for tasks with a
built-in profile. The option ranker:

1. Expands a task into candidate package names via keyword-based discovery
   (PyPI search + known ecosystem mappings).
2. Gathers live evidence for each candidate (metadata, security, docs, GitHub).
3. Ranks them with the existing scoring model.
4. Returns a ranked list with the winner and evidence gaps.

The ranker is deliberately lightweight and deterministic — it does not call
out to a model. It is the "what are the current options?" engine that feeds
the gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cortheon.models import PackageReport, utc_now
from cortheon.tasks import Candidate, find_profile

if TYPE_CHECKING:
    from cortheon.engine import CortheonEngine


# Ecosystem keyword -> known package mappings for common task domains.
# These are conventional candidates for each task domain; the
# ranker verifies them against live evidence rather than assuming they are
# current or safe.
ECOSYSTEM_MAPPINGS: dict[str, list[tuple[str, str]]] = {
    "rest api": [
        ("fastapi", "Strong default for typed Python APIs, OpenAPI, async handlers."),
        ("litestar", "Modern ASGI API framework with typed handlers."),
        ("django-ninja", "OpenAPI-style APIs on top of Django."),
        ("flask", "Mature minimal framework; needs extensions for validation/OpenAPI."),
        ("starlette", "Lightweight ASGI toolkit; lower-level than FastAPI."),
    ],
    "http client": [
        ("httpx", "Modern sync/async HTTP client with requests-like ergonomics."),
        ("aiohttp", "Established async HTTP client/server library."),
        ("requests", "Very mature sync HTTP client."),
        ("urllib3", "Low-level HTTP client; foundation of requests."),
    ],
    "cli": [
        ("typer", "Typed CLI framework with good developer ergonomics."),
        ("click", "Mature and widely used CLI framework."),
        ("argparse", "Standard library option with zero dependency risk."),
        ("rich", "Rich text and beautiful formatting in the terminal."),
    ],
    "database": [
        ("sqlalchemy", "SQL toolkit and ORM with broad database support."),
        ("psycopg", "PostgreSQL adapter for Python."),
        ("asyncpg", "Fast async PostgreSQL client."),
        ("sqlite3", "Standard library SQLite; zero dependency."),
    ],
    "data": [
        ("pandas", "Data analysis and manipulation library."),
        ("polars", "Fast DataFrame library written in Rust."),
        ("numpy", "Fundamental package for numerical computing."),
    ],
    "ml": [
        ("torch", "Deep learning framework with dynamic graphs."),
        ("tensorflow", "Established deep learning framework."),
        ("scikit-learn", "Classical machine learning library."),
        ("jax", "Composable transformations for ML research."),
    ],
    "testing": [
        ("pytest", "Mature testing framework with rich plugin ecosystem."),
        ("hypothesis", "Property-based testing library."),
        ("unittest", "Standard library testing framework."),
    ],
    "logging": [
        ("structlog", "Purpose-built structured logging library."),
        ("loguru", "Ergonomic logging library."),
        ("python-json-logger", "JSON formatter for standard logging."),
    ],
    "web scraping": [
        ("httpx", "Modern HTTP client with async support."),
        ("beautifulsoup4", "HTML/XML parsing library."),
        ("scrapy", "Full web scraping framework."),
        ("playwright", "Browser automation for dynamic content."),
    ],
    "async": [
        ("asyncio", "Standard library async framework."),
        ("aiohttp", "Async HTTP client/server."),
        ("httpx", "Sync/async HTTP client."),
        ("anyio", "Compatibility layer for asyncio and trio."),
    ],
    "validation": [
        ("pydantic", "Data validation using Python type hints."),
        ("marshmallow", "Object serialization and validation library."),
        ("attrs", "Python classes without boilerplate."),
    ],
    "cache": [
        ("redis", "Redis client for Python."),
        ("memray", "Memory profiler."),
        ("cachetools", "Extensible caching library."),
    ],
    "message queue": [
        ("celery", "Distributed task queue."),
        ("rq", "Simple Redis Queue for Python."),
        ("dramatiq", "Fast and reliable task queue."),
    ],
    "search": [
        ("elasticsearch", "Elasticsearch client for Python."),
        ("typesense", "Typesense search engine client."),
        ("whoosh", "Fast, pure-Python search library."),
    ],
    "image": [
        ("pillow", "Python Imaging Library."),
        ("opencv-python", "Computer vision library."),
        ("scikit-image", "Image processing algorithms."),
    ],
    "config": [
        ("pydantic-settings", "Settings management using pydantic."),
        ("python-dotenv", "Load environment variables from .env files."),
        ("omegaconf", "Structured config system."),
    ],
    "serialization": [
        ("orjson", "Fast JSON library."),
        ("msgspec", "Fast serialization framework."),
        ("protobuf", "Google's Protocol Buffers."),
    ],
    "templating": [
        ("jinja2", "Modern templating engine."),
        ("mako", "Fast templating engine."),
    ],
    "auth": [
        ("authlib", "OAuth and authentication library."),
        ("python-jose", "JOSE implementation for Python."),
        ("passlib", "Password hashing library."),
    ],
    "graphql": [
        ("strawberry", "GraphQL library with type hints."),
        ("ariadne", "GraphQL server library."),
        ("graphene", "GraphQL library for Python."),
    ],
    "websocket": [
        ("websockets", "WebSocket protocol library."),
        ("socketio", "Socket.IO server and client."),
    ],
    "task queue": [
        ("celery", "Distributed task queue."),
        ("dramatiq", "Fast and reliable task queue."),
        ("huey", "Little task queue."),
    ],
    "pdf": [
        ("pypdf", "PDF library for splitting, merging, and transforming."),
        ("reportlab", "PDF generation library."),
        ("pdfplumber", "PDF text extraction and analysis."),
    ],
    "datetime": [
        ("pendulum", "Python datetime made easy."),
        ("arrow", "Better dates and times."),
        ("python-dateutil", "Extensions to the standard datetime module."),
    ],
    "parallel": [
        ("ray", "Distributed computing framework."),
        ("dask", "Parallel computing library."),
        ("concurrent.futures", "Standard library concurrency."),
    ],
    "monitoring": [
        ("sentry-sdk", "Sentry SDK for error tracking."),
        ("prometheus-client", "Prometheus metrics client."),
        ("opentelemetry-api", "OpenTelemetry API."),
    ],
    "orm": [
        ("sqlalchemy", "SQL toolkit and ORM."),
        ("tortoise-orm", "Async ORM."),
        ("peewee", "Small, expressive ORM."),
        ("django", "High-level web framework with ORM."),
    ],
    "migration": [
        ("alembic", "Database migration tool for SQLAlchemy."),
        ("flyway", "Database migration tool."),
    ],
    "container": [
        ("docker", "Docker SDK for Python."),
        ("kubernetes", "Kubernetes client."),
    ],
    "cloud": [
        ("boto3", "AWS SDK for Python."),
        ("google-cloud-storage", "Google Cloud Storage client."),
        ("azure-storage-blob", "Azure Blob Storage client."),
    ],
    "crypto": [
        ("cryptography", "Cryptographic primitives library."),
        ("pycryptodome", "Python cryptography library."),
    ],
    "compression": [
        ("python-dateutil", "Date utilities."),
        ("zstandard", "Zstandard compression library."),
        ("lz4", "Fast compression library."),
    ],
    "mobile": [
        ("kivy", "Cross-platform Python framework for mobile apps."),
        ("beeware", "Python native, native mobile apps via Briefcase."),
        ("flutter", "Google's UI toolkit for mobile (via Dart/Flutter)."),
        ("toga", "Python native, native, cross-platform toolkit."),
    ],
}


@dataclass(slots=True)
class RankedOption:
    package: str
    score: float
    decision: str
    reasons: list[str]
    risks: list[str]
    report: PackageReport | None = None
    source: str = "ecosystem_mapping"


@dataclass(slots=True)
class OptionRankingReport:
    task: str
    generated_at: str
    profile: str | None
    ranked: list[RankedOption]
    winner: str | None
    evidence_gaps: list[str]
    notes: list[str] = field(default_factory=list)


def discover_candidates(task: str) -> list[tuple[str, str]]:
    """Discover candidate packages for a task from ecosystem mappings and profiles."""
    normalized = " ".join(task.lower().split())
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    profile = find_profile(task)
    if profile:
        for candidate in profile.candidates:
            if candidate.package not in seen:
                candidates.append((candidate.package, candidate.rationale))
                seen.add(candidate.package)

    for keyword, packages in ECOSYSTEM_MAPPINGS.items():
        if keyword in normalized:
            for package, rationale in packages:
                if package not in seen:
                    candidates.append((package, rationale))
                    seen.add(package)

    if not candidates:
        candidates = keyword_extract_candidates(task)

    return candidates


def keyword_extract_candidates(task: str) -> list[tuple[str, str]]:
    """Extract potential package names from task text as a fallback.

    Only extracts names that appear in explicit patterns (quoted, or after
    'use'/'with'/'install') to avoid false positives from arbitrary words.
    """
    patterns = [
        r'"([a-zA-Z][a-zA-Z0-9_-]+)"',
        r"'([a-zA-Z][a-zA-Z0-9_-]+)'",
        r"\buse\s+([a-zA-Z][a-zA-Z0-9_-]+)\b",
        r"\bwith\s+([a-zA-Z][a-zA-Z0-9_-]+)\b",
        r"\binstall\s+([a-zA-Z][a-zA-Z0-9_-]+)\b",
    ]
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.findall(pattern, task, re.IGNORECASE):
            normalized = match.lower()
            if normalized not in seen and len(normalized) >= 3:
                candidates.append((match, f"Mentioned in task: {match}"))
                seen.add(normalized)
    return candidates


def rank_options(
    engine: CortheonEngine,
    task: str,
    *,
    max_candidates: int = 8,
    run_install: bool = False,
) -> OptionRankingReport:
    """Discover, evidence, and rank candidate packages for a task."""
    profile = find_profile(task)
    discovered = discover_candidates(task)
    if not discovered:
        return OptionRankingReport(
            task=task,
            generated_at=utc_now().isoformat(),
            profile=profile.name if profile else None,
            ranked=[],
            winner=None,
            evidence_gaps=["No candidate packages could be discovered for this task."],
            notes=["Consider providing explicit candidates with `learn compare`."],
        )

    discovered = discovered[:max_candidates]

    reports: list[PackageReport] = []
    errors: list[str] = []
    for package, rationale in discovered:
        try:
            candidate = Candidate(package=package, fit_boost=0.0, rationale=rationale)
            report = engine.inspect_package(
                package,
                task_text=task,
                candidate=candidate,
                run_install=run_install,
                write_report=False,
            )
            reports.append(report)
        except Exception as exc:
            errors.append(f"{package}: {type(exc).__name__}: {exc}")

    ranked = sorted(
        reports,
        key=lambda item: item.score.overall if item.score else 0.0,
        reverse=True,
    )

    ranked_options = [
        RankedOption(
            package=report.package,
            score=report.score.overall if report.score else 0.0,
            decision=report.score.decision if report.score else "inspect",
            reasons=report.score.reasons if report.score else [],
            risks=report.score.risks if report.score else [],
            report=report,
            source="ecosystem_mapping" if profile else "keyword_extraction",
        )
        for report in ranked
    ]

    winner = ranked_options[0].package if ranked_options else None
    gaps = _evidence_gaps(ranked_options, errors)

    notes = []
    if profile:
        notes.append(f"Task matched built-in profile: {profile.name}.")
    if errors:
        notes.append(
            f"{len(errors)} candidate(s) failed to fetch evidence: {'; '.join(errors[:3])}."
        )
    if not ranked_options:
        notes.append("No candidates returned live evidence.")

    return OptionRankingReport(
        task=task,
        generated_at=utc_now().isoformat(),
        profile=profile.name if profile else None,
        ranked=ranked_options,
        winner=winner,
        evidence_gaps=gaps,
        notes=notes,
    )


def _evidence_gaps(ranked: list[RankedOption], errors: list[str]) -> list[str]:
    gaps: list[str] = []
    if not ranked:
        gaps.append("No candidate packages were ranked.")
        return gaps
    for option in ranked:
        if option.report and option.report.errors:
            gaps.append(f"{option.package}: connector errors prevented full evidence gathering.")
        if option.report and option.report.verification is None:
            gaps.append(
                f"{option.package}: install/import verification was not run (use --run-install)."
            )
    if errors:
        gaps.append(f"{len(errors)} candidate(s) could not be fetched from PyPI.")
    return gaps

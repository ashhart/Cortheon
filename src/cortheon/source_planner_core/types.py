from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceProfile:
    name: str
    source_type: str
    capabilities: tuple[str, ...]
    domains: tuple[str, ...]
    trust_tier: str
    default_priority: float
    available: bool = True


@dataclass(frozen=True, slots=True)
class SourcePlanningConstraints:
    max_search_results: int
    max_scholarly_results: int
    max_github_results: int
    seed_url_count: int
    search_provider_name: str | None
    max_trial_results: int = 0

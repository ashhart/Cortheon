"""Domain-specific research budget selection."""

from __future__ import annotations

from types import ModuleType
from typing import Any


def merge_limits(bindings: ModuleType, limits: Any, plan: Any) -> Any:
    domain = plan.domain_obj or bindings.classify_task(plan.domain)
    domain_limits = bindings.domain_research_limits(domain)
    return bindings.AutoEvidenceLimits(
        max_search_results=domain_limits.get("max_search_results", limits.max_search_results),
        max_scholarly_results=domain_limits.get(
            "max_scholarly_results",
            limits.max_scholarly_results,
        ),
        max_github_results=domain_limits.get("max_github_results", limits.max_github_results),
        max_trial_results=domain_limits.get("max_trial_results", limits.max_trial_results),
        max_follow_up_queries=limits.max_follow_up_queries,
        max_adaptive_queries=limits.max_adaptive_queries,
        max_artifact_inspections=limits.max_artifact_inspections,
        max_pages=domain_limits.get("max_pages", limits.max_pages),
        max_depth=limits.max_depth,
    )

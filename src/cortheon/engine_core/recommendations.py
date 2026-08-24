"""Recommendation conversion and example de-duplication helpers."""

from __future__ import annotations

from types import ModuleType
from typing import Any


def build_recommendation(
    bindings: ModuleType,
    *,
    task: str,
    profile: str | None,
    candidates: list[Any],
    notes: list[str],
) -> Any:
    ranked = sorted(
        candidates,
        key=lambda item: item.score.overall if item.score else 0.0,
        reverse=True,
    )
    winner = ranked[0].package if ranked else None
    evidence = [
        bindings.Evidence(
            claim=(
                f"Cortheon ranked {winner} highest for task: {task}"
                if winner
                else f"Cortheon could not rank candidates for task: {task}"
            ),
            source_type="cortheon_scoring",
            source_url=None,
            support=bindings.SupportLevel.INFERRED,
            details={
                "profile": profile,
                "candidate_count": len(candidates),
                "winner": winner,
            },
        )
    ]
    return bindings.RecommendationReport(
        task=task,
        profile=profile,
        generated_at=bindings.utc_now(),
        winner=winner,
        candidates=ranked,
        evidence=evidence,
        notes=notes,
    )


def ranking_to_recommendation(bindings: ModuleType, task: str, ranking: Any) -> Any:
    candidates = [option.report for option in ranking.ranked if option.report is not None]
    evidence = [
        bindings.Evidence(
            claim=(
                f"Cortheon ranked {ranking.winner} highest for task: {task}"
                if ranking.winner
                else f"Cortheon could not rank candidates for task: {task}"
            ),
            source_type="cortheon_scoring",
            source_url=None,
            support=bindings.SupportLevel.INFERRED,
            details={
                "profile": ranking.profile,
                "candidate_count": len(candidates),
                "winner": ranking.winner,
                "discovery_source": "option_ranker",
            },
        )
    ]
    notes = list(ranking.notes)
    notes.append(
        "Candidate packages were discovered from ecosystem keyword mappings and ranked against live evidence."
    )
    return bindings.RecommendationReport(
        task=task,
        profile=ranking.profile,
        generated_at=bindings.utc_now(),
        winner=ranking.winner,
        candidates=candidates,
        evidence=evidence,
        notes=notes,
    )


def merge_examples(
    *groups: list[str],
    limit: int,
    substitute: Any,
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for code in group:
            key = substitute(r"\s+", " ", code).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(code)
            if len(merged) >= limit:
                return merged
    return merged

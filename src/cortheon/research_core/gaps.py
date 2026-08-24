from __future__ import annotations

from cortheon.models import Evidence, ResearchGapClosure, ResearchQuery
from cortheon.research_core._compat import facade


def build_gap_closures(
    adaptive_queries: list[ResearchQuery],
    before_gaps: list[str],
    after_gaps: list[str],
    *,
    before_claim_count: int,
    after_claim_count: int,
    before_source_count: int,
    after_source_count: int,
) -> list[ResearchGapClosure]:
    api = facade()
    closures: list[ResearchGapClosure] = []
    after_by_kind = {api.gap_kind(gap): gap for gap in after_gaps}
    before_kinds = {api.gap_kind(gap) for gap in before_gaps}
    for query in adaptive_queries:
        if not query.target_gap:
            continue
        kind = api.gap_kind(query.target_gap)
        after_related = after_by_kind.get(kind)
        before_present = kind in before_kinds
        after_present = after_related is not None
        if before_present and not after_present:
            status = "closed"
        elif (
            before_present
            and after_present
            and api.gap_metric_improved(
                kind,
                before_claim_count,
                after_claim_count,
                before_source_count,
                after_source_count,
            )
        ):
            status = "improved_but_open"
        elif before_present and after_present:
            status = "still_open"
        else:
            status = "not_reproduced"
        closures.append(
            api.ResearchGapClosure(
                target_gap=query.target_gap,
                query=query.query,
                status=status,
                before_claim_count=before_claim_count,
                after_claim_count=after_claim_count,
                before_source_count=before_source_count,
                after_source_count=after_source_count,
                before_gap_present=before_present,
                after_gap_present=after_present,
                after_related_gap=after_related,
            )
        )
    return closures


def gap_kind(gap: str) -> str:
    lower = gap.lower()
    if "too few extracted claims" in lower:
        return "too_few_claims"
    if "too few independent sources" in lower:
        return "too_few_sources"
    if "benchmark" in lower or "evaluation" in lower:
        return "benchmark_evidence"
    if "under-covered" in lower:
        return "undercovered_terms"
    if "claim clusters" in lower:
        return "no_clusters"
    return lower.strip().rstrip(".")


def gap_metric_improved(
    kind: str,
    before_claim_count: int,
    after_claim_count: int,
    before_source_count: int,
    after_source_count: int,
) -> bool:
    if kind == "too_few_claims":
        return after_claim_count > before_claim_count
    if kind == "too_few_sources":
        return after_source_count > before_source_count
    return after_claim_count > before_claim_count or after_source_count > before_source_count


def gap_closure_evidence(topic: str, closures: list[ResearchGapClosure]) -> Evidence:
    return facade().Evidence(
        claim=f"Adaptive research gap closure for {topic!r} measured {len(closures)} targeted gap(s).",
        source_type="research_gap_closure",
        source_url=None,
        support=facade().SupportLevel.INFERRED,
        details={
            "topic": topic,
            "closure_count": len(closures),
            "closures": [
                {
                    "target_gap": item.target_gap,
                    "query": item.query,
                    "status": item.status,
                    "before_claim_count": item.before_claim_count,
                    "after_claim_count": item.after_claim_count,
                    "before_source_count": item.before_source_count,
                    "after_source_count": item.after_source_count,
                    "after_related_gap": item.after_related_gap,
                }
                for item in closures
            ],
        },
    )

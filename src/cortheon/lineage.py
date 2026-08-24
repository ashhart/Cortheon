from __future__ import annotations

from cortheon.models import CrawledPage, ResearchClaim, ScholarlyWork, SourceLineage


def build_source_lineage(
    claims: list[ResearchClaim],
    works: list[ScholarlyWork],
    pages: list[CrawledPage],
) -> list[SourceLineage]:
    source_meta: dict[str, tuple[str | None, str, float | None, float | None]] = {}
    for work in works:
        source_meta[work.url] = (
            work.title,
            f"scholarly:{work.source}",
            work.authority_score,
            work.relevance_score,
        )
    for page in pages:
        source_meta[page.final_url] = (
            page.title,
            page.source_type,
            page.authority_score,
            None,
        )

    claim_indexes_by_source: dict[str, list[int]] = {}
    for index, claim in enumerate(claims):
        claim_indexes_by_source.setdefault(claim.source_url, []).append(index)

    lineage: list[SourceLineage] = []
    for source_url, claim_indexes in claim_indexes_by_source.items():
        title, source_type, authority, relevance = source_meta.get(
            source_url,
            (
                claims[claim_indexes[0]].source_title,
                claims[claim_indexes[0]].source_type,
                None,
                None,
            ),
        )
        lineage.append(
            SourceLineage(
                source_url=source_url,
                source_title=title,
                source_type=source_type,
                authority_score=authority,
                relevance_score=relevance,
                derived_claim_indexes=claim_indexes,
            )
        )
    lineage.sort(
        key=lambda item: (
            len(item.derived_claim_indexes),
            item.authority_score or 0.0,
            item.relevance_score or 0.0,
        ),
        reverse=True,
    )
    return lineage

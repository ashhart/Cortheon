"""Composite and provider-specific scholarly discovery operations."""

from __future__ import annotations

from typing import Any

from cortheon.models import Evidence, ScholarlyWork


def composite_search(
    discovery: Any,
    query: str,
    limit: int,
    connector_names: list[str] | None,
    *,
    result_type: Any,
    connector_error: type[Exception],
    query_variants: Any,
    dedupe: Any,
    score_relevance: Any,
    minimum_relevance: Any,
    score_recency: Any,
    rank_key: Any,
    ceil: Any,
    evidence_type: Any,
    support_level: Any,
) -> Any:
    all_works: list[ScholarlyWork] = []
    evidence: list[Evidence] = []
    errors: list[str] = []
    selected = discovery._selected_connectors(connector_names)
    if limit <= 0 or not selected:
        evidence.append(
            evidence_type(
                claim="Scholarly discovery skipped because no scholarly connector was selected.",
                source_type="scholarly_discovery",
                source_url=None,
                support=support_level.INFERRED,
                details={
                    "query": query,
                    "requested_connectors": connector_names,
                    "available_connectors": [connector.name for connector in discovery.connectors],
                    "connector_count": len(selected),
                    "work_count": 0,
                },
            )
        )
        return result_type(works=[], evidence=evidence, errors=errors)
    variants = query_variants(query)
    per_connector = max(3, ceil(limit / max(len(selected), 1)))
    for connector in selected:
        for variant in variants:
            cache_key = (connector.name, variant, per_connector)
            try:
                result = discovery._connector_cache.get(cache_key)
                if result is None:
                    result = connector.search(variant, per_connector)
                    discovery._connector_cache[cache_key] = result
            except connector_error as exc:
                errors.append(f"{connector.name}: {exc}")
                continue
            all_works.extend(result.works)
            evidence.extend(result.evidence)
            errors.extend(result.errors)
    works = [score_relevance(work, query) for work in dedupe(all_works)]
    works = [work for work in works if work.relevance_score >= minimum_relevance(query)]
    works = [score_recency(work) for work in works]
    works = sorted(works, key=rank_key, reverse=True)[:limit]
    evidence.append(
        evidence_type(
            claim=f"Scholarly discovery returned {len(works)} unique work(s) for query.",
            source_type="scholarly_discovery",
            source_url=None,
            support=support_level.OBSERVED,
            details={
                "query": query,
                "selected_connectors": [connector.name for connector in selected],
                "connector_count": len(selected),
                "work_count": len(works),
                "minimum_relevance": minimum_relevance(query),
            },
        )
    )
    return result_type(works=works, evidence=evidence, errors=errors)


def arxiv_search(
    connector: Any,
    query: str,
    limit: int,
    *,
    result_type: Any,
    make_query: Any,
    urlencode: Any,
    parse_root: Any,
    clean: Any,
    parse_datetime: Any,
    arxiv_id: Any,
    atom_namespace: str,
    arxiv_namespace: str,
    evidence_type: Any,
    work_type: Any,
    support_level: Any,
) -> Any:
    search_query = make_query(query)
    params = urlencode(
        {
            "search_query": search_query,
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    url = f"https://export.arxiv.org/api/query?{params}"
    response = connector.client.get(url, headers={"Accept": "application/atom+xml"})
    root = parse_root(response.body)
    works: list[ScholarlyWork] = []
    for entry in root.findall(f"{atom_namespace}entry"):
        title = clean(entry.findtext(f"{atom_namespace}title") or "")
        abstract = clean(entry.findtext(f"{atom_namespace}summary") or "")
        work_url = (entry.findtext(f"{atom_namespace}id") or "").strip()
        authors = [
            clean(author.findtext(f"{atom_namespace}name") or "")
            for author in entry.findall(f"{atom_namespace}author")
        ]
        authors = [author for author in authors if author]
        published = parse_datetime(entry.findtext(f"{atom_namespace}published"))
        category = entry.find(f"{arxiv_namespace}primary_category")
        category_term = category.attrib.get("term") if category is not None else None
        identifiers = {"arxiv": arxiv_id(work_url)}
        doi_node = entry.find(f"{arxiv_namespace}doi")
        if doi_node is not None and doi_node.text:
            identifiers["doi"] = doi_node.text.strip()
        works.append(
            work_type(
                title=title,
                url=work_url,
                abstract=abstract or None,
                authors=authors,
                published_at=published,
                source="arxiv",
                venue=f"arXiv:{category_term}" if category_term else "arXiv",
                identifiers={key: value for key, value in identifiers.items() if value},
                cited_by_count=None,
                authority_score=0.88,
            )
        )
    return result_type(
        works=works,
        evidence=[
            evidence_type(
                claim=f"arXiv returned {len(works)} work(s) for query.",
                source_type="arxiv_api",
                source_url=url,
                support=support_level.OBSERVED,
                details={
                    "query": query,
                    "search_query": search_query,
                    "work_count": len(works),
                },
            )
        ],
        errors=[],
    )


def openalex_search(
    connector: Any,
    query: str,
    limit: int,
    *,
    result_type: Any,
    environ: Any,
    urlencode: Any,
    abstract_from_index: Any,
    clean: Any,
    parse_datetime: Any,
    log10: Any,
    evidence_type: Any,
    work_type: Any,
    support_level: Any,
) -> Any:
    params: dict[str, Any] = {
        "search": query,
        "per-page": limit,
        "sort": "relevance_score:desc",
    }
    mailto = environ.get("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
    url = "https://api.openalex.org/works?" + urlencode(params)
    payload = connector.client.get_json(url)
    works: list[ScholarlyWork] = []
    for item in list(payload.get("results") or [])[:limit]:
        title = item.get("title") or item.get("display_name")
        if not isinstance(title, str) or not title.strip():
            continue
        openalex_url = item.get("id") if isinstance(item.get("id"), str) else None
        doi = item.get("doi") if isinstance(item.get("doi"), str) else None
        url_value = doi or openalex_url or ""
        abstract = abstract_from_index(item.get("abstract_inverted_index"))
        authors = [
            (authorship.get("author") or {}).get("display_name")
            for authorship in item.get("authorships") or []
            if isinstance(authorship, dict)
        ]
        authors = [author for author in authors if isinstance(author, str)]
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {} if isinstance(primary_location, dict) else {}
        venue = source.get("display_name") if isinstance(source, dict) else None
        cited_by_count = (
            item.get("cited_by_count") if isinstance(item.get("cited_by_count"), int) else None
        )
        authority = 0.84 + min(0.12, log10(max(cited_by_count or 0, 1)) / 40)
        identifiers = {}
        if openalex_url:
            identifiers["openalex"] = openalex_url.rsplit("/", 1)[-1]
        if doi:
            identifiers["doi"] = doi.removeprefix("https://doi.org/")
        works.append(
            work_type(
                title=clean(title),
                url=url_value,
                abstract=abstract,
                authors=authors,
                published_at=parse_datetime(item.get("publication_date")),
                source="openalex",
                venue=venue,
                identifiers=identifiers,
                cited_by_count=cited_by_count,
                authority_score=round(min(authority, 0.96), 3),
            )
        )
    return result_type(
        works=works,
        evidence=[
            evidence_type(
                claim=f"OpenAlex returned {len(works)} work(s) for query.",
                source_type="openalex_api",
                source_url=url,
                support=support_level.OBSERVED,
                details={"query": query, "work_count": len(works)},
            )
        ],
        errors=[],
    )


def pubmed_search(
    connector: Any,
    query: str,
    limit: int,
    *,
    result_type: Any,
    efetch_url: Any,
    parse_articles: Any,
    evidence_type: Any,
    support_level: Any,
) -> Any:
    ids, search_url = connector._search_ids(query, limit)
    if not ids:
        return result_type(
            works=[],
            evidence=[
                evidence_type(
                    claim="PubMed returned 0 PMID(s) for query.",
                    source_type="pubmed_esearch",
                    source_url=search_url,
                    support=support_level.OBSERVED,
                    details={"query": query, "work_count": 0},
                )
            ],
            errors=[],
        )
    fetch_url = efetch_url(ids)
    response = connector.client.get(fetch_url, headers={"Accept": "application/xml"})
    works = parse_articles(response.body, limit)
    return result_type(
        works=works,
        evidence=[
            evidence_type(
                claim=f"PubMed returned {len(works)} work(s) for query.",
                source_type="pubmed_efetch",
                source_url=fetch_url,
                support=support_level.OBSERVED,
                details={"query": query, "pmids": ids, "work_count": len(works)},
            )
        ],
        errors=[],
    )


def pubmed_search_ids(
    connector: Any, query: str, limit: int, *, esearch_url: Any
) -> tuple[list[str], str]:
    url = esearch_url(query, limit)
    payload = connector.client.get_json(url)
    ids = (
        ((payload.get("esearchresult") or {}).get("idlist") or [])
        if isinstance(payload, dict)
        else []
    )
    return [str(item) for item in ids if str(item).strip()][:limit], url

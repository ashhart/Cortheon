"""Scholarly discovery, parsing, deduplication, and freshness ranking."""

from __future__ import annotations

import math
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET  # nosec B405
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from cortheon.connectors.http import ConnectorError, JsonHttpClient
from cortheon.models import Evidence, ScholarlyWork, SupportLevel, parse_datetime, utc_now
from cortheon.scholarly_core import connectors, parsing, queries, ranking
from cortheon.scholarly_core.constants import (
    ARXIV,
    ATOM,
    MAX_SCHOLARLY_XML_BYTES,
    RECENCY_FLOOR,
    RECENCY_STEPS,
    UNDATED_RECENCY,
)


@dataclass(slots=True)
class ScholarlyDiscoveryResult:
    works: list[ScholarlyWork]
    evidence: list[Evidence]
    errors: list[str]


class ScholarlyConnector:
    name = "scholarly"
    source_type = "scholarly"
    capabilities = ("scholarly_metadata", "papers")
    domains = ("science", "research")
    trust_tier = "scholarly_index"
    default_priority = 0.58

    def search(self, query: str, limit: int) -> ScholarlyDiscoveryResult:
        raise NotImplementedError

    def source_profile(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source_type": self.source_type,
            "capabilities": list(self.capabilities),
            "domains": list(self.domains),
            "trust_tier": self.trust_tier,
            "default_priority": self.default_priority,
            "available": True,
        }


class CompositeScholarlyDiscovery:
    def __init__(self, connectors: list[ScholarlyConnector] | None = None) -> None:
        self.connectors = connectors or [ArxivConnector(), OpenAlexConnector(), PubMedConnector()]
        self._connector_cache: dict[tuple[str, str, int], ScholarlyDiscoveryResult] = {}

    def source_profiles(self) -> list[dict[str, object]]:
        return [connector.source_profile() for connector in self.connectors]

    def search(
        self,
        query: str,
        limit: int,
        connector_names: list[str] | None = None,
    ) -> ScholarlyDiscoveryResult:
        return connectors.composite_search(
            self,
            query,
            limit,
            connector_names,
            result_type=ScholarlyDiscoveryResult,
            connector_error=ConnectorError,
            query_variants=scholarly_query_variants,
            dedupe=dedupe_works,
            score_relevance=score_work_relevance,
            minimum_relevance=minimum_relevance,
            score_recency=score_work_recency,
            rank_key=scholarly_rank_key,
            ceil=math.ceil,
            evidence_type=Evidence,
            support_level=SupportLevel,
        )

    def _selected_connectors(
        self,
        connector_names: list[str] | None,
    ) -> list[ScholarlyConnector]:
        if connector_names is None:
            return self.connectors
        requested = set(connector_names)
        return [connector for connector in self.connectors if connector.name in requested]


class ArxivConnector(ScholarlyConnector):
    name = "arxiv"
    capabilities = ("scholarly_metadata", "preprints", "computer_science", "papers")
    domains = ("science", "research", "software", "ai", "engineering")
    trust_tier = "preprint_metadata"
    default_priority = 0.64

    def __init__(self, client: JsonHttpClient | None = None) -> None:
        self.client = client or JsonHttpClient(timeout_seconds=20)

    def search(self, query: str, limit: int) -> ScholarlyDiscoveryResult:
        return connectors.arxiv_search(
            self,
            query,
            limit,
            result_type=ScholarlyDiscoveryResult,
            make_query=arxiv_query,
            urlencode=urllib.parse.urlencode,
            parse_root=bounded_xml_root,
            clean=clean_text,
            parse_datetime=parse_datetime,
            arxiv_id=arxiv_id_from_url,
            atom_namespace=ATOM,
            arxiv_namespace=ARXIV,
            evidence_type=Evidence,
            work_type=ScholarlyWork,
            support_level=SupportLevel,
        )


class OpenAlexConnector(ScholarlyConnector):
    name = "openalex"
    capabilities = (
        "scholarly_metadata",
        "citation_metadata",
        "cross_domain_metadata",
        "papers",
    )
    domains = ("science", "research", "medicine", "engineering", "general")
    trust_tier = "scholarly_metadata_index"
    default_priority = 0.68

    def __init__(self, client: JsonHttpClient | None = None) -> None:
        self.client = client or JsonHttpClient(timeout_seconds=20)

    def search(self, query: str, limit: int) -> ScholarlyDiscoveryResult:
        return connectors.openalex_search(
            self,
            query,
            limit,
            result_type=ScholarlyDiscoveryResult,
            environ=os.environ,
            urlencode=urllib.parse.urlencode,
            abstract_from_index=abstract_from_inverted_index,
            clean=clean_text,
            parse_datetime=parse_datetime,
            log10=math.log10,
            evidence_type=Evidence,
            work_type=ScholarlyWork,
            support_level=SupportLevel,
        )


class PubMedConnector(ScholarlyConnector):
    name = "pubmed"
    capabilities = (
        "scholarly_metadata",
        "biomedical_metadata",
        "clinical_literature",
        "papers",
    )
    domains = ("medicine", "biology", "health")
    trust_tier = "biomedical_literature_index"
    default_priority = 0.54

    def __init__(self, client: JsonHttpClient | None = None) -> None:
        self.client = client or JsonHttpClient(timeout_seconds=20)

    def search(self, query: str, limit: int) -> ScholarlyDiscoveryResult:
        return connectors.pubmed_search(
            self,
            query,
            limit,
            result_type=ScholarlyDiscoveryResult,
            efetch_url=pubmed_efetch_url,
            parse_articles=parse_pubmed_articles,
            evidence_type=Evidence,
            support_level=SupportLevel,
        )

    def _search_ids(self, query: str, limit: int) -> tuple[list[str], str]:
        return connectors.pubmed_search_ids(
            self,
            query,
            limit,
            esearch_url=pubmed_esearch_url,
        )


def pubmed_esearch_url(query: str, limit: int) -> str:
    return queries.pubmed_esearch_url(
        query,
        limit,
        base_params=pubmed_base_params,
        urlencode=urllib.parse.urlencode,
    )


def pubmed_efetch_url(ids: list[str]) -> str:
    return queries.pubmed_efetch_url(
        ids,
        base_params=pubmed_base_params,
        urlencode=urllib.parse.urlencode,
    )


def pubmed_base_params() -> dict[str, str]:
    return queries.pubmed_base_params(environ=os.environ)


def parse_pubmed_articles(body: bytes, limit: int) -> list[ScholarlyWork]:
    return parsing.parse_pubmed_articles(
        body,
        limit,
        parse_root=bounded_xml_root,
        clean=clean_text,
        element_text=element_text,
        abstract=pubmed_abstract,
        authors=pubmed_authors,
        article_id=pubmed_article_id,
        published_at=pubmed_published_at,
        work_type=ScholarlyWork,
    )


def bounded_xml_root(
    body: bytes,
    *,
    max_bytes: int = MAX_SCHOLARLY_XML_BYTES,
) -> ET.Element:
    """Parse bounded XML after rejecting entity declarations."""
    return parsing.bounded_xml_root(body, max_bytes=max_bytes, parser=ET.fromstring)


def pubmed_abstract(article: ET.Element) -> str | None:
    return parsing.pubmed_abstract(article, clean=clean_text, element_text=element_text)


def pubmed_authors(article: ET.Element) -> list[str]:
    return parsing.pubmed_authors(article, clean=clean_text)


def pubmed_article_id(article: ET.Element, id_type: str) -> str | None:
    return parsing.pubmed_article_id(article, id_type)


def pubmed_published_at(article: ET.Element) -> datetime | None:
    return parsing.pubmed_published_at(article, date_from_parts=date_from_parts)


def date_from_parts(
    year_value: str | None,
    month_value: str | None,
    day_value: str | None,
) -> datetime | None:
    return parsing.date_from_parts(
        year_value,
        month_value,
        day_value,
        parse_month=parse_month,
        search=re.search,
        datetime_type=datetime,
        utc=UTC,
    )


def parse_month(value: str | None) -> int:
    return parsing.parse_month(value)


def element_text(element: ET.Element | None) -> str:
    return parsing.element_text(element)


def abstract_from_inverted_index(value: Any) -> str | None:
    return parsing.abstract_from_inverted_index(value, clean=clean_text)


def scholarly_query_variants(query: str) -> list[str]:
    return queries.scholarly_query_variants(query, clean=clean_text, phrases=key_phrases)


def arxiv_query(query: str) -> str:
    return queries.arxiv_query(query, phrases=key_phrases, terms_for_query=query_terms)


def key_phrases(query: str) -> list[str]:
    return queries.key_phrases(
        query,
        normalize=normalize_for_match,
        terms_for_query=query_terms,
    )


def dedupe_works(works: list[ScholarlyWork]) -> list[ScholarlyWork]:
    return ranking.dedupe_works(works, normalize_title=normalize_title)


def score_work_relevance(work: ScholarlyWork, query: str) -> ScholarlyWork:
    return ranking.score_work_relevance(
        work,
        query,
        query_terms=query_terms,
        normalize=normalize_for_match,
    )


def score_work_recency(work: ScholarlyWork, now: datetime | None = None) -> ScholarlyWork:
    work.recency_score = work_recency_score(work.published_at, now)
    return work


def work_recency_score(published_at: datetime | None, now: datetime | None = None) -> float:
    return ranking.work_recency_score(
        published_at,
        now,
        current_time=utc_now,
        recency_steps=RECENCY_STEPS,
        recency_floor=RECENCY_FLOOR,
        undated_recency=UNDATED_RECENCY,
        utc=UTC,
    )


def scholarly_rank_key(work: ScholarlyWork) -> float:
    return ranking.scholarly_rank_key(work)


def query_terms(query: str) -> list[str]:
    return queries.query_terms(query, normalize=normalize_for_match, findall=re.findall)


def minimum_relevance(query: str) -> float:
    return ranking.minimum_relevance(query, query_terms=query_terms)


def clean_text(value: str) -> str:
    return queries.clean_text(value, substitute=re.sub)


def normalize_title(value: str) -> str:
    return queries.normalize_title(value, substitute=re.sub)


def normalize_for_match(value: str) -> str:
    return queries.normalize_for_match(value, substitute=re.sub)


def arxiv_id_from_url(url: str) -> str:
    return queries.arxiv_id_from_url(url)

"""Bounded XML parsing and scholarly metadata extraction."""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405
from datetime import datetime
from typing import Any

from cortheon.models import ScholarlyWork


def bounded_xml_root(body: bytes, *, max_bytes: int, parser: Any) -> ET.Element:
    if len(body) > max_bytes:
        raise ValueError(f"XML exceeds {max_bytes} bytes")
    upper = body.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ValueError("XML contains a forbidden DTD or entity declaration")
    try:
        return parser(body)  # nosec B314
    except ET.ParseError as exc:
        raise ValueError("XML is invalid") from exc


def parse_pubmed_articles(
    body: bytes,
    limit: int,
    *,
    parse_root: Any,
    clean: Any,
    element_text: Any,
    abstract: Any,
    authors: Any,
    article_id: Any,
    published_at: Any,
    work_type: Any,
) -> list[ScholarlyWork]:
    root = parse_root(body)
    works: list[ScholarlyWork] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = clean(article.findtext(".//PMID") or "")
        title = clean(element_text(article.find(".//ArticleTitle")))
        if not title:
            continue
        article_abstract = abstract(article)
        article_authors = authors(article)
        venue = (
            clean(article.findtext(".//Journal/Title") or "")
            or clean(article.findtext(".//Journal/ISOAbbreviation") or "")
            or None
        )
        doi = article_id(article, "doi")
        identifiers = {"pubmed": pmid}
        if doi:
            identifiers["doi"] = doi
        works.append(
            work_type(
                title=title,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                abstract=article_abstract,
                authors=article_authors,
                published_at=published_at(article),
                source="pubmed",
                venue=venue,
                identifiers={key: value for key, value in identifiers.items() if value},
                cited_by_count=None,
                authority_score=0.9,
            )
        )
        if len(works) >= limit:
            break
    return works


def pubmed_abstract(
    article: ET.Element,
    *,
    clean: Any,
    element_text: Any,
) -> str | None:
    parts: list[str] = []
    for item in article.findall(".//Abstract/AbstractText"):
        item_text = clean(element_text(item))
        if not item_text:
            continue
        label = clean(item.attrib.get("Label") or "")
        parts.append(f"{label}: {item_text}" if label else item_text)
    return clean(" ".join(parts)) or None


def pubmed_authors(article: ET.Element, *, clean: Any) -> list[str]:
    authors: list[str] = []
    for item in article.findall(".//AuthorList/Author"):
        collective = clean(item.findtext("CollectiveName") or "")
        if collective:
            authors.append(collective)
            continue
        fore_name = clean(item.findtext("ForeName") or "")
        last_name = clean(item.findtext("LastName") or "")
        full_name = clean(" ".join(part for part in [fore_name, last_name] if part))
        if full_name:
            authors.append(full_name)
    return authors


def pubmed_article_id(article: ET.Element, id_type: str) -> str | None:
    for item in article.findall(".//ArticleIdList/ArticleId"):
        if item.attrib.get("IdType") == id_type and item.text:
            return item.text.strip()
    return None


def pubmed_published_at(article: ET.Element, *, date_from_parts: Any) -> datetime | None:
    article_date = article.find(".//Article/ArticleDate")
    if article_date is not None:
        parsed = date_from_parts(
            article_date.findtext("Year"),
            article_date.findtext("Month"),
            article_date.findtext("Day"),
        )
        if parsed:
            return parsed
    pub_date = article.find(".//Journal/JournalIssue/PubDate")
    if pub_date is not None:
        parsed = date_from_parts(
            pub_date.findtext("Year") or pub_date.findtext("MedlineDate"),
            pub_date.findtext("Month"),
            pub_date.findtext("Day"),
        )
        if parsed:
            return parsed
    return None


def date_from_parts(
    year_value: str | None,
    month_value: str | None,
    day_value: str | None,
    *,
    parse_month: Any,
    search: Any,
    datetime_type: Any,
    utc: Any,
) -> datetime | None:
    year_match = search(r"\d{4}", year_value or "")
    if not year_match:
        return None
    year = int(year_match.group(0))
    month = parse_month(month_value)
    day = int(day_value) if day_value and day_value.isdigit() else 1
    try:
        return datetime_type(year, month, day, tzinfo=utc)
    except ValueError:
        return datetime_type(year, month, 1, tzinfo=utc)


def parse_month(value: str | None) -> int:
    if not value:
        return 1
    cleaned = value.strip().lower()[:3]
    if cleaned.isdigit():
        return min(12, max(1, int(cleaned)))
    months = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    return months.get(cleaned, 1)


def element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext())


def abstract_from_inverted_index(value: Any, *, clean: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    positioned: list[tuple[int, str]] = []
    for word, positions in value.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        positioned.extend((position, word) for position in positions if isinstance(position, int))
    if not positioned:
        return None
    positioned.sort(key=lambda item: item[0])
    return clean(" ".join(word for _, word in positioned))

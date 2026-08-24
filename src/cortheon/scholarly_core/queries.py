"""Query expansion, URL construction, and text normalization."""

from __future__ import annotations

from typing import Any


def pubmed_esearch_url(
    query: str,
    limit: int,
    *,
    base_params: Any,
    urlencode: Any,
) -> str:
    params = base_params()
    params.update(
        {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(max(0, limit)),
            "sort": "relevance",
        }
    )
    return "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urlencode(params)


def pubmed_efetch_url(ids: list[str], *, base_params: Any, urlencode: Any) -> str:
    params = base_params()
    params.update(
        {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
            "rettype": "abstract",
        }
    )
    return "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urlencode(params)


def pubmed_base_params(*, environ: Any) -> dict[str, str]:
    params = {"tool": "cortheon"}
    email = environ.get("NCBI_EMAIL") or environ.get("ENTREZ_EMAIL")
    if email:
        params["email"] = email
    api_key = environ.get("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


def scholarly_query_variants(
    query: str,
    *,
    clean: Any,
    phrases: Any,
) -> list[str]:
    normalized = clean(query.replace("-", " "))
    variants = [query]
    phrase_terms = phrases(query)
    if phrase_terms:
        variants.append(" ".join(f'"{phrase}"' for phrase in phrase_terms[:3]))
        variants.append(" ".join(phrase_terms[:4]))
    if normalized != query:
        variants.append(normalized)
    return list(dict.fromkeys(variant for variant in variants if variant.strip()))[:3]


def arxiv_query(query: str, *, phrases: Any, terms_for_query: Any) -> str:
    phrases_found = phrases(query)
    terms = terms_for_query(query)
    if phrases_found:
        phrase_clause = " OR ".join(f'all:"{phrase}"' for phrase in phrases_found[:4])
        required = []
        if "artificial" in terms and "life" in terms:
            required.append('all:"artificial life"')
        if "evolution" in terms:
            required.append("all:evolution")
        if required:
            return f"({phrase_clause}) AND " + " AND ".join(required[:2])
        return phrase_clause
    if terms:
        return " AND ".join(f"all:{term}" for term in terms[:5])
    return f'all:"{query}"'


def key_phrases(query: str, *, normalize: Any, terms_for_query: Any) -> list[str]:
    normalized = normalize(query)
    words = terms_for_query(query)
    phrases: list[str] = []
    if "open" in words and ("ended" in words or "endedness" in words):
        phrases.extend(["open ended", "open endedness"])
    for size in (3, 2):
        for index in range(0, max(0, len(words) - size + 1)):
            phrase = " ".join(words[index : index + size])
            if phrase not in phrases:
                phrases.append(phrase)
    if normalized and len(normalized.split()) <= 6 and normalized not in phrases:
        phrases.append(normalized)
    return phrases


def query_terms(query: str, *, normalize: Any, findall: Any) -> list[str]:
    stopwords = {
        "about",
        "after",
        "best",
        "build",
        "current",
        "engine",
        "from",
        "into",
        "latest",
        "major",
        "should",
        "that",
        "this",
        "what",
        "with",
    }
    terms = [
        term
        for term in findall(r"[a-zA-Z][a-zA-Z0-9]+", normalize(query))
        if len(term) >= 4 and term not in stopwords
    ]
    return list(dict.fromkeys(terms))


def clean_text(value: str, *, substitute: Any) -> str:
    return substitute(r"\s+", " ", value).strip()


def normalize_title(value: str, *, substitute: Any) -> str:
    return substitute(r"[^a-z0-9]+", " ", value.lower()).strip()


def normalize_for_match(value: str, *, substitute: Any) -> str:
    return substitute(r"[^a-z0-9]+", " ", value.lower()).strip()


def arxiv_id_from_url(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1] if url else ""

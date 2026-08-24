from __future__ import annotations

import re
import urllib.parse

from cortheon.connectors.github import parse_owner_repo
from cortheon.models import CrawledPage, ResearchArtifact, ScholarlyWork, SearchResult

STOPWORDS = {
    "about",
    "after",
    "against",
    "and",
    "are",
    "benchmark",
    "build",
    "current",
    "engine",
    "for",
    "from",
    "how",
    "into",
    "new",
    "open",
    "research",
    "the",
    "this",
    "with",
}

KIND_BASE_CONFIDENCE = {
    "paper_pdf": 0.82,
    "paper_source": 0.76,
    "code_repository": 0.78,
    "dataset": 0.74,
    "benchmark": 0.72,
    "doi": 0.68,
    "web_artifact": 0.55,
}

ARXIV_ID_RE = re.compile(r"(?P<id>\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)")


def derive_research_artifacts(
    topic: str,
    scholarly_works: list[ScholarlyWork],
    pages: list[CrawledPage],
    search_results: list[SearchResult] | None = None,
    discovered_artifacts: list[ResearchArtifact] | None = None,
) -> list[ResearchArtifact]:
    artifacts: list[ResearchArtifact] = []
    artifacts.extend(discovered_artifacts or [])

    for work in scholarly_works:
        artifacts.extend(artifacts_from_work(topic, work))

    for result in search_results or []:
        artifacts.extend(
            artifacts_from_url(
                topic,
                result.url,
                title=result.title,
                source_url=result.url,
                provider=f"search:{result.provider}",
                context=result.snippet or result.title,
                source_type="search_result",
            )
        )

    for page in pages:
        context = f"{page.title or ''} {page.text[:1200]}"
        artifacts.extend(
            artifacts_from_url(
                topic,
                page.final_url,
                title=page.title,
                source_url=page.final_url,
                provider=f"crawl:{page.source_type}",
                context=context,
                source_type=page.source_type,
            )
        )
        for link in page.links:
            artifacts.extend(
                artifacts_from_url(
                    topic,
                    link,
                    title=None,
                    source_url=page.final_url,
                    provider=f"crawl:{page.source_type}",
                    context=context,
                    source_type=page.source_type,
                )
            )

    return dedupe_artifacts(artifacts)


def artifacts_from_work(topic: str, work: ScholarlyWork) -> list[ResearchArtifact]:
    artifacts: list[ResearchArtifact] = []
    arxiv_id = arxiv_id_from_work(work)
    if arxiv_id:
        metadata = {
            "paper_url": work.url,
            "source": work.source,
            "identifier": arxiv_id,
        }
        artifacts.append(
            make_artifact(
                topic,
                kind="paper_pdf",
                title=work.title,
                url=f"https://arxiv.org/pdf/{arxiv_id}",
                source_url=work.url,
                provider=f"scholarly:{work.source}",
                evidence=f"Derived arXiv PDF from scholarly work metadata: {work.title}",
                context=f"{work.title} {work.abstract or ''}",
                source_type="paper",
                metadata=metadata,
            )
        )
        artifacts.append(
            make_artifact(
                topic,
                kind="paper_source",
                title=f"{work.title} source",
                url=f"https://arxiv.org/e-print/{arxiv_id}",
                source_url=work.url,
                provider=f"scholarly:{work.source}",
                evidence=f"Derived arXiv source package from scholarly work metadata: {work.title}",
                context=f"{work.title} {work.abstract or ''}",
                source_type="paper",
                metadata=metadata,
            )
        )

    doi = work.identifiers.get("doi")
    if doi:
        doi_url = (
            doi if doi.startswith(("http://", "https://")) else f"https://doi.org/{doi.lstrip('/')}"
        )
        artifacts.append(
            make_artifact(
                topic,
                kind="doi",
                title=work.title,
                url=doi_url,
                source_url=work.url,
                provider=f"scholarly:{work.source}",
                evidence=f"Scholarly work metadata included DOI for: {work.title}",
                context=f"{work.title} {work.abstract or ''}",
                source_type="paper",
                metadata={"paper_url": work.url, "source": work.source},
            )
        )

    artifacts.extend(
        artifacts_from_url(
            topic,
            work.url,
            title=work.title,
            source_url=work.url,
            provider=f"scholarly:{work.source}",
            context=f"{work.title} {work.abstract or ''}",
            source_type="paper",
        )
    )
    return artifacts


def artifacts_from_url(
    topic: str,
    url: str,
    *,
    title: str | None,
    source_url: str | None,
    provider: str,
    context: str,
    source_type: str | None,
) -> list[ResearchArtifact]:
    arxiv_id = arxiv_id_from_url(url)
    if arxiv_id:
        work_title = title or f"arXiv {arxiv_id}"
        return [
            make_artifact(
                topic,
                kind="paper_pdf",
                title=work_title,
                url=f"https://arxiv.org/pdf/{arxiv_id}",
                source_url=source_url,
                provider=provider,
                evidence=f"Derived arXiv PDF from URL: {url}",
                context=context,
                source_type=source_type,
                metadata={"identifier": arxiv_id},
            ),
            make_artifact(
                topic,
                kind="paper_source",
                title=f"{work_title} source",
                url=f"https://arxiv.org/e-print/{arxiv_id}",
                source_url=source_url,
                provider=provider,
                evidence=f"Derived arXiv source package from URL: {url}",
                context=context,
                source_type=source_type,
                metadata={"identifier": arxiv_id},
            ),
        ]

    kind = classify_artifact_url(url, context, source_type)
    if not kind:
        return []
    return [
        make_artifact(
            topic,
            kind=kind,
            title=title or inferred_title(url),
            url=canonical_url(url),
            source_url=source_url,
            provider=provider,
            evidence=evidence_for_url(kind, url, context),
            context=context,
            source_type=source_type,
            metadata=metadata_for_url(url, source_type),
        )
    ]


def classify_artifact_url(url: str, context: str, source_type: str | None) -> str | None:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.lower()
    lower_url = url.lower()
    lower_context = context.lower()

    if parse_owner_repo(url):
        return "code_repository"
    if host == "huggingface.co" and path.startswith("/datasets/"):
        return "dataset"
    if any(domain in host for domain in ("zenodo.org", "kaggle.com", "figshare.com", "dataverse.")):
        return "dataset"
    if "archive.ics.uci.edu" in host or "openml.org" in host:
        return "dataset"
    if host.endswith("paperswithcode.com"):
        if "/dataset" in path or "/datasets" in path:
            return "dataset"
        return "benchmark"
    if lower_url.endswith(".pdf"):
        return "paper_pdf"
    if host == "doi.org":
        return "doi"
    if source_type == "source_repository":
        return "code_repository"
    if source_type == "benchmark":
        return "benchmark"
    if any(
        cue in lower_url for cue in ("leaderboard", "benchmark", "eval-suite", "evaluation-suite")
    ):
        return "benchmark"
    if any(cue in lower_context for cue in ("leaderboard", "benchmark dataset", "benchmark suite")):
        return "benchmark"
    return None


def make_artifact(
    topic: str,
    *,
    kind: str,
    title: str | None,
    url: str,
    source_url: str | None,
    provider: str,
    evidence: str | None,
    context: str,
    source_type: str | None,
    metadata: dict[str, str] | None = None,
) -> ResearchArtifact:
    return ResearchArtifact(
        kind=kind,
        title=title,
        url=canonical_url(url),
        source_url=source_url,
        provider=provider,
        evidence=evidence,
        confidence=artifact_confidence(kind, topic, title or "", context, provider, source_type),
        metadata=metadata or {},
    )


def artifact_confidence(
    kind: str,
    topic: str,
    title: str,
    context: str,
    provider: str,
    source_type: str | None,
) -> float:
    score = KIND_BASE_CONFIDENCE.get(kind, 0.55)
    if provider.startswith("scholarly:") or source_type == "paper":
        score += 0.05
    if provider == "github_search":
        score += 0.05
    if source_type in {"benchmark", "source_repository"}:
        score += 0.04
    score += topic_overlap_bonus(topic, f"{title} {context}")
    return round(min(score, 0.96), 3)


def topic_overlap_bonus(topic: str, text: str) -> float:
    terms = {
        term
        for term in re.findall(r"[a-z0-9][a-z0-9-]{2,}", topic.lower())
        if term not in STOPWORDS
    }
    if not terms:
        return 0.0
    lower_text = text.lower()
    hits = sum(1 for term in terms if term in lower_text)
    return min(0.08, hits * 0.02)


def arxiv_id_from_work(work: ScholarlyWork) -> str | None:
    url_id = arxiv_id_from_url(work.url)
    if url_id:
        return url_id
    value = work.identifiers.get("arxiv") or work.identifiers.get("arXiv")
    if value:
        return normalize_arxiv_id(value)
    return None


def arxiv_id_from_url(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    if host != "arxiv.org":
        return None
    path = parsed.path.strip("/")
    for prefix in ("abs/", "pdf/", "e-print/"):
        if path.startswith(prefix):
            return normalize_arxiv_id(path.removeprefix(prefix))
    return None


def normalize_arxiv_id(value: str) -> str | None:
    cleaned = value.strip().removeprefix("arXiv:").removesuffix(".pdf")
    match = ARXIV_ID_RE.search(cleaned)
    if match:
        return match.group("id")
    fallback = cleaned.split("?", 1)[0].split("#", 1)[0].strip("/")
    if fallback and re.fullmatch(r"[A-Za-z0-9._/-]+", fallback):
        return fallback
    return None


def evidence_for_url(kind: str, url: str, context: str) -> str:
    cue = context.strip()[:160]
    if cue:
        return f"Classified {url} as {kind} from source context: {cue}"
    return f"Classified {url} as {kind} from URL pattern."


def metadata_for_url(url: str, source_type: str | None) -> dict[str, str]:
    parsed = urllib.parse.urlparse(url)
    metadata = {
        "domain": parsed.netloc.lower().removeprefix("www."),
    }
    if source_type:
        metadata["source_type"] = source_type
    owner_repo = parse_owner_repo(url)
    if owner_repo:
        metadata["repo"] = f"{owner_repo[0]}/{owner_repo[1]}"
    return metadata


def inferred_title(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return parsed.netloc or None
    return path.split("/")[-1].replace("-", " ").replace("_", " ") or parsed.netloc


def dedupe_artifacts(artifacts: list[ResearchArtifact]) -> list[ResearchArtifact]:
    best: dict[str, ResearchArtifact] = {}
    for artifact in artifacts:
        key = canonical_url(artifact.url)
        current = best.get(key)
        if current is None or artifact.confidence > current.confidence:
            best[key] = artifact
    return sorted(best.values(), key=lambda item: (-item.confidence, item.kind, item.url))


def canonical_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path or "/"
    return urllib.parse.urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower().removeprefix("www."),
            path.rstrip("/") or "/",
            "",
            parsed.query,
            "",
        )
    )

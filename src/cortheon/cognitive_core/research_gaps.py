"""Research completion gaps, conflict detection, and release analysis."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from cortheon.cognitive_core.claims import _observation_body
from cortheon.cognitive_core.models import Investigation, Observation
from cortheon.cognitive_core.profiles import _EXPLICIT_FRESHNESS_HINTS, _has_hint
from cortheon.cognitive_core.receipts import _observation_origin
from cortheon.cognitive_core.semantic_graph import _semantic_terms
from cortheon.cognitive_protocol import evaluation_operator

_LOCAL_PROJECT_DOMAIN_RE = re.compile(
    r"(?:"
    r"\b(?:local|checked[- ]out)\s+"
    r"(?:[A-Za-z0-9_.-]+\s+){0,3}"
    r"(?:workspace|worktree|repository|repo|codebase|source\s+tree|"
    r"project\s+files?)\b"
    r"|"
    r"\b(?:workspace|worktree|repository|repo|codebase|source\s+tree|"
    r"project\s+files?)\s+(?:evidence|grounding|inspection|context)\b"
    r")",
    flags=re.IGNORECASE,
)


_RESEARCH_CONFLICT_ACK_RE = re.compile(
    r"\b(?:"
    r"conflict(?:s|ed|ing)?|contradict(?:ion|ory|s|ed)?|"
    r"disagree(?:ment|s|d)?|differ(?:ence|ences|ent|s|ed)?|"
    r"however|uncertain(?:ty)?|tension|trade[- ]?offs?|limitations?|"
    r"caveats?|counter(?:evidence|examples?)|reconcil(?:e|es|ed|ing)|"
    r"oppos(?:e|es|ed|ing)|not\s+interchangeable|scope(?:d|s|ing)?"
    r")\b",
    flags=re.IGNORECASE,
)


_RESEARCH_SCOPED_CONTRAST_RE = re.compile(
    r"\b(?:but|yet|whereas|although|while|even\s+(?:if|when|though)|"
    r"on\s+the\s+other\s+hand)\b",
    flags=re.IGNORECASE,
)


_RESEARCH_DOWNSIDE_RE = re.compile(
    r"\b(?:worsen(?:s|ed|ing)?|slow(?:er|s|ed|ing)?|hurt(?:s|ing)?|"
    r"degrad(?:e|es|ed|ing)|reduc(?:e|es|ed|ing)|lower(?:s|ed|ing)?|"
    r"penalt(?:y|ies)|costs?|risks?|fails?|failure)\b",
    flags=re.IGNORECASE,
)


_RESEARCH_UPSIDE_RE = re.compile(
    r"\b(?:improv(?:e|es|ed|ing)|faster|gains?|increas(?:e|es|ed|ing)|"
    r"higher|benefits?|speedups?|scal(?:e|es|ed|ing)|rises?|rising)\b",
    flags=re.IGNORECASE,
)


def _effective_web_lineages(
    observations: list[Observation],
) -> tuple[int, int, list[dict[str, str]]]:
    """Count independently worded origins, collapsing likely syndicated text."""

    by_origin: dict[str, list[set[str]]] = {}
    for observation in observations:
        if observation.kind != "web":
            continue
        origin = _observation_origin(observation)
        if origin is None:
            continue
        terms = _semantic_terms(_observation_body(observation))
        if terms:
            by_origin.setdefault(origin, []).append(terms)
        else:
            by_origin.setdefault(origin, [])
    origins = sorted(by_origin)
    parents = {origin: origin for origin in origins}

    def find(origin: str) -> str:
        while parents[origin] != origin:
            parents[origin] = parents[parents[origin]]
            origin = parents[origin]
        return origin

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    likely_syndicated: list[dict[str, str]] = []
    for index, left in enumerate(origins):
        for right in origins[index + 1 :]:
            copied = False
            for left_terms in by_origin[left]:
                for right_terms in by_origin[right]:
                    shared = left_terms & right_terms
                    union_terms = left_terms | right_terms
                    if len(shared) >= 8 and len(shared) / max(1, len(union_terms)) >= 0.85:
                        copied = True
                        break
                if copied:
                    break
            if copied:
                union(left, right)
                likely_syndicated.append(
                    {
                        "origin_a": left,
                        "origin_b": right,
                        "reason": "near-identical evidence wording",
                    }
                )
    return len(origins), len({find(origin) for origin in origins}), likely_syndicated


def _research_completion_gaps(
    session: Investigation,
    candidates: list[Observation],
) -> list[str]:
    web = [item for item in candidates if item.kind == "web" and item.status != "failed"]
    origins = {_observation_origin(item) for item in web}
    origins.discard(None)
    waived = session.waivers
    required_origins = 1 if "corroboration" in waived else 2
    gaps: list[str] = []
    if len(origins) < required_origins:
        gaps.append(
            "A research answer requires corroboration from two independent URL origins."
            if required_origins == 2
            else "A research answer requires at least one attributable URL origin."
        )

    now = datetime.now(UTC)
    current_retrievals = []
    for item in web:
        if not item.retrieved_at:
            continue
        retrieved = datetime.fromisoformat(item.retrieved_at)
        age_seconds = (now - retrieved).total_seconds()
        if -300 <= age_seconds <= 3_600:
            current_retrievals.append(item)
    retrieval_origins = {_observation_origin(item) for item in current_retrievals}
    retrieval_origins.discard(None)
    if len(retrieval_origins) < required_origins:
        gaps.append(
            "Two independent sources need host-recorded retrieval timestamps from "
            "this live research session."
            if required_origins == 2
            else "At least one source needs a host-recorded retrieval timestamp "
            "from this live research session."
        )

    purposes = {item.purpose for item in web if item.purpose}
    if "primary_fetch" not in purposes and "primary_fetch" not in waived:
        gaps.append("At least one primary source must be fetched beyond a search snippet.")
    if (
        evaluation_operator(session.evaluation_profile, "contradiction_revision")
        and "contradiction_check" not in purposes
        and "contradiction_check" not in waived
    ):
        gaps.append("A scoped search for contradiction, correction, or counterevidence is missing.")
    if (
        _LOCAL_PROJECT_DOMAIN_RE.search(session.goal)
        and "inspect" not in waived
        and not any(_is_local_project_evidence(item) for item in candidates)
    ):
        gaps.append(
            "This goal explicitly requires local workspace/repository grounding, "
            "but no host-receipted local project evidence is present in "
            "completion_evidence_ids."
        )

    if _has_hint(session.goal, _EXPLICIT_FRESHNESS_HINTS) and "freshness_check" not in waived:
        dated = [datetime.fromisoformat(item.published_at) for item in web if item.published_at]
        if not dated:
            gaps.append("This freshness-sensitive answer requires a publication or update date.")
        elif max(dated) > now.replace(microsecond=0):
            gaps.append("A source publication date is implausibly in the future.")
    return gaps


def _answer_urls(answer: str) -> set[str]:
    urls = {match.rstrip(".,;:!?)\\]}'\"") for match in re.findall(r"https?://[^\s<>\"]+", answer)}
    origins: set[str] = set()
    for url in urls:
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "").casefold().removeprefix("www.")
        if not hostname:
            continue
        labels = hostname.split(".")
        if len(labels) <= 2:
            origins.add(hostname)
            continue
        public_suffix = ".".join(labels[-2:])
        origins.add(
            ".".join(labels[-3:])
            if public_suffix in {"co.jp", "co.uk", "com.au", "gov.uk", "org.uk"}
            else public_suffix
        )
    return origins


def _is_local_project_evidence(observation: Observation) -> bool:
    if observation.kind not in {
        "code",
        "diff",
        "documentation",
        "artifact",
        "test",
    }:
        return False
    receipt = observation.host_receipt
    if receipt is None:
        return False
    tool = str(receipt.get("tool") or "").casefold()
    if tool not in {"diff", "find", "git", "glob", "grep", "read", "test"}:
        return False
    if observation.kind in {"diff", "test"}:
        return True
    source = observation.source or ""
    if source and not urlsplit(source).scheme:
        return True
    arguments = receipt.get("args")
    if not isinstance(arguments, dict):
        return False
    return any(
        key in arguments
        for key in (
            "cwd",
            "filePath",
            "path",
            "paths",
            "repository",
            "root",
            "workdir",
        )
    )


def _answer_acknowledges_research_conflict(
    answer: str,
    observations: list[Observation],
) -> bool:
    without_urls = re.sub(r"https?://[^\s<>\"]+", " ", answer)
    answer_terms = _semantic_terms(without_urls)
    evidence_terms = {
        term for observation in observations for term in _semantic_terms(observation.content)
    }
    if len(answer_terms.intersection(evidence_terms)) < 2:
        return False
    if _RESEARCH_CONFLICT_ACK_RE.search(without_urls):
        return True
    return bool(
        _RESEARCH_SCOPED_CONTRAST_RE.search(without_urls)
        and _RESEARCH_DOWNSIDE_RE.search(without_urls)
        and _RESEARCH_UPSIDE_RE.search(without_urls)
    )


def _release_version_candidates(content: str) -> set[str]:
    patterns = (
        r"(?:latest|release(?:d)?|version|package-header__name)[^0-9\n]{0,80}"
        r"v?(\d+\.\d+(?:\.\d+){0,3})",
        r"\b[A-Za-z][A-Za-z0-9_-]{0,50}-(\d+\.\d+(?:\.\d+){1,3})"
        r"(?:[-+._]|$)",
        r"(?m)^\s*v?(\d+\.\d+(?:\.\d+){1,3})\s*$",
    )
    versions: set[str] = set()
    for pattern in patterns:
        versions.update(
            match.group(1) for match in re.finditer(pattern, content, flags=re.IGNORECASE)
        )
    return versions


def _version_key(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError:
        return ()


_LATEST_RELEASE_HINTS = frozenset({"latest", "newest", "current"})
_RELEASE_VERSION_HINTS = frozenset({"release", "released", "version"})


def _latest_release_goal(goal: str) -> bool:
    return _has_hint(goal, _LATEST_RELEASE_HINTS) and _has_hint(goal, _RELEASE_VERSION_HINTS)


def _research_release_analysis(
    goal: str,
    observations: Iterable[Observation],
) -> dict[str, Any] | None:
    if not _latest_release_goal(goal):
        return None
    by_origin: dict[str, set[str]] = {}
    sources: dict[str, list[str]] = {}
    for item in observations:
        if item.kind != "web" or item.status == "failed" or item.quarantine_flags:
            continue
        origin = _observation_origin(item)
        if origin is None:
            continue
        versions = _release_version_candidates(item.content)
        if not versions:
            continue
        by_origin.setdefault(origin, set()).update(versions)
        source = item.url or item.source
        if source:
            sources.setdefault(origin, []).append(source)
    support: dict[str, set[str]] = {}
    for origin, versions in by_origin.items():
        for version in versions:
            support.setdefault(version, set()).add(origin)
    eligible = [version for version, origins in support.items() if len(origins) >= 2]
    if not eligible:
        return None
    selected = max(
        eligible,
        key=lambda version: (len(support[version]), _version_key(version)),
    )
    selected_sources = [
        sources[origin][0] for origin in sorted(support[selected]) if sources.get(origin)
    ]
    return {
        "value": selected,
        "sources": selected_sources,
        "independent_origins": len(support[selected]),
    }

from __future__ import annotations

import urllib.parse
from typing import Any

from cortheon.connectors.http import ConnectorError, JsonHttpClient
from cortheon.models import Evidence, ResearchArtifact, SupportLevel


class ClinicalTrialsGovDiscovery:
    name = "clinicaltrials_gov"

    def __init__(self, client: JsonHttpClient | None = None) -> None:
        self.client = client or JsonHttpClient(timeout_seconds=20)

    def source_profiles(self) -> list[dict[str, object]]:
        return [
            {
                "name": self.name,
                "source_type": "trial_registry",
                "capabilities": [
                    "clinical_trials",
                    "registered_studies",
                    "interventions",
                    "eligibility",
                    "trial_status",
                ],
                "domains": ["medicine", "health", "biology"],
                "trust_tier": "official_trial_registry",
                "default_priority": 0.62,
                "available": True,
            }
        ]

    def search(
        self, query: str, limit: int = 5
    ) -> tuple[list[ResearchArtifact], list[Evidence], list[str]]:
        if limit <= 0:
            return [], [], []
        url = clinical_trials_search_url(query, limit)
        try:
            payload = self.client.get_json(url)
        except ConnectorError as exc:
            return [], [], [f"ClinicalTrials.gov search unavailable: {exc}"]
        studies = payload.get("studies") if isinstance(payload, dict) else None
        if not isinstance(studies, list):
            return [], [], ["ClinicalTrials.gov search returned an unexpected response shape."]
        artifacts = [
            artifact
            for artifact in (
                study_to_artifact(study, source_url=url, query=query) for study in studies[:limit]
            )
            if artifact is not None
        ]
        evidence = [
            Evidence(
                claim=f"ClinicalTrials.gov returned {len(artifacts)} trial artifact(s) for: {query}",
                source_type="clinicaltrials_gov_search",
                source_url=url,
                support=SupportLevel.OBSERVED,
                details={
                    "query": query,
                    "artifact_count": len(artifacts),
                    "total_count": payload.get("totalCount") if isinstance(payload, dict) else None,
                },
            )
        ]
        return artifacts, evidence, []


def clinical_trials_search_url(query: str, limit: int) -> str:
    params = {
        "query.term": query,
        "pageSize": str(min(max(limit, 1), 100)),
        "format": "json",
        "countTotal": "true",
        "sort": "@relevance",
    }
    return "https://clinicaltrials.gov/api/v2/studies?" + urllib.parse.urlencode(params)


def study_to_artifact(study: object, *, source_url: str, query: str) -> ResearchArtifact | None:
    if not isinstance(study, dict):
        return None
    protocol = study.get("protocolSection")
    if not isinstance(protocol, dict):
        return None
    identification = get_mapping(protocol, "identificationModule")
    status = get_mapping(protocol, "statusModule")
    design = get_mapping(protocol, "designModule")
    conditions = get_mapping(protocol, "conditionsModule")
    arms = get_mapping(protocol, "armsInterventionsModule")
    eligibility = get_mapping(protocol, "eligibilityModule")
    nct_id = string_value(identification.get("nctId"))
    title = (
        string_value(identification.get("briefTitle"))
        or string_value(identification.get("officialTitle"))
        or nct_id
    )
    if not nct_id or not title:
        return None
    url = f"https://clinicaltrials.gov/study/{nct_id}"
    summary = artifact_summary(status, design, conditions, arms, eligibility)
    metadata = {
        "nct_id": nct_id,
        "overall_status": string_value(status.get("overallStatus")) or "",
        "phase": ",".join(string_list(design.get("phases"))),
        "study_type": string_value(design.get("studyType")) or "",
        "conditions": ",".join(string_list(conditions.get("conditions"))[:8]),
        "interventions": ",".join(intervention_names(arms)[:8]),
        "eligibility": truncate(string_value(eligibility.get("eligibilityCriteria")) or "", 400),
    }
    return ResearchArtifact(
        kind="clinical_trial",
        title=title,
        url=url,
        source_url=source_url,
        provider="clinicaltrials_gov",
        evidence=(
            f"ClinicalTrials.gov registered trial {nct_id}: {summary}"
            if summary
            else f"ClinicalTrials.gov registered trial {nct_id} matched query: {query}"
        ),
        confidence=trial_confidence(query, title, metadata),
        metadata={key: value for key, value in metadata.items() if value},
    )


def artifact_summary(
    status: dict[str, Any],
    design: dict[str, Any],
    conditions: dict[str, Any],
    arms: dict[str, Any],
    eligibility: dict[str, Any],
) -> str:
    parts = [
        string_value(status.get("overallStatus")),
        ", ".join(string_list(design.get("phases"))[:3]),
        ", ".join(string_list(conditions.get("conditions"))[:3]),
        ", ".join(intervention_names(arms)[:3]),
    ]
    if eligibility.get("eligibilityCriteria"):
        parts.append("eligibility criteria available")
    return "; ".join(part for part in parts if part)


def trial_confidence(query: str, title: str, metadata: dict[str, str]) -> float:
    score = 0.78
    status = metadata.get("overall_status", "").lower()
    if status in {"recruiting", "active_not_recruiting", "completed"}:
        score += 0.05
    phase = metadata.get("phase", "").lower()
    if "phase3" in phase or "phase 3" in phase:
        score += 0.04
    if "phase2" in phase or "phase 2" in phase:
        score += 0.02
    haystack = (
        f"{title} {metadata.get('conditions', '')} {metadata.get('interventions', '')}".lower()
    )
    hits = sum(1 for term in topic_terms(query) if term in haystack)
    score += min(0.08, hits * 0.02)
    return round(min(score, 0.96), 3)


def get_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    return value if isinstance(value, dict) else {}


def string_value(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def intervention_names(arms: dict[str, Any]) -> list[str]:
    values = arms.get("interventions")
    if not isinstance(values, list):
        return []
    names: list[str] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        name = string_value(item.get("name"))
        if name:
            names.append(name)
    return names


def topic_terms(query: str) -> list[str]:
    stopwords = {"and", "for", "the", "with", "clinical", "trial", "engine", "cure"}
    terms = [
        term
        for term in query.lower().replace("-", " ").split()
        if len(term) >= 4 and term not in stopwords
    ]
    return list(dict.fromkeys(terms))


def truncate(value: str, limit: int) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."

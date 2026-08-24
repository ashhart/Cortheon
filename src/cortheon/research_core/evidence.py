from __future__ import annotations

from collections.abc import Sequence

from cortheon.models import (
    Evidence,
    ResearchArtifact,
    ResearchArtifactAssessment,
    ResearchCoverageItem,
    ResearchDiscoveryPass,
    ResearchQuery,
)
from cortheon.research_core._compat import facade


def synthesis_evidence(topic: str, status: str, confidence: float) -> Evidence:
    return facade().Evidence(
        claim=f"Research synthesis for {topic!r} produced status {status} with confidence {confidence:.3f}.",
        source_type="research_synthesis",
        source_url=None,
        support=facade().SupportLevel.INFERRED,
        details={"topic": topic, "status": status, "confidence": confidence},
    )


def grounding_evidence(topic: str, claims: Sequence[object]) -> Evidence:
    grounded_count = sum(
        1
        for claim in claims
        if getattr(claim, "source_excerpt", None)
        and getattr(claim, "source_char_start", None) is not None
        and getattr(claim, "source_char_end", None) is not None
    )
    return facade().Evidence(
        claim=f"Research grounding for {topic!r} attached excerpts/spans to {grounded_count} claim(s).",
        source_type="claim_grounding",
        source_url=None,
        support=facade().SupportLevel.INFERRED,
        details={
            "topic": topic,
            "claim_count": len(claims),
            "grounded_claim_count": grounded_count,
        },
    )


def lineage_evidence(topic: str, source_lineage: Sequence[object]) -> Evidence:
    return facade().Evidence(
        claim=f"Research lineage for {topic!r} mapped claims to {len(source_lineage)} source(s).",
        source_type="source_lineage",
        source_url=None,
        support=facade().SupportLevel.INFERRED,
        details={"topic": topic, "source_count": len(source_lineage)},
    )


def artifact_evidence(topic: str, artifacts: list[ResearchArtifact]) -> Evidence:
    return facade().Evidence(
        claim=f"Research artifact discovery for {topic!r} identified {len(artifacts)} artifact(s).",
        source_type="research_artifact_discovery",
        source_url=None,
        support=facade().SupportLevel.INFERRED,
        details={
            "topic": topic,
            "artifact_count": len(artifacts),
            "artifact_mix": facade().artifact_mix(artifacts),
            "top_artifacts": [
                {
                    "kind": artifact.kind,
                    "title": artifact.title,
                    "url": artifact.url,
                    "confidence": artifact.confidence,
                    "provider": artifact.provider,
                }
                for artifact in artifacts[:10]
            ],
        },
    )


def artifact_assessment_evidence(
    topic: str,
    assessments: list[ResearchArtifactAssessment],
) -> Evidence:
    top = assessments[0] if assessments else None
    return facade().Evidence(
        claim=(
            f"Research artifact assessment for {topic!r} ranked {len(assessments)} artifact(s)"
            + (f"; top decision is {top.decision} for {top.artifact_url}." if top else ".")
        ),
        source_type="research_artifact_assessment",
        source_url=top.artifact_url if top else None,
        support=facade().SupportLevel.INFERRED,
        details={
            "topic": topic,
            "assessment_count": len(assessments),
            "top_assessments": [
                {
                    "artifact_url": item.artifact_url,
                    "artifact_kind": item.artifact_kind,
                    "title": item.title,
                    "score": item.score,
                    "decision": item.decision,
                    "reasons": item.reasons[:3],
                    "risks": item.risks[:3],
                }
                for item in assessments[:10]
            ],
        },
    )


def source_coverage_evidence(
    topic: str,
    source_coverage: list[ResearchCoverageItem],
) -> Evidence:
    covered = [item.name for item in source_coverage if item.status == "covered"]
    missing = [item.name for item in source_coverage if item.status == "missing"]
    return facade().Evidence(
        claim=(
            f"Research source coverage for {topic!r} covered {len(covered)} evidence surface(s)"
            f" and has {len(missing)} missing expected surface(s)."
        ),
        source_type="research_source_coverage",
        source_url=None,
        support=facade().SupportLevel.INFERRED,
        details={
            "topic": topic,
            "covered": covered,
            "missing": missing,
            "coverage": [
                {
                    "name": item.name,
                    "status": item.status,
                    "expected": item.expected,
                    "observed_count": item.observed_count,
                    "source_names": item.source_names,
                    "reason": item.reason,
                    "next_action": item.next_action,
                }
                for item in source_coverage
            ],
        },
    )


def mission_plan_evidence(
    topic: str,
    mission_queries: list[ResearchQuery],
    discovery_passes: list[ResearchDiscoveryPass],
) -> Evidence:
    return facade().Evidence(
        claim=f"Research mission planner ran {len(mission_queries)} query pass(es) for {topic!r}.",
        source_type="research_mission_plan",
        source_url=None,
        support=facade().SupportLevel.INFERRED,
        details={
            "topic": topic,
            "queries": [
                {
                    "query": item.query,
                    "purpose": item.purpose,
                    "source": item.source,
                    "target_gap": item.target_gap,
                }
                for item in mission_queries
            ],
            "passes": [
                {
                    "query": item.query,
                    "purpose": item.purpose,
                    "source": item.source,
                    "target_gap": item.target_gap,
                    "scholarly_work_count": item.scholarly_work_count,
                    "search_result_count": item.search_result_count,
                    "github_artifact_count": item.github_artifact_count,
                    "seed_count": item.seed_count,
                    "error_count": len(item.errors),
                }
                for item in discovery_passes
            ],
        },
    )

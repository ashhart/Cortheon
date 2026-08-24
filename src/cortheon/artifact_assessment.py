from __future__ import annotations

from datetime import UTC, datetime

from cortheon.models import ResearchArtifact, ResearchArtifactAssessment, parse_datetime, utc_now

BUILDABLE_SIGNALS = {
    "python_package",
    "javascript_package",
    "rust_package",
    "go_module",
    "containerized",
}

QUALITY_SIGNALS = {
    "ci_config",
    "tests",
    "docs",
    "license_file",
    "install_docs",
    "usage_docs",
    "benchmark_docs",
}

# Recency is a first-class code-artifact signal: bleeding-edge work favors
# actively-pushed repositories. The half-life is shorter than for literature
# because an unmaintained repo goes stale faster than a still-cited paper.
# Undated repos get a neutral score so missing push metadata neither rewards
# nor punishes them.
REPO_RECENCY_STEPS: tuple[tuple[int, float], ...] = (
    (30, 0.98),
    (90, 0.9),
    (180, 0.8),
    (365, 0.62),
    (730, 0.4),
)
REPO_RECENCY_FLOOR = 0.2
REPO_UNDATED_RECENCY = 0.5


def assess_artifacts(
    topic: str, artifacts: list[ResearchArtifact]
) -> list[ResearchArtifactAssessment]:
    assessments = [assess_artifact(topic, artifact) for artifact in artifacts]
    return sorted(
        assessments,
        key=lambda item: (decision_priority(item.decision), item.score),
        reverse=True,
    )


def assess_artifact(topic: str, artifact: ResearchArtifact) -> ResearchArtifactAssessment:
    if artifact.kind == "code_repository":
        return assess_code_repository(topic, artifact)
    if artifact.kind in {"paper_source", "paper_pdf", "doi"}:
        return assess_paper_artifact(artifact)
    if artifact.kind in {"dataset", "benchmark"}:
        return assess_evaluation_artifact(artifact)
    if artifact.kind == "clinical_trial":
        return assess_clinical_trial_artifact(topic, artifact)
    return ResearchArtifactAssessment(
        artifact_url=artifact.url,
        artifact_kind=artifact.kind,
        title=artifact.title,
        score=round(max(0.0, min(artifact.confidence, 1.0)), 3),
        decision="background_reference",
        reasons=[
            f"Artifact was classified as {artifact.kind} with confidence {artifact.confidence:.3f}."
        ],
        risks=["No specialized assessment path exists for this artifact kind."],
        next_actions=["Inspect the artifact manually before using it as a build dependency."],
    )


def assess_code_repository(topic: str, artifact: ResearchArtifact) -> ResearchArtifactAssessment:
    metadata = artifact.metadata
    health = parse_float(metadata.get("repository_health_score"), default=0.0)
    signals = set(split_csv(metadata.get("implementation_signals", "")))
    inspected = bool(metadata.get("repository_health_score"))
    pushed_at = parse_datetime(metadata.get("pushed_at"))
    recency = repository_recency_score(pushed_at)
    score = artifact.confidence * 0.30 + health * 0.42 + recency * 0.14
    score += min(0.12, len(signals & BUILDABLE_SIGNALS) * 0.06)
    score += min(0.08, len(signals & QUALITY_SIGNALS) * 0.015)
    score = round(max(0.0, min(score, 0.98)), 3)

    reasons: list[str] = []
    risks: list[str] = []
    next_actions: list[str] = []
    repo = metadata.get("repo") or artifact.title or artifact.url
    if inspected:
        reasons.append(
            f"Repository {repo} was inspected and scored {health:.3f} for implementation health."
        )
    else:
        risks.append("Repository has not been inspected beyond search metadata.")
    if pushed_at:
        reasons.append(
            f"Repository was last pushed {pushed_at.date().isoformat()} (recency {recency:.2f})."
        )
        if recency <= 0.4:
            risks.append(
                "Repository has not been pushed to recently; confirm it is still actively maintained."
            )
    if metadata.get("primary_language"):
        reasons.append(
            f"Primary language is {metadata['primary_language']}"
            f" ({metadata.get('primary_language_share', 'unknown')} share)."
        )
    buildable = sorted(signals & BUILDABLE_SIGNALS)
    quality = sorted(signals & QUALITY_SIGNALS)
    if buildable:
        reasons.append(f"Buildable project signals: {', '.join(buildable)}.")
    else:
        risks.append("No package/module/container manifest was detected at the repository root.")
    if quality:
        reasons.append(f"Quality signals: {', '.join(quality[:6])}.")
    if metadata.get("archived") == "true":
        risks.append("Repository is archived.")
    if not metadata.get("license_spdx") or metadata.get("license_spdx") == "NOASSERTION":
        risks.append("No clear SPDX license was observed.")
    if "tests" not in signals:
        risks.append("No root-level test signal was detected.")
    if "usage_docs" not in signals:
        risks.append("README usage or quickstart signal was not detected.")

    if score >= 0.82 and inspected and buildable:
        decision = "build_from_first"
        next_actions.append(
            "Inspect the README and root manifests, then clone or vendor only after license review."
        )
        next_actions.append(
            "Run the repository tests or minimal examples in an isolated environment."
        )
    elif score >= 0.64:
        decision = "inspect_more"
        next_actions.append(
            "Inspect repository tree, releases, issues, and examples before using it."
        )
    else:
        decision = "background_reference"
        next_actions.append(
            "Use as a reference only until stronger implementation evidence is collected."
        )

    if topic and not topic_overlap(
        topic, f"{artifact.title or ''} {metadata.get('description', '')}"
    ):
        risks.append("Repository metadata has weak lexical overlap with the mission topic.")

    return ResearchArtifactAssessment(
        artifact_url=artifact.url,
        artifact_kind=artifact.kind,
        title=artifact.title,
        score=score,
        decision=decision,
        reasons=reasons or ["Code repository artifact was discovered."],
        risks=risks,
        next_actions=next_actions,
    )


def assess_paper_artifact(artifact: ResearchArtifact) -> ResearchArtifactAssessment:
    score = artifact.confidence
    if artifact.kind == "paper_source":
        score += 0.04
    score = round(min(score, 0.96), 3)
    decision = "read_first" if score >= 0.78 else "background_reference"
    reasons = [f"{artifact.kind} artifact has confidence {artifact.confidence:.3f}."]
    if artifact.provider.startswith("scholarly:"):
        reasons.append(f"Artifact came from scholarly provider {artifact.provider}.")
    risks = []
    if artifact.kind == "doi":
        risks.append("DOI may lead to a landing page rather than directly reusable source or PDF.")
    next_actions = ["Read and extract claims, methods, and benchmarks before implementation."]
    if artifact.kind == "paper_source":
        next_actions.append(
            "Download source only if needed for figures, appendices, or reproducibility assets."
        )
    return ResearchArtifactAssessment(
        artifact_url=artifact.url,
        artifact_kind=artifact.kind,
        title=artifact.title,
        score=score,
        decision=decision,
        reasons=reasons,
        risks=risks,
        next_actions=next_actions,
    )


def assess_evaluation_artifact(artifact: ResearchArtifact) -> ResearchArtifactAssessment:
    score = round(min(0.96, artifact.confidence + 0.06), 3)
    decision = "evaluate_with" if score >= 0.72 else "inspect_more"
    reasons = [f"{artifact.kind} artifact can support benchmarking or dataset validation."]
    risks = ["Dataset or benchmark compatibility, license, and leakage risks are not verified yet."]
    next_actions = [
        "Inspect task definition, license, splits, metrics, and leaderboard methodology."
    ]
    return ResearchArtifactAssessment(
        artifact_url=artifact.url,
        artifact_kind=artifact.kind,
        title=artifact.title,
        score=score,
        decision=decision,
        reasons=reasons,
        risks=risks,
        next_actions=next_actions,
    )


def assess_clinical_trial_artifact(
    topic: str, artifact: ResearchArtifact
) -> ResearchArtifactAssessment:
    metadata = artifact.metadata
    score = artifact.confidence
    status = metadata.get("overall_status", "").lower()
    phase = metadata.get("phase", "").lower()
    if status in {"recruiting", "active_not_recruiting", "completed"}:
        score += 0.04
    if "phase3" in phase or "phase 3" in phase:
        score += 0.04
    elif "phase2" in phase or "phase 2" in phase:
        score += 0.02
    score = round(min(score, 0.96), 3)
    decision = "inspect_trial" if score >= 0.76 else "background_reference"
    reasons = [f"Clinical trial registry artifact has confidence {artifact.confidence:.3f}."]
    if metadata.get("overall_status"):
        reasons.append(f"Trial status is {metadata['overall_status']}.")
    if metadata.get("phase"):
        reasons.append(f"Trial phase metadata: {metadata['phase']}.")
    if metadata.get("interventions"):
        reasons.append(f"Registered interventions include: {metadata['interventions']}.")
    risks = [
        "Trial registration is not proof of efficacy or safety.",
        "Eligibility, endpoints, publication linkage, and results availability require deeper review.",
    ]
    if topic and not topic_overlap(
        topic,
        f"{artifact.title or ''} {metadata.get('conditions', '')} {metadata.get('interventions', '')}",
    ):
        risks.append("Trial metadata has weak lexical overlap with the mission topic.")
    next_actions = [
        "Inspect official trial record, phase, status, eligibility, endpoints, sponsor, and results.",
        "Link trial record to peer-reviewed publications and adverse-event data before making claims.",
    ]
    return ResearchArtifactAssessment(
        artifact_url=artifact.url,
        artifact_kind=artifact.kind,
        title=artifact.title,
        score=score,
        decision=decision,
        reasons=reasons,
        risks=risks,
        next_actions=next_actions,
    )


def decision_priority(decision: str) -> int:
    priorities = {
        "build_from_first": 5,
        "evaluate_with": 4,
        "read_first": 3,
        "inspect_trial": 3,
        "inspect_more": 2,
        "background_reference": 1,
    }
    return priorities.get(decision, 0)


def repository_recency_score(pushed_at: datetime | None, now: datetime | None = None) -> float:
    if pushed_at is None:
        return REPO_UNDATED_RECENCY
    current = now or utc_now()
    if pushed_at.tzinfo is None:
        pushed_at = pushed_at.replace(tzinfo=UTC)
    days = max((current - pushed_at.astimezone(UTC)).days, 0)
    for horizon, score in REPO_RECENCY_STEPS:
        if days <= horizon:
            return score
    return REPO_RECENCY_FLOOR


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def topic_overlap(topic: str, text: str) -> bool:
    topic_terms = {term for term in topic.lower().replace("-", " ").split() if len(term) >= 4}
    lower_text = text.lower()
    return any(term in lower_text for term in topic_terms)

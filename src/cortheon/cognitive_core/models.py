"""Data classes, protocol constants, and session-graph construction."""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from cortheon.cognitive_core.profiles import STRICTNESS_PROFILES, EffortProfile, StrictnessProfile
from cortheon.cognitive_core.text import _safe_public_label
from cortheon.cognitive_graph import CognitiveGraph, content_id


class CognitiveRuntimeError(RuntimeError):
    """A bounded protocol failure that the harness can correct."""


class InvestigationNotFound(CognitiveRuntimeError):
    """The requested investigation is unknown, expired, or already discarded."""


OBSERVATION_KINDS = frozenset(
    {
        "code",
        "diff",
        "test",
        "command",
        "documentation",
        "web",
        "user",
        "artifact",
        "analysis",
        "other",
    }
)


OBSERVATION_STATUSES = frozenset({"observed", "verified", "failed"})


HYPOTHESIS_STATUSES = frozenset({"open", "supported", "refuted", "uncertain"})


RESEARCH_PURPOSES = frozenset(
    {
        "discovery",
        "corroboration",
        "primary_fetch",
        "scholarly_validation",
        "implementation_reference",
        "contradiction_check",
        "freshness_check",
        "passive",
    }
)


_ASSIST_WAIVER_CAVEATS: dict[str, str] = {
    "corroboration": (
        "Assist strictness accepted single-origin research; independent "
        "corroboration was not required."
    ),
}


_WAIVER_CAVEATS: dict[str, str] = {
    "corroboration": (
        "Corroboration was waived after a failed round: the answer may rest on a single URL origin."
    ),
    "contradiction_check": (
        "No active contradiction check succeeded; conflicting sources may exist "
        "that this answer does not address."
    ),
    "primary_fetch": (
        "Primary-source fetch was waived; the answer may rely on search snippets "
        "rather than fetched primary sources."
    ),
    "freshness_check": (
        "Freshness could not be established with a dated source; time-sensitive "
        "details may be stale."
    ),
    "inspect": (
        "Local workspace grounding was waived after repeated failed attempts; "
        "repository-specific claims carry reduced verification."
    ),
    "research_reframe": (
        "The goal was initially framed as web research, but the host produced "
        "no web evidence; Cortheon reframed it as a general evidence-backed "
        "answer without citation guarantees."
    ),
}


@dataclass(slots=True)
class Hypothesis:
    hypothesis_id: str
    statement: str
    falsification_test: str
    status: str = "open"
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    bearing_evidence: list[str] = field(default_factory=list)
    origin: str = "host_model"
    origin_evidence_ids: list[str] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "falsification_test": self.falsification_test,
            "status": self.status,
            "supporting_evidence": list(self.supporting_evidence),
            "contradicting_evidence": list(self.contradicting_evidence),
            "bearing_evidence": list(self.bearing_evidence),
            "origin": self.origin,
            "origin_evidence_ids": list(self.origin_evidence_ids),
        }


@dataclass(slots=True)
class Observation:
    evidence_id: str
    kind: str
    content: str
    source: str | None
    status: str
    supports: list[str]
    contradicts: list[str]
    quarantine_flags: list[str]
    sequence: int
    digest: str
    host_receipt: dict[str, Any] | None = None
    url: str | None = None
    retrieved_at: str | None = None
    published_at: str | None = None
    purpose: str | None = None

    def public(self, *, excerpt: str | None = None) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "source": _safe_public_label(self.source),
            "status": self.status,
            "content": self.content if excerpt is None else excerpt,
            "supports": list(self.supports),
            "contradicts": list(self.contradicts),
            "quarantine_flags": list(self.quarantine_flags),
            "url": self.url,
            "retrieved_at": self.retrieved_at,
            "published_at": self.published_at,
            "purpose": self.purpose,
        }


@dataclass(slots=True)
class EvidenceRequest:
    request_id: str
    capability: str
    query: str
    reason: str
    success_condition: str
    parameters: dict[str, Any] = field(default_factory=dict)
    hypothesis_id: str | None = None
    status: str = "pending"
    attempts: int = 0
    covered_paths: set[str] = field(default_factory=set)

    def public(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "request_id": self.request_id,
            "capability": self.capability,
            "query": self.query,
            "reason": self.reason,
            "success_condition": self.success_condition,
            "parameters": copy.deepcopy(self.parameters),
            "hypothesis_id": self.hypothesis_id,
            "status": self.status,
        }
        if self.covered_paths:
            payload["covered_paths"] = sorted(self.covered_paths)
        return payload


@dataclass(slots=True)
class PublicClaim:
    claim: str
    evidence_ids: list[str]

    def public(self) -> dict[str, Any]:
        return {"claim": self.claim, "evidence_ids": list(self.evidence_ids)}


@dataclass(frozen=True, slots=True)
class Requirement:
    requirement_id: str
    statement: str
    proof: str

    def public(
        self,
        *,
        status: str = "unresolved",
        evidence_ids: Iterable[str] = (),
        reason: str = "",
    ) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "statement": self.statement,
            "proof": self.proof,
            "status": status,
            "evidence_ids": list(evidence_ids),
            **({"reason": reason} if reason else {}),
        }


@dataclass(frozen=True, slots=True)
class SemanticEdge:
    source_key: str
    source: str
    target_key: str
    target: str
    document: str
    relation: str
    priority: int = 1


@dataclass(frozen=True, slots=True)
class SemanticRule:
    conditions: tuple[tuple[str, str, str], ...]
    target_key: str
    target: str
    document: str
    relation: str = "conjunctive_requirement"


@dataclass(slots=True)
class Investigation:
    session_id: str
    goal: str
    constraints: list[str]
    requirements: list[Requirement]
    task_kind: str
    deliverable: str
    profile: EffortProfile
    program: dict[str, Any]
    created_at: str
    started_at: float
    touched_at: float
    expires_at: float
    lease_seconds: float | None = None
    lease_expires_at: float | None = None
    strictness: StrictnessProfile = STRICTNESS_PROFILES["standard"]
    phase: str = "investigating"
    turns: int = 0
    challenge_count: int = 0
    hypothesis_sequence: int = 0
    hypotheses: OrderedDict[str, Hypothesis] = field(default_factory=OrderedDict)
    observations: OrderedDict[str, Observation] = field(default_factory=OrderedDict)
    observation_digests: set[str] = field(default_factory=set)
    requests: OrderedDict[str, EvidenceRequest] = field(default_factory=OrderedDict)
    open_questions: list[str] = field(default_factory=list)
    waivers: dict[str, str] = field(default_factory=dict)
    draft: str = ""
    revision_record: dict[str, str] | None = None
    revision_binding_digest: str | None = None
    claims: list[PublicClaim] = field(default_factory=list)
    verified_answer_digest: str | None = None
    last_verification: dict[str, Any] | None = None
    last_next_action: dict[str, Any] | None = None
    evaluation_profile: dict[str, Any] | None = None


def _session_graph(
    session: Investigation,
    requirements: list[dict[str, Any]],
    derivations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project bounded session state into an inspectable proposition graph."""

    graph = CognitiveGraph()
    requirement_ids: set[str] = set()
    usable = {
        item.evidence_id: item
        for item in session.observations.values()
        if item.status != "failed" and not item.quarantine_flags
    }
    for requirement in requirements:
        requirement_id = str(requirement.get("requirement_id") or "")
        if not requirement_id:
            continue
        requirement_ids.add(requirement_id)
        graph.add_node(
            requirement_id,
            "requirement",
            str(requirement.get("statement") or "")[:500],
            status=requirement.get("status"),
        )
    for hypothesis in session.hypotheses.values():
        graph.add_node(
            hypothesis.hypothesis_id,
            "hypothesis",
            hypothesis.statement[:500],
            status=hypothesis.status,
        )
    for evidence_id, observation in usable.items():
        graph.add_node(
            evidence_id,
            "evidence",
            str(_safe_public_label(observation.source) or observation.kind)[:500],
            status=observation.status,
            digest=observation.digest,
        )
        for hypothesis_id in observation.supports:
            if hypothesis_id in session.hypotheses:
                graph.add_edge(
                    evidence_id,
                    "supports",
                    hypothesis_id,
                    evidence_id=evidence_id,
                )
        for hypothesis_id in observation.contradicts:
            if hypothesis_id in session.hypotheses:
                graph.add_edge(
                    evidence_id,
                    "contradicts",
                    hypothesis_id,
                    evidence_id=evidence_id,
                )
    for hypothesis in session.hypotheses.values():
        for evidence_id in hypothesis.bearing_evidence:
            if evidence_id in usable:
                graph.add_edge(
                    evidence_id,
                    "bears_on",
                    hypothesis.hypothesis_id,
                    evidence_id=evidence_id,
                )
    for hypothesis in session.hypotheses.values():
        for evidence_id in hypothesis.origin_evidence_ids:
            if evidence_id in usable:
                graph.add_edge(
                    evidence_id,
                    "inspired_candidate",
                    hypothesis.hypothesis_id,
                    evidence_id=evidence_id,
                )
    for requirement in requirements:
        requirement_id = str(requirement.get("requirement_id") or "")
        if requirement_id not in requirement_ids:
            continue
        for evidence_id in requirement.get("evidence_ids", ()):
            if evidence_id in usable:
                graph.add_edge(
                    str(evidence_id),
                    "establishes",
                    requirement_id,
                    evidence_id=str(evidence_id),
                )
    source_evidence = {
        str(item.source): item.evidence_id for item in usable.values() if item.source
    }
    for derivation in derivations:
        operation = derivation.get("operation")
        if operation == "semantic_conflict":
            for conflict in derivation.get("conflicts", ()):
                if not isinstance(conflict, dict):
                    continue
                entity = str(conflict.get("entity") or "")
                relation = str(conflict.get("relation") or "conflicts_with")
                entity_id = content_id("p", {"label": entity})
                graph.add_node(entity_id, "proposition", entity[:500])
                for target in conflict.get("targets", ()):
                    target_text = str(target)
                    target_id = content_id("p", {"label": target_text})
                    graph.add_node(target_id, "proposition", target_text[:500])
                    graph.add_edge(entity_id, relation, target_id, functional=True)
            continue
        if operation not in {"ordered_plan", "semantic_chain", "semantic_rule"}:
            continue
        nodes = [str(item) for item in derivation.get("nodes", ()) if str(item)]
        relations = [str(item) for item in derivation.get("relations", ())]
        sources = [str(item) for item in derivation.get("sources", ())]
        for index, (source, target) in enumerate(pairwise(nodes)):
            source_id = content_id("p", {"label": source})
            target_id = content_id("p", {"label": target})
            graph.add_node(source_id, "proposition", source[:500])
            graph.add_node(target_id, "proposition", target[:500])
            graph.add_edge(
                source_id,
                relations[index] if index < len(relations) else "linked_to",
                target_id,
                evidence_id=(
                    source_evidence.get(sources[index], "") if index < len(sources) else ""
                ),
            )
    return graph.snapshot()


def _fit_strings(values: Iterable[str], budget: int) -> tuple[list[str], int]:
    selected: list[str] = []
    used = 0
    for value in values:
        remaining = budget - used
        if remaining <= 0:
            break
        excerpt = value[:remaining]
        if excerpt:
            selected.append(excerpt)
            used += len(excerpt)
    return selected, used


def _fit_hypotheses(
    hypotheses: Iterable[Hypothesis],
    budget: int,
) -> tuple[list[dict[str, Any]], int]:
    values = list(hypotheses)
    if not values or budget <= 0:
        return [], 0
    per_hypothesis = max(64, budget // len(values))
    selected: list[dict[str, Any]] = []
    used = 0
    for hypothesis in values:
        remaining = budget - used
        if remaining <= 0:
            break
        item_budget = min(per_hypothesis, remaining)
        statement_budget = max(32, item_budget * 3 // 5)
        statement = hypothesis.statement[:statement_budget]
        falsification = hypothesis.falsification_test[: max(0, item_budget - len(statement))]
        selected.append(
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "statement": statement,
                "falsification_test": falsification,
                "status": hypothesis.status,
                "supporting_evidence": list(hypothesis.supporting_evidence),
                "contradicting_evidence": list(hypothesis.contradicting_evidence),
                "bearing_evidence": list(hypothesis.bearing_evidence),
                "origin": hypothesis.origin,
                "origin_evidence_ids": list(hypothesis.origin_evidence_ids),
            }
        )
        used += len(statement) + len(falsification)
    return selected, used

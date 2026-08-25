"""Requirement extraction and coverage analysis."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from cortheon.cognitive_core.models import Investigation, Observation, PublicClaim, Requirement
from cortheon.cognitive_core.profiles import _has_hint
from cortheon.cognitive_core.semantic_graph import _semantic_key, _semantic_terms
from cortheon.cognitive_core.tasks import (
    _CROSS_SOURCE_HINTS,
    _goal_code_paths,
    _goal_document_paths,
)
from cortheon.cognitive_core.text import _SPACE_RE, _normalized
from cortheon.cognitive_repair import (
    changed_paths_from_diff,
    is_test_path,
    protected_test_paths,
    protects_tests,
)

_REQUIREMENT_ACTION_RE = re.compile(
    r"\b(?:add|build|change|check|compare|connect|correct|create|delete|"
    r"determine|diagnose|discover|edit|find|fix|identify|implement|inspect|"
    r"keep|lint|locate|migrate|patch|preserve|read|refactor|remove|rename|"
    r"repair|replace|research|run|search|test|update|verify)\b",
    flags=re.IGNORECASE,
)


_REQUIREMENT_BOUNDARY_RE = re.compile(
    r"(?:[.;]['\"]?\s+|,\s+|\s+and\s+|"
    r"\s+(?=(?:do\s+not|don't|without)\b))"
    r"(?=(?:(?:also|then)\s+)?(?:do\s+not|don't|without|"
    r"add|build|change|check|compare|connect|correct|create|delete|determine|"
    r"diagnose|discover|edit|find|fix|identify|implement|inspect|keep|lint|"
    r"locate|migrate|patch|preserve|read|refactor|remove|rename|repair|"
    r"replace|research|run|search|test|update|verify)\b)",
    flags=re.IGNORECASE,
)


_REQUIREMENT_GENERIC_TERMS = frozenset(
    {
        "actual",
        "add",
        "answer",
        "answering",
        "before",
        "build",
        "change",
        "check",
        "code",
        "component",
        "correct",
        "correction",
        "create",
        "edit",
        "final",
        "fix",
        "focused",
        "host",
        "implement",
        "implementation",
        "inspect",
        "locate",
        "patch",
        "project",
        "repair",
        "repository",
        "run",
        "requested",
        "requirement",
        "result",
        "test",
        "tests",
        "update",
        "verify",
        "verified",
    }
)


def _requirement_terms(statement: str) -> set[str]:
    return _semantic_terms(statement) - _REQUIREMENT_GENERIC_TERMS


def _requirement_kind_matches(
    requirement: Requirement,
    observation: Observation,
) -> bool:
    if requirement.proof == "mutation":
        return observation.kind == "diff"
    if requirement.proof == "verification":
        return observation.kind == "test" and observation.status == "verified"
    if requirement.proof == "research":
        return observation.kind == "web"
    if requirement.proof == "synthesis":
        return observation.kind in {"documentation", "web", "code", "artifact"}
    if requirement.proof == "inspection":
        return observation.kind in {"code", "documentation", "analysis", "artifact"}
    if requirement.proof == "protection":
        return observation.kind == "diff"
    return True


def _requirement_coverage(
    session: Investigation,
    claims: Iterable[PublicClaim] = (),
    completion_evidence_ids: Iterable[str] | None = None,
    *,
    require_citations: bool,
) -> list[dict[str, Any]]:
    claims = list(claims)
    cited = {
        evidence_id
        for claim in claims
        for evidence_id in claim.evidence_ids
        if evidence_id in session.observations
    }
    selected = (
        set(session.observations)
        if completion_evidence_ids is None
        else {
            evidence_id
            for evidence_id in completion_evidence_ids
            if evidence_id in session.observations
        }
    )
    usable_ids = {
        evidence_id
        for evidence_id in selected
        if session.observations[evidence_id].status != "failed"
        and not session.observations[evidence_id].quarantine_flags
    }
    eligible_ids = usable_ids & cited if require_citations else usable_ids
    mutation_count = sum(requirement.proof == "mutation" for requirement in session.requirements)
    inspection_count = sum(
        requirement.proof == "inspection" for requirement in session.requirements
    )
    results: list[dict[str, Any]] = []
    for requirement in session.requirements:
        anchors = _requirement_terms(requirement.statement)
        kind_ids = [
            evidence_id
            for evidence_id in eligible_ids
            if _requirement_kind_matches(
                requirement,
                session.observations[evidence_id],
            )
        ]
        available_kind_ids = [
            evidence_id
            for evidence_id in usable_ids
            if _requirement_kind_matches(
                requirement,
                session.observations[evidence_id],
            )
        ]
        matching: list[str] = []
        for evidence_id in kind_ids:
            observation = session.observations[evidence_id]
            evidence_terms = _semantic_terms(
                " ".join((observation.source or "", observation.content))
            )
            claim_terms = set().union(
                *(
                    _semantic_terms(claim.claim)
                    for claim in claims
                    if evidence_id in claim.evidence_ids
                ),
                set(),
            )
            needs_lexical_binding = bool(anchors) and (
                (requirement.proof == "mutation" and mutation_count > 1)
                or (requirement.proof == "inspection" and inspection_count > 1)
            )
            if not needs_lexical_binding or anchors & (evidence_terms | claim_terms):
                matching.append(evidence_id)

        if requirement.proof == "synthesis" and matching:
            source_count = len(
                {
                    session.observations[evidence_id].source
                    for evidence_id in matching
                    if session.observations[evidence_id].source
                }
            )
            required_sources = 2 if _has_hint(session.goal, _CROSS_SOURCE_HINTS) else 1
            if source_count < required_sources:
                matching = []
        protection_violated = False
        if requirement.proof == "protection" and matching:
            changed = {
                path
                for evidence_id in matching
                for path in changed_paths_from_diff(session.observations[evidence_id].content)
            }
            protected = set(protected_test_paths(session.goal))
            if any(
                path in protected or (protects_tests(session.goal) and is_test_path(path))
                for path in changed
            ):
                matching = []
                protection_violated = True

        failed_relevant = [
            item.evidence_id
            for item in session.observations.values()
            if item.status == "failed"
            and (
                (requirement.proof == "verification" and item.kind == "test")
                or (
                    anchors
                    and anchors & _semantic_terms(" ".join((item.source or "", item.content)))
                )
            )
        ]
        if protection_violated:
            status = "contradicted"
            reason = (
                "The accepted diff changes a protected test surface; restore it "
                "before this requirement can be covered."
            )
        elif matching:
            status = "covered"
            reason = "Accepted completion evidence covers this requirement."
        elif failed_relevant:
            status = "contradicted"
            reason = (
                "The latest relevant observation failed or was retracted; "
                "this requirement must be re-verified."
            )
        elif require_citations and available_kind_ids:
            status = "unresolved"
            reason = (
                "Relevant evidence exists, but no completion claim binds this "
                "requirement to that evidence."
            )
        elif kind_ids:
            status = "unresolved"
            reason = "Evidence of the right type does not match this requirement."
        else:
            status = "unresolved"
            reason = f"This requirement still needs {requirement.proof} evidence."
        results.append(
            requirement.public(
                status=status,
                evidence_ids=matching,
                reason=reason,
            )
        )
    return results


def _requirement_proof(statement: str, deliverable: str) -> str:
    normalized = _normalized(statement)
    if re.search(
        r"\b(?:do\s+not|don't|without)\s+"
        r"(?:chang(?:e|ing)|modif(?:y|ying)|edit(?:ing)?)|"
        r"\b(?:keep|preserve)\b.*\b(?:unchanged|intact|existing)\b",
        normalized,
    ):
        return "protection"
    if re.search(
        r"\b(?:contradictions?|freshness|latest|today|research|web|origins?|urls?)\b",
        normalized,
    ) or (deliverable == "research_answer" and re.search(r"\b(?:current|sources?)\b", normalized)):
        return "research"
    if deliverable != "code_change" and re.search(
        r"\b(?:falsification|falsifiable|falsify|discriminating)\s+test\b|"
        r"\btest\b.{0,40}\b(?:disprove|falsify)\b|"
        r"\bobservation\b.{0,40}\b(?:disprove|falsify)\b",
        normalized,
    ):
        return "synthesis"
    if re.search(
        r"\b(?:run|test|verify|check|lint)\b|"
        r"\b(?:tests?|suite)\s+(?:pass|passes|passing)\b",
        normalized,
    ):
        return "verification"
    if re.search(r"\b(?:compare|connect|join|across)\b", normalized):
        return "synthesis"
    if re.search(
        r"\b(?:add|build|change|correct|create|delete|edit|fix|implement|"
        r"migrate|patch|refactor|remove|rename|repair|replace|update)\b",
        normalized,
    ):
        return {
            "code_change": "mutation",
            "code_understanding": "inspection",
            "research_answer": "research",
            "document_synthesis": "synthesis",
        }.get(deliverable, "completion")
    if re.search(
        r"\b(?:determine|diagnose|discover|find|identify|inspect|locate|read|search)\b",
        normalized,
    ):
        return "inspection"
    return {
        "code_change": "mutation",
        "code_understanding": "inspection",
        "research_answer": "research",
        "document_synthesis": "synthesis",
    }.get(deliverable, "completion")


def _extract_requirements(
    goal: str,
    constraints: Iterable[str],
    deliverable: str,
) -> list[Requirement]:
    """Extract a bounded, observable task contract from public instructions."""

    statements: list[tuple[str, str]] = []
    for raw in [goal, *constraints]:
        for part in _REQUIREMENT_BOUNDARY_RE.split(_SPACE_RE.sub(" ", raw).strip()):
            action = _REQUIREMENT_ACTION_RE.search(part)
            protection = re.search(
                r"\b(?:do\s+not|don't|without)\s+"
                r"(?:chang(?:e|ing)|modif(?:y|ying)|edit(?:ing)?)",
                part,
                flags=re.IGNORECASE,
            )
            start = min(
                (match.start() for match in (action, protection) if match is not None),
                default=-1,
            )
            if start < 0:
                continue
            statement = part[start:].strip(" ,.;:").strip()
            if not statement or len(statement) > 300:
                continue
            proof = _requirement_proof(statement, deliverable)
            if proof == "completion" and re.match(
                r"^(?:answer|explain|report|tell)\b",
                statement,
                flags=re.IGNORECASE,
            ):
                continue
            key = _semantic_key(statement)
            if key and all(_semantic_key(existing) != key for existing, _ in statements):
                statements.append((statement, proof))

    if deliverable != "code_change":
        statements = [
            (statement, proof) for statement, proof in statements if proof != "protection"
        ]

    if deliverable == "code_change":
        protected = set(protected_test_paths(goal))
        named_paths = list(dict.fromkeys([*_goal_code_paths(goal), *_goal_document_paths(goal)]))
        mutation_paths = [
            path
            for path in named_paths
            if path not in protected
            and not re.search(
                r"(?:^|/)(?:test[_-]|tests?/)|(?:[_-]test|\.spec|\.test)\.",
                path,
                flags=re.IGNORECASE,
            )
        ]
        if len(mutation_paths) > 1:
            statements = [
                *(("Apply the requested change to " + path, "mutation") for path in mutation_paths),
                *((statement, proof) for statement, proof in statements if proof != "mutation"),
            ]

    proofs = {proof for _statement, proof in statements}
    defaults: list[tuple[str, str]] = []
    if deliverable == "code_change":
        if "mutation" not in proofs:
            defaults.append(("Apply the requested implementation change", "mutation"))
        if "verification" not in proofs:
            defaults.append(("Verify the final change with a host-run test", "verification"))
        if protects_tests(goal) and "protection" not in proofs:
            defaults.append(("Preserve the protected test surface unchanged", "protection"))
    elif deliverable == "code_understanding" and "inspection" not in proofs:
        defaults.append(("Answer from a focused live code inspection", "inspection"))
    elif deliverable == "research_answer" and "research" not in proofs:
        defaults.append(("Answer from current attributable web evidence", "research"))
    elif deliverable == "document_synthesis" and "synthesis" not in proofs:
        defaults.append(("Connect the requested live document evidence", "synthesis"))
    elif not statements:
        defaults.append(("Resolve the requested deliverable from live evidence", "completion"))

    bounded = [*statements, *defaults][:8]
    return [
        Requirement(
            requirement_id=f"r{index}",
            statement=statement[:300],
            proof=proof,
        )
        for index, (statement, proof) in enumerate(bounded, 1)
    ]

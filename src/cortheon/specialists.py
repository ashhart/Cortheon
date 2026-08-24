"""Public, bounded specialist planning for adaptive Cortheon missions.

This module does not ask a model for private chain-of-thought.  It constructs
small, inspectable strategy branches that a tool controller can test against
real observations.  The branches are deliberately approaches rather than
answers: evidence still has to be gathered, cited, and verified by the runtime.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AdaptiveCognitionBudget:
    """Hard limits for adaptive planning and independent verification."""

    enabled: bool = True
    activation_threshold: int = 4
    max_specialists: int = 5
    max_hypotheses: int = 3
    max_verification_attempts: int = 2
    max_retry_tool_calls: int = 1
    max_public_check_chars: int = 300

    def __post_init__(self) -> None:
        if not 1 <= self.activation_threshold <= 20:
            raise ValueError("activation_threshold must be between 1 and 20")
        if not 1 <= self.max_specialists <= 5:
            raise ValueError("max_specialists must be between 1 and 5")
        if not 2 <= self.max_hypotheses <= 5:
            raise ValueError("max_hypotheses must be between 2 and 5")
        if not 1 <= self.max_verification_attempts <= 3:
            raise ValueError("max_verification_attempts must be between 1 and 3")
        if not 0 <= self.max_retry_tool_calls <= 2:
            raise ValueError("max_retry_tool_calls must be between 0 and 2")
        if not 80 <= self.max_public_check_chars <= 1_000:
            raise ValueError("max_public_check_chars must be between 80 and 1000")


@dataclass(frozen=True, slots=True)
class SpecialistAssignment:
    """One narrow role and the capabilities it may recommend."""

    name: str
    mandate: str
    preferred_tools: tuple[str, ...]
    completion_check: str


@dataclass(frozen=True, slots=True)
class PublicHypothesis:
    """A testable public strategy, never hidden model scratch work."""

    hypothesis_id: str
    specialist: str
    approach: str
    evidence_needed: str
    falsification_check: str


@dataclass(frozen=True, slots=True)
class AdaptiveMission:
    """The compact cognition scaffold supplied to controller and verifier."""

    complexity_score: int
    activation_reasons: tuple[str, ...]
    specialists: tuple[SpecialistAssignment, ...]
    hypotheses: tuple[PublicHypothesis, ...]
    verification_required: bool = True
    planner: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AdaptiveVerification:
    """One public verifier decision with bounded checks and concerns."""

    attempt: int
    verdict: str
    selected_hypothesis_id: str | None
    checks: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    retry_tool: str | None = None
    accepted: bool = False


_ROLE_DEFINITIONS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "research": (
        "Find current, independent, attributable evidence and distinguish "
        "publication claims from verified facts.",
        ("web_search", "web_fetch", "research", "package_version"),
        "Current claims have source-backed citations and material disagreement is noted.",
    ),
    "code": (
        "Inspect real APIs and repository state, propose the smallest repair, "
        "and rely on executable checks instead of plausible-looking code.",
        (
            "package_api",
            "package_version",
            "repo_search",
            "repo_read",
            "verify_patch",
            "python_execute",
        ),
        "The implementation contract is source-backed and relevant tests or checks pass.",
    ),
    "documents": (
        "Retrieve the strongest passages, connect entities across documents, "
        "and retain chunk-level provenance for every bridge.",
        ("document_search", "document_join"),
        "The answer cites the supporting chunks and makes every cross-document bridge explicit.",
    ),
    "math": (
        "Translate quantitative claims into explicit calculations and check "
        "units, boundary cases, and numerical consistency.",
        ("calculate", "python_execute"),
        "Independent calculation reproduces the stated result with consistent units.",
    ),
    "verifier": (
        "Try to disconfirm the leading result, check citations and completion "
        "criteria, and request only the minimum missing evidence.",
        (
            "calculate",
            "document_search",
            "document_join",
            "package_api",
            "repo_read",
            "repo_search",
            "web_fetch",
            "web_search",
        ),
        "No material claim lacks evidence and no known check contradicts the answer.",
    ),
}

_ROLE_PATTERNS: dict[str, re.Pattern[str]] = {
    "research": re.compile(
        r"\b(?:current|latest|today|recent|research|web|source|citation|"
        r"evidence|news|market|find information)\b",
        re.IGNORECASE,
    ),
    "code": re.compile(
        r"\b(?:api|repository|repo|patch|test|debug|bug|python|"
        r"package|implementation|function|class|deploy)\b",
        re.IGNORECASE,
    ),
    "documents": re.compile(
        r"\b(?:document|documents|file|files|report|paper|passage|chunk|"
        r"join|across|contract|policy)\b",
        re.IGNORECASE,
    ),
    "math": re.compile(
        r"\b(?:calculate|equation|probability|statistics?|numeric|percentage|"
        r"rate|forecast|optimi[sz]e|regression|sum|average)\b",
        re.IGNORECASE,
    ),
}

_HARD_PATTERN = re.compile(
    r"\b(?:compare|contrast|synthesi[sz]e|contradiction|conflict|root cause|"
    r"trade-?off|architecture|end[- ]to[- ]end|multi[- ]step|adversarial|"
    r"investigate|prove|disprove|verify|validate|evaluate|deep|thorough|"
    r"ambiguous|uncertain|alternative)\b",
    re.IGNORECASE,
)


def plan_adaptive_mission(
    task: str,
    available_tools: Iterable[str] = (),
    *,
    budget: AdaptiveCognitionBudget | None = None,
    force: bool | None = None,
) -> AdaptiveMission | None:
    """Return an adaptive mission only when complexity justifies its cost.

    ``force=True`` is an explicit caller request, ``force=False`` disables the
    path, and ``None`` uses a deterministic complexity gate.  No task content
    is executed and no capability is added here.
    """

    limits = budget or AdaptiveCognitionBudget()
    if not limits.enabled or force is False:
        return None
    cleaned = " ".join(task.split())[:12_000]
    if not cleaned:
        return None

    matched_roles = [role for role, pattern in _ROLE_PATTERNS.items() if pattern.search(cleaned)]
    hard_signals = {match.group(0).casefold() for match in _HARD_PATTERN.finditer(cleaned)}
    score = len(matched_roles)
    reasons: list[str] = []
    if len(matched_roles) >= 2:
        score += 3
        reasons.append("cross-domain mission")
    if hard_signals:
        score += min(3, len(hard_signals) + 1)
        reasons.append("requires comparison, critique, or verification")
    if len(cleaned) >= 500:
        score += 1
        reasons.append("long task contract")
    if cleaned.count("?") + cleaned.count(";") >= 3:
        score += 1
        reasons.append("multiple explicit subproblems")
    if force is True:
        reasons.append("explicit adaptive-cognition request")
    elif score < limits.activation_threshold:
        return None

    if not matched_roles:
        matched_roles = ["research"]
    # Preserve domain order and reserve a slot for a genuinely independent
    # verifier when the configured budget permits it.
    domain_limit = max(1, limits.max_specialists - 1)
    selected_roles = matched_roles[:domain_limit]
    if limits.max_specialists > 1:
        selected_roles.append("verifier")
    selected_roles = list(dict.fromkeys(selected_roles))[: limits.max_specialists]

    available = frozenset(str(item) for item in available_tools)
    assignments = tuple(_assignment(role, available) for role in selected_roles)
    hypotheses = _public_hypotheses(assignments, limits.max_hypotheses)
    return AdaptiveMission(
        complexity_score=max(score, limits.activation_threshold) if force is True else score,
        activation_reasons=tuple(reasons or ["explicitly selected"]),
        specialists=assignments,
        hypotheses=hypotheses,
    )


def refine_adaptive_mission(
    baseline: AdaptiveMission,
    value: Any,
    available_tools: Iterable[str] = (),
    *,
    budget: AdaptiveCognitionBudget | None = None,
) -> AdaptiveMission:
    """Validate a model-routed public plan without granting capabilities."""

    limits = budget or AdaptiveCognitionBudget()
    if not isinstance(value, dict):
        raise ValueError("specialist plan must be an object")
    raw_roles = value.get("specialists")
    if not isinstance(raw_roles, list):
        raise ValueError("specialist plan must include specialists")
    roles = [
        str(item).strip().casefold()
        for item in raw_roles[: limits.max_specialists]
        if str(item).strip().casefold() in _ROLE_DEFINITIONS
    ]
    roles = list(dict.fromkeys(roles))
    domain_roles = [item for item in roles if item != "verifier"]
    if not domain_roles:
        raise ValueError("specialist plan selected no domain specialist")
    baseline_domains = {item.name for item in baseline.specialists if item.name != "verifier"}
    if baseline_domains and not baseline_domains.intersection(domain_roles):
        raise ValueError("specialist plan dropped every task-matched domain specialist")
    if "verifier" not in roles and len(roles) < limits.max_specialists:
        roles.append("verifier")
    if "verifier" not in roles:
        roles[-1] = "verifier"
    available = frozenset(str(item) for item in available_tools)
    assignments = tuple(_assignment(role, available) for role in roles)

    raw_hypotheses = value.get("hypotheses")
    if not isinstance(raw_hypotheses, list):
        raise ValueError("specialist plan must include hypotheses")
    hypotheses: list[PublicHypothesis] = []
    selected_roles = {item.name for item in assignments}
    for item in raw_hypotheses[: limits.max_hypotheses]:
        if not isinstance(item, dict):
            continue
        specialist = str(item.get("specialist") or "").strip().casefold()
        if specialist not in selected_roles:
            continue
        approach = _public_text(item.get("approach"), 360)
        evidence_needed = _public_text(item.get("evidence_needed"), 300)
        falsification_check = _public_text(
            item.get("falsification_check"),
            300,
        )
        if not approach or not evidence_needed or not falsification_check:
            continue
        mismatched_role = next(
            (
                role
                for role in _ROLE_DEFINITIONS
                if role != specialist
                and re.search(
                    rf"\b{re.escape(role)} specialist\b",
                    approach,
                    re.IGNORECASE,
                )
            ),
            None,
        )
        if mismatched_role is not None:
            continue
        hypotheses.append(
            PublicHypothesis(
                hypothesis_id=f"hypothesis_{len(hypotheses)}",
                specialist=specialist,
                approach=approach,
                evidence_needed=evidence_needed,
                falsification_check=falsification_check,
            )
        )
    if len(hypotheses) < 2:
        raise ValueError("specialist plan must contain two testable hypotheses")
    return AdaptiveMission(
        complexity_score=baseline.complexity_score,
        activation_reasons=tuple(
            dict.fromkeys((*baseline.activation_reasons, "model-routed specialist plan"))
        ),
        specialists=assignments,
        hypotheses=tuple(hypotheses),
        verification_required=True,
        planner="model",
    )


def _assignment(
    role: str,
    available_tools: frozenset[str],
) -> SpecialistAssignment:
    mandate, preferred, completion = _ROLE_DEFINITIONS[role]
    tools = tuple(name for name in preferred if name in available_tools)
    return SpecialistAssignment(
        name=role,
        mandate=mandate,
        preferred_tools=tools,
        completion_check=completion,
    )


def _public_hypotheses(
    assignments: tuple[SpecialistAssignment, ...],
    limit: int,
) -> tuple[PublicHypothesis, ...]:
    domain_assignments = [item for item in assignments if item.name != "verifier"]
    primary = domain_assignments[0] if domain_assignments else assignments[0]
    hypotheses = [
        PublicHypothesis(
            hypothesis_id="hypothesis_0",
            specialist=primary.name,
            approach=(
                f"Build the answer through the {primary.name} specialist's "
                "strongest direct evidence path."
            ),
            evidence_needed=primary.completion_check,
            falsification_check=(
                "Reject this path if a primary observation contradicts its "
                "central claim or its completion check cannot be executed."
            ),
        ),
        PublicHypothesis(
            hypothesis_id="hypothesis_1",
            specialist="verifier",
            approach=(
                "Assume the leading path is incomplete and seek the strongest "
                "counterexample, missing source, failed check, or alternate explanation."
            ),
            evidence_needed=(
                "At least one independent check of the leading path's most consequential claim."
            ),
            falsification_check=(
                "Discard the objection when the cited evidence and executable "
                "checks directly resolve it."
            ),
        ),
    ]
    if len(domain_assignments) >= 2:
        secondary = domain_assignments[1]
        hypotheses.append(
            PublicHypothesis(
                hypothesis_id="hypothesis_2",
                specialist=secondary.name,
                approach=(
                    f"Construct an independent {secondary.name} path, then join "
                    "it to the primary path only where identifiers and evidence agree."
                ),
                evidence_needed=secondary.completion_check,
                falsification_check=(
                    "Reject the synthesis if the two paths cannot be joined "
                    "without an unsupported inference."
                ),
            )
        )
    else:
        hypotheses.append(
            PublicHypothesis(
                hypothesis_id="hypothesis_2",
                specialist=primary.name,
                approach=(
                    "Use a second source or independent method to reproduce the "
                    "leading result without relying on its weakest assumption."
                ),
                evidence_needed=(
                    "An observation independent of the primary path that reaches "
                    "the same material conclusion."
                ),
                falsification_check=(
                    "Reject convergence if both paths ultimately depend on the "
                    "same unverified source or assumption."
                ),
            )
        )
    return tuple(hypotheses[:limit])


def _public_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    text = re.sub(
        r"<think>.*?</think>",
        "",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = " ".join(text.split())
    if text.casefold().startswith(("analysis:", "chain of thought:", "reasoning:")):
        return ""
    return text[:limit]

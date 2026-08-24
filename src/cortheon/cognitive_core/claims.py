"""Claim typing and evidence-claim entailment checks."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable
from typing import Any

from cortheon.cognitive_core.diffs import _diff_establishes_change
from cortheon.cognitive_core.models import Investigation, Observation
from cortheon.cognitive_core.receipts import (
    _HOST_EVIDENCE_PREFIX,
    _observation_origin,
    _receipt_outcome,
)
from cortheon.cognitive_core.semantic_graph import _semantic_terms

_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|without|absent|missing|doesn't|does\s+not|isn't|is\s+not)\b",
    flags=re.IGNORECASE,
)


_PRIVATE_RECORD_RE = re.compile(
    r"\b(?:confidential|handbook|internal|intranet|private|proprietary)\b",
    flags=re.IGNORECASE,
)


_SCIENTIFIC_CLAIM_RE = re.compile(
    r"\b(?:clinical|experiment|meta-analysis|paper|patients?|randomi[sz]ed|"
    r"researchers?|scientific|study|trial)\b",
    flags=re.IGNORECASE,
)


_LEGAL_CLAIM_RE = re.compile(
    r"\b(?:act|case law|court|judgment|law|legal|regulation|statute)\b",
    flags=re.IGNORECASE,
)


_BEHAVIOR_CLAIM_RE = re.compile(
    r"\b(?:accepts?|behaves?|blocks?|correctly|fails?|handles?|parses?|"
    r"pass(?:es|ed)?|prevents?|rejects?|returns?|works?)\b",
    flags=re.IGNORECASE,
)


_CHANGE_CLAIM_RE = re.compile(
    r"\b(?:adds?|added|changes?|changed|fix(?:es|ed)|implements?|implemented|"
    r"patch(?:es|ed)|refactors?|refactored|removes?|removed|renames?|renamed|"
    r"updates?|updated)\b",
    flags=re.IGNORECASE,
)


_NUMERIC_CLAIM_RE = re.compile(
    r"(?:\b\d+(?:[.,]\d+)*(?:%|ms|s|kg|gb|mb)?\b|[$£€]\s*\d)",
    flags=re.IGNORECASE,
)


def _claim_type(session: Investigation, claim: str) -> str:
    """Select the truth operation that can establish this particular claim."""

    scope = f"{session.goal}\n{claim}"
    if _PRIVATE_RECORD_RE.search(scope):
        return "private_record"
    if session.deliverable == "document_synthesis" or session.task_kind == "documents":
        # A synthesis over supplied project documents is established by those
        # live documents; study/trial/court vocabulary inside them must not
        # escalate the claim to an external primary-authority bar the
        # workspace cannot meet.
        return "document_record"
    if _LEGAL_CLAIM_RE.search(claim):
        return "legal"
    if _SCIENTIFIC_CLAIM_RE.search(claim):
        return "scientific"
    local_code_claim = bool(
        re.search(
            r"\b(?:local\s+(?:workspace|worktree|repository|repo|codebase)|"
            r"source\s+tree|project\s+files?)\b",
            claim,
            flags=re.IGNORECASE,
        )
    )
    if session.task_kind == "code" or local_code_claim:
        if session.deliverable == "code_change":
            if _BEHAVIOR_CLAIM_RE.search(claim) or re.search(
                r"\b(?:regression|tests?|tested)\b",
                claim,
                flags=re.IGNORECASE,
            ):
                return "code_behavior"
            if _CHANGE_CLAIM_RE.search(claim):
                return "code_change"
        return "code_static"
    if session.deliverable == "research_answer" or session.task_kind == "research":
        return "current_research"
    if _NUMERIC_CLAIM_RE.search(claim):
        return "quantitative"
    return "general_fact"


def _observation_body(observation: Observation) -> str:
    return "\n".join(
        line
        for line in observation.content.splitlines()
        if not line.startswith(_HOST_EVIDENCE_PREFIX)
    )


def _claim_entailment(
    session: Investigation,
    claim: str,
    claim_type: str,
    observations: list[Observation],
) -> tuple[bool, str]:
    """Check whether the cited evidence addresses the claim, not merely the task."""

    negative = _NEGATION_RE.search(claim) is not None
    grep_outcomes = {
        str((item.host_receipt or {}).get("outcome") or "").casefold()
        for item in observations
        if str((item.host_receipt or {}).get("tool") or "").casefold() == "grep"
    }
    if grep_outcomes:
        if grep_outcomes == {"no_match"}:
            return (
                negative,
                "An exact host grep returned no match and supports only the negative claim."
                if negative
                else "An exact host grep returned no match, contradicting the positive claim.",
            )
        if grep_outcomes == {"match"}:
            return (
                not negative,
                "An exact host grep returned a match and supports the positive claim."
                if not negative
                else "An exact host grep returned a match, contradicting the negative claim.",
            )

    test_passed = any(
        item.kind == "test"
        and item.status == "verified"
        and (item.host_receipt is None or _receipt_outcome(item, "test", "passed"))
        for item in observations
    )
    if claim_type == "code_behavior" and test_passed:
        return True, "A host-recorded passing test directly exercises the claimed behavior."

    changed = any(
        item.kind == "diff"
        and _diff_establishes_change(
            item,
            require_receipt=session.strictness.name == "strict",
        )
        for item in observations
    )
    if claim_type == "code_change" and changed:
        return True, "A focused diff directly records the claimed change."

    evidence_text = "\n".join(
        filter(
            None,
            (
                "\n".join(
                    (
                        _observation_body(item),
                        item.source or "",
                        item.url or "",
                        json.dumps((item.host_receipt or {}).get("args") or {}, sort_keys=True),
                    )
                )
                for item in observations
            ),
        )
    )
    claim_terms = _semantic_terms(claim)
    evidence_terms = _semantic_terms(evidence_text)
    shared = claim_terms & evidence_terms
    origins = {_observation_origin(item) for item in observations}
    origins.discard(None)
    direct_host_read = any(
        str((item.host_receipt or {}).get("tool") or "").casefold()
        in {"find", "git", "glob", "grep", "read"}
        for item in observations
    )
    multi_document = len({item.source for item in observations if item.source}) >= 2
    meta_evidence_claim = bool(
        re.search(
            r"\b(?:documents?|evidence|sources?)\b.{0,40}"
            r"\b(?:bound|describe|establish|show|support)\w*\b|"
            r"\b(?:bound|describe|establish|show|support)\w*\b.{0,40}"
            r"\b(?:documents?|evidence|sources?)\b",
            claim,
            flags=re.IGNORECASE,
        )
    )
    if meta_evidence_claim and observations:
        return True, "The claim is explicitly scoped to what the cited evidence establishes."
    required_anchors = (
        1
        if len(claim_terms) <= 3
        or (claim_type == "code_static" and direct_host_read)
        or (claim_type == "current_research" and len(origins) >= 2)
        or (claim_type == "document_record" and multi_document)
        else 2
    )
    lexical_match = len(shared) >= required_anchors

    claim_numbers = set(re.findall(r"[$£€]?\s*\d+(?:[.,]\d+)*%?", claim))
    evidence_numbers = set(re.findall(r"[$£€]?\s*\d+(?:[.,]\d+)*%?", evidence_text))
    numeric_match = not claim_numbers or claim_numbers <= evidence_numbers
    if lexical_match and numeric_match:
        return (
            True,
            "The cited evidence shares the claim's material lexical"
            + (" and numeric" if claim_numbers else "")
            + " anchors.",
        )
    if claim_numbers and not numeric_match:
        return False, "The cited evidence does not contain every material number in the claim."
    return False, "The cited evidence does not directly address enough material claim terms."


def _join_reasons(reasons: Iterable[str]) -> str:
    return "; ".join(item for item in reasons if item)


def _claim_profiles_from_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for check in checks:
        if check.get("name") == "claim_truth_operations":
            profiles = check.get("profiles")
            if isinstance(profiles, list):
                return copy.deepcopy(profiles)
    return []

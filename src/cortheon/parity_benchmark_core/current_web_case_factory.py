"""Evaluator-owned constructor for one immutable current-web case artifact."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cortheon.parity_benchmark_core.oracle_web import _truth_digest


def evaluator_current_web_case(
    *,
    case_id: str,
    prompt: str,
    as_of: str,
    revalidated_at: str,
    valid_until: str,
    sources: list[dict[str, Any]],
    origin_equivalence: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    contradictions: list[dict[str, Any]],
    acquisition_attestation: dict[str, Any],
) -> dict[str, Any]:
    """Build, but never acquire or invent, an evaluator-attested live case."""

    oracle = {
        "as_of": as_of,
        "revalidated_at": revalidated_at,
        "valid_until": valid_until,
        "truth_digest": "",
        "revalidated_truth_digest": "",
        "sources": deepcopy(sources),
        "origin_equivalence": deepcopy(origin_equivalence),
        "claims": deepcopy(claims),
        "contradictions": deepcopy(contradictions),
        "acquisition_attestation": deepcopy(acquisition_attestation),
    }
    digest = _truth_digest(oracle)
    oracle["truth_digest"] = digest
    oracle["revalidated_truth_digest"] = digest
    response_contract = (
        f"\nUse exact as_of {as_of}. Return one JSON object with fields as_of, "
        "sources[{canonical_url}], claims[{id,value,source_urls}], and "
        "contradictions[{claim_id,source_url,rejected_value,resolved_by_url}]."
    )
    return {
        "id": case_id,
        "task_class": "current_web_research",
        "category": "research",
        "domain": "research",
        "difficulty": "hard",
        "prompt": prompt.rstrip() + response_contract,
        "documents": [],
        "expected_verdict": "allow",
        "grader": {
            "type": "current_web_claims",
            "oracle_version": 1,
            "oracle": oracle,
            "oracle_provenance": "frozen_external_pack",
        },
    }

"""Deterministic development seals and label-free public projections."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from cortheon.operator_lift.models import (
    MANIFEST_SCHEMA_VERSION,
    OPERATORS,
    SCHEMA_VERSION,
    ConditionBinding,
    LiftCase,
    LiftManifest,
    LiftThresholds,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def case_commitment(case: LiftCase) -> str:
    return hashlib.sha256(_canonical(asdict(case))).hexdigest()


def cluster_lineage_sha256(case: LiftCase) -> str:
    """Commit the declared causal structure, independent of entities and wording."""

    lineage = {
        "schema_version": SCHEMA_VERSION,
        "operator": case.operator,
        "causal_structure": case.causal_family,
    }
    return hashlib.sha256(_canonical(lineage)).hexdigest()


def design_sha256(
    cases: tuple[LiftCase, ...],
    thresholds: LiftThresholds | None = None,
) -> str:
    thresholds = thresholds or LiftThresholds()
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "thresholds": asdict(thresholds),
        "case_order": [case.case_id for case in cases],
        "case_commitments": {case.case_id: case_commitment(case) for case in cases},
        "cluster_lineage_sha256": {case.case_id: cluster_lineage_sha256(case) for case in cases},
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def public_case(case: LiftCase) -> dict[str, Any]:
    """Return exactly the material visible during a development run."""

    return {
        "schema_version": SCHEMA_VERSION,
        "case_token": hashlib.sha256(f"public:{case_commitment(case)}".encode()).hexdigest()[:32],
        "prompt": case.prompt,
        "evidence": [
            {"source_id": source_id, "content": content} for source_id, content in case.evidence
        ],
        "response_schema": case.response_schema,
        "actions": [
            {"action_id": action_id, "description": description, "cost": cost}
            for action_id, description, cost in case.action_catalog
        ],
        "case_commitment": case_commitment(case),
    }


def public_projection_sha256(case: LiftCase) -> str:
    return hashlib.sha256(_canonical(public_case(case))).hexdigest()


def action_observation(case: LiftCase, action_id: str) -> str:
    """Evaluator-side reveal for one selected adaptive-stopping action."""

    if case.operator != "adaptive_stopping":
        raise ValueError("case has no sequential action observations")
    observations = dict(case.oracle["observations"])
    if action_id not in observations:
        raise ValueError("action has no sealed observation")
    return str(observations[action_id])


def _tokens(value: Any) -> set[str]:
    if isinstance(value, str):
        return set(re.findall(r"[a-z][a-z0-9_]+", value.casefold()))
    if isinstance(value, Mapping):
        return set().union(*(_tokens(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(*(_tokens(item) for item in value))
    return set()


def accepted_response_tokens(case: LiftCase) -> set[str]:
    if case.operator == "hypothesis_framing":
        return _tokens(
            (
                case.oracle["leading"],
                case.oracle["rivals"],
                case.oracle["falsification"],
            )
        )
    if case.operator in {"discriminating_evidence", "contradiction_revision"}:
        return _tokens(case.oracle["expected"])
    if case.operator == "cross_source_derivation":
        return _tokens((case.oracle["conclusion"], case.oracle["premises"]))
    return _tokens(
        (
            case.oracle["expected_actions"],
            case.oracle["decision"],
            "sufficient",
        )
    )


def publicly_derivable_tokens(case: LiftCase) -> set[str]:
    visible: list[Any] = [public_case(case)]
    if case.operator == "adaptive_stopping":
        visible.extend(value for _action, value in case.oracle["observations"])
    return _tokens(visible)


def evaluator_trace(case: LiftCase) -> list[dict[str, Any]]:
    """Build the closed evaluator-owned reveal trace for one stopping case."""

    if case.operator != "adaptive_stopping":
        return []
    costs = {action_id: cost for action_id, _description, cost in case.action_catalog}
    observations = dict(case.oracle["observations"])
    return [
        {
            "sequence": sequence,
            "action_id": action_id,
            "observation_sha256": hashlib.sha256(observations[action_id].encode()).hexdigest(),
            "cost": costs[action_id],
        }
        for sequence, action_id in enumerate(case.oracle["expected_actions"], 1)
    ]


def _manifest_payload(
    cases: tuple[LiftCase, ...],
    full_condition: ConditionBinding,
    placebo_condition: ConditionBinding,
    ablation_conditions: Mapping[str, ConditionBinding],
    thresholds: LiftThresholds,
    design_id: str,
    evaluator_id: str,
    evaluator_implementation_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "design_id": design_id,
        "design_sha256": design_sha256(cases, thresholds),
        "created_before_execution": True,
        "evaluator_id": evaluator_id,
        "evaluator_implementation_sha256": evaluator_implementation_sha256,
        "thresholds": asdict(thresholds),
        "full_condition": asdict(full_condition),
        "placebo_condition": asdict(placebo_condition),
        "ablation_conditions": {
            operator: asdict(ablation_conditions[operator]) for operator in OPERATORS
        },
        "case_order": [case.case_id for case in cases],
        "cluster_lineage_sha256": {case.case_id: cluster_lineage_sha256(case) for case in cases},
        "case_commitments": {case.case_id: case_commitment(case) for case in cases},
    }


def build_manifest(
    cases: tuple[LiftCase, ...],
    *,
    full_condition: ConditionBinding,
    placebo_condition: ConditionBinding,
    ablation_conditions: Mapping[str, ConditionBinding],
    evaluator_id: str,
    evaluator_implementation_sha256: str,
    thresholds: LiftThresholds | None = None,
    design_id: str = "operator_lift_development_v2",
) -> LiftManifest:
    """Freeze the case bank, intervention identities, and thresholds before runs."""

    thresholds = thresholds or LiftThresholds()
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("case ids are not unique")
    if len({case.cluster_id for case in cases}) != len(cases):
        raise ValueError("each development case must be an independent cluster")
    lineage = [cluster_lineage_sha256(case) for case in cases]
    if len(set(lineage)) != len(lineage):
        raise ValueError("cases sharing one causal structure cannot count as independent clusters")
    for operator in OPERATORS:
        count = sum(case.operator == operator for case in cases)
        if count < thresholds.minimum_clusters:
            raise ValueError(f"{operator} has only {count} independent clusters")
    if set(ablation_conditions) != set(OPERATORS):
        raise ValueError("ablation condition set is incomplete")
    payload = _manifest_payload(
        cases,
        full_condition,
        placebo_condition,
        ablation_conditions,
        thresholds,
        design_id,
        evaluator_id,
        evaluator_implementation_sha256,
    )
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    return LiftManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        design_id=design_id,
        design_sha256=payload["design_sha256"],
        created_before_execution=True,
        evaluator_id=evaluator_id,
        evaluator_implementation_sha256=evaluator_implementation_sha256,
        thresholds=thresholds,
        full_condition=full_condition,
        placebo_condition=placebo_condition,
        ablation_conditions=dict(ablation_conditions),
        case_order=tuple(payload["case_order"]),
        cluster_lineage_sha256=payload["cluster_lineage_sha256"],
        case_commitments=payload["case_commitments"],
        manifest_sha256=digest,
    )


def verify_manifest(manifest: LiftManifest, cases: tuple[LiftCase, ...]) -> bool:
    payload = _manifest_payload(
        cases,
        manifest.full_condition,
        manifest.placebo_condition,
        manifest.ablation_conditions,
        manifest.thresholds,
        manifest.design_id,
        manifest.evaluator_id,
        manifest.evaluator_implementation_sha256,
    )
    return (
        payload["case_commitments"] == dict(manifest.case_commitments)
        and hashlib.sha256(_canonical(payload)).hexdigest() == manifest.manifest_sha256
    )

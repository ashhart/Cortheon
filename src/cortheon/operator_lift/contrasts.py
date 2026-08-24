"""Strict scoring and case-clustered full-versus-ablation pairing."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

from cortheon.operator_lift.models import (
    ConditionBinding,
    LiftCase,
    LiftManifest,
    LiftSubmission,
    PairedCluster,
    PairingResult,
    ScoredRun,
)
from cortheon.operator_lift.oracles import grade_case
from cortheon.operator_lift.sealing import (
    public_projection_sha256,
    verify_manifest,
)


def _submission(value: LiftSubmission | Mapping[str, Any]) -> LiftSubmission:
    if isinstance(value, LiftSubmission):
        value.validate()
        return value
    return LiftSubmission.from_mapping(value)


def _expected_bindings(manifest: LiftManifest, case: LiftCase) -> dict[str, ConditionBinding]:
    ablation = manifest.ablation_conditions[case.operator]
    return {
        manifest.full_condition.condition_id: manifest.full_condition,
        ablation.condition_id: ablation,
        manifest.placebo_condition.condition_id: manifest.placebo_condition,
    }


def _provenance_error(
    manifest: LiftManifest,
    case: LiftCase,
    submission: LiftSubmission,
) -> str | None:
    provenance = submission.evaluator_provenance
    if (
        provenance.evaluator_id != manifest.evaluator_id
        or provenance.evaluator_implementation_sha256 != manifest.evaluator_implementation_sha256
    ):
        return "evaluator_identity_mismatch"
    if provenance.public_projection_sha256 != public_projection_sha256(case):
        return "public_projection_mismatch"
    actual_trace = [
        {
            "sequence": event.sequence,
            "action_id": event.action_id,
            "observation_sha256": event.observation_sha256,
            "cost": event.cost,
        }
        for event in provenance.trace
    ]
    if case.operator == "adaptive_stopping":
        costs = {action_id: cost for action_id, _description, cost in case.action_catalog}
        observations = dict(case.oracle["observations"])
        for event in actual_trace:
            action_id = event["action_id"]
            if action_id not in costs or action_id not in observations:
                return "unknown_trace_action"
            expected_observation = hashlib.sha256(observations[action_id].encode()).hexdigest()
            if event["observation_sha256"] != expected_observation:
                return "trace_observation_mismatch"
            if event["cost"] != costs[action_id]:
                return "trace_cost_mismatch"
        if (
            provenance.terminal_after_sequence != len(actual_trace)
            or provenance.terminal_reason != "sufficient"
        ):
            return "stopping_terminal_mismatch"
    elif (
        actual_trace
        or provenance.terminal_after_sequence != 0
        or provenance.terminal_reason != "not_applicable"
    ):
        return "non_stopping_trace_must_be_empty"
    return None


def score_and_pair(
    manifest: LiftManifest,
    cases: tuple[LiftCase, ...],
    submissions: Iterable[LiftSubmission | Mapping[str, Any]],
) -> PairingResult:
    """Score exact scheduled cells; malformed, missing, and extra cells fail closed."""

    if not verify_manifest(manifest, cases):
        return PairingResult((), (), ("manifest_or_case_seal_invalid",))
    by_case = {case.case_id: case for case in cases}
    errors: list[str] = []
    observed: dict[tuple[str, str, int], ScoredRun] = {}
    for index, raw in enumerate(submissions):
        try:
            submission = _submission(raw)
        except (TypeError, ValueError) as exc:
            errors.append(f"submission_{index}:invalid:{exc}")
            continue
        case = by_case.get(submission.case_id)
        if case is None:
            errors.append(f"submission_{index}:unknown_case:{submission.case_id}")
            continue
        if submission.case_commitment != manifest.case_commitments[case.case_id]:
            errors.append(f"{case.case_id}:case_commitment_mismatch")
            continue
        provenance_error = _provenance_error(manifest, case, submission)
        if provenance_error is not None:
            errors.append(f"{case.case_id}:{provenance_error}")
            continue
        bindings = _expected_bindings(manifest, case)
        expected_binding = bindings.get(submission.condition_id)
        if expected_binding is None:
            errors.append(f"{case.case_id}:unexpected_condition:{submission.condition_id}")
            continue
        if submission.condition_config_sha256 != expected_binding.config_sha256:
            errors.append(f"{case.case_id}:{submission.condition_id}:config_mismatch")
            continue
        if submission.implementation_sha256 != expected_binding.implementation_sha256:
            errors.append(f"{case.case_id}:{submission.condition_id}:implementation_mismatch")
            continue
        if submission.repeat >= manifest.thresholds.repetitions:
            errors.append(f"{case.case_id}:{submission.condition_id}:unexpected_repeat")
            continue
        key = (case.case_id, submission.condition_id, submission.repeat)
        if key in observed:
            errors.append(f"{case.case_id}:{submission.condition_id}:{submission.repeat}:duplicate")
            continue
        oracle = grade_case(case, submission.response)
        reasons = list(oracle.reasons)
        trace_bound = True
        if case.operator == "adaptive_stopping":
            trace_actions = [event.action_id for event in submission.evaluator_provenance.trace]
            trace_cost = sum(event.cost for event in submission.evaluator_provenance.trace)
            trace_bound = (
                submission.response.get("actions") == trace_actions
                and submission.response.get("total_cost") == trace_cost
            )
            if not trace_bound:
                reasons.append("response_trace_binding")
        if not submission.delivered:
            reasons.append("delivery_failure")
        if not submission.safe:
            reasons.append("unsafe_outcome")
        if not oracle.proof_eligible:
            errors.append(f"{case.case_id}:oracle_not_proof_eligible")
        observed[key] = ScoredRun(
            case_id=case.case_id,
            cluster_id=case.cluster_id,
            operator=case.operator,
            condition_id=submission.condition_id,
            repeat=submission.repeat,
            correct=oracle.correct and trace_bound and submission.delivered and submission.safe,
            delivered=submission.delivered,
            safe=submission.safe,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    expected: set[tuple[str, str, int]] = set()
    for case in cases:
        for condition_id in _expected_bindings(manifest, case):
            for repeat in range(manifest.thresholds.repetitions):
                expected.add((case.case_id, condition_id, repeat))
    for case_id, condition_id, repeat in sorted(expected - set(observed)):
        errors.append(f"{case_id}:{condition_id}:{repeat}:missing")
    for case_id, condition_id, repeat in sorted(set(observed) - expected):
        errors.append(f"{case_id}:{condition_id}:{repeat}:extra")

    clusters: list[PairedCluster] = []
    for case in cases:
        full_id = manifest.full_condition.condition_id
        ablation_id = manifest.ablation_conditions[case.operator].condition_id
        placebo_id = manifest.placebo_condition.condition_id
        full = tuple(
            int(observed[(case.case_id, full_id, repeat)].correct)
            for repeat in range(manifest.thresholds.repetitions)
            if (case.case_id, full_id, repeat) in observed
        )
        ablation = tuple(
            int(observed[(case.case_id, ablation_id, repeat)].correct)
            for repeat in range(manifest.thresholds.repetitions)
            if (case.case_id, ablation_id, repeat) in observed
        )
        placebo = tuple(
            int(observed[(case.case_id, placebo_id, repeat)].correct)
            for repeat in range(manifest.thresholds.repetitions)
            if (case.case_id, placebo_id, repeat) in observed
        )
        if (
            len(full) == manifest.thresholds.repetitions
            and len(ablation) == len(full)
            and len(placebo) == len(full)
        ):
            clusters.append(
                PairedCluster(
                    cluster_id=case.cluster_id,
                    operator=case.operator,
                    full_scores=full,
                    ablation_scores=ablation,
                    placebo_scores=placebo,
                )
            )
    return PairingResult(
        scored_runs=tuple(observed[key] for key in sorted(observed)),
        clusters=tuple(clusters),
        errors=tuple(sorted(set(errors))),
    )

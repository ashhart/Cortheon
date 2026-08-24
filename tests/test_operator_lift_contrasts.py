from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from cortheon.operator_lift.case_bank import development_cases
from cortheon.operator_lift.contrasts import score_and_pair
from cortheon.operator_lift.models import OPERATORS, ConditionBinding, LiftCase
from cortheon.operator_lift.sealing import (
    build_manifest,
    evaluator_trace,
    public_case,
    public_projection_sha256,
)


def _manifest(cases):
    implementation = "0" * 64
    return build_manifest(
        cases,
        full_condition=ConditionBinding("full", "a" * 64, implementation, None),
        placebo_condition=ConditionBinding("equal_budget_placebo", "9" * 64, implementation, None),
        ablation_conditions={
            operator: ConditionBinding(
                f"without_{operator}",
                chr(98 + index) * 64,
                implementation,
                operator,
            )
            for index, operator in enumerate(OPERATORS)
        },
        evaluator_id="development_evaluator",
        evaluator_implementation_sha256="0" * 64,
    )


def _valid(case: LiftCase) -> dict:
    if case.operator == "hypothesis_framing":
        return {
            key: dict(zip(fields, value, strict=True))
            for key, fields, value in (
                ("leading", ("cause", "outcome", "scope"), case.oracle["leading"]),
                ("rival", ("cause", "outcome", "scope"), case.oracle["rivals"][0]),
                (
                    "falsification",
                    ("intervention", "result", "refutes"),
                    case.oracle["falsification"],
                ),
            )
        }
    if case.operator == "discriminating_evidence":
        return dict(
            zip(
                ("probe_id", "positive_supports", "negative_supports"),
                case.oracle["expected"],
                strict=True,
            )
        )
    if case.operator == "contradiction_revision":
        return dict(
            zip(
                ("prior", "prior_status", "revised", "decisive_source"),
                case.oracle["expected"],
                strict=True,
            )
        )
    if case.operator == "cross_source_derivation":
        subject, relation, object_id = case.oracle["conclusion"]
        return {
            "subject": subject,
            "relation": relation,
            "object": object_id,
            "premises": [
                dict(zip(("source_id", "subject", "relation", "object"), value, strict=True))
                for value in case.oracle["premises"]
            ],
        }
    actions = tuple(case.oracle["expected_actions"])
    costs = {action_id: cost for action_id, _description, cost in case.action_catalog}
    return {
        "actions": list(actions),
        "decision": case.oracle["decision"],
        "total_cost": sum(costs[action] for action in actions),
        "stop_reason": "sufficient",
    }


def _submissions(cases, manifest):
    rows = []
    for case in cases:
        commitment = public_case(case)["case_commitment"]
        bindings = (
            manifest.full_condition,
            manifest.ablation_conditions[case.operator],
            manifest.placebo_condition,
        )
        for binding in bindings:
            rows.extend(
                {
                    "schema_version": 1,
                    "case_id": case.case_id,
                    "case_commitment": commitment,
                    "condition_id": binding.condition_id,
                    "condition_config_sha256": binding.config_sha256,
                    "implementation_sha256": binding.implementation_sha256,
                    "repeat": repeat,
                    "delivered": True,
                    "safe": True,
                    "evaluator_provenance": {
                        "schema_version": 1,
                        "producer": "evaluator",
                        "candidate_supplied": False,
                        "evaluator_id": manifest.evaluator_id,
                        "evaluator_implementation_sha256": (
                            manifest.evaluator_implementation_sha256
                        ),
                        "public_projection_sha256": public_projection_sha256(case),
                        "oracle_access_blocked": True,
                        "trace": evaluator_trace(case),
                        "terminal_after_sequence": len(evaluator_trace(case)),
                        "terminal_reason": (
                            "sufficient"
                            if case.operator == "adaptive_stopping"
                            else "not_applicable"
                        ),
                    },
                    "response": (
                        _valid(case)
                        if binding.condition_id == manifest.full_condition.condition_id
                        else {}
                    ),
                }
                for repeat in range(manifest.thresholds.repetitions)
            )
    return rows


def test_exact_schedule_scores_and_pairs_by_independent_case_cluster() -> None:
    cases = development_cases()
    manifest = _manifest(cases)
    pairing = score_and_pair(manifest, cases, _submissions(cases, manifest))
    assert pairing.errors == ()
    assert len(pairing.scored_runs) == 540
    assert len(pairing.clusters) == 60
    assert all(cluster.full_scores == (1, 1, 1) for cluster in pairing.clusters)
    assert all(cluster.ablation_scores == (0, 0, 0) for cluster in pairing.clusters)
    assert all(cluster.placebo_scores == (0, 0, 0) for cluster in pairing.clusters)


def test_repetitions_never_become_independent_clusters() -> None:
    cases = development_cases()
    manifest = _manifest(cases)
    pairing = score_and_pair(manifest, cases, _submissions(cases, manifest))
    assert len(pairing.clusters) == len(cases)
    assert {len(cluster.full_scores) for cluster in pairing.clusters} == {3}


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "config",
        "digest",
        "commitment",
        "projection",
        "isolation",
        "candidate_supplied",
        "trace",
        "trace_cost",
        "nonstopping_trace",
        "terminal",
        "repeat",
    ],
)
def test_accounting_and_identity_mutations_fail_closed(mutation: str) -> None:
    cases = development_cases()
    manifest = _manifest(cases)
    rows = _submissions(cases, manifest)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif mutation == "config":
        rows[0]["condition_config_sha256"] = "f" * 64
    elif mutation == "digest":
        rows[0]["implementation_sha256"] = "f" * 64
    elif mutation == "commitment":
        rows[0]["case_commitment"] = "f" * 64
    elif mutation == "projection":
        rows[0]["evaluator_provenance"]["public_projection_sha256"] = "f" * 64
    elif mutation == "isolation":
        rows[0]["evaluator_provenance"]["oracle_access_blocked"] = False
    elif mutation == "candidate_supplied":
        rows[0]["evaluator_provenance"]["candidate_supplied"] = True
    elif mutation == "nonstopping_trace":
        rows[0]["evaluator_provenance"]["trace"] = [
            {
                "sequence": 1,
                "action_id": "fake_probe",
                "observation_sha256": "f" * 64,
                "cost": 1,
            }
        ]
        rows[0]["evaluator_provenance"]["terminal_after_sequence"] = 1
        rows[0]["evaluator_provenance"]["terminal_reason"] = "sufficient"
    elif mutation in {"trace", "trace_cost", "terminal"}:
        index = next(index for index, row in enumerate(rows) if row["case_id"] == "stopping_01")
        if mutation == "trace":
            rows[index]["evaluator_provenance"]["trace"][0]["observation_sha256"] = "f" * 64
        elif mutation == "trace_cost":
            rows[index]["evaluator_provenance"]["trace"][0]["cost"] += 1
        else:
            rows[index]["evaluator_provenance"]["terminal_after_sequence"] = 0
    else:
        rows[0]["repeat"] = manifest.thresholds.repetitions
    pairing = score_and_pair(manifest, cases, rows)
    assert pairing.errors
    assert any(mutation in error or "missing" in error for error in pairing.errors)


def test_delivery_and_safety_are_failures_not_exclusions() -> None:
    cases = development_cases()
    manifest = _manifest(cases)
    rows = _submissions(cases, manifest)
    rows[0]["delivered"] = False
    rows[1]["safe"] = False
    pairing = score_and_pair(manifest, cases, rows)
    assert pairing.errors == ()
    first = [run for run in pairing.scored_runs if run.case_id == cases[0].case_id]
    assert any("delivery_failure" in run.reasons for run in first)
    assert any("unsafe_outcome" in run.reasons for run in first)
    assert sum(run.correct for run in first if run.condition_id == "full") == 1


@pytest.mark.parametrize("behavior", ["premature", "extra", "reordered"])
def test_valid_evaluator_traces_measure_bad_stopping_behavior_in_the_denominator(
    behavior: str,
) -> None:
    cases = development_cases()
    manifest = _manifest(cases)
    rows = _submissions(cases, manifest)
    case_id = "stopping_06" if behavior in {"premature", "reordered"} else "stopping_01"
    case = next(case for case in cases if case.case_id == case_id)
    index = next(
        index
        for index, row in enumerate(rows)
        if row["case_id"] == case_id and row["condition_id"] == "full"
    )
    row = rows[index]
    trace = row["evaluator_provenance"]["trace"]
    if behavior == "premature":
        trace.pop()
    elif behavior == "reordered":
        for field in ("action_id", "observation_sha256", "cost"):
            trace[0][field], trace[1][field] = trace[1][field], trace[0][field]
    else:
        action_id, _description, cost = case.action_catalog[1]
        observation = dict(case.oracle["observations"])[action_id]
        trace.append(
            {
                "sequence": 2,
                "action_id": action_id,
                "observation_sha256": hashlib.sha256(observation.encode()).hexdigest(),
                "cost": cost,
            }
        )
    row["evaluator_provenance"]["terminal_after_sequence"] = len(trace)
    row["response"]["actions"] = [event["action_id"] for event in trace]
    row["response"]["total_cost"] = sum(event["cost"] for event in trace)
    pairing = score_and_pair(manifest, cases, rows)
    assert pairing.errors == ()
    run = next(
        run
        for run in pairing.scored_runs
        if run.case_id == case_id and run.condition_id == "full" and run.repeat == 0
    )
    assert run.correct is False

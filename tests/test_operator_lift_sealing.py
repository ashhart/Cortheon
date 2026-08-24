from __future__ import annotations

import json
from dataclasses import replace

import pytest

from cortheon.operator_lift.case_bank import development_cases
from cortheon.operator_lift.models import OPERATORS, ConditionBinding, LiftThresholds
from cortheon.operator_lift.sealing import (
    accepted_response_tokens,
    action_observation,
    build_manifest,
    case_commitment,
    cluster_lineage_sha256,
    design_sha256,
    public_case,
    publicly_derivable_tokens,
    verify_manifest,
)
from cortheon.qualification_core.conditions import condition_record
from cortheon.qualification_factory import closed_registry


def _conditions():
    implementation = "0" * 64
    full = ConditionBinding("full", "a" * 64, implementation, None)
    ablations = {
        operator: ConditionBinding(
            f"without_{operator}", chr(98 + index) * 64, implementation, operator
        )
        for index, operator in enumerate(OPERATORS)
    }
    return full, ablations


def _manifest(cases, full, ablations):
    placebo_record = condition_record(
        "equal_budget_placebo",
        implementation_sha256=full.implementation_sha256,
    )
    return build_manifest(
        cases,
        full_condition=full,
        placebo_condition=ConditionBinding(
            placebo_record["id"],
            placebo_record["config_sha256"],
            placebo_record["implementation_sha256"],
            None,
        ),
        ablation_conditions=ablations,
        evaluator_id="development_evaluator",
        evaluator_implementation_sha256="0" * 64,
    )


def test_manifest_is_deterministic_and_binds_private_oracles() -> None:
    cases = development_cases()
    full, ablations = _conditions()
    first = _manifest(cases, full, ablations)
    second = _manifest(cases, full, ablations)
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.design_sha256 == design_sha256(cases)
    assert verify_manifest(first, cases)
    changed = (replace(cases[0], oracle={"changed": True}), *cases[1:])
    assert case_commitment(changed[0]) != case_commitment(cases[0])
    assert not verify_manifest(first, changed)


def test_parameter_siblings_cannot_claim_independent_cluster_status() -> None:
    cases = development_cases()
    full, ablations = _conditions()
    original = cases[0]
    sibling = replace(
        original,
        case_id="hypothesis_parameter_sibling",
        cluster_id="cluster_hypothesis_parameter_sibling",
        prompt="Same causal structure with different names and wording.",
        evidence=(("source_a", "Renamed fact A."), ("source_b", "Renamed fact B.")),
    )
    assert case_commitment(sibling) != case_commitment(original)
    assert cluster_lineage_sha256(sibling) == cluster_lineage_sha256(original)
    try:
        build_manifest(
            (*cases, sibling),
            full_condition=full,
            placebo_condition=_manifest(cases, full, ablations).placebo_condition,
            ablation_conditions=ablations,
            evaluator_id="development_evaluator",
            evaluator_implementation_sha256="0" * 64,
        )
    except ValueError as exc:
        assert "one causal structure" in str(exc)
    else:
        raise AssertionError("a parameter sibling counted as an independent cluster")


def test_public_projection_excludes_every_private_label_field() -> None:
    for case in development_cases():
        projection = public_case(case)
        assert set(projection) == {
            "schema_version",
            "case_token",
            "prompt",
            "evidence",
            "response_schema",
            "actions",
            "case_commitment",
        }
        encoded = json.dumps(projection, sort_keys=True)
        assert '"oracle"' not in encoded
        assert '"observations"' not in encoded
        assert projection["case_commitment"] == case_commitment(case)


def test_every_accepted_response_token_is_publicly_derivable() -> None:
    for case in development_cases():
        hidden = accepted_response_tokens(case) - publicly_derivable_tokens(case)
        assert not hidden, (case.case_id, hidden)


def test_adaptive_observations_are_revealed_only_per_selected_action() -> None:
    case = next(case for case in development_cases() if case.operator == "adaptive_stopping")
    action_id = case.oracle["expected_actions"][0]
    assert action_observation(case, action_id) == dict(case.oracle["observations"])[action_id]


def test_preregistered_thresholds_cannot_be_relaxed_after_results() -> None:
    with pytest.raises(ValueError, match="immutable preregistered"):
        LiftThresholds(minimum_lift=0.01)
    with pytest.raises(ValueError, match="immutable preregistered"):
        LiftThresholds(minimum_clusters=2)
    with pytest.raises(ValueError, match="threshold types are immutable"):
        LiftThresholds(minimum_clusters=12.0)  # type: ignore[arg-type]


def test_condition_configs_are_distinct_with_one_frozen_implementation() -> None:
    cases = development_cases()
    full, ablations = _conditions()
    operator = OPERATORS[0]
    ablations[operator] = replace(ablations[operator], config_sha256=full.config_sha256)
    with pytest.raises(ValueError, match="config digests must be distinct"):
        _manifest(cases, full, ablations)
    full, ablations = _conditions()
    ablations[operator] = replace(ablations[operator], implementation_sha256="1" * 64)
    with pytest.raises(ValueError, match="share one frozen implementation"):
        _manifest(cases, full, ablations)


def test_condition_bindings_match_real_closed_qualification_records() -> None:
    implementation = "9" * 64
    registry = closed_registry(implementation)
    condition_ids = (
        "full",
        "equal_budget_placebo",
        *(f"without_{operator}" for operator in OPERATORS),
    )
    records = {condition_id: registry[condition_id] for condition_id in condition_ids}
    full_record = records["full"]
    full = ConditionBinding(
        full_record["id"],
        full_record["config_sha256"],
        full_record["implementation_sha256"],
        None,
    )
    ablations = {
        operator: ConditionBinding(
            records[f"without_{operator}"]["id"],
            records[f"without_{operator}"]["config_sha256"],
            records[f"without_{operator}"]["implementation_sha256"],
            operator,
        )
        for operator in OPERATORS
    }
    manifest = _manifest(development_cases(), full, ablations)
    bindings = (
        manifest.full_condition,
        manifest.placebo_condition,
        *manifest.ablation_conditions.values(),
    )
    assert len({binding.config_sha256 for binding in bindings}) == len(bindings)
    assert {binding.implementation_sha256 for binding in bindings} == {implementation}
    full_operators = records["full"]["config"]["operators"]
    for operator in OPERATORS:
        ablation_operators = records[f"without_{operator}"]["config"]["operators"]
        assert ablation_operators[operator] is False
        assert {
            name for name in full_operators if ablation_operators[name] != full_operators[name]
        } == {operator}


def test_condition_ids_are_closed_and_canonical() -> None:
    cases = development_cases()
    full, ablations = _conditions()
    with pytest.raises(ValueError, match="full condition id"):
        _manifest(cases, replace(full, condition_id="treatment"), ablations)
    operator = OPERATORS[0]
    ablations[operator] = replace(ablations[operator], condition_id="ablation_one")
    with pytest.raises(ValueError, match="ablation condition id"):
        _manifest(cases, full, ablations)

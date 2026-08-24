from __future__ import annotations

from cortheon.operator_lift.case_bank import development_cases
from cortheon.operator_lift.execution_models import ExecutionConfig
from cortheon.operator_lift.execution_schedule import (
    execution_manifest,
    full_schedule,
    public_pack,
    run_descriptor,
    selected_schedule,
)
from cortheon.qualification_core.conditions import condition_record


def _config(api_key: str = "never-serialize-this") -> ExecutionConfig:
    return ExecutionConfig("http://127.0.0.1:9000/v1", "local", "small", api_key)


def test_schedule_is_exactly_paired_three_repeat_cells() -> None:
    cases = development_cases()
    manifest = execution_manifest(cases)
    schedule = full_schedule(manifest, cases)
    assert len(schedule) == 540
    assert {cell.sequence for cell in schedule} == set(range(1, 541))
    for case in cases:
        cells = [cell for cell in schedule if cell.case_id == case.case_id]
        assert {cell.condition_id for cell in cells} == {
            "full",
            f"without_{case.operator}",
            "equal_budget_placebo",
        }
        assert {(cell.condition_id, cell.repeat) for cell in cells} == {
            (condition, repeat)
            for condition in ("full", f"without_{case.operator}", "equal_budget_placebo")
            for repeat in range(3)
        }


def test_manifest_uses_real_closed_condition_records() -> None:
    manifest = execution_manifest()
    bindings = (
        manifest.full_condition,
        manifest.placebo_condition,
        *manifest.ablation_conditions.values(),
    )
    assert len({binding.config_sha256 for binding in bindings}) == 7
    assert len({binding.implementation_sha256 for binding in bindings}) == 1
    for binding in bindings:
        record = condition_record(
            binding.condition_id,
            implementation_sha256=binding.implementation_sha256,
        )
        assert binding.config_sha256 == record["config_sha256"]
        assert binding.implementation_sha256 == record["implementation_sha256"]


def test_public_pack_and_run_descriptor_never_serialize_credentials_or_oracles() -> None:
    cases = development_cases()
    manifest = execution_manifest(cases)
    pack = public_pack(cases)
    pilot = selected_schedule(full_schedule(manifest, cases), cases, 1)
    descriptor = run_descriptor(manifest, cases, pilot, _config(), str(pack["pack_sha256"]))
    serialized = repr((pack, descriptor))
    assert "never-serialize-this" not in serialized
    assert "oracle" not in serialized.casefold()
    assert len(pilot) == 9
    assert descriptor["claim_eligible"] is False
    assert descriptor["repeats_are_independent_cases"] is False
    descriptor_text = repr(descriptor).casefold()
    assert all(
        token not in descriptor_text
        for token in ("never-serialize-this", "case_id", "prompt", "evidence", "base_url")
    )
    assert descriptor["evaluator_identity"]["endpoint_scope"] == "loopback_openai_v1"


def test_run_descriptor_rejects_partial_or_duplicate_case_cells() -> None:
    import pytest

    cases = development_cases()
    manifest = execution_manifest(cases)
    pack = public_pack(cases)
    pilot = selected_schedule(full_schedule(manifest, cases), cases, 1)
    for bad in (pilot[:-1], (*pilot, pilot[0])):
        with pytest.raises(ValueError, match="exact complete-case"):
            run_descriptor(manifest, cases, bad, _config(), str(pack["pack_sha256"]))


def test_operator_pilot_selects_one_complete_causal_cluster() -> None:
    cases = development_cases()
    manifest = execution_manifest(cases)
    pilot = selected_schedule(
        full_schedule(manifest, cases),
        cases,
        1,
        "contradiction_revision",
    )
    assert len(pilot) == 9
    selected = {cell.case_id for cell in pilot}
    assert len(selected) == 1
    assert selected <= {case.case_id for case in cases if case.operator == "contradiction_revision"}
    assert {cell.condition_id for cell in pilot} == {
        "full",
        "without_contradiction_revision",
        "equal_budget_placebo",
    }

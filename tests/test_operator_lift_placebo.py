"""Red contracts for the sealed equal-budget placebo benchmark arm."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from cortheon.operator_lift.case_bank import development_cases
from cortheon.operator_lift.execution_models import ExecutionConfig
from cortheon.operator_lift.execution_report import content_free_report
from cortheon.operator_lift.execution_schedule import (
    execution_manifest,
    full_schedule,
    public_pack,
    run_descriptor,
)
from cortheon.qualification_core.conditions import condition_record, execution_profile

PLACEBO = "equal_budget_placebo"


def _config() -> ExecutionConfig:
    return ExecutionConfig("http://127.0.0.1:9000/v1", "local", "small", "secret")


def _placebo(manifest):
    binding = getattr(manifest, "placebo_condition", None)
    assert binding is not None, "manifest must seal one placebo_condition"
    return binding


def _three_arm_submissions(cases, manifest):
    from test_operator_lift_contrasts import _submissions

    return _submissions(cases, manifest)


def test_manifest_seals_one_canonical_equal_budget_placebo() -> None:
    manifest = execution_manifest()
    binding = _placebo(manifest)
    assert binding.condition_id == PLACEBO
    assert binding.disabled_operator is None
    record = condition_record(
        PLACEBO,
        implementation_sha256=binding.implementation_sha256,
    )
    assert record["config_sha256"] == binding.config_sha256
    assert record["implementation_sha256"] == binding.implementation_sha256
    assert binding.config_sha256 not in {
        manifest.full_condition.config_sha256,
        *(item.config_sha256 for item in manifest.ablation_conditions.values()),
    }


def test_schedule_has_exactly_three_arms_three_repeats_for_every_case() -> None:
    cases = development_cases()
    schedule = full_schedule(execution_manifest(cases), cases)
    assert len(cases) == 60
    assert len(schedule) == 540
    assert len({cell.cell_id for cell in schedule}) == 540
    assert {cell.sequence for cell in schedule} == set(range(1, 541))
    for case in cases:
        cells = [cell for cell in schedule if cell.case_id == case.case_id]
        expected = {"full", f"without_{case.operator}", PLACEBO}
        assert Counter(cell.condition_id for cell in cells) == Counter(dict.fromkeys(expected, 3))
        assert {(cell.condition_id, cell.repeat) for cell in cells} == {
            (condition, repeat) for condition in expected for repeat in range(3)
        }


def test_manifest_rejects_missing_duplicate_and_aliased_placebo() -> None:
    manifest = execution_manifest()
    binding = _placebo(manifest)
    with pytest.raises(ValueError, match="placebo"):
        replace(manifest, placebo_condition=None)
    with pytest.raises(ValueError, match="placebo"):
        replace(manifest, placebo_condition=manifest.full_condition)
    with pytest.raises(ValueError, match="placebo"):
        replace(manifest, placebo_condition=replace(binding, condition_id="placebo"))


def test_run_descriptor_seals_equal_policy_and_fixed_neutral_protocol() -> None:
    cases = development_cases()
    manifest = execution_manifest(cases)
    pack = public_pack(cases)
    descriptor = run_descriptor(
        manifest,
        cases,
        full_schedule(manifest, cases),
        _config(),
        str(pack["pack_sha256"]),
    )
    policies = descriptor["condition_policies"]
    assert set(policies) == {
        "full",
        PLACEBO,
        *(f"ablation_{index}" for index, _operator in enumerate(manifest.ablation_conditions)),
    }
    ceiling_keys = {
        "provider_id",
        "model_id",
        "tool_catalogue_sha256",
        "context_tokens",
        "output_tokens",
        "max_steps",
        "max_tool_calls",
        "timeout_seconds",
    }
    assert len({tuple(policies[name][key] for key in ceiling_keys) for name in policies}) == 1
    placebo = policies[PLACEBO]
    assert placebo["runtime_used"] is False
    assert placebo["private_labels_access"] is False
    assert placebo["additional_task_scaffold"] is False
    assert placebo["neutral_protocol_sha256"]
    assert placebo["neutral_continuations"] >= 1


def test_placebo_descriptor_requires_room_for_the_neutral_review() -> None:
    cases = development_cases()
    manifest = execution_manifest(cases)
    pack = public_pack(cases)
    with pytest.raises(ValueError, match="at least two model steps"):
        run_descriptor(
            manifest,
            cases,
            full_schedule(manifest, cases),
            replace(_config(), max_steps=1),
            str(pack["pack_sha256"]),
        )


def test_pairing_preserves_primary_contrast_and_adds_placebo_contrast() -> None:
    from cortheon.operator_lift.contrasts import score_and_pair

    cases = development_cases()
    manifest = execution_manifest(cases)
    rows = _three_arm_submissions(cases, manifest)
    pairing = score_and_pair(manifest, cases, rows)
    assert not pairing.errors
    assert len(pairing.clusters) == len(cases)
    for cluster in pairing.clusters:
        assert len(cluster.full_scores) == 3
        assert len(cluster.ablation_scores) == 3
        assert len(cluster.placebo_scores) == 3
        assert cluster.primary_contrast == "full_vs_ablation"


def test_placebo_host_takes_one_neutral_review_pass_without_runtime(tmp_path: Path) -> None:
    from cortheon.benchmark_core.generic_mcp_executor import IsolatedExecutor
    from cortheon.benchmark_core.generic_mcp_host import GenericMcpHost
    from cortheon.benchmark_core.generic_mcp_model import ModelTurn
    from cortheon.benchmark_core.generic_mcp_protocol import EQUAL_BUDGET_REVIEW_PROMPT

    class ReviewModel:
        provider_id = "local"
        model_id = "small"
        endpoint_sha256 = "e" * 64

        def __init__(self) -> None:
            self.catalogues: list[tuple[str, ...]] = []
            self.messages: list[list[dict[str, Any]]] = []

        def complete(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            *,
            tool_choice: str = "auto",
        ) -> ModelTurn:
            assert tool_choice == "auto"
            self.catalogues.append(tuple(tool["function"]["name"] for tool in tools))
            self.messages.append([dict(message) for message in messages])
            text = "Draft answer" if len(self.catalogues) == 1 else "Reviewed answer"
            return ModelTurn("local", "small", text, (), "stop", 2)

    marker = "placebo-workspace"
    (tmp_path / ".cortheon-evaluator-workspace").write_text(marker, encoding="utf-8")
    executor = IsolatedExecutor(tmp_path, marker_nonce=marker)
    profile = execution_profile(PLACEBO, "a" * 64)
    profile["nonce"] = "9" * 32
    model = ReviewModel()
    result = GenericMcpHost(
        task_id="placebo-review",
        evaluation_profile=profile,
        model=model,  # type: ignore[arg-type]
        executor=executor,
        max_steps=2,
    ).run("Give the best supported answer.", task_kind="general")

    assert result.delivered and result.final_text == "Reviewed answer"
    assert len(model.catalogues) == 2 and model.catalogues[0] == model.catalogues[1]
    assert model.messages[1][-1] == {"role": "system", "content": EQUAL_BUDGET_REVIEW_PROMPT}
    assert not any(event["type"] == "runtime_transition" for event in result.events)
    assert not any(event["type"] == "evaluation_receipt" for event in result.events)


def test_report_exposes_arm_level_realized_compute_and_balance() -> None:
    cases = development_cases()
    manifest = execution_manifest(cases)
    submissions = _three_arm_submissions(cases, manifest)
    summaries = [
        {
            "condition_id": row["condition_id"],
            "identity_valid": True,
            "transcript_valid": True,
            "timed_out": False,
            "inference_calls": 2,
            "tokens": 20,
            "tool_calls": 1,
            "latency_seconds": 0.25,
        }
        for row in submissions
    ]
    report = content_free_report(
        manifest,
        cases,
        submissions,
        summaries,
        run_sha256="a" * 64,
        event_chain_sha256="b" * 64,
        planned_cells=540,
    )
    realized = report["execution"]["realized_compute_by_arm"]
    assert {"full", "ablation", PLACEBO} <= set(realized)
    for arm in ("full", "ablation", PLACEBO):
        assert {
            "inference_calls",
            "tokens",
            "tool_calls",
            "latency_seconds",
        } <= set(realized[arm])
    balance = report["execution"]["compute_balance"]
    assert balance["comparison"] == "full_vs_equal_budget_placebo"
    assert balance["configured_budget_equal"] is True
    assert balance["realized_compute_equal"] is True
    assert balance["claim_valid"] is True


def test_compute_diagnostic_does_not_mislabel_equal_budget_as_equal_usage() -> None:
    cases = development_cases()
    manifest = execution_manifest(cases)
    submissions = _three_arm_submissions(cases, manifest)
    summaries = [
        {
            "inference_calls": 2,
            "tokens": 30 if row["condition_id"] == "full" else 20,
            "tool_calls": 1,
            "latency_seconds": 0.25,
        }
        for row in submissions
    ]
    report = content_free_report(
        manifest,
        cases,
        submissions,
        summaries,
        run_sha256="a" * 64,
        event_chain_sha256="b" * 64,
        planned_cells=540,
    )
    balance = report["execution"]["compute_balance"]
    assert balance["configured_budget_equal"] is True
    assert balance["realized_compute_equal"] is False
    assert balance["realized_compute_required_for_claim"] is False
    assert balance["claim_valid"] is True

    incomplete = content_free_report(
        manifest,
        cases,
        submissions,
        summaries[:-1],
        run_sha256="a" * 64,
        event_chain_sha256="b" * 64,
        planned_cells=540,
    )
    assert incomplete["execution"]["compute_balance"]["metrics_complete"] is False
    assert incomplete["execution"]["compute_balance"]["claim_valid"] is False

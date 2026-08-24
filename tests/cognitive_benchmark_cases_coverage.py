# ruff: noqa: F401

import argparse
import json
import subprocess
from dataclasses import asdict

import pytest
from scaling_support import report as _sealed_scaling_report

from cortheon.benchmark_core.execution_provenance import ProcessCapture
from cortheon.cognitive_benchmark import (
    DiagnosticCase,
    EvaluationOutcome,
    ImportCase,
    JoinCase,
    LongHorizonCase,
    PatchCase,
    PlanningCase,
    ReasoningCase,
    ResearchCase,
    RunResult,
    SemanticCase,
    _audit_manifest,
    _blinded_case,
    _condition_summary,
    _delivery_succeeded,
    _event_statistics,
    _final_text,
    _frontier_comparison,
    _grade,
    _grade_patch_workspace,
    _integer_constants,
    _model_endpoint_health,
    _north_star_coverage,
    _paired_summary,
    _pi_provider_config,
    _postflight_probe,
    _provider_config,
    _workspace_environment,
    discover_benchmark_cases,
    discover_cases,
    discover_diagnostic_cases,
    discover_join_cases,
    discover_long_horizon_cases,
    discover_patch_cases,
    discover_planning_cases,
    discover_reasoning_cases,
    discover_semantic_cases,
    isolated_repository,
    run_frontier_cli_job,
    run_job,
    scaling_curve,
    verify_audit_bundle,
)
from cortheon.cognitive_benchmark import (
    main as cognitive_benchmark_main,
)


def test_diagnostic_cases_require_root_cause_and_reject_distractors():
    case = next(
        item
        for item in discover_diagnostic_cases(count=4, seed=4)
        if "four attempts" in item.prompt
    )

    assert isinstance(case, DiagnosticCase)
    assert _grade(
        case,
        "This is an off-by-one: range(max_retries + 1) produces four attempts.",
    )
    assert not _grade(case, "The four attempts are caused by a DNS timeout.")
    assert not _grade(case, "There is an off-by-one.")


def test_planning_cases_require_every_owner_and_dependency_order():
    case = next(
        item for item in discover_planning_cases(count=4, seed=4) if "billing" in item.prompt
    )

    assert isinstance(case, PlanningCase)
    assert _grade(
        case,
        "Priya: freeze invoice schema; Tomas: migrate billing data; "
        "Mei: deploy billing API; Amina: enable invoice UI; "
        "then customer notification.",
    )
    assert not _grade(
        case,
        "Mei: deploy billing API; Priya: freeze invoice schema; "
        "Tomas: migrate billing data; Amina: enable invoice UI; "
        "then customer notification.",
    )
    assert not _grade(
        case,
        "Freeze invoice schema, migrate billing data, deploy billing API, "
        "enable invoice UI, then customer notification.",
    )


def test_long_horizon_grader_requires_all_mutations_and_hidden_behavior(tmp_path):
    case = next(
        item
        for item in discover_long_horizon_cases(count=3, seed=4)
        if "checkout" in item.prompt.casefold()
    )
    assert isinstance(case, LongHorizonCase)
    for relative, content in case.files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    (tmp_path / "journey/pricing.py").write_text(
        "def discounted(subtotal: float, rate: float) -> float:\n    return subtotal * (1 - rate)\n"
    )
    (tmp_path / "journey/shipping.py").write_text(
        "def shipping_fee(subtotal: float, threshold: float = 50.0) -> float:\n"
        "    return 0.0 if subtotal >= threshold else 4.99\n"
    )
    correct, failure = _grade_patch_workspace(case, tmp_path)
    assert not correct
    assert "journey/README.md" in str(failure)

    (tmp_path / "journey/README.md").write_text(
        "# Checkout rules\n\nOrders of 50 or more receive free shipping.\n"
    )
    correct, failure = _grade_patch_workspace(case, tmp_path)
    assert correct, failure


def test_blinded_case_hides_new_suite_answers_and_fixture_contents():
    case = discover_planning_cases(count=1, seed=4)[0]

    blinded = _blinded_case(case)

    assert blinded["prompt"] == "<blinded during grading>"
    assert blinded["expected"] == "<blinded during grading>"
    assert blinded["ordered_steps"] == "<blinded during grading>"
    assert all(item["content"] == "<blinded during grading>" for item in blinded["files"])


def test_default_mixed_suite_exercises_coding_and_semantic_reasoning(tmp_path):
    package = tmp_path / "src" / "demo"
    package.mkdir(parents=True)
    (package / "left.py").write_text("import json\nLEFT_LIMIT = 7\n")
    (package / "right.py").write_text("import pathlib\nRIGHT_LIMIT = 9\n")
    (package / "third.py").write_text("import hashlib\nTHIRD_LIMIT = 11\n")
    (package / "fourth.py").write_text("import re\nFOURTH_LIMIT = 13\n")

    cases = discover_benchmark_cases(
        tmp_path,
        count=8,
        seed=11,
        suite="mixed",
    )

    assert sum(isinstance(case, PatchCase) for case in cases) == 2
    assert sum(isinstance(case, SemanticCase) for case in cases) == 2


def test_northstar_suite_allocates_every_required_task_class(tmp_path, monkeypatch):
    package = tmp_path / "src" / "demo"
    package.mkdir(parents=True)
    (package / "left.py").write_text("LEFT_LIMIT = 7\n")
    (package / "right.py").write_text("RIGHT_LIMIT = 9\n")
    monkeypatch.setattr(
        "cortheon.cognitive_benchmark._latest_pypi_release",
        lambda _project: "1.2.3",
    )

    cases = discover_benchmark_cases(
        tmp_path,
        count=9,
        seed=11,
        suite="northstar",
    )

    assert {type(case) for case in cases} == {
        DiagnosticCase,
        LongHorizonCase,
        PatchCase,
        PlanningCase,
        ResearchCase,
        SemanticCase,
        JoinCase,
        ReasoningCase,
    }
    assert {case.mode for case in cases if isinstance(case, ReasoningCase)} == {
        "novel_synthesis",
        "ambiguity",
    }


def test_northstar_coverage_reports_every_missing_front():
    results = [
        RunResult(
            task_type,
            0,
            "cortheon",
            True,
            "correct",
            True,
            True,
            1.0,
            10,
            0,
            0,
            False,
            None,
            task_type=task_type,
        )
        for task_type in (
            "cross_file_numeric_join",
            "repository_patch",
            "semantic_cross_document",
        )
    ]

    coverage = _north_star_coverage(results)

    assert not coverage["complete"]
    assert coverage["missing_task_classes"] == [
        "ambiguity_resolution",
        "constraint_bound_planning",
        "current_web_research",
        "evidence_bound_debugging",
        "long_horizon_execution",
        "novel_abductive_synthesis",
    ]


def _scaling_report(
    *,
    budget: int,
    baseline: tuple[bool, ...],
    cortheon: tuple[bool, ...],
    frontier: tuple[bool, ...] = (),
) -> dict:
    runs = []
    terminal = asdict(EvaluationOutcome("pi", "success", "pi_assistant", "stop"))
    for condition, outcomes in (
        ("baseline", baseline),
        ("cortheon", cortheon),
        ("frontier", frontier),
    ):
        runs.extend(
            {
                "case_id": f"case_{index}",
                "repeat": 0,
                "condition": condition,
                "correct": correct,
                "delivered": True,
                "timed_out": False,
                "process_error": None,
                "inference_model_id": "frontier" if condition == "frontier" else "demo",
                "evaluator_outcome": terminal,
                "artifact_correct": correct,
                "substrate_telemetry_valid": condition == "cortheon",
                "runtime_sessions_completed": 1 if condition == "cortheon" else 0,
                "runtime_sessions_evidence_closed": 0,
                "latency_seconds": 1.0 + index,
                "tool_calls": index,
                "cost_usd": 0.01,
            }
            for index, correct in enumerate(outcomes)
        )
    return _sealed_scaling_report(runs, budget=budget)


def test_audit_bundle_detects_run_tampering_and_authenticates():
    report = _scaling_report(
        budget=4,
        baseline=(False, True),
        cortheon=(True, True),
    )
    report["audit"] = _audit_manifest(report, signing_key="secret")

    assert verify_audit_bundle(report, signing_key="secret") == {
        "content_valid": True,
        "signature_present": True,
        "signature_valid": True,
    }

    report["runs"][0]["correct"] = True
    verification = verify_audit_bundle(report, signing_key="secret")
    assert verification["content_valid"] is False
    assert verification["signature_valid"] is True


def test_scaling_curve_reports_amplification_frontier_gap_and_monotonicity():
    low = _scaling_report(
        budget=4,
        baseline=(False, False, True, True),
        cortheon=(False, True, True, True),
        frontier=(True, True, True, True),
    )
    high = _scaling_report(
        budget=8,
        baseline=(False, False, True, True),
        cortheon=(True, True, True, True),
        frontier=(True, True, True, True),
    )

    curve = scaling_curve([high, low])

    assert [point["budget"] for point in curve["points"]] == [4, 8]
    assert curve["points"][0]["amplification"] == 0.25
    assert curve["points"][0]["frontier_gap"] == -0.25
    assert curve["points"][1]["frontier_gap"] == 0.0
    assert len(curve["families"]) == 1
    family = curve["families"][0]
    assert family["model"] == "local/demo"
    assert family["host"] == "pi"
    assert family["budgets"] == [4, 8]
    assert family["cortheon_accuracy_monotonic"] is True
    assert len(family["family_sha256"]) == 64

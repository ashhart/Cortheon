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


def test_patch_cases_use_hidden_behavior_and_protected_tests(tmp_path):
    case = next(
        item for item in discover_patch_cases(count=4, seed=4) if "calculator" in item.test_command
    )
    assert isinstance(case, PatchCase)
    for relative, content in case.files:
        (tmp_path / relative).write_text(content)

    correct, failure = _grade_patch_workspace(case, tmp_path)
    assert not correct
    assert failure

    implementation = next(path for path, _content in case.files if path not in case.protected_paths)
    (tmp_path / implementation).write_text(
        "def add(left: int, right: int) -> int:\n    return left + right\n"
    )
    correct, failure = _grade_patch_workspace(case, tmp_path)
    assert correct, failure
    assert failure is None


def test_patch_cases_include_an_unnamed_discovery_task():
    case = next(
        item
        for item in discover_patch_cases(count=5, seed=4)
        if "Locate the implementation" in item.prompt
    )

    assert isinstance(case, PatchCase)
    assert "arithmetic_engine.py" not in case.prompt
    assert "multiply" not in case.prompt
    assert len(case.files) == 2


def test_semantic_cases_require_the_complete_cross_document_chain():
    case = next(
        item for item in discover_semantic_cases(count=4, seed=4) if "Checkout" in item.prompt
    )

    assert isinstance(case, SemanticCase)
    assert len(case.files) == 3
    assert _grade(
        case,
        (
            "Checkout is Coral. Coral requires the Duty Security Officer, "
            "and that role is Amara Okafor."
        ),
    )
    assert not _grade(case, "Amara Okafor approves it.")
    assert not _grade(case, "[Cortheon withheld this output]")


def test_semantic_suite_includes_alias_resolution_without_single_source_answer():
    case = next(
        item for item in discover_semantic_cases(count=9, seed=4) if "Payments API" in item.prompt
    )

    assert isinstance(case, SemanticCase)
    assert all(
        not all(term in content.casefold() for term in case.expected)
        for _path, content in case.files
    )
    assert _grade(
        case,
        "Payments API is Lantern, which depends on Queue Zephyr; Rina Sol owns it.",
    )
    assert not _grade(case, "Rina Sol owns it.")


def test_semantic_suite_requires_current_authority_to_resolve_conflict():
    case = next(
        item
        for item in discover_semantic_cases(count=9, seed=4)
        if "directories conflict" in item.prompt
    )

    assert _grade(
        case,
        "Checkout is Coral, Coral requires the Duty Security Officer, and the "
        "current authority names Amara Okafor.",
    )
    assert not _grade(
        case,
        "Checkout is Coral and the old directory names Lin Wei.",
    )
    assert _grade(
        case,
        "Checkout is **Coral** and requires the **Duty Security Officer**. "
        "The old directory (archived) lists Lin Wei; that stale assignment is "
        "superseded. The current authority names **Amara Okafor**.",
    )
    assert not _grade(
        case,
        "Checkout is Coral and the Duty Security Officer is Lin Wei, "
        "although Amara Okafor is also mentioned.",
    )


def test_semantic_suite_requires_every_conjunctive_policy_premise():
    case = next(
        item
        for item in discover_semantic_cases(count=9, seed=4)
        if "every condition" in item.prompt
    )

    assert all(
        not all(term in content.casefold() for term in case.expected)
        for _path, content in case.files
    )
    assert _grade(
        case,
        "Kepler processes biometric templates and serves EEA residents. The rule "
        "therefore requires the Data Protection Officer, Noor Patel.",
    )
    assert not _grade(
        case,
        "Kepler processes biometric templates, so Noor Patel approves it.",
    )
    assert _grade(
        case,
        "Kepler processes **biometric templates** and serves **EEA residents**; "
        "the **Data Protection Officer**, **Noor Patel**, must approve.",
    )


def test_semantic_suite_includes_cross_document_markdown_tables():
    case = next(
        item
        for item in discover_semantic_cases(count=9, seed=4)
        if "matching table rows" in item.prompt
    )

    assert all("|" in content for _path, content in case.files)
    assert all(
        not all(term in content.casefold() for term in case.expected)
        for _path, content in case.files
    )
    assert _grade(
        case,
        "Order Console maps to Helios, which uses Ledger Aurora; "
        "the dataset's current steward is Imani Brooks.",
    )
    assert not _grade(case, "Pavel Novak owns the dataset.")


def test_semantic_suite_includes_unnamed_document_discovery():
    case = next(
        item
        for item in discover_semantic_cases(count=9, seed=4)
        if "without assuming filenames" in item.prompt
    )

    assert all(path not in case.prompt for path, _content in case.files)
    assert len(case.files) == 4
    assert _grade(
        case,
        "Atlas Portal maps to Meridian, which uses Dataset Ember; "
        "Leila Hassan is its current steward.",
    )
    assert not _grade(case, "Marco Silva stewards Dataset Frost.")


def test_reasoning_cases_require_synthesis_hypotheses_and_falsification():
    case = next(
        item
        for item in discover_reasoning_cases(
            count=12,
            seed=4,
            mode="novel_synthesis",
        )
        if "activation" in item.prompt
    )
    complete = (
        "The leading hypothesis is the legacy token broker because migration bursts "
        "reach 900 while its limit is 500. An alternative explanation is cohort selection, "
        "but the boundary supports the broker interaction. Test and falsify this by "
        "routing a weekend cohort through the new broker."
    )
    assert _grade(case, complete)
    assert not _grade(
        case,
        "The legacy token broker limit is 500 while bursts reach 900.",
    )
    blinded = _blinded_case(case)
    assert blinded["expected"] == "<blinded during grading>"
    assert blinded["required_any"] == "<blinded during grading>"
    assert blinded["derived_relations"] == "<blinded during grading>"
    assert all(item["content"] == "<blinded during grading>" for item in blinded["files"])


def test_synthesis_grader_requires_a_derived_cross_source_relation():
    case = next(
        item
        for item in discover_reasoning_cases(
            count=12,
            seed=4,
            mode="novel_synthesis",
        )
        if "Northstar" in " ".join(content for _path, content in item.files)
    )
    derived = (
        "The leading hypothesis is a compact-nonce cache collision. Northstar is "
        "Android v9, where truncation to the first 8 characters makes household "
        "members share a cache key; parallel refresh therefore returns the wrong "
        "member token and causes the authentication failure. An identity-provider "
        "outage is an alternative but cannot explain serial success. Test this by "
        "keying the cache with the full nonce; continued failures would falsify it."
    )
    disconnected_fact_dump = (
        "Hypothesis: compact nonce. Northstar is Android v9. The first 8 characters "
        "are retained. There is a household prefix. There is a cache collision. "
        "Authentication failures occurred. An identity-provider outage is an "
        "alternative. Test with a full nonce to falsify the hypothesis."
    )

    assert _grade(case, derived)
    assert not _grade(case, disconnected_fact_dump)


def test_novel_synthesis_suite_contains_twelve_distinct_problem_surfaces():
    cases = discover_reasoning_cases(
        count=12,
        seed=4,
        mode="novel_synthesis",
    )

    assert len(cases) == 12
    assert len({case.case_id for case in cases}) == 12
    prompts_and_files = " ".join(
        text
        for case in cases
        for text in (case.prompt, *(content for _path, content in case.files))
    ).casefold()
    for surface in (
        "northstar",
        "quartz",
        "heron",
        "orchid",
        "vega",
        "lumen",
        "sparrow",
        "zone c",
    ):
        assert surface in prompts_and_files


def test_hard_synthesis_grader_accepts_structured_causal_paraphrases():
    cases = discover_reasoning_cases(
        count=12,
        seed=4,
        mode="novel_synthesis",
    )
    vega = next(
        case
        for case in cases
        if "Vega dashboards" in " ".join(content for _path, content in case.files)
    )
    orchid = next(
        case
        for case in cases
        if "Orchid denotes" in " ".join(content for _path, content in case.files)
    )

    assert _grade(
        vega,
        "The best explanation is a namespace mismatch. Vega moved from legacy "
        "observability to the new observability-v2 namespace, but alert rules require "
        "an exact namespace match; therefore the rules no longer see the metrics and "
        "no alert fires although telemetry is present. A metrics outage is an "
        "alternative, but raw queries refute it. Test by updating one rule to v2; "
        "continued silence would falsify this explanation.",
    )
    assert _grade(
        orchid,
        "The causal explanation is that Orchid annual prepaid accounts have hundreds "
        "of micro-credits entering CAD conversion, where each allocation is rounded. "
        "Those rounding errors therefore accumulate into a multi-cent statement "
        "mismatch. An exchange-rate feed fault is an alternative. Test by summing "
        "before rounding; persistent drift would falsify the selected explanation.",
    )


def test_hard_synthesis_sources_do_not_contradict_the_intended_join():
    cases = discover_reasoning_cases(
        count=12,
        seed=4,
        mode="novel_synthesis",
    )
    lumen_text = " ".join(
        content
        for case in cases
        for path, content in case.files
        if path in {"access_model.md", "migration_design.md"}
    )
    sparrow_text = " ".join(
        content
        for case in cases
        for path, content in case.files
        if path in {"release_notes.md", "index_baseline.md", "index_pipeline.md"}
    )

    assert "its Lumen deny from the Restricted-Research parent policy" in lumen_text
    assert "omits inherited denies" in lumen_text
    assert "canonical record names did not change" in sparrow_text
    assert "before the Sparrow aliases were added" in sparrow_text
    assert "Alias-only changes are skipped" in sparrow_text


def test_reasoning_grader_accepts_lexical_variants_and_rejected_competitor():
    case = next(
        item
        for item in discover_reasoning_cases(
            count=12,
            seed=4,
            mode="novel_synthesis",
        )
        if "queue stalls" in item.prompt
    )
    answer = (
        "The leading explanation is that top-of-hour reconciliation sends every "
        "account job for a tenant id to one saturated partition. A competing global "
        "queue outage is weaker and contradicted because other partitions remain free. "
        "This explains why only large tenants stall. Test the explanation by sharding "
        "one large tenant across partition keys; persistence would falsify it."
    )

    assert _grade(case, answer)
    assert not _grade(
        case,
        "The most likely root cause is a global queue outage. Hourly reconciliation, "
        "tenant id, and a hot partition are unrelated. Test the outage hypothesis.",
    )


def test_ambiguity_grader_accepts_conflict_and_underspecification_language():
    case = next(
        item
        for item in discover_reasoning_cases(
            count=12,
            seed=4,
            mode="ambiguity",
        )
        if "improve conversion" in item.prompt
    )

    assert _grade(
        case,
        "The request is not actionable because the definitions conflict: one "
        "interpretation is visitor-to-account signup and the other is "
        "checkout-to-paid-order. Minimum clarification: which funnel is intended?",
    )
    assert _grade(
        case,
        "The request is not actionable. Commerce means checkout \u2192 paid order, "
        "while Growth means visitor \u2192 account signup. Which funnel is intended?",
    )


def test_ambiguity_grader_accepts_clear_two_option_language_without_magic_word():
    case = next(
        item
        for item in discover_reasoning_cases(
            count=12,
            seed=4,
            mode="ambiguity",
        )
        if "stable release" in item.prompt
    )

    assert _grade(
        case,
        "The request does not specify a product channel. The two live "
        "interpretations are Desktop 4.2 and Server 5.0. Which channel should "
        "be deployed?",
    )


def test_ambiguity_suite_contains_twelve_distinct_problem_surfaces():
    cases = discover_reasoning_cases(
        count=12,
        seed=4,
        mode="ambiguity",
    )

    assert len(cases) == 12
    assert len({case.case_id for case in cases}) == 12
    prompts = " ".join(case.prompt for case in cases).casefold()
    for surface in ("ttl", "margin", "17:00", "account", "reliability", "west", "batch"):
        assert surface in prompts


def test_ambiguity_cases_reward_clarification_and_penalize_guessing():
    case = next(
        item
        for item in discover_reasoning_cases(
            count=12,
            seed=4,
            mode="ambiguity",
        )
        if "deploy Atlas" in item.prompt
    )
    correct = (
        "The request is ambiguous and cannot determine whether Atlas Portal or "
        "Atlas Pipeline is intended. Either interpretation is viable. Clarify which "
        "component and environment before acting."
    )
    assert _grade(case, correct)
    assert _grade(
        case,
        "The request is ambiguous across two interpretations:\n"
        "1. Deploy Atlas Portal — the customer-facing application.\n"
        "2. Deploy Atlas Pipeline — the internal service.\n"
        "Either could be intended. Clarify which component and environment.",
    )
    assert not _grade(
        case,
        "Deploy Atlas Portal because it is customer-facing; Atlas Pipeline is internal.",
    )
    assert not _grade(
        case,
        "The request is ambiguous, but the recommended action is therefore deploy "
        "Atlas Portal. Atlas Pipeline is the other interpretation. Clarify later.",
    )

"""Hostile web, abduction, patch, taxonomy, and secrecy tests."""

from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest
from north_star_oracle_support import cases, encoded

from cortheon.parity import public_case_projection
from cortheon.parity_benchmark_core.casepack import _case_has_frozen_oracle, _normalize_cases
from cortheon.parity_benchmark_core.cases_builtin import _builtin_cases
from cortheon.parity_benchmark_core.grading import grade_answer
from cortheon.parity_benchmark_core.oracle_taxonomy import ORACLE_SPECS, TASK_CLASSES
from cortheon.parity_benchmark_core.oracle_web import _truth_digest


def test_taxonomy_is_exact_and_every_class_has_one_versioned_oracle() -> None:
    assert {
        "ambiguity_resolution",
        "constraint_bound_planning",
        "cross_file_numeric_join",
        "current_web_research",
        "evidence_bound_debugging",
        "long_horizon_execution",
        "novel_abductive_synthesis",
        "repository_patching",
        "semantic_cross_document_reasoning",
    } == TASK_CLASSES
    assert len({spec.grader_type for spec in ORACLE_SPECS.values()}) == 9
    assert {spec.oracle_version for spec in ORACLE_SPECS.values()} == {1}


def test_public_projection_contains_no_private_oracle_labels_or_hints() -> None:
    private = [case for case, _answer in cases().values()]
    projection = public_case_projection(private)

    assert all(
        not ({"task_class", "expected_verdict", "grader", "oracle", "oracle_version"} & set(case))
        for case in projection
    )
    assert "origin_equivalence" not in repr(projection)
    assert "mutation" not in repr(projection).casefold()


def test_abduction_requires_competing_hypotheses_discriminator_and_every_source() -> None:
    case, answer = cases()["novel_abductive_synthesis"]
    _normalize_cases({"cases": [case]}, built_in=False, allow_external_patch_tests=True)
    assert grade_answer(case, encoded(answer, "The prose may be paraphrased."))["passed"] is True

    one_hypothesis = deepcopy(answer)
    one_hypothesis["hypotheses"].pop()
    wrong_discriminator = deepcopy(answer)
    wrong_discriminator["discriminator"]["supports"] = "network_loss"
    leave_one_out = deepcopy(answer)
    leave_one_out["premises"].pop(1)

    assert grade_answer(case, encoded(one_hypothesis))["passed"] is False
    assert (
        "wrong_discriminating_observation"
        in grade_answer(case, encoded(wrong_discriminator))["failures"]
    )
    assert "leave_one_source_out_failure" in grade_answer(case, encoded(leave_one_out))["failures"]


def test_abductive_issuance_rejects_a_conclusion_copied_from_a_source() -> None:
    case, _answer = cases()["novel_abductive_synthesis"]
    case["documents"][0]["text"] += " east_signing_key_mismatch"
    case["grader"]["oracle"]["source_bindings"][0]["sha256"] = hashlib.sha256(
        case["documents"][0]["text"].encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="must be novel"):
        _normalize_cases({"cases": [case]}, built_in=False, allow_external_patch_tests=True)


def test_current_web_accepts_valid_attested_graph_and_rejects_near_misses() -> None:
    case, answer = cases()["current_web_research"]
    assert grade_answer(case, encoded(answer))["passed"] is True

    syndicated = deepcopy(answer)
    syndicated["sources"][1]["canonical_url"] = "https://mirror.example/story"
    stale_value = deepcopy(answer)
    stale_value["claims"][0]["value"] = "1.9"
    wrong_primary = deepcopy(answer)
    wrong_primary["contradictions"][0]["resolved_by_url"] = "https://analysis.example/report"

    assert grade_answer(case, encoded(syndicated))["passed"] is False
    assert "wrong_current_claims" in grade_answer(case, encoded(stale_value))["failures"]
    assert (
        "wrong_contradiction_resolution" in grade_answer(case, encoded(wrong_primary))["failures"]
    )


def test_current_web_invalidates_changed_bytes_stale_truth_and_fake_acquisition() -> None:
    case, answer = cases()["current_web_research"]
    changed = deepcopy(case)
    changed["grader"]["oracle"]["revalidated_truth_digest"] = "f" * 64
    stale = deepcopy(case)
    stale["grader"]["oracle"]["valid_until"] = "2026-08-22T00:00:00+00:00"
    fake = deepcopy(case)
    fake["grader"]["oracle"]["acquisition_attestation"]["records"][0]["revalidated_sha256"] = (
        "e" * 64
    )

    assert "truth_changed_since_seal" in grade_answer(changed, encoded(answer))["failures"]
    assert "stale_or_unvalidated_truth" in grade_answer(stale, encoded(answer))["failures"]
    assert (
        "acquisition_does_not_match_current_source_bytes"
        in grade_answer(fake, encoded(answer))["failures"]
    )


def test_current_web_truth_digest_cannot_be_recomputed_to_hide_changed_content() -> None:
    case, _answer = cases()["current_web_research"]
    oracle = case["grader"]["oracle"]
    oracle["sources"][0]["content_sha256"] = "9" * 64
    oracle["truth_digest"] = oracle["revalidated_truth_digest"] = _truth_digest(oracle)

    with pytest.raises(ValueError, match="acquisition_does_not_match"):
        _normalize_cases({"cases": [case]}, built_in=False, allow_external_patch_tests=True)


def test_patch_oracle_binds_pristine_fixture_tests_and_command() -> None:
    case = next(item for item in _builtin_cases() if item["id"] == "repository_patch_verified")
    normalized = _normalize_cases(
        {"cases": [case]}, built_in=True, allow_external_patch_tests=True
    )[0]
    assert normalized["task_class"] == "repository_patching"

    for field, value in (
        ("pristine_sha256", "0" * 64),
        ("tests_sha256", "1" * 64),
        ("test_command", ["python", "-m", "unittest", "fake"]),
    ):
        tampered = deepcopy(case)
        tampered["grader"]["oracle"][field] = value
        with pytest.raises(ValueError, match="digest or command mismatch"):
            _normalize_cases({"cases": [tampered]}, built_in=True, allow_external_patch_tests=True)

    allowed_test_edit = deepcopy(case)
    allowed_test_edit["grader"]["allowed_files"].append("test_calculator.py")
    with pytest.raises(ValueError, match="invalid test_files"):
        _normalize_cases(
            {"cases": [allowed_test_edit]}, built_in=True, allow_external_patch_tests=True
        )


def test_patch_grader_rejects_a_direct_test_tampering_diff_before_execution() -> None:
    case = next(item for item in _builtin_cases() if item["id"] == "repository_patch_verified")
    patch = """```diff
--- a/test_calculator.py
+++ b/test_calculator.py
@@ -4,3 +4,3 @@
 class CalculatorTests(unittest.TestCase):
     def test_division(self):
-        self.assertEqual(divide(12, 4), 3)
+        self.assertTrue(True)
```"""

    result = grade_answer(case, patch)
    assert result["passed"] is False
    assert "patch_apply:changed_files_outside_allowlist" in result["failures"]


def test_current_web_prompt_must_expose_exact_as_of_and_closed_response_shape() -> None:
    case, _answer = cases()["current_web_research"]
    case["prompt"] = "Research the current release and answer briefly."

    with pytest.raises(ValueError, match="prompt omits"):
        _normalize_cases({"cases": [case]}, built_in=False, allow_external_patch_tests=True)


def test_diagnostic_graders_cannot_claim_a_task_class_or_frozen_proof() -> None:
    case = {
        "id": "diagnostic_case",
        "task_class": "ambiguity_resolution",
        "prompt": "Say ok",
        "expected_verdict": "allow",
        "grader": {"type": "patterns", "required_patterns": ["ok"]},
    }
    with pytest.raises(ValueError, match="diagnostic grader"):
        _normalize_cases({"cases": [case]}, built_in=False, allow_external_patch_tests=True)
    case.pop("task_class")
    case["grader"]["oracle_provenance"] = "frozen_external_pack"
    assert _case_has_frozen_oracle(case) is False

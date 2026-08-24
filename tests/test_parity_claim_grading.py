from __future__ import annotations

from unittest import mock

import pytest

from cortheon.benchmark import Contender, ModelResult
from cortheon.benchmark_core.outcomes import EvaluationOutcome
from cortheon.parity_benchmark_core.casepack import _case_has_frozen_oracle, _normalize_cases
from cortheon.parity_benchmark_core.cases_builtin import _builtin_cases
from cortheon.parity_benchmark_core.grading import grade_answer
from cortheon.parity_benchmark_core.runner import run_benchmark


def _private_join() -> dict:
    return next(case for case in _builtin_cases() if case["id"] == "private_document_join")


@pytest.mark.parametrize(
    "answer",
    [
        (
            "According to the Launch plan, the authorization code is ORBIT-17. "
            "The Station roster names Maya Chen as the security reviewer."
        ),
        (
            "Launch plan identifies ORBIT-17 as the launch authorization code.\n"
            "Security reviewer: Maya Chen (Station roster)."
        ),
        (
            "Authorization code = ORBIT-17 [benchmark://launch-plan].\n"
            "Maya Chen is the reviewer [benchmark://station-roster]."
        ),
    ],
)
def test_private_join_accepts_relation_bound_paraphrases(answer: str) -> None:
    grade = grade_answer(_private_join(), answer)

    assert grade == {
        "passed": True,
        "method": "document_relations",
        "failures": [],
        "proof_eligible": False,
        "assurance": "diagnostic_source_relation",
    }


@pytest.mark.parametrize(
    ("answer", "failure"),
    [
        ("ORBIT-17 Maya Chen Launch plan Station roster", "missing_relation"),
        (
            "Launch plan says Maya Chen is the authorization code. "
            "Station roster says ORBIT-17 is the reviewer.",
            "missing_relation",
        ),
        (
            "Station roster says the authorization code is ORBIT-17. "
            "Launch plan says the reviewer is Maya Chen.",
            "missing_source_binding",
        ),
        (
            "The Launch plan says the authorization code is not ORBIT-17. "
            "The Station roster says Maya Chen is not the reviewer.",
            "wrong_polarity",
        ),
        (
            "What is the launch authorization code and who is the named reviewer? "
            "Explain which supplied document supports each fact.",
            "missing_relation",
        ),
    ],
)
def test_private_join_rejects_adversarial_near_misses(answer: str, failure: str) -> None:
    grade = grade_answer(_private_join(), answer)

    assert grade["passed"] is False
    assert any(item.startswith(failure) for item in grade["failures"])


def test_removing_relation_or_source_binding_kills_proof() -> None:
    valid = (
        "Launch plan: the authorization code is ORBIT-17. "
        "Station roster: the reviewer is Maya Chen."
    )
    without_relations = valid.replace(" is ", " ")
    without_sources = valid.replace("Launch plan: ", "").replace("Station roster: ", "")

    assert grade_answer(_private_join(), valid)["passed"] is True
    assert grade_answer(_private_join(), without_relations)["passed"] is False
    assert grade_answer(_private_join(), without_sources)["passed"] is False


def test_pypi_oracle_binds_fields_and_live_source() -> None:
    case = {
        "grader": {
            "type": "pypi_metadata",
            "package": "uv",
            "answer_key": {"version": "9.8.7", "requires_python": ">=3.11"},
        }
    }

    valid = grade_answer(
        case,
        "uv==9.8.7\nRequires-Python: >=3.11\nSource: PyPI",
    )
    salad = grade_answer(case, "uv 9.8.7 Requires-Python >=3.11 PyPI")

    assert valid["passed"] is True and valid["proof_eligible"] is False
    assert salad["passed"] is False


def test_regex_graders_are_diagnostic_and_cannot_verify_completion() -> None:
    case = {
        "id": "diagnostic",
        "category": "custom",
        "domain": "custom",
        "difficulty": "medium",
        "prompt": "Return ok",
        "expected_verdict": "allow",
        "grader": {"type": "patterns", "required_patterns": ["ok"]},
        "documents": [],
    }
    contender = Contender("candidate", "stock", "http://unused", "model", "")
    with mock.patch(
        "cortheon.benchmark.call_contender",
        return_value=ModelResult(
            answer="ok",
            latency_ms=1.0,
            metadata={},
            evaluator_outcome=EvaluationOutcome(
                "openai_chat",
                "success",
                "chat_finish_reason",
                "stop",
            ),
        ),
    ):
        report = run_benchmark(
            [contender],
            [case],
            repetitions=1,
            seed=1,
            timeout=1,
            max_tokens=8,
            include_answers=False,
        )
    row = report["rows"][0]

    assert row["passed"] is True
    assert row["verified_completion"] is False
    assert report["summary"]["candidate_1"]["verified_completion_rate"] == 0.0


def test_ordered_regex_grader_is_also_diagnostic() -> None:
    grade = grade_answer(
        {
            "grader": {
                "type": "ordered_patterns",
                "required_patterns": ["first", "second"],
            }
        },
        "first, then second",
    )

    assert grade["passed"] is True
    assert grade["proof_eligible"] is False
    assert grade["assurance"] == "diagnostic_regex"


def test_current_version_text_matching_cannot_be_promoted() -> None:
    case = {
        "grader": {
            "type": "current_versions",
            "answer_key": {"fastapi": "1.2.3", "httpx": "4.5.6"},
        }
    }
    negated = grade_answer(
        case,
        "fastapi==1.2.3 is not current; httpx==4.5.6 is not current",
    )
    copied_prompt = grade_answer(case, "State the exact current stable PyPI versions.")
    keyword_salad = grade_answer(case, "fastapi 1.2.3 httpx 4.5.6 PyPI")

    assert negated["passed"] is True
    assert negated["proof_eligible"] is False
    assert negated["assurance"] == "diagnostic_text_match"
    assert copied_prompt["passed"] is False
    assert keyword_salad["passed"] is False


def test_only_structured_frozen_oracles_are_independent() -> None:
    pattern = {
        "grader": {
            "type": "patterns",
            "required_patterns": ["answer"],
            "oracle_provenance": "frozen_external_pack",
        }
    }
    structured = _private_join()
    structured["grader"] = dict(structured["grader"])
    structured["grader"]["oracle_provenance"] = "frozen_external_pack"

    assert _case_has_frozen_oracle(pattern) is False
    assert _case_has_frozen_oracle(structured) is False


def test_document_relation_schema_rejects_unbound_claims() -> None:
    case = _private_join()
    case["grader"] = dict(case["grader"])
    case["grader"]["claims"] = [dict(case["grader"]["claims"][0])]
    case["grader"]["claims"][0].pop("source_aliases")

    with pytest.raises(ValueError, match="source_aliases"):
        _normalize_cases(
            {"cases": [case]},
            built_in=False,
            allow_external_patch_tests=False,
        )


def test_document_relation_oracle_must_match_its_named_fixture() -> None:
    case = _private_join()
    case["grader"] = dict(case["grader"])
    case["grader"]["claims"] = [dict(claim) for claim in case["grader"]["claims"]]
    case["grader"]["claims"][0]["source_aliases"] = ["Imaginary source"]

    with pytest.raises(ValueError, match="not bound to its source document"):
        _normalize_cases(
            {"cases": [case]},
            built_in=False,
            allow_external_patch_tests=False,
        )

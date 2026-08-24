from __future__ import annotations

from collections import Counter

from cortheon.operator_lift.case_bank import development_cases
from cortheon.operator_lift.models import OPERATORS


def test_bank_has_twelve_independent_clusters_per_operator() -> None:
    cases = development_cases()
    assert len(cases) == 60
    assert Counter(case.operator for case in cases) == dict.fromkeys(OPERATORS, 12)
    assert len({case.case_id for case in cases}) == 60
    assert len({case.cluster_id for case in cases}) == 60
    assert len({case.causal_family for case in cases}) == 60


def test_every_case_has_multiple_source_bound_evidence_records() -> None:
    for case in development_cases():
        source_ids = [source_id for source_id, _content in case.evidence]
        assert 2 <= len(source_ids) <= 6
        assert len(source_ids) == len(set(source_ids))
        assert all(content.strip() for _source_id, content in case.evidence)


def test_operator_protocols_are_not_lexical_graders() -> None:
    by_operator = {case.operator: case for case in development_cases()}
    assert set(by_operator) == set(OPERATORS)
    assert by_operator["hypothesis_framing"].oracle.keys() == {
        "leading",
        "rivals",
        "falsification",
    }
    assert by_operator["discriminating_evidence"].oracle.keys() == {
        "hypotheses",
        "expected",
    }
    assert by_operator["contradiction_revision"].oracle.keys() == {"expected"}
    assert by_operator["cross_source_derivation"].oracle.keys() == {
        "conclusion",
        "premises",
    }
    assert by_operator["adaptive_stopping"].oracle.keys() == {
        "expected_actions",
        "decision",
        "observations",
    }


def test_hypothesis_cases_publish_field_bound_vocabulary() -> None:
    for case in development_cases():
        if case.operator != "hypothesis_framing":
            continue
        fields = case.response_schema["field_vocabulary"]
        leading = case.oracle["leading"]
        rival = case.oracle["rivals"][0]
        falsification = case.oracle["falsification"]
        assert set(fields["leading"]["cause"]) == {leading[0], rival[0]}
        assert fields["leading"]["outcome"] == [leading[1]]
        assert fields["rival"]["outcome"] == [rival[1]]
        assert fields["falsification"]["intervention"] == [falsification[0]]
        assert fields["falsification"]["result"] == [falsification[1]]
        assert set(fields["falsification"]["refutes"]) == {leading[0], rival[0]}

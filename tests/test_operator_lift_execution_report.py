from __future__ import annotations

from test_operator_lift_contrasts import _submissions

from cortheon.operator_lift.case_bank import development_cases
from cortheon.operator_lift.execution_report import content_free_report
from cortheon.operator_lift.execution_schedule import execution_manifest


def test_pilot_report_is_content_free_incomplete_and_nonclaiming() -> None:
    cases = development_cases()
    manifest = execution_manifest(cases)
    submissions = _submissions(cases, manifest)[:9]
    summaries = [
        {
            "identity_valid": True,
            "transcript_valid": True,
            "timed_out": False,
            "tokens": 3,
            "cost_usd": 0.0,
        }
        for _ in submissions
    ]
    report = content_free_report(
        manifest,
        cases,
        submissions,
        summaries,
        run_sha256="a" * 64,
        event_chain_sha256="b" * 64,
        planned_cells=9,
    )
    assert report["pilot"] is True
    assert report["pilot_claim_eligible"] is False
    assert report["development_gate_passes"] is False
    assert report["raw_content_included"] is False
    assert report["event_record_schema_version"] == 1
    assert report["event_chain_sha256"] == "b" * 64
    assert "submission_freeze_sha256" not in report
    assert report["accounting"]["complete"] is False
    assert report["accounting"]["pairing_error_summary"]["count"] == 531
    assert "pairing_errors" not in report["accounting"]
    assert all("integrity_errors" not in operator for operator in report["operators"].values())
    assert report["execution"]["delivered_cells"] == 9
    assert report["execution"]["safe_cells"] == 9
    assert report["execution"]["nonempty_response_cells"] == 3
    assert report["execution"]["terminal_status_counts"] == {"missing": 9}
    serialized = repr(report)
    assert '"response"' not in serialized
    assert '"prompt"' not in serialized
    assert '"evidence"' not in serialized

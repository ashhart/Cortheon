from __future__ import annotations

from dataclasses import replace

from cortheon.threat_model.models import ReviewerSignoff, ValidationResult
from cortheon.threat_model.promotion import evaluate_promotion
from cortheon.threat_model.report import report_sha256

REPORT = (
    b'{"catalog_valid":true,"collection_valid":true,"hostile_tests_executed":true,'
    b'"hostile_tests_passed":true,"manifest_binding_valid":true,"manifest_sha256":"'
    + b"b"
    * 64
    + b'","promotion_approved":false,"reviewer_signoffs":[],'
    b'"signature_verification":"external_not_performed"}'
)


def _validation(*, passed: bool = True) -> ValidationResult:
    return ValidationResult(
        valid=passed,
        catalog_valid=True,
        collection_valid=True,
        hostile_tests_executed=passed,
        hostile_tests_passed=passed,
        manifest_sha256="b" * 64,
        errors=(),
        collected_node_ids=("tests/test_x.py::test_x",),
        test_source_sha256=(("tests/test_x.py", "a" * 64),),
        pytest_command=("python", "-m", "pytest"),
    )


def _signoff(reviewer: str, key: str) -> ReviewerSignoff:
    return ReviewerSignoff(
        reviewer_id=reviewer,
        reviewer_role="independent security reviewer",
        reviewed_at_utc="2026-08-23T12:00:00Z",
        report_sha256=report_sha256(REPORT),
        key_id=key,
        external_signature=f"detached-test-fixture-for-{reviewer}",
        decision="approve",
    )


def test_repository_state_with_no_signoffs_cannot_promote() -> None:
    decision = evaluate_promotion(REPORT, _validation(), ())
    assert decision.eligible is False
    assert decision.accepted_reviewer_ids == ()
    assert "exactly_two_reviewer_signoffs_required" in decision.reasons
    assert "external_signature_verification_required" in decision.reasons


def test_two_shaped_signoffs_are_not_called_verified_without_external_verifier() -> None:
    signoffs = (_signoff("reviewer_alpha", "key-alpha"), _signoff("reviewer_beta", "key-beta"))
    decision = evaluate_promotion(REPORT, _validation(), signoffs)
    assert decision.eligible is False
    assert decision.cryptographic_verification == "external"
    assert decision.accepted_reviewer_ids == ()


def test_external_verification_still_requires_distinct_people_and_keys() -> None:
    first = _signoff("reviewer_alpha", "key-alpha")
    decision = evaluate_promotion(
        REPORT,
        _validation(),
        (first, replace(first)),
        external_signature_verifier=lambda _signoff, _report: True,
    )
    assert decision.eligible is False
    assert "reviewer_identities_not_independent" in decision.reasons
    assert "reviewer_keys_not_independent" in decision.reasons


def test_two_external_signatures_over_exact_bytes_can_promote_a_passing_run() -> None:
    signoffs = (_signoff("reviewer_alpha", "key-alpha"), _signoff("reviewer_beta", "key-beta"))
    decision = evaluate_promotion(
        REPORT,
        _validation(),
        signoffs,
        external_signature_verifier=lambda signoff, report: (
            signoff.report_sha256 == report_sha256(report)
            and signoff.external_signature.startswith("detached-test-fixture")
        ),
    )
    assert decision.eligible is True
    assert decision.cryptographic_verification == "verified"
    assert decision.accepted_reviewer_ids == ("reviewer_alpha", "reviewer_beta")


def test_stale_report_digest_or_failed_hostile_run_blocks_promotion() -> None:
    signoffs = (_signoff("reviewer_alpha", "key-alpha"), _signoff("reviewer_beta", "key-beta"))
    stale = replace(signoffs[0], report_sha256="f" * 64)

    def verifier(_signoff: ReviewerSignoff, _report: bytes) -> bool:
        return True

    assert not evaluate_promotion(
        REPORT, _validation(), (stale, signoffs[1]), external_signature_verifier=verifier
    ).eligible
    assert not evaluate_promotion(
        REPORT, _validation(passed=False), signoffs, external_signature_verifier=verifier
    ).eligible

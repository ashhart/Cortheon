"""External-review promotion boundary for exact threat-report bytes."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from cortheon.threat_model.models import PromotionDecision, ReviewerSignoff, ValidationResult
from cortheon.threat_model.report import report_sha256

SignatureVerifier = Callable[[ReviewerSignoff, bytes], bool]


def evaluate_promotion(
    report_bytes: bytes,
    validation: ValidationResult,
    signoffs: Sequence[ReviewerSignoff],
    *,
    external_signature_verifier: SignatureVerifier | None = None,
) -> PromotionDecision:
    """Require two independent, externally authenticated approvals."""

    digest = report_sha256(report_bytes)
    reasons: list[str] = []
    if (
        not validation.valid
        or not validation.hostile_tests_executed
        or not validation.hostile_tests_passed
    ):
        reasons.append("validated_hostile_test_run_required")
    try:
        payload = json.loads(report_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = None
    expected_report_fields = {
        "manifest_sha256": validation.manifest_sha256,
        "manifest_binding_valid": True,
        "catalog_valid": validation.catalog_valid,
        "collection_valid": validation.collection_valid,
        "hostile_tests_executed": validation.hostile_tests_executed,
        "hostile_tests_passed": validation.hostile_tests_passed,
        "promotion_approved": False,
        "reviewer_signoffs": [],
        "signature_verification": "external_not_performed",
    }
    if not isinstance(payload, dict) or any(
        payload.get(field) != value for field, value in expected_report_fields.items()
    ):
        reasons.append("report_validation_binding_mismatch")
    if len(signoffs) != 2:
        reasons.append("exactly_two_reviewer_signoffs_required")
    reviewer_ids: list[str] = []
    key_ids: list[str] = []
    for index, signoff in enumerate(signoffs):
        try:
            signoff.validate_shape()
        except ValueError as exc:
            reasons.append(f"signoff_{index}_invalid:{exc}")
            continue
        reviewer_ids.append(signoff.reviewer_id)
        key_ids.append(signoff.key_id)
        if signoff.report_sha256 != digest:
            reasons.append(f"signoff_{index}_report_digest_mismatch")
        if signoff.decision != "approve":
            reasons.append(f"signoff_{index}_not_approved")
    if len(reviewer_ids) != len(set(reviewer_ids)):
        reasons.append("reviewer_identities_not_independent")
    if len(key_ids) != len(set(key_ids)):
        reasons.append("reviewer_keys_not_independent")
    cryptographic_verification = "external"
    if external_signature_verifier is None:
        reasons.append("external_signature_verification_required")
    elif len(signoffs) == 2:
        verified = []
        for index, signoff in enumerate(signoffs):
            try:
                verified.append(bool(external_signature_verifier(signoff, report_bytes)))
            except Exception:
                verified.append(False)
            if not verified[-1]:
                reasons.append(f"signoff_{index}_signature_not_verified")
        if all(verified):
            cryptographic_verification = "verified"
    unique_reasons = tuple(sorted(set(reasons)))
    return PromotionDecision(
        eligible=not unique_reasons,
        report_sha256=digest,
        cryptographic_verification=cryptographic_verification,
        accepted_reviewer_ids=tuple(reviewer_ids) if not unique_reasons else (),
        reasons=unique_reasons,
    )

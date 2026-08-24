from __future__ import annotations

import json
from pathlib import Path

from cortheon.threat_model.catalog import THREAT_MANIFEST
from cortheon.threat_model.report import build_report_bytes, report_sha256
from cortheon.threat_model.validation import validate_threat_model

ROOT = Path(__file__).parents[1]


def test_report_is_deterministic_unsigned_and_explicitly_unpromoted() -> None:
    validation = validate_threat_model(ROOT, THREAT_MANIFEST)
    first = build_report_bytes(THREAT_MANIFEST, validation)
    second = build_report_bytes(THREAT_MANIFEST, validation)
    assert first == second
    payload = json.loads(first)
    assert payload["promotion_approved"] is False
    assert payload["reviewer_signoffs"] == []
    assert payload["signature_verification"] == "external_not_performed"
    assert payload["manifest_binding_valid"] is True
    assert payload["manifest_sha256"] == validation.manifest_sha256
    assert payload["review_status"].startswith("awaiting_two_independent")
    assert payload["claim_scope"] == "repository_threat_control_coverage_only"
    assert len(payload["lower_residuals"]) == 6
    assert len(report_sha256(first)) == 64


def test_collection_only_report_does_not_claim_hostile_tests_passed() -> None:
    validation = validate_threat_model(ROOT, THREAT_MANIFEST)
    payload = json.loads(build_report_bytes(THREAT_MANIFEST, validation))
    assert payload["collection_valid"] is True
    assert payload["hostile_tests_executed"] is False
    assert payload["hostile_tests_passed"] is False
    assert payload["promotion_approved"] is False

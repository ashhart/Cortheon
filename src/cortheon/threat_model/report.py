"""Canonical unsigned threat-model report bytes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from cortheon.threat_model.models import ThreatManifest, ValidationResult, manifest_sha256


def build_report_bytes(manifest: ThreatManifest, validation: ValidationResult) -> bytes:
    """Build exact review bytes; no local reviewer approval is implied."""

    current_manifest_sha256 = manifest_sha256(manifest)
    manifest_binding_valid = current_manifest_sha256 == validation.manifest_sha256
    payload = {
        "schema_version": manifest.schema_version,
        "model_version": manifest.model_version,
        "claim_scope": "repository_threat_control_coverage_only",
        "promotion_approved": False,
        "review_status": "awaiting_two_independent_externally_verified_signoffs",
        "reviewer_signoffs": [],
        "signature_verification": "external_not_performed",
        "manifest_sha256": current_manifest_sha256,
        "manifest_binding_valid": manifest_binding_valid,
        "catalog_valid": validation.catalog_valid and manifest_binding_valid,
        "collection_valid": validation.collection_valid,
        "hostile_tests_executed": validation.hostile_tests_executed,
        "hostile_tests_passed": validation.hostile_tests_passed,
        "validation_errors": list(validation.errors),
        "risks": [asdict(risk) for risk in manifest.risks],
        "lower_residuals": [asdict(residual) for residual in manifest.residuals],
        "collected_node_ids": list(validation.collected_node_ids),
        "test_source_sha256": dict(validation.test_source_sha256),
        "pytest_command": list(validation.pytest_command),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def report_sha256(report_bytes: bytes) -> str:
    return hashlib.sha256(report_bytes).hexdigest()

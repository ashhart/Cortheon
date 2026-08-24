"""Closed value types for the repository-only threat-model gate."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

SCHEMA_VERSION = 1
MODEL_VERSION = "2026.08.3"
SEVERITIES = ("critical", "high")
OWNERS = (
    "runtime_lifecycle",
    "host_integrations",
    "evidence_integrity",
    "execution_receipts",
    "benchmark_integrity",
    "sandbox_safety",
    "transport_safety",
    "measurement_integrity",
    "model_identity",
    "release_engineering",
    "privacy_boundary",
)
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{2,79}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _text(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} is invalid")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


@dataclass(frozen=True, slots=True)
class ThreatRisk:
    risk_id: str
    severity: Literal["critical", "high"]
    phase: str
    owner: str
    attack: str
    test_node_ids: tuple[str, ...]

    def validate(self) -> None:
        _identifier(self.risk_id, "risk_id")
        if self.severity not in SEVERITIES:
            raise ValueError("severity is invalid")
        _identifier(self.phase, "phase")
        if self.owner not in OWNERS:
            raise ValueError("owner is invalid or unowned")
        _text(self.attack, "attack")
        if not self.test_node_ids or len(set(self.test_node_ids)) != len(self.test_node_ids):
            raise ValueError("test node ids are empty or duplicated within a risk")
        for node_id in self.test_node_ids:
            if (
                not isinstance(node_id, str)
                or not node_id.startswith("tests/")
                or "::test_" not in node_id
            ):
                raise ValueError("test node id is invalid")


@dataclass(frozen=True, slots=True)
class ResidualRisk:
    residual_id: str
    severity: Literal["medium", "low"]
    statement: str
    disposition: str

    def validate(self) -> None:
        _identifier(self.residual_id, "residual_id")
        if self.severity not in {"medium", "low"}:
            raise ValueError("residual severity is invalid")
        _text(self.statement, "residual statement")
        _text(self.disposition, "residual disposition")


@dataclass(frozen=True, slots=True)
class ThreatManifest:
    schema_version: int
    model_version: str
    risks: tuple[ThreatRisk, ...]
    residuals: tuple[ResidualRisk, ...]


def manifest_sha256(manifest: ThreatManifest) -> str:
    payload = json.dumps(
        asdict(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    catalog_valid: bool
    collection_valid: bool
    hostile_tests_executed: bool
    hostile_tests_passed: bool
    manifest_sha256: str
    errors: tuple[str, ...]
    collected_node_ids: tuple[str, ...]
    test_source_sha256: tuple[tuple[str, str], ...]
    pytest_command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReviewerSignoff:
    reviewer_id: str
    reviewer_role: str
    reviewed_at_utc: str
    report_sha256: str
    key_id: str
    external_signature: str
    decision: Literal["approve", "reject"]

    def validate_shape(self) -> None:
        _identifier(self.reviewer_id, "reviewer_id")
        _text(self.reviewer_role, "reviewer_role", 160)
        if (
            not isinstance(self.reviewed_at_utc, str)
            or _UTC.fullmatch(self.reviewed_at_utc) is None
        ):
            raise ValueError("reviewed_at_utc is invalid")
        _digest(self.report_sha256, "report_sha256")
        _text(self.key_id, "key_id", 256)
        _text(self.external_signature, "external_signature", 8_000)
        if self.decision not in {"approve", "reject"}:
            raise ValueError("review decision is invalid")


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    eligible: bool
    report_sha256: str
    cryptographic_verification: Literal["external", "verified"]
    accepted_reviewer_ids: tuple[str, ...]
    reasons: tuple[str, ...]

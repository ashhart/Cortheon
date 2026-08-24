"""Versioned repository threat gate; it makes no claim of deployment security."""

from cortheon.threat_model.catalog import THREAT_MANIFEST
from cortheon.threat_model.models import ReviewerSignoff
from cortheon.threat_model.promotion import evaluate_promotion
from cortheon.threat_model.report import build_report_bytes, report_sha256
from cortheon.threat_model.validation import validate_threat_model

__all__ = [
    "THREAT_MANIFEST",
    "ReviewerSignoff",
    "build_report_bytes",
    "evaluate_promotion",
    "report_sha256",
    "validate_threat_model",
]

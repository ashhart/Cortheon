"""Canonical deterministic projection of a graded parity report.

``grade_blind_submission`` output is fully deterministic given the same
sealed pack, attested submission, and inner contract, except for the
top-level ``generated_at`` timestamp. The evaluation receipt is the entire
report with exactly that field removed; comparing receipt SHA-256 digests
therefore detects any other difference between a stored report and a
recomputed one, fail-closed, without hand-picking fields.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

RECEIPT_EXCLUDED_FIELDS = ("generated_at",)


def evaluation_receipt(report: dict[str, Any]) -> dict[str, Any]:
    """Return the report with only the nondeterministic fields removed."""

    return {key: value for key, value in report.items() if key not in RECEIPT_EXCLUDED_FIELDS}


def evaluation_receipt_sha256(report: dict[str, Any]) -> str:
    """Digest the canonical receipt of a graded report."""

    canonical = json.dumps(
        evaluation_receipt(report),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()

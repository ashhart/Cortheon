"""Content-free run report built only after submission freeze."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any, cast

from cortheon.operator_lift.execution_schedule import canonical_bytes
from cortheon.operator_lift.models import LiftCase, LiftManifest, LiftSubmission
from cortheon.operator_lift.report import build_lift_report

_PLACEBO = "equal_budget_placebo"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _arm(condition_id: object) -> str:
    if condition_id == "full":
        return "full"
    if condition_id == _PLACEBO:
        return _PLACEBO
    return "ablation"


def _realized_compute(
    submissions: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> dict[str, dict[str, int | float | None]]:
    rows: dict[str, list[dict[str, Any]]] = {
        "full": [],
        "ablation": [],
        _PLACEBO: [],
    }
    for index, submission in enumerate(submissions):
        summary = summaries[index] if index < len(summaries) else {}
        rows[_arm(submission.get("condition_id"))].append(summary)

    def total(
        items: list[dict[str, Any]],
        key: str,
        fallback: str | None = None,
    ) -> int | float | None:
        values: list[int | float] = []
        for item in items:
            value = item.get(key)
            if value is None and fallback is not None:
                value = item.get(fallback)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            values.append(value)
        return sum(values) if values else None

    return {
        arm: {
            "cells": len(items),
            "inference_calls": total(items, "inference_calls", "steps"),
            "tokens": total(items, "tokens"),
            "tool_calls": total(items, "tool_calls"),
            "latency_seconds": total(items, "latency_seconds"),
        }
        for arm, items in rows.items()
    }


def _compute_balance(
    realized: dict[str, dict[str, int | float | None]],
    *,
    summary_count_matches: bool,
) -> dict[str, Any]:
    full = realized["full"]
    placebo = realized[_PLACEBO]
    measured = ("inference_calls", "tokens", "tool_calls", "latency_seconds")
    complete = (
        summary_count_matches
        and full["cells"] == placebo["cells"]
        and all(full[key] is not None and placebo[key] is not None for key in measured)
    )
    equal = complete and all(full[key] == placebo[key] for key in measured)
    return {
        "comparison": "full_vs_equal_budget_placebo",
        "configured_budget_equal": True,
        "realized_compute_equal": equal,
        "realized_compute_required_for_claim": False,
        "metrics_complete": complete,
        "claim_valid": bool(complete),
        "interpretation": "equal_budget_with_realized_compute_reported",
    }


def _error_summary(errors: list[str]) -> dict[str, Any]:
    kinds = Counter(error.rsplit(":", 1)[-1] for error in errors)
    return {
        "count": len(errors),
        "sha256": hashlib.sha256(canonical_bytes(errors)).hexdigest(),
        "kind_counts": dict(sorted(kinds.items())),
    }


def content_free_report(
    manifest: LiftManifest,
    cases: tuple[LiftCase, ...],
    submissions: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    *,
    run_sha256: str,
    event_chain_sha256: str,
    planned_cells: int,
) -> dict[str, Any]:
    if _SHA256.fullmatch(run_sha256) is None or _SHA256.fullmatch(event_chain_sha256) is None:
        raise ValueError("report execution digest is invalid")
    if type(planned_cells) is not int or planned_cells < 1:
        raise ValueError("planned_cells is invalid")
    typed_submissions = cast(list[LiftSubmission | Mapping[str, Any]], submissions)
    lift = build_lift_report(manifest, cases, typed_submissions)
    pairing_errors = list(lift["accounting"].pop("pairing_errors"))
    lift["accounting"]["pairing_error_summary"] = _error_summary(pairing_errors)
    for operator in lift["operators"].values():
        integrity_errors = list(operator.pop("integrity_errors"))
        operator["integrity_error_summary"] = _error_summary(integrity_errors)
    identity_failures = sum(item.get("identity_valid") is not True for item in summaries)
    transcript_failures = sum(item.get("transcript_valid") is not True for item in summaries)
    terminal_statuses = Counter(str(item.get("terminal_status", "missing")) for item in summaries)
    budget_reasons = Counter(
        str(item["budget_reason"]) for item in summaries if item.get("budget_reason")
    )
    realized_compute = _realized_compute(submissions, summaries)
    payload: dict[str, Any] = {
        **lift,
        "claim_scope": "development_operator_lift_execution_only",
        "run_sha256": run_sha256,
        "event_record_schema_version": 1,
        "event_chain_sha256": event_chain_sha256,
        "event_replay_scope": "structural_accounting_and_evaluator_recorded_grades",
        "event_authenticity_scope": "requires_external_chain_root",
        "planned_cells": planned_cells,
        "completed_cells": len(submissions),
        "pilot": planned_cells < len(cases) * 9,
        "pilot_claim_eligible": False,
        "raw_content_included": False,
        "execution": {
            "identity_failures": identity_failures,
            "transcript_failures": transcript_failures,
            "delivered_cells": sum(item.get("delivered") is True for item in submissions),
            "safe_cells": sum(item.get("safe") is True for item in submissions),
            "nonempty_response_cells": sum(bool(item.get("response")) for item in submissions),
            "terminal_status_counts": dict(sorted(terminal_statuses.items())),
            "budget_reason_counts": dict(sorted(budget_reasons.items())),
            "timeouts": sum(item.get("timed_out") is True for item in summaries),
            "total_tokens": sum(
                int(item["tokens"]) for item in summaries if type(item.get("tokens")) is int
            ),
            "total_cost_usd": sum(
                float(item["cost_usd"])
                for item in summaries
                if isinstance(item.get("cost_usd"), (int, float))
                and not isinstance(item.get("cost_usd"), bool)
            ),
            "realized_compute_by_arm": realized_compute,
            "compute_balance": _compute_balance(
                realized_compute,
                summary_count_matches=len(summaries) == len(submissions),
            ),
        },
    }
    if payload["pilot"]:
        payload["development_gate_passes"] = False
    payload.pop("report_sha256", None)
    payload["report_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload

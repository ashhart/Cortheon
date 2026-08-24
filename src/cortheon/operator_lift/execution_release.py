"""Content-free release records for operator-lift executions."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from cortheon.operator_lift.contrasts import score_and_pair
from cortheon.operator_lift.execution_models import ScheduledCell
from cortheon.operator_lift.execution_schedule import canonical_bytes
from cortheon.operator_lift.models import (
    OPERATORS,
    LiftCase,
    LiftManifest,
    LiftSubmission,
)

RELEASE_SCHEMA_VERSION = 1
_ZERO_DIGEST = "0" * 64
_DIGEST = re.compile(r"[0-9a-f]{64}")
_TERMINAL_STATUSES = frozenset(
    {"success", "withheld", "incomplete", "tool_only", "transport_error", "missing"}
)
_BUDGET_REASONS = frozenset({"max_steps", "max_tool_calls", "output_bytes"})
_RELEASE_FIELDS = frozenset(
    {
        "schema_version",
        "content_free",
        "run_sha256",
        "manifest_sha256",
        "report_sha256",
        "records",
        "chain_root_sha256",
        "claim_eligible",
        "trust_anchor",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "type",
        "sequence",
        "case_ordinal",
        "case_commitment",
        "condition_id",
        "repeat",
        "delivered",
        "safe",
        "correct",
        "output_present",
        "identity_valid",
        "transcript_valid",
        "measurements",
        "previous_record_sha256",
        "record_sha256",
    }
)
_MEASUREMENT_FIELDS = frozenset(
    {
        "measurements_valid",
        "model_steps",
        "tokens",
        "tool_calls",
        "latency_seconds",
        "cost_usd",
        "timed_out",
        "budget_reason",
        "terminal_status",
    }
)


def _closed(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} is invalid")
    return value


def _integer(value: Any, label: str, *, nullable: bool = False) -> int | None:
    if nullable and value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} is invalid")
    return value


def _number(value: Any, label: str, *, nullable: bool = False) -> float | None:
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is invalid")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{label} is invalid")
    return parsed


def _condition_arm(condition_id: str) -> str:
    if condition_id == "full":
        return "full"
    if condition_id == "equal_budget_placebo":
        return "equal_budget_placebo"
    if condition_id == "ablation" or re.fullmatch(r"ablation_[0-4]", condition_id):
        return "ablation"
    raise ValueError("release condition is invalid")


def _public_condition(operator: str, condition_id: str) -> str:
    if condition_id == "full":
        return "full"
    if condition_id == "equal_budget_placebo":
        return "equal_budget_placebo"
    if condition_id == f"without_{operator}":
        return f"ablation_{OPERATORS.index(operator)}"
    raise ValueError("source condition is invalid")


def _record_sha256(unsigned: Mapping[str, Any], run_sha256: str, manifest_sha256: str) -> str:
    bound = {
        "run_sha256": run_sha256,
        "manifest_sha256": manifest_sha256,
        "record": unsigned,
    }
    return hashlib.sha256(canonical_bytes(bound)).hexdigest()


def _measurement(summary: Mapping[str, Any]) -> dict[str, Any]:
    steps = summary.get("model_steps", summary.get("steps", summary.get("inference_calls")))
    terminal = summary.get("terminal_status", "missing")
    budget = summary.get("budget_reason")
    item = {
        "measurements_valid": summary.get("measurements_valid", True),
        "model_steps": steps,
        "tokens": summary.get("tokens"),
        "tool_calls": summary.get("tool_calls"),
        "latency_seconds": summary.get("latency_seconds"),
        "cost_usd": summary.get("cost_usd"),
        "timed_out": summary.get("timed_out", False),
        "budget_reason": budget,
        "terminal_status": terminal,
    }
    _validate_measurement(item)
    return item


def _validate_measurement(value: Any) -> Mapping[str, Any]:
    item = _closed(value, _MEASUREMENT_FIELDS, "release measurement")
    _boolean(item["measurements_valid"], "measurements_valid")
    _integer(item["model_steps"], "model_steps")
    _integer(item["tokens"], "tokens", nullable=True)
    _integer(item["tool_calls"], "tool_calls")
    _number(item["latency_seconds"], "latency_seconds")
    _number(item["cost_usd"], "cost_usd", nullable=True)
    _boolean(item["timed_out"], "timed_out")
    budget = item["budget_reason"]
    if budget is not None and budget not in _BUDGET_REASONS:
        raise ValueError("budget_reason is invalid")
    if item["terminal_status"] not in _TERMINAL_STATUSES:
        raise ValueError("terminal_status is invalid")
    return item


def _submission_key(value: LiftSubmission | Mapping[str, Any]) -> tuple[str, str, int]:
    submission = value if isinstance(value, LiftSubmission) else LiftSubmission.from_mapping(value)
    submission.validate()
    return submission.case_id, submission.condition_id, submission.repeat


def _project_records(
    manifest: LiftManifest,
    cases: tuple[LiftCase, ...],
    schedule: tuple[ScheduledCell, ...],
    submissions: Sequence[LiftSubmission | Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    run_sha256: str,
) -> list[dict[str, Any]]:
    if len(submissions) != len(summaries) or len(schedule) != len(submissions):
        raise ValueError("release cell accounting is incomplete")
    case_by_id = {case.case_id: case for case in cases}
    case_ordinal = {case_id: index for index, case_id in enumerate(manifest.case_order)}
    submission_by_key: dict[tuple[str, str, int], LiftSubmission | Mapping[str, Any]] = {}
    summary_by_key: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for submission, summary in zip(submissions, summaries, strict=True):
        key = _submission_key(submission)
        if key in submission_by_key or not isinstance(summary, Mapping):
            raise ValueError("release submissions are duplicated or malformed")
        submission_by_key[key] = submission
        summary_by_key[key] = summary
    pairing = score_and_pair(manifest, cases, submissions)
    scored = {(run.case_id, run.condition_id, run.repeat): run for run in pairing.scored_runs}
    previous = _ZERO_DIGEST
    records: list[dict[str, Any]] = []
    schedule_sequences = [cell.sequence for cell in schedule]
    if schedule_sequences != sorted(set(schedule_sequences)):
        raise ValueError("release schedule ordering is invalid")
    for expected_sequence, cell in enumerate(schedule, 1):
        if cell.case_id not in case_by_id:
            raise ValueError("release schedule is invalid")
        key = (cell.case_id, cell.condition_id, cell.repeat)
        raw = submission_by_key.get(key)
        run = scored.get(key)
        summary = summary_by_key.get(key)
        if raw is None or run is None or summary is None:
            raise ValueError("scheduled cell has no valid evaluator grade")
        submission = raw if isinstance(raw, LiftSubmission) else LiftSubmission.from_mapping(raw)
        if (
            submission.case_commitment != manifest.case_commitments[cell.case_id]
            or submission.condition_id != cell.condition_id
            or summary.get("condition_id", cell.condition_id) != cell.condition_id
        ):
            raise ValueError("scheduled cell identity is invalid")
        unsigned = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "type": "cell_evaluated",
            "sequence": expected_sequence,
            "case_ordinal": case_ordinal[cell.case_id],
            "case_commitment": submission.case_commitment,
            "condition_id": _public_condition(cell.operator, cell.condition_id),
            "repeat": cell.repeat,
            "delivered": run.delivered,
            "safe": run.safe,
            "correct": run.correct,
            "output_present": bool(submission.response),
            "identity_valid": summary.get("identity_valid"),
            "transcript_valid": summary.get("transcript_valid"),
            "measurements": _measurement(summary),
            "previous_record_sha256": previous,
        }
        _boolean(unsigned["identity_valid"], "identity_valid")
        _boolean(unsigned["transcript_valid"], "transcript_valid")
        record = {
            **unsigned,
            "record_sha256": _record_sha256(
                unsigned,
                run_sha256,
                manifest.manifest_sha256,
            ),
        }
        records.append(record)
        previous = record["record_sha256"]
    return records


def build_release(
    manifest: LiftManifest,
    cases: tuple[LiftCase, ...],
    schedule: tuple[ScheduledCell, ...],
    submissions: Sequence[LiftSubmission | Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    run_sha256: str,
) -> dict[str, Any]:
    """Project private evaluator state into a deterministic public record chain."""

    run_digest = _digest(run_sha256, "run_sha256")
    report_digest = _report_digest(report)
    if (
        report.get("run_sha256") != run_digest
        or report.get("manifest_sha256") != manifest.manifest_sha256
    ):
        raise ValueError("release report identity is invalid")
    records = release_records(
        manifest,
        cases,
        schedule,
        submissions,
        summaries,
        run_digest,
    )
    chain_root = records[-1]["record_sha256"] if records else _ZERO_DIGEST
    if report.get("event_chain_sha256") != chain_root:
        raise ValueError("release chain does not match report")
    expected_cells = len(cases) * 3 * manifest.thresholds.repetitions
    release = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "content_free": True,
        "run_sha256": run_digest,
        "manifest_sha256": manifest.manifest_sha256,
        "report_sha256": report_digest,
        "records": records,
        "chain_root_sha256": chain_root,
        "claim_eligible": len(schedule) == expected_cells,
        "trust_anchor": "external_chain_root",
    }
    verify_release(release, report)
    return release


def release_records(
    manifest: LiftManifest,
    cases: tuple[LiftCase, ...],
    schedule: tuple[ScheduledCell, ...],
    submissions: Sequence[LiftSubmission | Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    run_sha256: str,
) -> list[dict[str, Any]]:
    """Return the public chain before its report binding is finalized."""

    run_digest = _digest(run_sha256, "run_sha256")
    return _project_records(manifest, cases, schedule, submissions, summaries, run_digest)


def _report_digest(report: Mapping[str, Any]) -> str:
    stored = _digest(report.get("report_sha256"), "report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if hashlib.sha256(canonical_bytes(unsigned)).hexdigest() != stored:
        raise ValueError("report digest is invalid")
    return stored


def verify_release(
    release: Mapping[str, Any],
    report: Mapping[str, Any],
    descriptor: Mapping[str, Any] | None = None,
    expected_chain_root: str | None = None,
) -> dict[str, Any]:
    import cortheon.operator_lift.execution_release_verify as verifier

    return verifier.verify_release(release, report, descriptor, expected_chain_root)

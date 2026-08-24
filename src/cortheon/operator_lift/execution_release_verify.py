"""Independent replay checks for content-free operator-lift releases."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from cortheon.operator_lift.execution_release import (
    _RECORD_FIELDS,
    _RELEASE_FIELDS,
    _ZERO_DIGEST,
    RELEASE_SCHEMA_VERSION,
    _boolean,
    _closed,
    _condition_arm,
    _digest,
    _integer,
    _record_sha256,
    _report_digest,
    _validate_measurement,
)
from cortheon.operator_lift.execution_schedule import canonical_bytes
from cortheon.operator_lift.models import OPERATORS, LiftThresholds, PairedCluster
from cortheon.operator_lift.statistics import summarize_operator, summarize_placebo


def _validate_record(
    value: Any,
    sequence: int,
    previous: str,
    run_sha256: str,
    manifest_sha256: str,
) -> Mapping[str, Any]:
    item = _closed(value, _RECORD_FIELDS, "release record")
    if item["schema_version"] != RELEASE_SCHEMA_VERSION or item["type"] != "cell_evaluated":
        raise ValueError("release record header is invalid")
    if item["sequence"] != sequence or item["previous_record_sha256"] != previous:
        raise ValueError("release record ordering is invalid")
    _integer(item["case_ordinal"], "case_ordinal")
    _digest(item["case_commitment"], "case_commitment")
    condition = item["condition_id"]
    if not isinstance(condition, str):
        raise ValueError("condition_id is invalid")
    _condition_arm(condition)
    _integer(item["repeat"], "repeat")
    for field in (
        "delivered",
        "safe",
        "correct",
        "output_present",
        "identity_valid",
        "transcript_valid",
    ):
        _boolean(item[field], field)
    _validate_measurement(item["measurements"])
    _digest(item["previous_record_sha256"], "previous_record_sha256")
    stored = _digest(item["record_sha256"], "record_sha256")
    unsigned = {key: child for key, child in item.items() if key != "record_sha256"}
    if _record_sha256(unsigned, run_sha256, manifest_sha256) != stored:
        raise ValueError("release record digest is invalid")
    return item


def _validate_case_cells(
    records: Sequence[Mapping[str, Any]],
    claim_eligible: bool,
) -> dict[int, list[Mapping[str, Any]]]:
    case_cells: Counter[tuple[str, str, int]] = Counter()
    by_case: dict[int, list[Mapping[str, Any]]] = {}
    commitment_ordinals: dict[str, int] = {}
    for record in records:
        ordinal = int(record["case_ordinal"])
        commitment = str(record["case_commitment"])
        prior = commitment_ordinals.setdefault(commitment, ordinal)
        if prior != ordinal:
            raise ValueError("case commitment is bound to multiple ordinals")
        by_case.setdefault(ordinal, []).append(record)
        case_cells[(commitment, str(record["condition_id"]), int(record["repeat"]))] += 1
    if any(count != 1 for count in case_cells.values()):
        raise ValueError("release contains a duplicate cell identity")
    if len(commitment_ordinals) != len(by_case):
        raise ValueError("case ordinals are not unique")
    for rows in by_case.values():
        commitments = {row["case_commitment"] for row in rows}
        ablations = {
            str(row["condition_id"])
            for row in rows
            if _condition_arm(str(row["condition_id"])) == "ablation"
        }
        if len(rows) != 9 or len(commitments) != 1 or len(ablations) != 1:
            raise ValueError("release case cell set is incomplete")
        expected = {
            (condition, repeat)
            for condition in ("full", "equal_budget_placebo", next(iter(ablations)))
            for repeat in range(3)
        }
        observed = {(str(row["condition_id"]), int(row["repeat"])) for row in rows}
        if observed != expected:
            raise ValueError("release case arms or repeats are invalid")
    if claim_eligible and set(by_case) != set(range(len(by_case))):
        raise ValueError("claim-eligible release has incomplete case ordinals")
    return by_case


def _clusters(
    by_case: Mapping[int, list[Mapping[str, Any]]],
) -> tuple[list[PairedCluster], dict[str, list[Mapping[str, Any]]]]:
    clusters: list[PairedCluster] = []
    operator_rows: dict[str, list[Mapping[str, Any]]] = {operator: [] for operator in OPERATORS}
    for ordinal, rows in sorted(by_case.items()):
        ablation_id = next(
            str(row["condition_id"])
            for row in rows
            if _condition_arm(str(row["condition_id"])) == "ablation"
        )
        operator = OPERATORS[int(ablation_id.rsplit("_", 1)[1])]
        operator_rows[operator].extend(rows)

        def scores(
            condition: str,
            case_rows: tuple[Mapping[str, Any], ...] = tuple(rows),
        ) -> tuple[int, ...]:
            selected = sorted(
                (row for row in case_rows if row["condition_id"] == condition),
                key=lambda row: int(row["repeat"]),
            )
            return tuple(int(row["correct"] is True) for row in selected)

        clusters.append(
            PairedCluster(
                cluster_id=f"case_{ordinal}",
                operator=operator,
                full_scores=scores("full"),
                ablation_scores=scores(ablation_id),
                placebo_scores=scores("equal_budget_placebo"),
            )
        )
    return clusters, operator_rows


def _replay(records: Sequence[Mapping[str, Any]], claim_eligible: bool) -> dict[str, Any]:
    arm_rows: dict[str, list[Mapping[str, Any]]] = {
        "full": [],
        "ablation": [],
        "equal_budget_placebo": [],
    }
    totals: dict[str, int | float] = {
        "model_steps": 0,
        "tokens": 0,
        "tool_calls": 0,
        "latency_seconds": 0.0,
    }
    for record in records:
        arm_rows[_condition_arm(str(record["condition_id"]))].append(record)
        measurement = record["measurements"]
        assert isinstance(measurement, Mapping)
        totals["model_steps"] += int(measurement["model_steps"])
        if measurement["tokens"] is not None:
            totals["tokens"] += int(measurement["tokens"])
        totals["tool_calls"] += int(measurement["tool_calls"])
        totals["latency_seconds"] += float(measurement["latency_seconds"])
    by_case = _validate_case_cells(records, claim_eligible)
    clusters, operator_rows = _clusters(by_case)
    thresholds = LiftThresholds()
    operators = {
        operator: {
            **summarize_operator(clusters, operator, thresholds),
            "delivery_failures": sum(not row["delivered"] for row in operator_rows[operator]),
            "unsafe_outcomes": sum(not row["safe"] for row in operator_rows[operator]),
        }
        for operator in OPERATORS
    }
    return {
        "record_count": len(records),
        "chain_root_sha256": records[-1]["record_sha256"] if records else _ZERO_DIGEST,
        "claim_eligible": claim_eligible,
        "by_arm": {
            arm: {
                "cells": len(rows),
                "correct": sum(row["correct"] is True for row in rows),
                "delivered": sum(row["delivered"] is True for row in rows),
                "safe": sum(row["safe"] is True for row in rows),
            }
            for arm, rows in arm_rows.items()
        },
        "measurements": totals,
        "operators": operators,
        "placebo_control": summarize_placebo(clusters, thresholds),
    }


def _verify_accounting(
    replay: Mapping[str, Any], report: Mapping[str, Any], release: Mapping[str, Any]
) -> None:
    for field in ("run_sha256", "manifest_sha256"):
        if report.get(field) != release[field]:
            raise ValueError(f"release {field} does not match report")
    if report.get("event_chain_sha256") != release["chain_root_sha256"]:
        raise ValueError("release chain root does not match report")
    if report.get("completed_cells") != replay["record_count"]:
        raise ValueError("release cell count does not match report")
    if report.get("planned_cells") != replay["record_count"]:
        raise ValueError("release schedule count does not match report")
    accounting = report.get("accounting")
    if not isinstance(accounting, Mapping):
        raise ValueError("report accounting projection is invalid")
    expected_rows = accounting.get("expected_rows")
    if type(expected_rows) is not int or expected_rows < replay["record_count"]:
        raise ValueError("report expected cell count is invalid")
    if release["claim_eligible"] is not (replay["record_count"] == expected_rows):
        raise ValueError("release claim eligibility is invalid")
    if accounting.get("scored_rows") != replay["record_count"]:
        raise ValueError("release scored count does not match report")
    pairing = accounting.get("pairing_error_summary")
    if not isinstance(pairing, Mapping) or pairing.get("count") != (
        expected_rows - replay["record_count"]
    ):
        raise ValueError("release pairing count does not match report")


def _verify_results(
    replay: Mapping[str, Any], report: Mapping[str, Any], release: Mapping[str, Any]
) -> None:
    execution = report.get("execution")
    records = release["records"]
    if not isinstance(execution, Mapping) or not isinstance(records, list):
        raise ValueError("report execution projection is invalid")
    expected = {
        "identity_failures": sum(row["identity_valid"] is not True for row in records),
        "transcript_failures": sum(row["transcript_valid"] is not True for row in records),
        "delivered_cells": sum(row["delivered"] is True for row in records),
        "safe_cells": sum(row["safe"] is True for row in records),
        "nonempty_response_cells": sum(row["output_present"] is True for row in records),
        "timeouts": sum(row["measurements"]["timed_out"] is True for row in records),
        "total_tokens": replay["measurements"]["tokens"],
        "total_cost_usd": sum(float(row["measurements"]["cost_usd"] or 0.0) for row in records),
    }
    if any(execution.get(field) != value for field, value in expected.items()):
        raise ValueError("release execution totals do not match report")
    report_operators = report.get("operators")
    if not isinstance(report_operators, Mapping):
        raise ValueError("report operator projection is invalid")
    fields = {
        "operator",
        "independent_clusters",
        "repetitions_per_arm",
        "full_rate",
        "ablation_rate",
        "paired_lift",
        "one_sided_confidence",
        "clustered_lower_bound",
        "negative_effect_clusters",
        "zero_effect_clusters",
        "cluster_effects",
        "delivery_failures",
        "unsafe_outcomes",
    }
    for operator, expected_operator in replay["operators"].items():
        actual = report_operators.get(operator)
        if not isinstance(actual, Mapping) or any(
            actual.get(field) != expected_operator[field] for field in fields
        ):
            raise ValueError("release operator results do not match report")
    if report.get("placebo_control") != replay["placebo_control"]:
        raise ValueError("release placebo results do not match report")
    if report.get("pilot") is not (not release["claim_eligible"]):
        raise ValueError("release claim eligibility does not match report")


def _verify_descriptor(
    descriptor: Mapping[str, Any],
    release: Mapping[str, Any],
) -> None:
    fields = frozenset(
        {
            "schema_version",
            "manifest_sha256",
            "public_pack_sha256",
            "evaluator_identity",
            "condition_policies",
            "schedule",
            "repeats_are_independent_cases",
            "claim_eligible",
            "run_sha256",
        }
    )
    item = _closed(descriptor, fields, "run descriptor")
    if item["schema_version"] != 3 or item["repeats_are_independent_cases"] is not False:
        raise ValueError("run descriptor header is invalid")
    stored = _digest(item["run_sha256"], "descriptor run_sha256")
    unsigned = {key: value for key, value in item.items() if key != "run_sha256"}
    if hashlib.sha256(canonical_bytes(unsigned)).hexdigest() != stored:
        raise ValueError("run descriptor digest is invalid")
    if stored != release["run_sha256"] or item["manifest_sha256"] != release["manifest_sha256"]:
        raise ValueError("run descriptor identity does not match release")
    if item["claim_eligible"] is not release["claim_eligible"]:
        raise ValueError("run descriptor claim scope does not match release")
    schedule = item["schedule"]
    records = release["records"]
    if (
        not isinstance(schedule, list)
        or not isinstance(records, list)
        or len(schedule) != len(records)
    ):
        raise ValueError("run descriptor schedule length is invalid")
    schedule_fields = frozenset(
        {"sequence", "case_ordinal", "case_commitment", "condition_id", "repeat"}
    )
    for expected, record in zip(schedule, records, strict=True):
        projected = _closed(expected, schedule_fields, "run descriptor schedule cell")
        if any(projected[field] != record[field] for field in schedule_fields):
            raise ValueError("run descriptor schedule does not match release")


def verify_release(
    release: Mapping[str, Any],
    report: Mapping[str, Any],
    descriptor: Mapping[str, Any] | None = None,
    expected_chain_root: str | None = None,
) -> dict[str, Any]:
    """Revalidate chain integrity and evaluator-recorded aggregate facts."""

    item = _closed(release, _RELEASE_FIELDS, "release")
    if item["schema_version"] != RELEASE_SCHEMA_VERSION or item["content_free"] is not True:
        raise ValueError("release header is invalid")
    if item["trust_anchor"] != "external_chain_root":
        raise ValueError("release trust anchor is invalid")
    _digest(item["run_sha256"], "run_sha256")
    _digest(item["manifest_sha256"], "manifest_sha256")
    if item["report_sha256"] != _report_digest(report):
        raise ValueError("release report digest is invalid")
    records = item["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("release records are invalid")
    previous = _ZERO_DIGEST
    validated: list[Mapping[str, Any]] = []
    for sequence, raw in enumerate(records, 1):
        record = _validate_record(
            raw,
            sequence,
            previous,
            str(item["run_sha256"]),
            str(item["manifest_sha256"]),
        )
        validated.append(record)
        previous = str(record["record_sha256"])
    if item["chain_root_sha256"] != previous:
        raise ValueError("release chain root is invalid")
    trust_anchor_verified = expected_chain_root is not None
    if (
        expected_chain_root is not None
        and _digest(expected_chain_root, "expected_chain_root") != previous
    ):
        raise ValueError("release does not match the externally pinned chain root")
    if descriptor is not None:
        _verify_descriptor(descriptor, item)
    claim_eligible = _boolean(item["claim_eligible"], "claim_eligible")
    replay = _replay(validated, claim_eligible)
    _verify_accounting(replay, report, item)
    _verify_results(replay, report, item)
    return {
        **replay,
        "descriptor_verified": descriptor is not None,
        "trust_anchor_verified": trust_anchor_verified,
    }

"""Sealed public pack, real condition bindings, and paired cell schedule."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from cortheon.benchmark_core.generic_mcp_protocol import EQUAL_BUDGET_REVIEW_PROMPT
from cortheon.benchmark_core.generic_mcp_source import generic_source_sha256
from cortheon.benchmark_core.generic_mcp_tools import host_tool_definitions
from cortheon.operator_lift.case_bank import development_cases
from cortheon.operator_lift.execution_models import ExecutionConfig, ScheduledCell
from cortheon.operator_lift.models import OPERATORS, ConditionBinding, LiftCase, LiftManifest
from cortheon.operator_lift.sealing import build_manifest, public_case
from cortheon.qualification_core.conditions import (
    EQUAL_BUDGET_PLACEBO,
    condition_record,
    implementation_digest,
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def case_goal(case: LiftCase) -> str:
    action_note = (
        "For each selected action, read only actions/<action_id>.txt and record its action_id. "
        if case.operator == "adaptive_stopping"
        else ""
    )
    return (
        f"Task: {case.prompt} You must call host_read before answering. Read "
        "public-projection.json, reason over its evidence, and solve its prompt. "
        f"{action_note}The file contains the required response_schema. Return exactly one JSON "
        "object populated with the solved values, with no Markdown. Never return the schema itself."
    )


def condition_policies(
    manifest: LiftManifest,
    cases: tuple[LiftCase, ...],
    config: ExecutionConfig,
) -> dict[str, dict[str, object]]:
    tools = hashlib.sha256(canonical_bytes(host_tool_definitions(web_enabled=False))).hexdigest()
    bindings = (
        manifest.full_condition,
        manifest.placebo_condition,
        *manifest.ablation_conditions.values(),
    )
    policies: dict[str, dict[str, object]] = {}
    for binding in bindings:
        placebo = binding.condition_id == manifest.placebo_condition.condition_id
        condition_key = (
            f"ablation_{OPERATORS.index(binding.disabled_operator)}"
            if binding.disabled_operator is not None
            else binding.condition_id
        )
        policies[condition_key] = {
            "provider_id": config.provider_id,
            "model_id": config.model_id,
            "tool_catalogue_sha256": tools,
            "context_tokens": config.context_tokens,
            "output_tokens": config.output_tokens,
            "max_steps": config.max_steps,
            "max_tool_calls": config.max_tool_calls,
            "timeout_seconds": float(config.timeout_seconds),
            "runtime_used": not placebo,
            "private_labels_access": False,
            "additional_task_scaffold": not placebo,
            "neutral_protocol_sha256": (
                hashlib.sha256(EQUAL_BUDGET_REVIEW_PROMPT.encode()).hexdigest() if placebo else None
            ),
            "neutral_continuations": int(placebo),
        }
    return policies


def evaluator_implementation_sha256() -> str:
    root = Path(__file__).parent
    names = (
        "execution_models.py",
        "execution_schedule.py",
        "execution_storage.py",
        "execution_release.py",
        "execution_release_verify.py",
        "execution_runner.py",
        "execution_report.py",
        "cli.py",
    )
    digest = hashlib.sha256()
    for name in names:
        payload = (root / name).read_bytes()
        digest.update(name.encode())
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    digest.update(generic_source_sha256().encode())
    return digest.hexdigest()


def execution_manifest(cases: tuple[LiftCase, ...] | None = None) -> LiftManifest:
    cases = cases or development_cases()
    implementation = implementation_digest()

    def binding(condition_id: str, operator: str | None) -> ConditionBinding:
        record = condition_record(condition_id, implementation_sha256=implementation)
        if not record["available"] or not isinstance(record["implementation_sha256"], str):
            raise ValueError(f"condition {condition_id} is unavailable")
        return ConditionBinding(
            condition_id,
            str(record["config_sha256"]),
            str(record["implementation_sha256"]),
            operator,
        )

    return build_manifest(
        cases,
        full_condition=binding("full", None),
        placebo_condition=binding(EQUAL_BUDGET_PLACEBO, None),
        ablation_conditions={
            operator: binding(f"without_{operator}", operator) for operator in OPERATORS
        },
        evaluator_id="operator_lift_generic_mcp_evaluator",
        evaluator_implementation_sha256=evaluator_implementation_sha256(),
    )


def full_schedule(manifest: LiftManifest, cases: tuple[LiftCase, ...]) -> tuple[ScheduledCell, ...]:
    cells = [
        ScheduledCell(0, case.case_id, case.operator, condition_id, repeat)
        for case in cases
        for condition_id in (
            manifest.full_condition.condition_id,
            manifest.ablation_conditions[case.operator].condition_id,
            manifest.placebo_condition.condition_id,
        )
        for repeat in range(manifest.thresholds.repetitions)
    ]
    random.Random(int(manifest.manifest_sha256[:16], 16)).shuffle(cells)
    return tuple(
        ScheduledCell(index, cell.case_id, cell.operator, cell.condition_id, cell.repeat)
        for index, cell in enumerate(cells, 1)
    )


def selected_schedule(
    schedule: tuple[ScheduledCell, ...],
    cases: tuple[LiftCase, ...],
    pilot_clusters: int | None,
    operator: str | None = None,
) -> tuple[ScheduledCell, ...]:
    eligible = [case for case in cases if operator is None or case.operator == operator]
    if operator is not None and not eligible:
        raise ValueError("pilot operator is invalid")
    if pilot_clusters is None and operator is None:
        return schedule
    count = len(eligible) if pilot_clusters is None else pilot_clusters
    if type(count) is not int or not 1 <= count <= len(eligible):
        raise ValueError("pilot_clusters is invalid")
    chosen = (
        _balanced_revision_subset(eligible, count)
        if operator == "contradiction_revision"
        else eligible[:count]
    )
    selected = {case.case_id for case in chosen}
    return tuple(cell for cell in schedule if cell.case_id in selected)


def _balanced_revision_subset(cases: list[LiftCase], count: int) -> tuple[LiftCase, ...]:
    strata = {case.case_id: _revision_strata(case) for case in cases}
    domains = [set(values) for values in zip(*strata.values(), strict=True)]

    def score(selected: tuple[LiftCase, ...]) -> tuple[int, int, tuple[str, ...]]:
        signatures = [strata[case.case_id] for case in selected]
        margin = 0
        for dimension, domain in enumerate(domains):
            totals = Counter(signature[dimension] for signature in signatures)
            values = [totals[value] for value in domain]
            margin += max(values) - min(values)
        interaction = 0
        for left, right in combinations(range(len(domains)), 2):
            totals = Counter((item[left], item[right]) for item in signatures)
            values = [
                totals[(left_value, right_value)]
                for left_value in domains[left]
                for right_value in domains[right]
            ]
            interaction += max(values) - min(values)
        return margin, interaction, tuple(case.case_id for case in selected)

    return min(combinations(cases, count), key=score)


def _revision_strata(case: LiftCase) -> tuple[bool, int, str]:
    expected = tuple(case.oracle["expected"])
    decisive_position = next(
        index for index, evidence in enumerate(case.evidence) if evidence[0] == expected[3]
    )
    vocabulary = {
        "status": case.response_schema["effect_status_map"],
        "change": case.response_schema["effect_changes_hypothesis"],
    }
    return (
        expected[0] != expected[2],
        decisive_position,
        hashlib.sha256(canonical_bytes(vocabulary)).hexdigest(),
    )


def public_pack(cases: tuple[LiftCase, ...]) -> dict[str, Any]:
    projections = [public_case(case) for case in cases]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "claim_scope": "public_operator_lift_development_inputs",
        "cases": projections,
    }
    payload["pack_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def run_descriptor(
    manifest: LiftManifest,
    cases: tuple[LiftCase, ...],
    schedule: tuple[ScheduledCell, ...],
    config: ExecutionConfig,
    public_pack_sha256: str,
) -> dict[str, Any]:
    complete = full_schedule(manifest, cases)
    selected_ids = {cell.case_id for cell in schedule}
    if not selected_ids or schedule != tuple(
        cell for cell in complete if cell.case_id in selected_ids
    ):
        raise ValueError("schedule is not an exact complete-case projection")
    if config.max_steps < 2:
        raise ValueError("equal-budget placebo requires at least two model steps")
    case_ordinal = {case_id: index for index, case_id in enumerate(manifest.case_order)}
    schedule_projection = [
        {
            "sequence": index,
            "case_ordinal": case_ordinal[cell.case_id],
            "case_commitment": manifest.case_commitments[cell.case_id],
            "condition_id": (
                f"ablation_{OPERATORS.index(cell.operator)}"
                if cell.condition_id.startswith("without_")
                else cell.condition_id
            ),
            "repeat": cell.repeat,
        }
        for index, cell in enumerate(schedule, 1)
    ]
    payload = {
        "schema_version": 3,
        "manifest_sha256": manifest.manifest_sha256,
        "public_pack_sha256": public_pack_sha256,
        "evaluator_identity": config.public_identity(),
        "condition_policies": condition_policies(manifest, cases, config),
        "schedule": schedule_projection,
        "repeats_are_independent_cases": False,
        "claim_eligible": schedule == complete,
    }
    payload["run_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload

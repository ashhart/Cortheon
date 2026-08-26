"""Deterministic runtime-decision replay over the sealed operator-lift bank.

Development instrument: drives the real CognitiveRuntime on each sealed case
under the full and per-operator-ablation conditions with a scripted
responder, resolving cells in milliseconds. Not claim-eligible: no model, no
host, no held-out grading.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cortheon.cognitive_core.models import CognitiveRuntimeError
from cortheon.cognitive_core.runtime import CognitiveRuntime
from cortheon.operator_lift.execution_schedule import case_goal
from cortheon.operator_lift.models import OPERATORS, LiftCase
from cortheon.operator_lift.oracles import grade_case
from cortheon.operator_lift.replay_responder import (
    ReplayResponder,
    _materialize_workspace,
    answer_payload,
)

__all__ = [
    "ReplayCell",
    "ReplayResponder",
    "_materialize_workspace",
    "_profile",
    "answer_payload",
    "replay_bank",
    "replay_case",
    "replay_summary",
]


@dataclass(frozen=True)
class ReplayCell:
    case_id: str
    operator: str
    condition_id: str
    certified: bool
    correct: bool
    delivered: bool
    requests: tuple[str, ...]
    withheld_reasons: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def request_digest(self) -> str:
        return hashlib.sha256("\n".join(self.requests).encode()).hexdigest()[:16]


def _profile(disabled_operator: str | None) -> dict[str, Any]:
    from cortheon.cognitive_protocol import EVALUATION_PROFILE_VERSION

    operators: dict[str, bool] = dict.fromkeys(
        ("retrieval", "verification", *OPERATORS),
        True,
    )
    if disabled_operator is not None:
        operators[disabled_operator] = False
    config: dict[str, Any] = {
        "schema_version": EVALUATION_PROFILE_VERSION,
        "operators": operators,
        "intercepts_final": True,
        "cleanup_before_answer": False,
        "hard_budgets_enforced": True,
        "sticky_terminal_safety": True,
        "transport_failure_fails_open": True,
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": EVALUATION_PROFILE_VERSION,
        "config": config,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "implementation_sha256": "a" * 64,
        "nonce": "c" * 32,
    }


def replay_case(
    case: LiftCase,
    workspace_root: Path,
    *,
    disabled_operator: str | None = None,
    runtime: CognitiveRuntime | None = None,
    effort: str = "deep",
    strictness: str = "standard",
) -> ReplayCell:
    """Run one case through the runtime under one condition, deterministically."""
    errors: list[str] = []
    condition_id = (
        "full" if disabled_operator is None else f"ablation_{OPERATORS.index(disabled_operator)}"
    )
    rt = runtime or CognitiveRuntime(require_host_receipts=True)
    responder: ReplayResponder | None = None
    final: dict[str, Any] = {}
    certified = False
    correct = False
    withheld_reasons: tuple[str, ...] = ()
    try:
        profile = _profile(disabled_operator)
        operators = profile["config"]["operators"]
        payload = rt.start(
            case_goal(case),
            task_kind="general",
            effort=effort,
            strictness=strictness,
            evaluation_profile=profile,
        )
        responder = ReplayResponder(case, workspace_root, operators)
        assert responder is not None
        session_id = str(payload["session"]["session_id"])
        final = responder.drive(rt, session_id, payload)
        verdict = None
        verification = final.get("verification") if isinstance(final, dict) else None
        if isinstance(verification, dict):
            verdict = verification.get("verdict")
        if verdict is None:
            verdict = getattr(responder, "_verdict", None)
        certified = verdict == "complete"
        if isinstance(verification, dict) and isinstance(verification.get("gaps"), list):
            withheld_reasons = tuple(str(item) for item in verification["gaps"])
        if not withheld_reasons:
            withheld_reasons = tuple(getattr(responder, "_gaps", ()))
        parsed = json.loads(responder.answer) if certified else None
        if parsed is not None:
            correct = grade_case(case, parsed).correct
    except (ValueError, KeyError, TypeError, CognitiveRuntimeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    drive_errors = tuple(getattr(responder, "_errors", ())) if responder is not None else ()
    return ReplayCell(
        case_id=case.case_id,
        operator=case.operator,
        condition_id=condition_id,
        certified=certified,
        correct=correct,
        delivered=bool(certified and correct),
        requests=tuple(responder.requests) if responder is not None else (),
        withheld_reasons=withheld_reasons,
        errors=tuple(errors) + drive_errors,
    )


def replay_bank(
    cases: tuple[LiftCase, ...],
    workspace_roots: dict[str, Path],
    *,
    conditions: tuple[str | None, ...] | None = None,
) -> list[ReplayCell]:
    """Run the bank over full and per-operator ablation conditions."""
    conditions = conditions if conditions is not None else (None, *OPERATORS)
    cells: list[ReplayCell] = []
    for case in cases:
        root = workspace_roots.get(case.case_id, workspace_roots.get(case.operator, Path.cwd()))
        cells.extend(
            replay_case(case, root, disabled_operator=condition) for condition in conditions
        )
    return cells


def replay_summary(cells: list[ReplayCell]) -> dict[str, Any]:
    """Compact per-condition tallies for fast development diffs."""

    by_condition: dict[str, list[ReplayCell]] = {}
    for cell in cells:
        by_condition.setdefault(cell.condition_id, []).append(cell)
    summary: dict[str, Any] = {}
    for condition_id, group in by_condition.items():
        summary[condition_id] = {
            "cells": len(group),
            "certified": sum(cell.certified for cell in group),
            "delivered": sum(cell.delivered for cell in group),
            "with_errors": sum(bool(cell.errors) for cell in group),
            "request_digests": sorted({cell.request_digest for cell in group})[:8],
        }
    summary["_as_of"] = datetime.now(UTC).isoformat()
    return summary

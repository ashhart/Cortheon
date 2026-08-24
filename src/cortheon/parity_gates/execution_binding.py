"""Execution-binding gates: which model actually answered, in which order.

Identity checks prove what was registered; these prove what ran. Every
model-backed row has to carry the registered model id, the matrix has to be
complete with no duplicated cell, and the executed order has to reproduce the
pre-registered schedule hash exactly -- otherwise cells could be reordered,
retried, or dropped after the fact.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from cortheon.parity_gates.context import ContenderIdentities, ParityContext
from cortheon.parity_gates.errors import ParityContractError
from cortheon.parity_gates.projection import evaluation_schedule, evaluation_schedule_hash


def evaluate_execution_binding(
    context: ParityContext,
    identities: ContenderIdentities,
) -> None:
    _check_model_identity(context, identities)
    _check_schedule(context, identities)


def _check_model_identity(context: ParityContext, identities: ContenderIdentities) -> None:
    """Every model-backed row names the registered model for its contender."""

    model_execution_evidence: dict[str, Any] = {}
    model_execution_bound = True
    for name, alias in identities.contender_aliases().items():
        expected_model = context.contender_models.get(name, "")
        contender_rows = [
            row for row in context.rows if alias is not None and row.get("candidate") == alias
        ]
        model_backed_rows = [
            row for row in contender_rows if row.get("completion_origin") != "controller_only"
        ]
        mismatches = [
            {
                "run": row.get("run"),
                "observed": row.get("observed_model_id"),
            }
            for row in model_backed_rows
            if row.get("observed_model_id") != expected_model
        ]
        controller_mismatches = [
            {
                "run": row.get("run"),
                "observed": row.get("observed_model_id"),
            }
            for row in contender_rows
            if row.get("completion_origin") == "controller_only"
            and row.get("observed_model_id") not in {None, "", expected_model}
        ]
        valid = bool(model_backed_rows) and not mismatches and not controller_mismatches
        model_execution_bound = model_execution_bound and valid
        model_execution_evidence[name] = {
            "expected": expected_model,
            "model_backed_rows": len(model_backed_rows),
            "mismatches": mismatches,
            "controller_mismatches": controller_mismatches,
        }
    context.check(
        "model_identity_bound_per_execution",
        model_execution_bound,
        contenders=model_execution_evidence,
    )


def _check_schedule(context: ParityContext, identities: ContenderIdentities) -> None:
    """The executed matrix is complete and reproduces the pre-registered order."""

    rows = context.rows
    expected_cells = len(context.cases) * len(context.candidates) * context.repetitions
    unique_cells = {
        (
            row.get("case_id"),
            row.get("candidate"),
            row.get("repetition"),
        )
        for row in rows
    }
    context.check(
        "complete_evaluation_schedule",
        bool(rows) and len(rows) == expected_cells and len(unique_cells) == expected_cells,
        rows=len(rows),
        unique_cells=len(unique_cells),
        expected=expected_cells,
    )
    case_ids = [str(case.get("id") or "") for case in context.cases]
    contender_names = [context.candidate_name, *context.frontier_names]
    seed = int(context.methodology.get("seed") or 0)
    try:
        expected_schedule = evaluation_schedule(
            case_ids,
            contender_names,
            context.repetitions,
            seed,
        )
        expected_schedule_hash = evaluation_schedule_hash(
            case_ids,
            contender_names,
            context.repetitions,
            seed,
        )
    except ParityContractError:
        expected_schedule = []
        expected_schedule_hash = ""
    observed_schedule = [
        {
            "run": row.get("run"),
            "repetition": row.get("repetition"),
            "case_id": row.get("case_id"),
            "candidate": row.get("candidate"),
            "contender_name": identities.observed_aliases.get(str(row.get("candidate")), ""),
        }
        for row in rows
    ]
    observed_schedule_hash = (
        hashlib.sha256(
            json.dumps(
                observed_schedule,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if observed_schedule
        else ""
    )
    case_bank = context.case_bank
    context.check(
        "schedule_matches_contract",
        bool(
            expected_schedule_hash
            and observed_schedule == expected_schedule
            and case_bank.get("schedule_precommitted") is True
            and hmac.compare_digest(
                str(case_bank.get("schedule_sha256") or ""),
                expected_schedule_hash,
            )
            and hmac.compare_digest(
                str(case_bank.get("precommitted_schedule_sha256") or ""),
                expected_schedule_hash,
            )
            and hmac.compare_digest(
                observed_schedule_hash,
                expected_schedule_hash,
            )
        ),
        expected=case_bank.get("precommitted_schedule_sha256"),
        recomputed=expected_schedule_hash,
        executed=observed_schedule_hash,
    )

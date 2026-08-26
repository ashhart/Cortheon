"""Deterministic P6 same-model qualification preregistration.

The powered same-model qualification is preregistered before the held-out
campaign freezes. This module emits the sealed preregistration fields from
the same constants the release plan pins, so the artifact is machine-checked
rather than prose. Nothing here is claim-eligible; the qualification itself
must still run and pass.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from cortheon.operator_lift.models import OPERATORS, LiftThresholds
from cortheon.operator_lift.replay import replay_case, replay_summary

# Accepted aggregate planning floors from PLAN.md section 6 P6 (external power
# plan): independent cases across the nine P6 task classes, balanced arms.
P6_CASE_FLOOR = 28_773
P6_MODEL_FAMILY_LIMIT = 3
P6_HOST_SET = ("pi", "opencode", "codex", "generic_mcp", "omp")


def _digest_of(value: Any) -> str:
    encoded = __import__("json").dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def preregistration(
    *,
    model_family_limit: int = P6_MODEL_FAMILY_LIMIT,
    host_set: tuple[str, ...] = P6_HOST_SET,
) -> dict[str, Any]:
    """Assemble the closed preregistration record for the P6 qualification."""
    thresholds = LiftThresholds()
    record: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "P6 same-model capability qualification",
        "hypothesis": (
            "The same fixed local model completes materially harder unseen work "
            "with Cortheon than without it."
        ),
        "primary_outcome": "verified task completion per independent case",
        "unit_of_independence": "case, clustered by task lineage; repetitions measure stability",
        "operators": list(OPERATORS),
        "contrasts": {
            "full_vs_bare": "full Cortheon versus the identical bare model",
            "full_vs_strongest_reduced": (
                "full Cortheon versus the reduced condition with the largest "
                "realized loss among the per-operator ablations, selected after "
                "campaign accounting under the familywise-corrected rule"
            ),
            "full_vs_placebo": "full Cortheon versus the equal-budget placebo",
        },
        "alpha": {
            "familywise": thresholds.familywise_alpha,
            "per_contrast": thresholds.per_contrast_alpha,
            "clustered_ci": "one-sided case-clustered 95% lower bound",
        },
        "effect_sizes_of_interest": {
            "full_vs_bare_lower_bound_points": 0.05,
            "full_vs_strongest_reduced_lower_bound_points": 0.03,
        },
        "case_floor": {
            "total_independent_cases": P6_CASE_FLOOR,
            "balanced_arms": True,
            "source": "accepted external power plan, PLAN.md section 6 P6",
        },
        "model_population": {
            "families": list(range(1, model_family_limit + 1)),
            "min_families": 1,
            "max_families": model_family_limit,
            "same_fixed_model_per_arm": True,
        },
        "hosts": list(host_set),
        "blinding": {
            "task_authors_grade_without_tuning": True,
            "maintainers_do_not_see_private_labels_pre_freeze": True,
            "sealed_packs": True,
        },
        "analysis_plan": {
            "estimand": "case-clustered one-sided 95% lower bound on verified completion",
            "families": "must reproduce across local model families",
            "replication": "two independent evaluators for promotion",
        },
        "stopping_rule": (
            "Stop for success only after complete accounting, zero candidate-caused "
            "treatment delivery failures, and every preregistered gate passing on the "
            "frozen bytes; never stop on interim trends."
        ),
        "claim_gate": (
            "No claim-eligible lift without this preregistered record and the sealed "
            "campaign passing its gates."
        ),
    }
    record["record_sha256"] = _digest_of({k: v for k, v in record.items() if k != "record_sha256"})
    record["_as_of"] = datetime.now(UTC).isoformat()
    return record


def replay_gate_snapshot(case_bank: Any, workspace_roots: dict[str, Any]) -> dict[str, Any]:
    """Snapshot the developer replay gate for the preregistration preamble.

    Development only: certifies that the sealed case bank resolves through the
    runtime under every operator condition before any live campaign runs.
    """
    cells: list[Any] = []
    for case in case_bank:
        root = workspace_roots.get(case.case_id)
        if root is None:
            continue
        cells.extend(
            replay_case(case, root, disabled_operator=disabled) for disabled in (None, *OPERATORS)
        )
    return replay_summary(cells)

"""Repository-only release-scale policy for broad frontier-parity claims.

The single source of truth for the pre-registration scale a parity contract
must meet before its report is eligible for a broad parity claim, and for
whether a registered contract meets it. Kept here so ``cortheon.parity``
stays a thin contract evaluator rather than a policy plus evaluator.

Tests may lower the case/domain/repetition floors through a *reduced
test-scale policy*; this is not a tightened-economics guarantee. Lowering
these numbers shrinks the test, while every authentication, identity,
digest, matrix, and regrading requirement remains fully enforced.
"""

from __future__ import annotations

import math
from typing import Any

# Release-scale thresholds a contract must pre-register before its report is
# eligible for a broad parity claim, plus the direction of each requirement:
# keys in ``UNIVERSAL_SCALE_CEILINGS`` are maxima, ``max_errors`` and the
# boolean flag are equalities, and every other key is a minimum.
UNIVERSAL_SCALE_REQUIREMENTS: dict[str, int | float | bool] = {
    "required_domains": 8,
    "min_frontiers": 2,
    "min_cases": 320,
    "min_cases_per_domain": 40,
    "min_cases_per_task_class": 24,
    "min_repetitions": 5,
    "min_safety_cases": 50,
    "max_errors": 0,
    "min_completion_rate": 0.9,
    "min_domain_completion_rate": 0.8,
    "min_substrate_plus_model_fraction": 1.0,
    "noninferiority_margin": 0.03,
    "domain_noninferiority_margin": 0.05,
    "max_ci_half_width": 0.03,
    "max_false_allow_rate": 0.01,
    "max_false_block_rate": 0.05,
    "max_unstable_case_fraction": 0.05,
    "max_latency_ratio": 1.25,
    "max_cost_ratio": 1.0,
    "require_metered_cost": True,
}
UNIVERSAL_SCALE_CEILINGS = frozenset(
    {
        "noninferiority_margin",
        "domain_noninferiority_margin",
        "max_ci_half_width",
        "max_false_allow_rate",
        "max_false_block_rate",
        "max_unstable_case_fraction",
        "max_latency_ratio",
        "max_cost_ratio",
    }
)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def universal_scale_ok(
    contract: dict[str, Any],
    release_scale: dict[str, Any],
    *,
    requirements: dict[str, int | float | bool] | None = None,
    ceilings: frozenset[str] = UNIVERSAL_SCALE_CEILINGS,
) -> bool:
    """Check a registered scale against the release-scale policy.

    ``requirements`` and ``ceilings`` default to this module's policy; callers
    may substitute a reduced test-scale policy without touching enforcement
    of anything else.
    """

    policy = requirements if requirements is not None else UNIVERSAL_SCALE_REQUIREMENTS
    for key, requirement in policy.items():
        value = release_scale.get(key)
        if isinstance(requirement, bool) or key == "max_errors":
            if value != requirement:
                return False
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return False
        if key in ceilings:
            if not float(value) <= float(requirement):
                return False
        elif not float(value) >= float(requirement):
            return False
    return all(
        float(value) >= float(release_scale["min_domain_completion_rate"])
        for value in _mapping(contract.get("domain_floors")).values()
    )

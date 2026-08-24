"""Tier gate policy: manifests may tighten a tier, never loosen it."""

from __future__ import annotations

from typing import Any

from cortheon.qualification_core.constants import GATE_KEYS, TIER_DEFAULTS
from cortheon.qualification_core.models import QualificationError
from cortheon.qualification_core.validation import (
    _bounded_int,
    _bounded_number,
    _reject_unknown,
)


def _strict_gate_overrides(
    tier: str,
    raw: Any,
) -> dict[str, int | float]:
    defaults = {
        key: value for key, value in TIER_DEFAULTS[tier].items() if key != "default_repeats"
    }
    if raw is None:
        return defaults
    if not isinstance(raw, dict):
        raise QualificationError("manifest.gates must be an object")
    _reject_unknown(raw, GATE_KEYS, "manifest.gates")
    minimum_fields = {
        "min_independent_cases",
        "min_full_accuracy",
        "min_full_vs_bare_accuracy_delta",
        "min_full_vs_bare_accuracy_delta_ci_lower",
        "min_full_vs_reduced_accuracy_delta_ci_lower",
    }
    maximum_fields = {
        "max_false_allows",
        "max_false_block_rate",
        "max_invalid_pairs",
    }
    for key, value in raw.items():
        if key in {"min_independent_cases", "max_false_allows", "max_invalid_pairs"}:
            parsed: int | float = _bounded_int(
                value,
                field=f"manifest.gates.{key}",
                minimum=0,
                maximum=1_000_000,
            )
        elif key in {"max_false_block_rate", "min_full_accuracy"}:
            parsed = _bounded_number(
                value,
                field=f"manifest.gates.{key}",
                minimum=0.0,
                maximum=1.0,
            )
        else:
            parsed = _bounded_number(
                value,
                field=f"manifest.gates.{key}",
                minimum=-1.0,
                maximum=1.0,
            )
        if key in minimum_fields and parsed < defaults[key]:
            raise QualificationError(
                f"manifest.gates.{key} may only tighten the {tier} tier (minimum {defaults[key]})"
            )
        if key in maximum_fields and parsed > defaults[key]:
            raise QualificationError(
                f"manifest.gates.{key} may only tighten the {tier} tier (maximum {defaults[key]})"
            )
        defaults[key] = parsed
    return defaults

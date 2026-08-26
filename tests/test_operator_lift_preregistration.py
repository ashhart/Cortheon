"""P6 preregistration record: closed fields, deterministic digest."""

from __future__ import annotations

from cortheon.operator_lift.models import OPERATORS, LiftThresholds
from cortheon.operator_lift.preregister import P6_CASE_FLOOR, preregistration


def test_preregistration_fields_are_closed_and_deterministic() -> None:
    first = preregistration()
    second = preregistration()
    assert "_as_of" in first
    first.pop("_as_of", None)
    second.pop("_as_of", None)
    assert first == second
    assert first["schema_version"] == 1
    assert first["purpose"].startswith("P6")
    assert set(first["alpha"]) == {"familywise", "per_contrast", "clustered_ci"}
    assert first["alpha"]["familywise"] == LiftThresholds().familywise_alpha
    assert first["effect_sizes_of_interest"]["full_vs_bare_lower_bound_points"] == 0.05
    assert first["effect_sizes_of_interest"]["full_vs_strongest_reduced_lower_bound_points"] == 0.03
    # The record digests itself so a frozen copy is verifiable byte-for-byte.
    assert first["record_sha256"] != ""


def test_case_floor_matches_the_accepted_power_plan() -> None:
    assert P6_CASE_FLOOR == 28_773


def test_contrasts_cover_every_operator_ablation() -> None:
    record = preregistration()
    assert list(record["operators"]) == list(OPERATORS)
    assert "full_vs_placebo" in record["contrasts"]
    assert "full_vs_bare" in record["contrasts"]
    # The strongest reduced arm is chosen by a preregistered decision rule,
    # not by peeking at the campaign result.
    assert "selected after" in record["contrasts"]["full_vs_strongest_reduced"]

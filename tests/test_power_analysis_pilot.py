from __future__ import annotations

from dataclasses import replace

import pytest

from cortheon.power_analysis import PilotArtifact, PilotPair, ResourceAssumptions, build_power_plan
from cortheon.power_analysis.pilot import (
    discordance_upper_bound,
    pilot_schedule_sha256,
    pilot_sha256,
)

RESOURCES = ResourceAssumptions(120, 32, 0.03, 2, 0, 0, 0)
REDUCED = "without_cross_source_derivation"


def _pilot(contrast_id: str = "full_vs_bare") -> PilotArtifact:
    pairs = tuple(PilotPair(f"pilot_case_{index:04d}", index < 50, False) for index in range(500))
    pilot = PilotArtifact(
        schema_version=1,
        pilot_id="external_pilot_one",
        contrast_id=contrast_id,
        evaluator_id="outside_evaluator",
        producer="evaluator",
        candidate_supplied=False,
        source_population_id="external_population_one",
        source_population_sha256="c" * 64,
        source_sha256="a" * 64,
        case_bank_sha256="b" * 64,
        schedule_sha256="0" * 64,
        created_at_utc="2026-08-01T12:00:00Z",
        claim_pack_created_at_utc="2026-08-02T12:00:00Z",
        registered_before_claim_pack=True,
        independent_of_claim_pack=True,
        pairs=pairs,
    )
    return replace(pilot, schedule_sha256=pilot_schedule_sha256(pilot))


def test_pilot_uses_conservative_one_sided_discordance_bound() -> None:
    pilot = _pilot()
    upper = discordance_upper_bound(pilot)
    assert 0.10 < upper < 0.14
    assert len(pilot_sha256(pilot)) == 64


def test_valid_source_bound_pilot_can_reduce_but_not_inflate_the_floor() -> None:
    worst = build_power_plan(RESOURCES, strongest_reduced_condition_id=REDUCED)
    calibrated = build_power_plan(
        RESOURCES,
        strongest_reduced_condition_id=REDUCED,
        pilots={"full_vs_bare": _pilot()},
    )
    assert calibrated.contrasts[0].calibration == "unsigned_development_pilot_ucb"
    assert calibrated.contrasts[0].pilot_sha256 == pilot_sha256(_pilot())
    assert (
        calibrated.contrasts[0].alternative_discordance < worst.contrasts[0].alternative_discordance
    )
    assert calibrated.contrasts[0].required_cases < worst.contrasts[0].required_cases
    assert calibrated.promotion_eligible is False


@pytest.mark.parametrize(
    "mutation",
    [
        {"source_sha256": "bad"},
        {"registered_before_claim_pack": False},
        {"independent_of_claim_pack": False},
        {"contrast_id": "full_vs_strongest_reduced"},
        {"producer": "candidate"},
        {"candidate_supplied": True},
        {"created_at_utc": "2026-08-03T12:00:00Z"},
    ],
)
def test_unbound_or_nonindependent_pilot_fails_closed(mutation: dict) -> None:
    pilot = replace(_pilot(), **mutation)
    with pytest.raises(ValueError):
        build_power_plan(
            RESOURCES,
            strongest_reduced_condition_id=REDUCED,
            pilots={"full_vs_bare": pilot},
        )

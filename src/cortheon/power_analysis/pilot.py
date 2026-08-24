"""Independent pilot calibration with a conservative discordance bound."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from cortheon.power_analysis.models import PilotArtifact
from cortheon.power_analysis.statistics import regularized_beta


def pilot_schedule_sha256(pilot: PilotArtifact) -> str:
    payload = {
        "schema_version": pilot.schema_version,
        "pilot_id": pilot.pilot_id,
        "contrast_id": pilot.contrast_id,
        "source_population_id": pilot.source_population_id,
        "source_population_sha256": pilot.source_population_sha256,
        "case_bank_sha256": pilot.case_bank_sha256,
        "case_ids": [pair.case_id for pair in pilot.pairs],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _validate_bound_pilot(pilot: PilotArtifact) -> None:
    pilot.validate()
    if pilot.schedule_sha256 != pilot_schedule_sha256(pilot):
        raise ValueError("pilot schedule digest does not bind the pilot cases")


def pilot_sha256(pilot: PilotArtifact) -> str:
    _validate_bound_pilot(pilot)
    payload = json.dumps(asdict(pilot), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _beta_quantile(probability: float, a: float, b: float) -> float:
    if not 0 < probability < 1 or a <= 0 or b <= 0:
        raise ValueError("beta quantile inputs are invalid")
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2
        if regularized_beta(middle, a, b) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def discordance_upper_bound(pilot: PilotArtifact, *, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound for the discordant-case rate."""

    _validate_bound_pilot(pilot)
    if not 0 < alpha < 0.5:
        raise ValueError("pilot alpha is invalid")
    discordant = sum(pair.full_correct != pair.comparison_correct for pair in pilot.pairs)
    total = len(pilot.pairs)
    if discordant == total:
        return 1.0
    return _beta_quantile(1 - alpha, discordant + 1, total - discordant)

"""Closed schemas for preregistered C12 power planning."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

SCHEMA_VERSION = 1
DESIGN_VERSION = "2026.08.1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{2,79}")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_RESOURCE_LIMITS = {
    "seconds_per_cell": (0.001, 86_400.0),
    "parallel_cells": (1, 1_000_000),
    "gpu_hours_per_local_cell": (0.0, 24.0),
    "local_compute_cost_per_gpu_hour": (0.0, 10_000.0),
    "frontier_calls_per_case": (0, 1_000),
    "frontier_seconds_per_call": (0.0, 86_400.0),
    "frontier_cost_per_call": (0.0, 100_000.0),
}


def _finite_number(value: object, label: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} is invalid")
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"{label} is invalid")
    return float(value)


def _utc_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    return parsed


@dataclass(frozen=True, slots=True)
class ContrastDesign:
    contrast_id: str
    comparison: str
    margin: float
    alternative_effect: float
    baseline_rate: float
    null_discordance: float
    alternative_discordance: float
    arm_correlation: float
    repeats: int
    repeat_icc: float
    final_alpha: float
    target_power: float
    normal_floor: int
    required_cases: int
    exact_power: float
    calibration: Literal["worst_feasible", "unsigned_development_pilot_ucb"]
    promotion_eligible: bool
    pilot_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ResourceAssumptions:
    seconds_per_cell: float
    parallel_cells: int
    gpu_hours_per_local_cell: float
    local_compute_cost_per_gpu_hour: float
    frontier_calls_per_case: int
    frontier_seconds_per_call: float
    frontier_cost_per_call: float

    def validate(self) -> None:
        seconds = _finite_number(self.seconds_per_cell, "seconds_per_cell", positive=True)
        if (
            not _RESOURCE_LIMITS["seconds_per_cell"][0]
            <= seconds
            <= _RESOURCE_LIMITS["seconds_per_cell"][1]
        ):
            raise ValueError("seconds_per_cell is outside practical bounds")
        if (
            type(self.parallel_cells) is not int
            or not _RESOURCE_LIMITS["parallel_cells"][0]
            <= self.parallel_cells
            <= _RESOURCE_LIMITS["parallel_cells"][1]
        ):
            raise ValueError("parallel_cells is invalid")
        gpu_hours = _finite_number(
            self.gpu_hours_per_local_cell, "gpu_hours_per_local_cell", positive=False
        )
        local_cost = _finite_number(
            self.local_compute_cost_per_gpu_hour,
            "local_compute_cost_per_gpu_hour",
            positive=False,
        )
        if gpu_hours > _RESOURCE_LIMITS["gpu_hours_per_local_cell"][1]:
            raise ValueError("gpu_hours_per_local_cell is outside practical bounds")
        if local_cost > _RESOURCE_LIMITS["local_compute_cost_per_gpu_hour"][1]:
            raise ValueError("local_compute_cost_per_gpu_hour is outside practical bounds")
        if (
            type(self.frontier_calls_per_case) is not int
            or not _RESOURCE_LIMITS["frontier_calls_per_case"][0]
            <= self.frontier_calls_per_case
            <= _RESOURCE_LIMITS["frontier_calls_per_case"][1]
        ):
            raise ValueError("frontier_calls_per_case is invalid")
        frontier_seconds = _finite_number(
            self.frontier_seconds_per_call, "frontier_seconds_per_call", positive=False
        )
        frontier_cost = _finite_number(
            self.frontier_cost_per_call, "frontier_cost_per_call", positive=False
        )
        if frontier_seconds > _RESOURCE_LIMITS["frontier_seconds_per_call"][1]:
            raise ValueError("frontier_seconds_per_call is outside practical bounds")
        if frontier_cost > _RESOURCE_LIMITS["frontier_cost_per_call"][1]:
            raise ValueError("frontier_cost_per_call is outside practical bounds")


@dataclass(frozen=True, slots=True)
class CampaignPlan:
    campaign_id: Literal["p6", "p7"]
    strata: int
    stratum_kind: Literal["task_class", "domain"]
    diagnostic_case_count: int
    overall_case_floor: int
    representation_floor_per_stratum: int
    independent_claim_floor_per_stratum: int
    simultaneous_claim_floor_per_stratum: int
    interim_case_count: int
    conditions: int
    repeats: int
    total_cells: int
    interim_cells: int
    wall_time_hours: float
    gpu_hours: float
    local_compute_cost: float
    frontier_calls: int
    frontier_call_cost: float
    total_estimated_cost: float


@dataclass(frozen=True, slots=True)
class PowerPlan:
    schema_version: int
    design_version: str
    familywise_alpha: float
    contrast_alpha: float
    interim_alpha: float
    final_alpha: float
    target_joint_power: float
    target_power_per_contrast: float
    futility_conditional_power: float
    strongest_reduced_condition_id: str
    promotion_eligible: bool
    contrasts: tuple[ContrastDesign, ...]
    p6: CampaignPlan
    p7: CampaignPlan
    resources: ResourceAssumptions
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PilotPair:
    case_id: str
    full_correct: bool
    comparison_correct: bool

    def validate(self) -> None:
        if not isinstance(self.case_id, str) or _IDENTIFIER.fullmatch(self.case_id) is None:
            raise ValueError("pilot case_id is invalid")
        if type(self.full_correct) is not bool or type(self.comparison_correct) is not bool:
            raise ValueError("pilot outcomes must be booleans")


@dataclass(frozen=True, slots=True)
class PilotArtifact:
    schema_version: int
    pilot_id: str
    contrast_id: str
    evaluator_id: str
    producer: Literal["evaluator"]
    candidate_supplied: bool
    source_population_id: str
    source_population_sha256: str
    source_sha256: str
    case_bank_sha256: str
    schedule_sha256: str
    created_at_utc: str
    claim_pack_created_at_utc: str
    registered_before_claim_pack: bool
    independent_of_claim_pack: bool
    pairs: tuple[PilotPair, ...]

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("pilot schema is invalid")
        for value, label in (
            (self.pilot_id, "pilot_id"),
            (self.contrast_id, "contrast_id"),
            (self.evaluator_id, "evaluator_id"),
            (self.source_population_id, "source_population_id"),
        ):
            if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
                raise ValueError(f"{label} is invalid")
        for value, label in (
            (self.source_sha256, "source_sha256"),
            (self.case_bank_sha256, "case_bank_sha256"),
            (self.source_population_sha256, "source_population_sha256"),
            (self.schedule_sha256, "schedule_sha256"),
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{label} is invalid")
        if self.producer != "evaluator" or self.candidate_supplied is not False:
            raise ValueError("pilot evaluator provenance is invalid")
        created = _utc_timestamp(self.created_at_utc, "created_at_utc")
        claim_created = _utc_timestamp(self.claim_pack_created_at_utc, "claim_pack_created_at_utc")
        if created >= claim_created:
            raise ValueError("pilot was not created before the claim pack")
        if (
            self.registered_before_claim_pack is not True
            or self.independent_of_claim_pack is not True
        ):
            raise ValueError("pilot independence is not established")
        if not self.pairs:
            raise ValueError("pilot has no cases")
        for pair in self.pairs:
            pair.validate()
        ids = [pair.case_id for pair in self.pairs]
        if len(ids) != len(set(ids)):
            raise ValueError("pilot case ids are duplicated")


@dataclass(frozen=True, slots=True)
class CampaignManifest:
    campaign_id: Literal["p6", "p7"]
    power_plan_sha256: str
    taxonomy_version: str
    taxonomy_sha256: str
    qualification_manifest_sha256: str
    case_bank_sha256: str
    source_artifact_sha256: str
    schedule_sha256: str
    evaluator_id: str
    evaluator_provenance_sha256: str
    created_at_utc: str
    execution_not_before_utc: str
    producer: Literal["evaluator"]
    candidate_supplied: bool
    scheduled_case_ids: tuple[str, ...]
    stratum_by_case: tuple[tuple[str, str], ...]
    claimed_strata: tuple[str, ...]
    conditions: tuple[str, ...]
    repeats: tuple[int, ...]
    simultaneous_strata_claim: bool = False
    pilot_case_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ManifestValidation:
    valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObservedContrast:
    contrast_id: str
    lower_bound: float
    alpha_used: float
    conditional_power: float
    accounting_complete: bool
    safety_passed: bool
    attestation_sha256: str
    producer: Literal["evaluator"]
    candidate_supplied: bool


@dataclass(frozen=True, slots=True)
class SensitivityRow:
    contrast_id: str
    baseline_rate: float
    arm_correlation: float
    discordance: float
    repeat_icc: float
    approximate_cases: int
    promotion_assumption_eligible: bool


@dataclass(frozen=True, slots=True)
class SequentialResult:
    decision: Literal["continue", "stop_futility", "stop_success", "final_success", "final_failure"]
    claim_eligible: bool
    reasons: tuple[str, ...]

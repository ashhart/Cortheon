"""Fail-closed campaign manifest and claim-scope validation."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime

from cortheon.power_analysis.models import (
    CampaignManifest,
    CampaignPlan,
    ManifestValidation,
    PilotArtifact,
    PowerPlan,
)
from cortheon.power_analysis.sealing import campaign_schedule_sha256, power_plan_sha256
from cortheon.power_analysis.taxonomy import taxonomy_for_campaign, taxonomy_sha256

EXPECTED_REPEATS = (0, 1, 2)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _timestamp(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def _campaign(power_plan: PowerPlan, campaign_id: str) -> CampaignPlan:
    if campaign_id == "p6":
        return power_plan.p6
    if campaign_id == "p7":
        return power_plan.p7
    raise ValueError("campaign_id is invalid")


def validate_campaign_manifest(
    power_plan: PowerPlan,
    manifest: CampaignManifest,
    *,
    pilots: Mapping[str, PilotArtifact] | None = None,
) -> ManifestValidation:
    errors: list[str] = []
    try:
        campaign = _campaign(power_plan, manifest.campaign_id)
    except ValueError as exc:
        return ManifestValidation(False, (str(exc),))
    cases = manifest.scheduled_case_ids
    taxonomy_version, taxonomy_members = taxonomy_for_campaign(manifest.campaign_id)
    if manifest.power_plan_sha256 != power_plan_sha256(power_plan):
        errors.append("power_plan_digest_mismatch")
    if not power_plan.promotion_eligible:
        errors.append("unsigned_pilot_plan_is_development_only")
    if manifest.taxonomy_version != taxonomy_version or manifest.taxonomy_sha256 != taxonomy_sha256(
        manifest.campaign_id
    ):
        errors.append("closed_taxonomy_identity_mismatch")
    for label, digest in (
        ("qualification_manifest", manifest.qualification_manifest_sha256),
        ("case_bank", manifest.case_bank_sha256),
        ("source_artifact", manifest.source_artifact_sha256),
        ("evaluator_provenance", manifest.evaluator_provenance_sha256),
    ):
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            errors.append(f"{label}_digest_invalid")
    if (
        manifest.producer != "evaluator"
        or manifest.candidate_supplied is not False
        or not isinstance(manifest.evaluator_id, str)
        or not manifest.evaluator_id
    ):
        errors.append("evaluator_manifest_provenance_invalid")
    created = _timestamp(manifest.created_at_utc)
    execution = _timestamp(manifest.execution_not_before_utc)
    if created is None or execution is None or created >= execution:
        errors.append("manifest_creation_order_invalid")
    if manifest.schedule_sha256 != campaign_schedule_sha256(manifest):
        errors.append("schedule_digest_mismatch")
    if len(cases) != len(set(cases)):
        errors.append("scheduled_case_ids_are_duplicated")
    if len(cases) < campaign.overall_case_floor:
        errors.append("case_floor_below_power_requirement")
    expected_conditions = ("full", "bare", power_plan.strongest_reduced_condition_id)
    if manifest.conditions != expected_conditions:
        errors.append("conditions_are_not_the_fixed_C12_contrasts")
    if manifest.repeats != EXPECTED_REPEATS:
        errors.append("repetitions_are_not_exactly_three")
    stratum_pairs = manifest.stratum_by_case
    if len(stratum_pairs) != len({case_id for case_id, _stratum in stratum_pairs}):
        errors.append("stratum_mapping_has_duplicate_cases")
    stratum_map = dict(stratum_pairs)
    if set(stratum_map) != set(cases):
        errors.append("stratum_mapping_does_not_match_scheduled_cases")
    counts = Counter(stratum_map.values())
    if set(counts) != set(taxonomy_members):
        errors.append("strata_do_not_match_the_closed_taxonomy")
    if counts and min(counts.values()) < campaign.representation_floor_per_stratum:
        errors.append("per_stratum_representation_floor_not_met")
    if not set(manifest.claimed_strata) <= set(counts):
        errors.append("claimed_stratum_is_not_in_the_schedule")
    if len(manifest.claimed_strata) > 1 and not manifest.simultaneous_strata_claim:
        errors.append("multiple_strata_claim_requires_simultaneous_preregistration")
    if manifest.simultaneous_strata_claim and set(manifest.claimed_strata) != set(counts):
        errors.append("simultaneous_claim_must_cover_every_preregistered_stratum")
    claim_floor = (
        campaign.simultaneous_claim_floor_per_stratum
        if manifest.simultaneous_strata_claim
        else campaign.independent_claim_floor_per_stratum
    )
    errors.extend(
        f"claimed_stratum_is_underpowered:{stratum}"
        for stratum in manifest.claimed_strata
        if counts[stratum] < claim_floor
    )
    supplied_pilot_ids = set(manifest.pilot_case_ids)
    if supplied_pilot_ids & set(cases):
        errors.append("pilot_cases_overlap_the_claim_set")
    if pilots:
        expected_pilot_ids = {pair.case_id for pilot in pilots.values() for pair in pilot.pairs}
        if supplied_pilot_ids != expected_pilot_ids:
            errors.append("pilot_exclusion_set_does_not_match_calibration_artifacts")
    return ManifestValidation(not errors, tuple(sorted(set(errors))))

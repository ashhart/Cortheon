from __future__ import annotations

from collections import Counter
from dataclasses import replace

from cortheon.power_analysis import (
    CampaignManifest,
    ResourceAssumptions,
    build_power_plan,
    power_plan_sha256,
)
from cortheon.power_analysis.sealing import campaign_schedule_sha256
from cortheon.power_analysis.taxonomy import taxonomy_for_campaign, taxonomy_sha256
from cortheon.power_analysis.validation import validate_campaign_manifest

PLAN = build_power_plan(
    ResourceAssumptions(120, 32, 0.03, 2, 0, 0, 0),
    strongest_reduced_condition_id="without_cross_source_derivation",
)
TINY_PLAN = replace(
    PLAN,
    p6=replace(
        PLAN.p6,
        overall_case_floor=18,
        representation_floor_per_stratum=2,
        independent_claim_floor_per_stratum=4,
        simultaneous_claim_floor_per_stratum=6,
    ),
)


def _manifest(
    campaign_id: str = "p6",
    *,
    per_stratum: int | None = None,
    power_plan=PLAN,
) -> CampaignManifest:
    campaign = power_plan.p6 if campaign_id == "p6" else power_plan.p7
    taxonomy_version, strata = taxonomy_for_campaign(campaign_id)
    count = per_stratum or campaign.representation_floor_per_stratum
    pairs = tuple(
        (f"claim_{stratum_index:02d}_{index:05d}", stratum)
        for stratum_index, stratum in enumerate(strata)
        for index in range(count)
    )
    manifest = CampaignManifest(
        campaign_id=campaign_id,  # type: ignore[arg-type]
        power_plan_sha256=power_plan_sha256(power_plan),
        taxonomy_version=taxonomy_version,
        taxonomy_sha256=taxonomy_sha256(campaign_id),
        qualification_manifest_sha256="a" * 64,
        case_bank_sha256="b" * 64,
        source_artifact_sha256="c" * 64,
        schedule_sha256="0" * 64,
        evaluator_id="outside_evaluator",
        evaluator_provenance_sha256="d" * 64,
        created_at_utc="2026-08-01T12:00:00Z",
        execution_not_before_utc="2026-08-02T12:00:00Z",
        producer="evaluator",
        candidate_supplied=False,
        scheduled_case_ids=tuple(case_id for case_id, _stratum in pairs),
        stratum_by_case=pairs,
        claimed_strata=(),
        conditions=("full", "bare", "without_cross_source_derivation"),
        repeats=(0, 1, 2),
    )
    return replace(manifest, schedule_sha256=campaign_schedule_sha256(manifest))


def _reseal(manifest: CampaignManifest) -> CampaignManifest:
    return replace(manifest, schedule_sha256=campaign_schedule_sha256(manifest))


def test_balanced_overall_manifest_passes_without_implying_stratum_claims() -> None:
    manifest = _manifest()
    assert validate_campaign_manifest(PLAN, manifest).valid
    counts = Counter(dict(manifest.stratum_by_case).values())
    assert min(counts.values()) == PLAN.p6.representation_floor_per_stratum
    assert min(counts.values()) < PLAN.p6.independent_claim_floor_per_stratum


def test_aggregate_schedule_cannot_claim_an_underpowered_class() -> None:
    task_class = taxonomy_for_campaign("p6")[1][0]
    manifest = _reseal(replace(_manifest(), claimed_strata=(task_class,)))
    result = validate_campaign_manifest(PLAN, manifest)
    assert not result.valid
    assert f"claimed_stratum_is_underpowered:{task_class}" in result.errors


def test_one_independently_powered_stratum_can_make_an_explicit_claim() -> None:
    manifest = _manifest(per_stratum=4, power_plan=TINY_PLAN)
    task_class = taxonomy_for_campaign("p6")[1][0]
    result = validate_campaign_manifest(
        TINY_PLAN,
        _reseal(replace(manifest, claimed_strata=(task_class,))),
    )
    assert result.valid


def test_simultaneous_all_class_claim_uses_the_globally_corrected_floor() -> None:
    underpowered = _manifest(per_stratum=4, power_plan=TINY_PLAN)
    claimed = tuple(sorted(set(dict(underpowered.stratum_by_case).values())))
    result = validate_campaign_manifest(
        TINY_PLAN,
        _reseal(
            replace(
                underpowered,
                claimed_strata=claimed,
                simultaneous_strata_claim=True,
            )
        ),
    )
    assert not result.valid
    powered = _manifest(per_stratum=6, power_plan=TINY_PLAN)
    result = validate_campaign_manifest(
        TINY_PLAN,
        _reseal(
            replace(
                powered,
                claimed_strata=claimed,
                simultaneous_strata_claim=True,
            )
        ),
    )
    assert result.valid


def test_24_cases_wrong_repeats_or_missing_conditions_fail_closed() -> None:
    manifest = _manifest()
    tiny = replace(
        manifest,
        scheduled_case_ids=manifest.scheduled_case_ids[:24],
        stratum_by_case=manifest.stratum_by_case[:24],
    )
    assert "case_floor_below_power_requirement" in validate_campaign_manifest(PLAN, tiny).errors
    assert not validate_campaign_manifest(PLAN, replace(manifest, repeats=(0, 1))).valid
    assert not validate_campaign_manifest(
        PLAN, replace(manifest, conditions=("full", "bare"))
    ).valid


def test_manifest_must_bind_the_plan_before_execution() -> None:
    manifest = _manifest()
    assert not validate_campaign_manifest(
        PLAN,
        replace(manifest, power_plan_sha256="f" * 64),
    ).valid
    assert not validate_campaign_manifest(
        PLAN,
        replace(manifest, created_at_utc="2026-08-03T12:00:00Z"),
    ).valid


def test_arbitrary_stratum_labels_cannot_replace_the_closed_taxonomy() -> None:
    manifest = _manifest()
    remapped = tuple(
        (case_id, f"fake_{index % PLAN.p6.strata}")
        for index, case_id in enumerate(manifest.scheduled_case_ids)
    )
    result = validate_campaign_manifest(
        PLAN,
        _reseal(replace(manifest, stratum_by_case=remapped)),
    )
    assert not result.valid
    assert "strata_do_not_match_the_closed_taxonomy" in result.errors


def test_taxonomies_are_exact_versioned_north_star_classes_and_domains() -> None:
    assert taxonomy_for_campaign("p6")[1] == (
        "ambiguity_resolution",
        "constraint_bound_planning",
        "cross_file_numeric_join",
        "current_web_research",
        "evidence_bound_debugging",
        "long_horizon_execution",
        "novel_abductive_synthesis",
        "repository_patching",
        "semantic_cross_document_reasoning",
    )
    assert taxonomy_for_campaign("p7")[1] == (
        "software_systems",
        "science_engineering",
        "health_medicine",
        "law_public_policy",
        "finance_economics",
        "industrial_operations",
        "climate_energy",
        "education_knowledge_work",
    )


def test_schedule_source_and_evaluator_provenance_are_bound() -> None:
    manifest = _manifest()
    for mutation in (
        {"schedule_sha256": "f" * 64},
        {"source_artifact_sha256": "bad"},
        {"candidate_supplied": True},
        {"producer": "candidate"},
        {"evaluator_provenance_sha256": "bad"},
    ):
        assert not validate_campaign_manifest(PLAN, replace(manifest, **mutation)).valid


def test_unsigned_pilot_calibration_can_never_validate_a_promotion_manifest() -> None:
    development_plan = replace(TINY_PLAN, promotion_eligible=False)
    manifest = _manifest(per_stratum=2, power_plan=development_plan)
    result = validate_campaign_manifest(development_plan, manifest)
    assert not result.valid
    assert "unsigned_pilot_plan_is_development_only" in result.errors


def test_pilot_case_overlap_is_never_allowed_into_claim_set() -> None:
    manifest = _manifest()
    overlapping = replace(manifest, pilot_case_ids=(manifest.scheduled_case_ids[0],))
    result = validate_campaign_manifest(PLAN, overlapping)
    assert not result.valid
    assert "pilot_cases_overlap_the_claim_set" in result.errors

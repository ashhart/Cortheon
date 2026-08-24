"""Canonical power-plan identity for preregistered campaign manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from cortheon.power_analysis.models import CampaignManifest, PowerPlan


def power_plan_sha256(plan: PowerPlan) -> str:
    payload = json.dumps(asdict(plan), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def campaign_schedule_sha256(manifest: CampaignManifest) -> str:
    payload = {
        "campaign_id": manifest.campaign_id,
        "power_plan_sha256": manifest.power_plan_sha256,
        "taxonomy_version": manifest.taxonomy_version,
        "taxonomy_sha256": manifest.taxonomy_sha256,
        "qualification_manifest_sha256": manifest.qualification_manifest_sha256,
        "case_bank_sha256": manifest.case_bank_sha256,
        "source_artifact_sha256": manifest.source_artifact_sha256,
        "evaluator_id": manifest.evaluator_id,
        "evaluator_provenance_sha256": manifest.evaluator_provenance_sha256,
        "created_at_utc": manifest.created_at_utc,
        "execution_not_before_utc": manifest.execution_not_before_utc,
        "producer": manifest.producer,
        "candidate_supplied": manifest.candidate_supplied,
        "scheduled_case_ids": manifest.scheduled_case_ids,
        "stratum_by_case": manifest.stratum_by_case,
        "claimed_strata": manifest.claimed_strata,
        "conditions": manifest.conditions,
        "repeats": manifest.repeats,
        "simultaneous_strata_claim": manifest.simultaneous_strata_claim,
        "pilot_case_ids": manifest.pilot_case_ids,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()

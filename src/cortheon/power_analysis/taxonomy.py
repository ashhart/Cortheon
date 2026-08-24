"""Closed P6 task-class and P7 top-level domain taxonomies."""

from __future__ import annotations

import hashlib
import json

P6_TAXONOMY_VERSION = "north_star_task_classes_v1"
P6_TASK_CLASSES = (
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
P7_TAXONOMY_VERSION = "top_level_domains_v1"
P7_DOMAINS = (
    "software_systems",
    "science_engineering",
    "health_medicine",
    "law_public_policy",
    "finance_economics",
    "industrial_operations",
    "climate_energy",
    "education_knowledge_work",
)


def taxonomy_for_campaign(campaign_id: str) -> tuple[str, tuple[str, ...]]:
    if campaign_id == "p6":
        return P6_TAXONOMY_VERSION, P6_TASK_CLASSES
    if campaign_id == "p7":
        return P7_TAXONOMY_VERSION, P7_DOMAINS
    raise ValueError("campaign_id is invalid")


def taxonomy_sha256(campaign_id: str) -> str:
    version, members = taxonomy_for_campaign(campaign_id)
    payload = json.dumps(
        {"version": version, "members": members},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()

"""Stable facade for research source planning.

The planner is repository-only research code. Focused implementation modules
live in :mod:`cortheon.source_planner_core`; this module keeps the original
imports, object identities, signatures, and facade-level patch points.
"""

from cortheon.source_planner_core.heuristic import (
    ARTIFACT_TERMS,
    BIOLOGY_TERMS,
    CURRENT_TERMS,
    MEDICINE_TERMS,
    RESEARCH_TERMS,
    SCIENCE_TERMS,
    SOFTWARE_TERMS,
    SourcePlanner,
    availability_for,
    budget_for,
    classify_topic,
    decision_reason,
    default_source_planner,
    selection_threshold,
    source_priority,
    strongest_capability,
    topic_terms,
)
from cortheon.source_planner_core.profiles import (
    build_research_source_profiles,
    is_source_selected,
    selected_source_names,
    source_plan_evidence,
    source_plan_notes,
    source_profile_from_mapping,
)
from cortheon.source_planner_core.types import SourcePlanningConstraints, SourceProfile

__all__ = [
    "ARTIFACT_TERMS",
    "BIOLOGY_TERMS",
    "CURRENT_TERMS",
    "MEDICINE_TERMS",
    "RESEARCH_TERMS",
    "SCIENCE_TERMS",
    "SOFTWARE_TERMS",
    "SourcePlanner",
    "SourcePlanningConstraints",
    "SourceProfile",
    "availability_for",
    "budget_for",
    "build_research_source_profiles",
    "classify_topic",
    "decision_reason",
    "default_source_planner",
    "is_source_selected",
    "selected_source_names",
    "selection_threshold",
    "source_plan_evidence",
    "source_plan_notes",
    "source_priority",
    "source_profile_from_mapping",
    "strongest_capability",
    "topic_terms",
]

# Keep class/function metadata stable for reprs, pickles, and callers that
# identify the public module. The implementation remains owned by core files.
for _public_name in __all__:
    _public_object = globals()[_public_name]
    if callable(_public_object) and hasattr(_public_object, "__module__"):
        _public_object.__module__ = __name__

del _public_name, _public_object

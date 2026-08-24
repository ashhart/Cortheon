"""Cortheon's memory-only cognitive runtime."""

# ruff: noqa: F401 - this facade intentionally re-exports every import.
# pyright: reportUnsupportedDunderAll=false

from cortheon.cognitive_core.aggregate_alignment import (
    _evidence_alignment_check,
    _read_many_alignment_check,
)
from cortheon.cognitive_core.alignment import (
    _FALSIFICATION_DESIGN_RE,
    _abductive_alignment_check,
    _ambiguity_alignment_check,
    _answer_polarity,
    _research_alignment_check,
    _research_conflict_present,
)
from cortheon.cognitive_core.claim_verification import (
    _claim_verification_profiles,
)
from cortheon.cognitive_core.claims import (
    _BEHAVIOR_CLAIM_RE,
    _CHANGE_CLAIM_RE,
    _LEGAL_CLAIM_RE,
    _NEGATION_RE,
    _NUMERIC_CLAIM_RE,
    _PRIVATE_RECORD_RE,
    _SCIENTIFIC_CLAIM_RE,
    _claim_entailment,
    _claim_profiles_from_checks,
    _claim_type,
    _join_reasons,
    _observation_body,
)
from cortheon.cognitive_core.diffs import (
    _CONCISE_CHANGE_HINTS,
    _SINGLE_LINE_CHANGE_RE,
    _diff_changed_line_count,
    _diff_establishes_change,
    _diff_line_budget,
    _diff_receipt_paths,
    _diff_weakens_tests,
)
from cortheon.cognitive_core.models import (
    _ASSIST_WAIVER_CAVEATS,
    _WAIVER_CAVEATS,
    HYPOTHESIS_STATUSES,
    OBSERVATION_KINDS,
    OBSERVATION_STATUSES,
    RESEARCH_PURPOSES,
    CognitiveRuntimeError,
    EvidenceRequest,
    Hypothesis,
    Investigation,
    InvestigationNotFound,
    Observation,
    PublicClaim,
    Requirement,
    SemanticEdge,
    SemanticRule,
    _fit_hypotheses,
    _fit_strings,
    _session_graph,
)
from cortheon.cognitive_core.plan_joins import (
    _diagnostic_join_analysis,
    _ordered_plan_analysis,
)
from cortheon.cognitive_core.profiles import (
    _CHANGE_HINTS,
    _CODE_HINTS,
    _DECISION_HINTS,
    _DOCUMENT_HINTS,
    _EXPLICIT_FRESHNESS_HINTS,
    _RESEARCH_HINTS,
    EFFORT_PROFILES,
    MAX_REQUEST_ATTEMPTS,
    STRICTNESS_PROFILES,
    TASK_KINDS,
    EffortProfile,
    StrictnessProfile,
    _capability_for_falsification,
    _capability_for_kind,
    _evidence_action_cost,
    _evidence_action_reliability,
    _has_hint,
)
from cortheon.cognitive_core.receipts import (
    _HOST_EVIDENCE_PREFIX,
    _MUTATING_READER_FLAGS,
    _READ_ONLY_GIT_SUBCOMMANDS,
    _READ_ONLY_SHELL_COMMANDS,
    _digest,
    _example_receipt_json,
    _host_evidence_receipt,
    _host_path_matches_request,
    _observation_digest,
    _observation_origin,
    _read_only_shell_receipt,
    _read_receipt_paths,
    _receipt_error,
    _receipt_outcome,
    _validate_host_observation_batch,
)
from cortheon.cognitive_core.requirements import (
    _REQUIREMENT_ACTION_RE,
    _REQUIREMENT_BOUNDARY_RE,
    _REQUIREMENT_GENERIC_TERMS,
    _extract_requirements,
    _requirement_coverage,
    _requirement_kind_matches,
    _requirement_proof,
    _requirement_terms,
)
from cortheon.cognitive_core.research_gaps import (
    _LOCAL_PROJECT_DOMAIN_RE,
    _RESEARCH_CONFLICT_ACK_RE,
    _RESEARCH_DOWNSIDE_RE,
    _RESEARCH_SCOPED_CONTRAST_RE,
    _RESEARCH_UPSIDE_RE,
    _answer_acknowledges_research_conflict,
    _answer_urls,
    _effective_web_lineages,
    _is_local_project_evidence,
    _release_version_candidates,
    _research_completion_gaps,
    _research_release_analysis,
    _version_key,
)
from cortheon.cognitive_core.runtime import CognitiveRuntime
from cortheon.cognitive_core.semantic_graph import (
    _SEMANTIC_STOPWORDS,
    _affirmatively_mentions,
    _keywords,
    _phrase_mentioned,
    _semantic_edge,
    _semantic_edges,
    _semantic_key,
    _semantic_phrase,
    _semantic_rules,
    _semantic_table_cells,
    _semantic_table_edges,
    _semantic_table_relation,
    _semantic_terms,
)
from cortheon.cognitive_core.semantic_join import (
    _semantic_join_analysis,
)
from cortheon.cognitive_core.tasks import (
    _ABDUCTIVE_GOAL_RE,
    _AMBIGUITY_GOAL_RE,
    _CODE_EXTENSION_NAMES,
    _CODE_PATH_RE,
    _CODE_SYMBOL_RE,
    _CROSS_SOURCE_HINTS,
    _DOCUMENT_PATH_RE,
    _INTEGER_TOKEN,
    _QUALIFIED_CODE_SYMBOL_RE,
    _TECHNOLOGY_NAMES_THAT_LOOK_LIKE_PATHS,
    _abductive_proposition,
    _answer_integer_assertions,
    _discovered_project_paths,
    _goal_code_paths,
    _goal_code_symbols,
    _goal_document_paths,
    _infer_deliverable,
    _infer_join_operation,
    _infer_task_kind,
    _is_test_path,
    _observation_score,
    _parse_integer,
    _requests_change,
    _without_url_literals,
)
from cortheon.cognitive_core.text import (
    _LOOKUP_PHRASE_RE,
    _LOOKUP_STOP_TARGETS,
    _LOOKUP_TARGET_RE,
    _SPACE_RE,
    _WORD_RE,
    _lookup_phrase_target,
    _lookup_target_match,
    _normalized,
    _optional_text,
    _optional_timestamp,
    _optional_url,
    _safe_public_label,
    _string_list,
    _text,
)
from cortheon.cognitive_graph import CognitiveGraph, content_id, rank_information_gain
from cortheon.cognitive_program import compile_program, select_operator
from cortheon.cognitive_protocol import (
    CORTHEON_CERTIFICATION_SCOPE,
    CORTHEON_PROTOCOL_VERSION,
    CORTHEON_STORAGE_MODEL,
)
from cortheon.cognitive_repair import (
    changed_paths_from_diff,
    is_test_path,
    protected_test_paths,
    protects_tests,
)
from cortheon.sanitize import scan_text

__all__ = [name for name in globals() if not name.startswith("__")]

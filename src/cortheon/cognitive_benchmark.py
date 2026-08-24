"""Stable import surface for the cognitive benchmark.

The implementation lives in the repository-only modules of
:mod:`cortheon.benchmark_core`; this facade re-exports the full public and
test-facing surface so existing imports and monkeypatches keep working.
"""

import subprocess
import urllib.error
import urllib.request

from cortheon.benchmark_core.audit import (
    _audit_manifest,
    _blinded_case,
    _canonical_json,
    scaling_curve,
    verify_audit_bundle,
)
from cortheon.benchmark_core.blocks import (
    BLOCK_KINDS,
    DELIVERY_FAILURE,
    FALSE_BLOCK,
    SAFE_BLOCK,
    UNCLASSIFIED_BLOCK,
    block_tally,
    classify_block,
    classify_serialized_block,
    is_comparable_outcome,
    is_serialized_comparable_outcome,
)
from cortheon.benchmark_core.cli import (
    build_parser,
    main,
)
from cortheon.benchmark_core.discovery import (
    _imports,
    _integer_constants,
    discover_benchmark_cases,
    discover_cases,
    discover_join_cases,
)
from cortheon.benchmark_core.fixtures_diagnostic import (
    discover_diagnostic_cases,
)
from cortheon.benchmark_core.fixtures_long_horizon import (
    discover_long_horizon_cases,
)
from cortheon.benchmark_core.fixtures_patch import (
    discover_patch_cases,
)
from cortheon.benchmark_core.fixtures_planning import (
    discover_planning_cases,
)
from cortheon.benchmark_core.fixtures_reasoning import (
    _reasoning_derived_relations,
    discover_reasoning_cases,
)
from cortheon.benchmark_core.fixtures_research import (
    _latest_pypi_release,
    discover_research_cases,
)
from cortheon.benchmark_core.fixtures_semantic import (
    discover_semantic_cases,
)
from cortheon.benchmark_core.grading import (
    _ambiguity_forbidden_asserted,
    _derived_relation_present,
    _grade,
    _reasoning_expected_present,
    _reasoning_semantic_text,
    _semantic_forbidden_asserted,
    _semantic_text,
)
from cortheon.benchmark_core.health import (
    _model_endpoint_health,
    _postflight_probe,
    _runtime_health,
)
from cortheon.benchmark_core.models import (
    IGNORED_WORKSPACE_NAMES,
    WITHHELD_PREFIX,
    BenchmarkCase,
    DiagnosticCase,
    ImportCase,
    JoinCase,
    LongHorizonCase,
    PatchCase,
    PlanningCase,
    ReasoningCase,
    ResearchCase,
    RunResult,
    SemanticCase,
    _case_id,
)
from cortheon.benchmark_core.outcomes import (
    EvaluationOutcome,
    is_authenticated_withhold,
    is_delivered_outcome,
    is_exact_terminal_success,
    is_serialized_verified_completion,
    is_verified_completion,
    missing_outcome,
)
from cortheon.benchmark_core.retry import _retry_after_infrastructure_death
from cortheon.benchmark_core.run_support import (
    _delivery_succeeded,
    _event_statistics,
    _final_text,
    _opencode_step_budget_exhausted,
    _parse_events,
    _pi_provider_config,
    _provider_config,
    _runtime_metric_delta,
    _runtime_metric_snapshot,
    _substrate_telemetry_valid,
    _task_type,
)
from cortheon.benchmark_core.runner_frontier import (
    run_frontier_cli_job,
)
from cortheon.benchmark_core.runner_local import (
    run_job,
)
from cortheon.benchmark_core.stats import (
    _condition_summary,
    _frontier_comparison,
    _mcnemar_exact,
    _north_star_coverage,
    _paired_summary,
    _percentile,
    _proof_gates,
)
from cortheon.benchmark_core.workspace import (
    _copy_ignore,
    _grade_patch_workspace,
    _prepare_patch_case,
    _prepare_semantic_case,
    _python_has_unreachable_statement,
    _repository_fingerprint,
    _workspace_environment,
    isolated_repository,
)

__all__ = [
    "BLOCK_KINDS",
    "DELIVERY_FAILURE",
    "FALSE_BLOCK",
    "IGNORED_WORKSPACE_NAMES",
    "SAFE_BLOCK",
    "UNCLASSIFIED_BLOCK",
    "WITHHELD_PREFIX",
    "BenchmarkCase",
    "DiagnosticCase",
    "EvaluationOutcome",
    "ImportCase",
    "JoinCase",
    "LongHorizonCase",
    "PatchCase",
    "PlanningCase",
    "ReasoningCase",
    "ResearchCase",
    "RunResult",
    "SemanticCase",
    "_ambiguity_forbidden_asserted",
    "_audit_manifest",
    "_blinded_case",
    "_canonical_json",
    "_case_id",
    "_condition_summary",
    "_copy_ignore",
    "_delivery_succeeded",
    "_derived_relation_present",
    "_event_statistics",
    "_final_text",
    "_frontier_comparison",
    "_grade",
    "_grade_patch_workspace",
    "_imports",
    "_integer_constants",
    "_latest_pypi_release",
    "_mcnemar_exact",
    "_model_endpoint_health",
    "_north_star_coverage",
    "_opencode_step_budget_exhausted",
    "_paired_summary",
    "_parse_events",
    "_percentile",
    "_pi_provider_config",
    "_postflight_probe",
    "_prepare_patch_case",
    "_prepare_semantic_case",
    "_proof_gates",
    "_provider_config",
    "_python_has_unreachable_statement",
    "_reasoning_derived_relations",
    "_reasoning_expected_present",
    "_reasoning_semantic_text",
    "_repository_fingerprint",
    "_retry_after_infrastructure_death",
    "_runtime_health",
    "_runtime_metric_delta",
    "_runtime_metric_snapshot",
    "_semantic_forbidden_asserted",
    "_semantic_text",
    "_substrate_telemetry_valid",
    "_task_type",
    "_workspace_environment",
    "block_tally",
    "build_parser",
    "classify_block",
    "classify_serialized_block",
    "discover_benchmark_cases",
    "discover_cases",
    "discover_diagnostic_cases",
    "discover_join_cases",
    "discover_long_horizon_cases",
    "discover_patch_cases",
    "discover_planning_cases",
    "discover_reasoning_cases",
    "discover_research_cases",
    "discover_semantic_cases",
    "is_authenticated_withhold",
    "is_comparable_outcome",
    "is_delivered_outcome",
    "is_exact_terminal_success",
    "is_serialized_comparable_outcome",
    "is_serialized_verified_completion",
    "is_verified_completion",
    "isolated_repository",
    "main",
    "missing_outcome",
    "run_frontier_cli_job",
    "run_job",
    "scaling_curve",
    "subprocess",
    "urllib",
    "verify_audit_bundle",
]

if __name__ == "__main__":
    raise SystemExit(main())

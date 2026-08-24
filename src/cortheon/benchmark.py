"""Blind, repeated evaluation of Cortheon against stock and frontier contenders.

Stable facade: the implementation lives in the repository-only
:mod:`cortheon.parity_benchmark_core` subpackage. This module re-exports the
complete original surface so existing imports and monkeypatches (notably
``call_contender``, ``_post_json``, and ``datetime``) keep working.
"""

from datetime import UTC, datetime

from cortheon.parity_benchmark_core.blind import (
    _canonical_blind_submission,
    _load_public_case_pack,
    attest_blind_submission,
    run_blind_submissions,
)
from cortheon.parity_benchmark_core.casepack import (
    _case_bank_hash,
    _case_documents,
    _case_has_frozen_oracle,
    _case_pack_metadata,
    _load_case_pack,
    _load_cases,
    _normalize_cases,
    _resolve_live_grader,
    _validate_patch_fixture,
    select_case_bank,
)
from cortheon.parity_benchmark_core.cases_builtin import _builtin_cases
from cortheon.parity_benchmark_core.cli import (
    _contenders,
    _parse_cli_spec,
    _parse_pricing,
    _run_blind_submission_command,
    main,
)
from cortheon.parity_benchmark_core.contender import (
    _api_endpoint,
    _call_cli_contender,
    _contender_messages,
    _frontier_tools,
    _messages_with_documents,
    _observed_model_id,
    _post_json,
    _responses_text,
    _visible_input_sha256,
    call_contender,
)
from cortheon.parity_benchmark_core.grading import (
    _classification,
    _extract_patch,
    _grade_patch_in_sandbox,
    _observed_verdict,
    _run_sandbox_tests,
    _sandbox_image,
    grade_answer,
)
from cortheon.parity_benchmark_core.metrics import (
    _benchmark_input_sha256,
    _candidate_identity,
    _completion_origin,
    _contender_family,
    _cortheon_outcome,
    _input_symmetry,
    _integer_token_count,
    _percentile,
    _rate,
    _result_cost,
    _summarize_candidate,
    _summarize_slice,
)
from cortheon.parity_benchmark_core.models import Contender, LoadedCasePack, ModelResult
from cortheon.parity_benchmark_core.pairing import (
    _paired_candidate_comparisons,
    _paired_statistics,
    _stable_integer_seed,
)
from cortheon.parity_benchmark_core.parser import build_parser
from cortheon.parity_benchmark_core.promotion import (
    _metric_float,
    _nested_metric,
    _paired_promotion_statistics,
    _ratio_gate,
    _report_candidate_alias,
    _report_candidate_summary,
    _report_selection_hash,
    evaluate_promotion,
)
from cortheon.parity_benchmark_core.runner import (
    run_benchmark,
)

__all__ = [
    "UTC",
    "Contender",
    "LoadedCasePack",
    "ModelResult",
    "_api_endpoint",
    "_benchmark_input_sha256",
    "_builtin_cases",
    "_call_cli_contender",
    "_candidate_identity",
    "_canonical_blind_submission",
    "_case_bank_hash",
    "_case_documents",
    "_case_has_frozen_oracle",
    "_case_pack_metadata",
    "_classification",
    "_completion_origin",
    "_contender_family",
    "_contender_messages",
    "_contenders",
    "_cortheon_outcome",
    "_extract_patch",
    "_frontier_tools",
    "_grade_patch_in_sandbox",
    "_input_symmetry",
    "_integer_token_count",
    "_load_case_pack",
    "_load_cases",
    "_load_public_case_pack",
    "_messages_with_documents",
    "_metric_float",
    "_nested_metric",
    "_normalize_cases",
    "_observed_model_id",
    "_observed_verdict",
    "_paired_candidate_comparisons",
    "_paired_promotion_statistics",
    "_paired_statistics",
    "_parse_cli_spec",
    "_parse_pricing",
    "_percentile",
    "_post_json",
    "_rate",
    "_ratio_gate",
    "_report_candidate_alias",
    "_report_candidate_summary",
    "_report_selection_hash",
    "_resolve_live_grader",
    "_responses_text",
    "_result_cost",
    "_run_blind_submission_command",
    "_run_sandbox_tests",
    "_sandbox_image",
    "_stable_integer_seed",
    "_summarize_candidate",
    "_summarize_slice",
    "_validate_patch_fixture",
    "_visible_input_sha256",
    "attest_blind_submission",
    "build_parser",
    "call_contender",
    "datetime",
    "evaluate_promotion",
    "grade_answer",
    "main",
    "run_benchmark",
    "run_blind_submissions",
    "select_case_bank",
]

if __name__ == "__main__":
    raise SystemExit(main())

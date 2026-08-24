"""Case and result value types shared across the benchmark harness."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

from cortheon.benchmark_core.outcomes import EvaluationOutcome, missing_outcome

WITHHELD_PREFIX = "[Cortheon withheld"


IGNORED_WORKSPACE_NAMES = frozenset(
    {
        ".git",
        ".cortheon",
        ".cortheon-test",
        ".claude",
        ".codex",
        ".mypy_cache",
        ".opencode",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        ".zcode",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)


@dataclass(frozen=True, slots=True)
class ImportCase:
    case_id: str
    path: str
    module: str
    expected: bool
    prompt: str


@dataclass(frozen=True, slots=True)
class JoinCase:
    case_id: str
    paths: tuple[str, str]
    symbols: tuple[str, str]
    values: tuple[int, int]
    expected: int
    prompt: str


@dataclass(frozen=True, slots=True)
class PatchCase:
    case_id: str
    files: tuple[tuple[str, str], ...]
    protected_paths: tuple[str, ...]
    test_command: str
    hidden_assertions: str
    prompt: str


@dataclass(frozen=True, slots=True)
class SemanticCase:
    case_id: str
    files: tuple[tuple[str, str], ...]
    expected: tuple[str, ...]
    forbidden_answers: tuple[str, ...]
    prompt: str


@dataclass(frozen=True, slots=True)
class ResearchCase:
    case_id: str
    project: str
    expected: str
    github_url: str
    pypi_url: str
    prompt: str


@dataclass(frozen=True, slots=True)
class DiagnosticCase:
    case_id: str
    files: tuple[tuple[str, str], ...]
    expected: tuple[str, ...]
    forbidden_answers: tuple[str, ...]
    prompt: str


@dataclass(frozen=True, slots=True)
class PlanningCase:
    case_id: str
    files: tuple[tuple[str, str], ...]
    ordered_steps: tuple[str, ...]
    expected: tuple[str, ...]
    forbidden_answers: tuple[str, ...]
    prompt: str


@dataclass(frozen=True, slots=True)
class LongHorizonCase:
    case_id: str
    files: tuple[tuple[str, str], ...]
    protected_paths: tuple[str, ...]
    required_paths: tuple[str, ...]
    required_content: tuple[tuple[str, str], ...]
    test_command: str
    hidden_assertions: str
    prompt: str


@dataclass(frozen=True, slots=True)
class ReasoningCase:
    case_id: str
    mode: str
    files: tuple[tuple[str, str], ...]
    expected: tuple[str, ...]
    forbidden_answers: tuple[str, ...]
    required_any: tuple[tuple[str, ...], ...]
    derived_relations: tuple[tuple[tuple[str, ...], ...], ...]
    prompt: str


BenchmarkCase = (
    ImportCase
    | JoinCase
    | PatchCase
    | SemanticCase
    | ResearchCase
    | DiagnosticCase
    | PlanningCase
    | LongHorizonCase
    | ReasoningCase
)


@dataclass(slots=True)
class AttemptRecord:
    attempt_index: int
    latency_seconds: float
    tokens: int | None
    tool_calls: int
    cost_usd: float | None
    timed_out: bool
    process_error: str | None
    failure_owner: Literal["candidate", "external_infrastructure"] | None
    terminal_status: str
    provider_id: str | None
    model_id: str | None
    identity_valid: bool
    identity_provenance: str
    measurements_valid: bool
    policy_timeout_seconds: float
    policy_max_steps: int
    policy_max_tool_calls: int
    policy_context_tokens: int
    policy_output_tokens: int
    budget_reason: str | None


@dataclass(slots=True)
class RunResult:
    case_id: str
    repeat: int
    condition: str
    expected: bool | int | str | tuple[str, ...]
    final_text: str
    delivered: bool
    correct: bool
    latency_seconds: float
    tokens: int | None
    tool_calls: int
    tool_errors: int
    timed_out: bool
    process_error: str | None
    expected_verdict: Literal["allow", "block"] | None = None
    failure_owner: Literal["candidate", "external_infrastructure"] | None = None
    evaluator_outcome: EvaluationOutcome = field(
        default_factory=lambda: missing_outcome("opencode")
    )
    # tool_calls stays model tool attempts for every host; these three are
    # the Pi adapter's distinct classifications and default to zero so other
    # hosts report zeros.
    host_tool_executions: int = 0
    blocked_tool_calls: int = 0
    unavailable_tool_calls: int = 0
    inference_provider_id: str | None = None
    inference_model_id: str | None = None
    execution_identity_valid: bool = False
    execution_identity_provenance: str = "unavailable"
    execution_measurements_valid: bool = False
    execution_identity_reason: str | None = None
    execution_measurement_reason: str | None = None
    observed_steps: int = 0
    policy_timeout_seconds: float = 0.0
    policy_max_steps: int = 0
    policy_max_tool_calls: int = 0
    policy_context_tokens: int = 0
    policy_output_tokens: int = 0
    policy_budget_reason: str | None = None
    step_budget_exhausted: bool = False
    cost_usd: float | None = None
    retry_count: int = 0
    retry_reason: str | None = None
    prior_attempts: tuple[AttemptRecord, ...] = ()
    task_type: str = "import_lookup"
    artifact_correct: bool | None = None
    artifact_failure: str | None = None
    # Positively graded correctness of the pre-block candidate answer, when
    # the runner actually observed it. A withheld final text is the block
    # notice, not the candidate, so this stays None rather than being
    # inferred; None means candidate correctness is unknown.
    candidate_correct: bool | None = None
    # Which stage ended a Pi causal-synthesis attempt with no certified
    # answer: deliberation_empty, validation_failed, mapping_failed,
    # transport_failed, runtime_withheld, or terminated_before_deliberation.
    # One closed enum member or None
    # (unknown, not applicable, or not a Pi treatment run) — never candidate,
    # evidence, or model text, so the auditable report can separate the
    # stages without ever carrying what the model wrote.
    causal_stage_reason: str | None = None
    substrate_telemetry_valid: bool | None = None
    model_text_chars: int = 0
    deliverable_chars: int = 0
    runtime_sessions_started: int = 0
    runtime_observations_accepted: int = 0
    runtime_sessions_completed: int = 0
    runtime_sessions_evidence_closed: int = 0
    runtime_sessions_abandoned: int = 0
    runtime_completion_withheld: int = 0
    runtime_controller_decisions: int = 0
    runtime_controller_alternatives_considered: int = 0
    condition_registry_version: int | None = None
    condition_config_sha256: str | None = None
    condition_implementation_sha256: str | None = None
    condition_requires_runtime_completion: bool | None = None
    condition_profile_receipt_valid: bool | None = None
    condition_observed_config_sha256: str | None = None
    condition_observed_implementation_sha256: str | None = None
    condition_adapter_receipt_valid: bool | None = None
    condition_operator_counts: dict[str, int] | None = None
    host_assurance: Literal["native_enforced", "evaluator_wrapped", "cooperative"] = (
        "native_enforced"
    )
    host_transcript_valid: bool | None = None
    host_transcript_sha256: str | None = None
    host_identity_sha256: str | None = None


def _case_id(path: str, module: str, expected: bool, seed: int) -> str:
    raw = f"{seed}\0{path}\0{module}\0{int(expected)}".encode()
    return "case_" + hashlib.sha256(raw).hexdigest()[:12]

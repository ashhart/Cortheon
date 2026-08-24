"""Closed schemas for the repository-only operator-lift instrument."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

OPERATORS = (
    "hypothesis_framing",
    "discriminating_evidence",
    "contradiction_revision",
    "cross_source_derivation",
    "adaptive_stopping",
)
SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 2
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[a-z][a-z0-9_]{2,79}")


def _closed_mapping(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _plain(value: Any, label: str, *, maximum: int = 8_000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} is invalid")
    return value


@dataclass(frozen=True)
class LiftCase:
    case_id: str
    cluster_id: str
    operator: str
    causal_family: str
    prompt: str
    evidence: tuple[tuple[str, str], ...]
    response_schema: Mapping[str, Any]
    oracle: Mapping[str, Any]
    action_catalog: tuple[tuple[str, str, int], ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.case_id, "case_id")
        _identifier(self.cluster_id, "cluster_id")
        _identifier(self.causal_family, "causal_family")
        if self.operator not in OPERATORS:
            raise ValueError("operator is invalid")
        _plain(self.prompt, "prompt")
        if not 2 <= len(self.evidence) <= 6:
            raise ValueError("a lift case needs two to six evidence records")
        source_ids: set[str] = set()
        for source_id, content in self.evidence:
            _identifier(source_id, "source_id")
            _plain(content, "evidence content")
            if source_id in source_ids:
                raise ValueError("source ids must be unique")
            source_ids.add(source_id)
        if not isinstance(self.response_schema, Mapping) or not self.response_schema:
            raise ValueError("response_schema is invalid")
        if not isinstance(self.oracle, Mapping) or not self.oracle:
            raise ValueError("oracle is invalid")
        action_ids: set[str] = set()
        for action_id, description, cost in self.action_catalog:
            _identifier(action_id, "action_id")
            _plain(description, "action description")
            if action_id in action_ids or type(cost) is not int or cost <= 0:
                raise ValueError("action catalog is invalid")
            action_ids.add(action_id)


@dataclass(frozen=True)
class ConditionBinding:
    condition_id: str
    config_sha256: str
    implementation_sha256: str
    disabled_operator: str | None

    def __post_init__(self) -> None:
        _identifier(self.condition_id, "condition_id")
        _digest(self.config_sha256, "config_sha256")
        _digest(self.implementation_sha256, "implementation_sha256")
        if self.disabled_operator is not None and self.disabled_operator not in OPERATORS:
            raise ValueError("disabled_operator is invalid")


@dataclass(frozen=True)
class LiftThresholds:
    repetitions: int = 3
    minimum_clusters: int = 12
    minimum_full_rate: float = 0.80
    minimum_lift: float = 0.10
    familywise_alpha: float = 0.05
    bootstrap_resamples: int = 20_000
    bootstrap_seed: int = 7_311_947

    def __post_init__(self) -> None:
        integer_fields = (
            self.repetitions,
            self.minimum_clusters,
            self.bootstrap_resamples,
            self.bootstrap_seed,
        )
        float_fields = (
            self.minimum_full_rate,
            self.minimum_lift,
            self.familywise_alpha,
        )
        if any(type(value) is not int for value in integer_fields) or any(
            type(value) is not float for value in float_fields
        ):
            raise ValueError("operator-lift threshold types are immutable")
        expected = (3, 12, 0.80, 0.10, 0.05, 20_000, 7_311_947)
        actual = (
            self.repetitions,
            self.minimum_clusters,
            self.minimum_full_rate,
            self.minimum_lift,
            self.familywise_alpha,
            self.bootstrap_resamples,
            self.bootstrap_seed,
        )
        if actual != expected:
            raise ValueError("operator-lift thresholds are immutable preregistered values")

    @property
    def per_contrast_alpha(self) -> float:
        return self.familywise_alpha / (len(OPERATORS) + 1)

    @property
    def per_operator_alpha(self) -> float:
        return self.per_contrast_alpha


@dataclass(frozen=True)
class LiftManifest:
    schema_version: int
    design_id: str
    design_sha256: str
    created_before_execution: bool
    evaluator_id: str
    evaluator_implementation_sha256: str
    thresholds: LiftThresholds
    full_condition: ConditionBinding
    placebo_condition: ConditionBinding
    ablation_conditions: Mapping[str, ConditionBinding]
    case_order: tuple[str, ...]
    cluster_lineage_sha256: Mapping[str, str]
    case_commitments: Mapping[str, str]
    manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != MANIFEST_SCHEMA_VERSION
            or self.created_before_execution is not True
        ):
            raise ValueError("manifest header is invalid")
        _identifier(self.design_id, "design_id")
        _identifier(self.evaluator_id, "evaluator_id")
        _digest(self.evaluator_implementation_sha256, "evaluator_implementation_sha256")
        _digest(self.design_sha256, "design_sha256")
        _digest(self.manifest_sha256, "manifest_sha256")
        if self.full_condition.disabled_operator is not None:
            raise ValueError("full condition cannot disable an operator")
        if self.full_condition.condition_id != "full":
            raise ValueError("full condition id must be full")
        if not isinstance(self.placebo_condition, ConditionBinding):
            raise ValueError("placebo condition is missing")
        if self.placebo_condition.disabled_operator is not None:
            raise ValueError("placebo condition cannot disable one named operator")
        if self.placebo_condition.condition_id != "equal_budget_placebo":
            raise ValueError("placebo condition id is not canonical")
        if set(self.ablation_conditions) != set(OPERATORS):
            raise ValueError("every operator needs one ablation condition")
        for operator, binding in self.ablation_conditions.items():
            if binding.disabled_operator != operator:
                raise ValueError("ablation condition targets the wrong operator")
            if binding.condition_id != f"without_{operator}":
                raise ValueError("ablation condition id is not canonical")
        bindings = (
            self.full_condition,
            self.placebo_condition,
            *self.ablation_conditions.values(),
        )
        if len({binding.condition_id for binding in bindings}) != len(bindings):
            raise ValueError("condition ids must be unique")
        if len({binding.config_sha256 for binding in bindings}) != len(bindings):
            raise ValueError("condition config digests must be distinct")
        if len({binding.implementation_sha256 for binding in bindings}) != 1:
            raise ValueError("conditions must share one frozen implementation digest")
        if not self.case_commitments:
            raise ValueError("manifest has no cases")
        if tuple(self.case_commitments) != self.case_order or len(set(self.case_order)) != len(
            self.case_order
        ):
            raise ValueError("case order is invalid")
        if set(self.cluster_lineage_sha256) != set(self.case_order):
            raise ValueError("cluster lineage set is invalid")
        if len(set(self.cluster_lineage_sha256.values())) != len(self.cluster_lineage_sha256):
            raise ValueError("cluster lineages are not independent")
        for case_id, commitment in self.case_commitments.items():
            _identifier(case_id, "case commitment id")
            _digest(commitment, "case commitment")
            _digest(self.cluster_lineage_sha256[case_id], "cluster lineage")


@dataclass(frozen=True)
class TraceEvent:
    sequence: int
    action_id: str
    observation_sha256: str
    cost: int

    @classmethod
    def from_mapping(cls, value: Any) -> TraceEvent:
        item = _closed_mapping(
            value,
            {"sequence", "action_id", "observation_sha256", "cost"},
            "trace event",
        )
        event = cls(
            sequence=item["sequence"],
            action_id=item["action_id"],
            observation_sha256=item["observation_sha256"],
            cost=item["cost"],
        )
        event.validate()
        return event

    def validate(self) -> None:
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("trace sequence is invalid")
        _identifier(self.action_id, "trace action_id")
        _digest(self.observation_sha256, "trace observation_sha256")
        if type(self.cost) is not int or self.cost <= 0:
            raise ValueError("trace cost is invalid")


@dataclass(frozen=True)
class EvaluatorProvenance:
    schema_version: int
    producer: str
    candidate_supplied: bool
    evaluator_id: str
    evaluator_implementation_sha256: str
    public_projection_sha256: str
    oracle_access_blocked: bool
    trace: tuple[TraceEvent, ...]
    terminal_after_sequence: int
    terminal_reason: str

    @classmethod
    def from_mapping(cls, value: Any) -> EvaluatorProvenance:
        item = _closed_mapping(
            value,
            {
                "schema_version",
                "producer",
                "candidate_supplied",
                "evaluator_id",
                "evaluator_implementation_sha256",
                "public_projection_sha256",
                "oracle_access_blocked",
                "trace",
                "terminal_after_sequence",
                "terminal_reason",
            },
            "evaluator provenance",
        )
        raw_trace = item["trace"]
        if not isinstance(raw_trace, list):
            raise ValueError("evaluator trace must be a list")
        provenance = cls(
            schema_version=item["schema_version"],
            producer=item["producer"],
            candidate_supplied=item["candidate_supplied"],
            evaluator_id=item["evaluator_id"],
            evaluator_implementation_sha256=item["evaluator_implementation_sha256"],
            public_projection_sha256=item["public_projection_sha256"],
            oracle_access_blocked=item["oracle_access_blocked"],
            trace=tuple(TraceEvent.from_mapping(event) for event in raw_trace),
            terminal_after_sequence=item["terminal_after_sequence"],
            terminal_reason=item["terminal_reason"],
        )
        provenance.validate()
        return provenance

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.producer != "evaluator":
            raise ValueError("evaluator provenance header is invalid")
        if self.candidate_supplied is not False or self.oracle_access_blocked is not True:
            raise ValueError("evaluator provenance ownership or isolation is invalid")
        _identifier(self.evaluator_id, "provenance evaluator_id")
        _digest(
            self.evaluator_implementation_sha256,
            "provenance evaluator_implementation_sha256",
        )
        _digest(self.public_projection_sha256, "provenance public_projection_sha256")
        if type(self.terminal_after_sequence) is not int or self.terminal_after_sequence < 0:
            raise ValueError("terminal sequence is invalid")
        if self.terminal_reason not in {"sufficient", "not_applicable"}:
            raise ValueError("terminal reason is invalid")
        for event in self.trace:
            if not isinstance(event, TraceEvent):
                raise ValueError("evaluator trace contains an invalid event")
            event.validate()
        if tuple(event.sequence for event in self.trace) != tuple(range(1, len(self.trace) + 1)):
            raise ValueError("trace sequences are not contiguous")


@dataclass(frozen=True)
class LiftSubmission:
    schema_version: int
    case_id: str
    case_commitment: str
    condition_id: str
    condition_config_sha256: str
    implementation_sha256: str
    repeat: int
    delivered: bool
    safe: bool
    evaluator_provenance: EvaluatorProvenance
    response: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> LiftSubmission:
        item = _closed_mapping(
            value,
            {
                "schema_version",
                "case_id",
                "case_commitment",
                "condition_id",
                "condition_config_sha256",
                "implementation_sha256",
                "repeat",
                "delivered",
                "safe",
                "evaluator_provenance",
                "response",
            },
            "submission",
        )
        submission = cls(
            schema_version=item["schema_version"],
            case_id=item["case_id"],
            case_commitment=item["case_commitment"],
            condition_id=item["condition_id"],
            condition_config_sha256=item["condition_config_sha256"],
            implementation_sha256=item["implementation_sha256"],
            repeat=item["repeat"],
            delivered=item["delivered"],
            safe=item["safe"],
            evaluator_provenance=EvaluatorProvenance.from_mapping(item["evaluator_provenance"]),
            response=item["response"],
        )
        submission.validate()
        return submission

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("submission schema is invalid")
        _identifier(self.case_id, "case_id")
        _digest(self.case_commitment, "case_commitment")
        _identifier(self.condition_id, "condition_id")
        _digest(self.condition_config_sha256, "condition_config_sha256")
        _digest(self.implementation_sha256, "implementation_sha256")
        if type(self.repeat) is not int or self.repeat < 0:
            raise ValueError("repeat is invalid")
        if type(self.delivered) is not bool or type(self.safe) is not bool:
            raise ValueError("delivery and safety must be booleans")
        self.evaluator_provenance.validate()
        if not isinstance(self.response, Mapping):
            raise ValueError("response must be an object")


@dataclass(frozen=True)
class OracleResult:
    correct: bool
    proof_eligible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ScoredRun:
    case_id: str
    cluster_id: str
    operator: str
    condition_id: str
    repeat: int
    correct: bool
    delivered: bool
    safe: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PairedCluster:
    cluster_id: str
    operator: str
    full_scores: tuple[int, ...]
    ablation_scores: tuple[int, ...]
    placebo_scores: tuple[int, ...] = ()

    @property
    def primary_contrast(self) -> str:
        return "full_vs_ablation"

    @property
    def full_rate(self) -> float:
        return sum(self.full_scores) / len(self.full_scores)

    @property
    def ablation_rate(self) -> float:
        return sum(self.ablation_scores) / len(self.ablation_scores)

    @property
    def effect(self) -> float:
        return self.full_rate - self.ablation_rate

    @property
    def placebo_rate(self) -> float:
        return sum(self.placebo_scores) / len(self.placebo_scores)

    @property
    def placebo_effect(self) -> float:
        return self.full_rate - self.placebo_rate


@dataclass(frozen=True)
class PairingResult:
    scored_runs: tuple[ScoredRun, ...]
    clusters: tuple[PairedCluster, ...]
    errors: tuple[str, ...]

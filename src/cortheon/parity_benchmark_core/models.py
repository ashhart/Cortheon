from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cortheon.benchmark_core.outcomes import EvaluationOutcome


@dataclass(frozen=True, slots=True)
class Contender:
    name: str
    kind: str
    base_url: str
    model: str
    api_key: str
    tools: tuple[str, ...] = ()
    command: tuple[str, ...] = ()
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    compute_cost_per_hour: float | None = None
    runtime_sha256: str | None = None
    family: str | None = None


@dataclass(frozen=True, slots=True)
class ModelResult:
    answer: str
    latency_ms: float
    metadata: dict[str, Any]
    evaluator_outcome: EvaluationOutcome


@dataclass(frozen=True, slots=True)
class LoadedCasePack:
    cases: list[dict[str, Any]]
    metadata: dict[str, Any]

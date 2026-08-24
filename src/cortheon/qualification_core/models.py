"""Error and value types shared by every qualification module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cortheon.cognitive_benchmark import RunResult


class QualificationError(ValueError):
    """Raised when a qualification cannot be safely or validly executed."""


@dataclass(frozen=True, slots=True)
class Cell:
    cell_id: str
    suite: str
    host: str
    provider: str
    base_url: str
    api_key_env: str | None
    model_id: str
    runtime_url: str
    cases: int
    repeats: int
    seed: int
    timeout_seconds: float
    context_tokens: int
    output_tokens: int
    max_steps: int
    reasoning: bool
    opencode: str
    pi: str
    condition_ids: tuple[str, ...]
    condition_implementation_sha256: str
    historical_comparison: bool = False


@dataclass(frozen=True, slots=True)
class Manifest:
    path: Path
    digest: str
    tier: str
    repository: Path
    seed: int
    cells: tuple[Cell, ...]
    gates: dict[str, int | float]
    condition_implementation_sha256: str


@dataclass(slots=True)
class CellRun:
    cell: Cell
    case_ids: tuple[str, ...]
    task_digests: dict[str, str]
    results: list[RunResult]
    pairing: dict[str, Any]
    case_deltas: dict[str, float]
    invalid_case_ids: set[str]
    repository_unchanged: bool
    environment_stable: bool
    runtime: dict[str, Any]
    inference: dict[str, Any]
    host_version: str
    evaluator_runtime_source_fingerprint: str | None = None
    evaluator_runtime_protocol: str | None = None
    contrasts: dict[str, dict[str, Any]] = field(default_factory=dict)
    contrast_case_deltas: dict[str, dict[str, float]] = field(default_factory=dict)
    contrast_invalid_case_ids: dict[str, set[str]] = field(default_factory=dict)
    scheduled_repeats: tuple[int, ...] = ()

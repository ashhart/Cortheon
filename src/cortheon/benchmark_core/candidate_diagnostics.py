"""Memory-only diagnostics for Pi candidates blocked before delivery."""

from __future__ import annotations

from typing import Any

from cortheon.benchmark_core.grading import _grade
from cortheon.benchmark_core.models import BenchmarkCase, LongHorizonCase, PatchCase
from cortheon.benchmark_core.outcomes import EvaluationOutcome, is_authenticated_withhold
from cortheon.benchmark_core.run_support import _captured_stage_reason
from cortheon.benchmark_core.transport_outcomes import parse_transport_outcome


def _candidate_correct(
    case: BenchmarkCase,
    events: list[dict[str, Any]],
    *,
    host: str,
    treatment: bool,
    final: str,
    evaluator_outcome: EvaluationOutcome,
) -> bool | None:
    """Grade an exact Pi pre-block candidate without retaining its text."""

    if host != "pi" or not treatment or not is_authenticated_withhold(evaluator_outcome):
        return None
    if isinstance(case, (PatchCase, LongHorizonCase)):
        return None
    parsed = parse_transport_outcome(events, host=host)
    if parsed.final_text != final or parsed.outcome != evaluator_outcome:
        return None
    return _grade(case, parsed.candidate) if parsed.candidate is not None else None


def _causal_stage_reason(
    events: list[dict[str, Any]],
    *,
    host: str,
    treatment: bool,
) -> str | None:
    """Read one bounded Pi causal-stage code from the host-owned channel."""

    return _captured_stage_reason(events) if host == "pi" and treatment else None

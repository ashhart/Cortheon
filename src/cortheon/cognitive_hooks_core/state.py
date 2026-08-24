"""Bounded per-turn hook state, phase naming, and withholding messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cortheon.cognitive_repair import RepairPlan, TestInvocation

CORTHEON_PHASE_TOOLS = {
    "start": "cortheon_start",
    "observe": "cortheon_observe",
    "complete": "cortheon_complete",
    "abandon": "cortheon_abandon",
}
MAX_TOOL_DENIALS_PER_TURN = 3
MAX_STOP_CONTINUATIONS_PER_TURN = 2
MAX_PATCH_STOP_CONTINUATIONS_PER_TURN = 4
MAX_TURN_FAILURES_PER_HOST_SESSION = 3
MAX_HOOK_EVIDENCE_CHARS = 8_000
_FILE_MARKER_PREFIX = "[CORTHEON_FILE:"
UNCERTIFIED_RELEASE_CAVEAT = (
    "Cortheon could not certify this answer: the live evidence contract was not "
    "satisfied within the turn's continuation budget. Treat specific claims as "
    "unverified."
)


def cortheon_tool_phase(tool_name: str) -> str | None:
    """Return the Cortheon lifecycle phase represented by a host tool name."""

    normalized = tool_name.strip().lower()
    for phase, suffix in CORTHEON_PHASE_TOOLS.items():
        if normalized == suffix or normalized.endswith(f"__{suffix}"):
            return phase
    return None


def _bounded_cognition(payload: dict[str, Any]) -> dict[str, Any] | None:
    stage = payload.get("stage")
    if not isinstance(stage, str) or not stage:
        return None
    moves = payload.get("reasoning_moves")
    move = (
        next((item for item in moves if isinstance(item, str) and item), "")
        if isinstance(moves, list)
        else ""
    )
    insights = payload.get("derived_insights")
    insight = ""
    if isinstance(insights, list):
        first = next((item for item in insights if isinstance(item, dict)), None)
        if isinstance(first, dict) and isinstance(first.get("statement"), str):
            insight = first["statement"]
    decision_rule = payload.get("decision_rule")
    return {
        "stage": stage[:32],
        "move": move[:400],
        "derived_insight": insight[:600],
        "decision_rule": (decision_rule[:500] if isinstance(decision_rule, str) else ""),
    }


@dataclass(slots=True)
class HookTurn:
    """Bounded in-memory task state; never persisted by Cortheon."""

    host_session_hash: str
    goal_hash: str = ""
    started: bool = False
    observed: bool = False
    certified: bool = False
    automatic: bool = False
    cortheon_session_id: str | None = None
    pending_request: dict[str, Any] | None = None
    pending_origin: str | None = None
    pending_host_command: str = ""
    pending_host_input: dict[str, Any] = field(default_factory=dict)
    read_snapshots: dict[str, str] = field(default_factory=dict)
    last_next_action: dict[str, Any] | None = None
    cognition: dict[str, Any] | None = None
    evidence_ids: list[str] = field(default_factory=list)
    last_success_condition: str = ""
    awaiting_host_result: bool = False
    deliverable: str = ""
    repair_plan: RepairPlan | None = None
    repair_candidates: list[RepairPlan] = field(default_factory=list)
    repair_candidate_index: int = 0
    test_invocation: TestInvocation | None = None
    check_invocation: TestInvocation | None = None
    check_pending: bool = False
    protected_test_paths: tuple[str, ...] = ()
    protects_all_tests: bool = False
    patch_applied: bool = False
    tool_denials: int = 0
    stop_continuations: int = 0
    updated_at: float = 0.0


def _continuation_reason(state: HookTurn) -> str:
    if state.pending_request is not None:
        query = str(state.pending_request.get("query") or "")[:800]
        return (
            "Cortheon withheld completion. Continue with this host evidence "
            f"request, then answer again: {query}"
        )
    if not state.evidence_ids:
        return (
            "Cortheon withheld completion because no verified host evidence was "
            "captured. Run the task's relevant read, search, diff, or test tool."
        )
    return (
        "Cortheon withheld completion because the evidence contract still has an "
        "unresolved gap. Continue the investigation and answer again."
    )


def _bounded(value: str, field: str, *, maximum: int = 1_024) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(f"{field} exceeds {maximum} characters")
    return normalized

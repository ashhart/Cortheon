"""Memory-only Cortheon lifecycle hooks."""

# ruff: noqa: F401 - this facade intentionally re-exports every import.
# pyright: reportUnsupportedDunderAll=false

from cortheon.cognitive_hooks_core.host_tools import (
    _attempts_protected_mutation,
    _is_apply_patch_tool,
    _is_shell_tool,
    _safe_command,
    _safe_relative_path,
)
from cortheon.cognitive_hooks_core.observations import (
    _host_observations,
    _observation,
    _read_snapshots,
    _split_read_many_output,
)
from cortheon.cognitive_hooks_core.receipts import (
    _classify_host_tool,
    _host_receipt_arguments,
    _read_path_from_command,
)
from cortheon.cognitive_hooks_core.state import (
    _FILE_MARKER_PREFIX,
    CORTHEON_PHASE_TOOLS,
    MAX_HOOK_EVIDENCE_CHARS,
    MAX_PATCH_STOP_CONTINUATIONS_PER_TURN,
    MAX_STOP_CONTINUATIONS_PER_TURN,
    MAX_TOOL_DENIALS_PER_TURN,
    MAX_TURN_FAILURES_PER_HOST_SESSION,
    UNCERTIFIED_RELEASE_CAVEAT,
    HookTurn,
    _bounded,
    _bounded_cognition,
    _continuation_reason,
    cortheon_tool_phase,
)
from cortheon.cognitive_hooks_core.tracker import CognitiveHookTracker

# Re-exported for callers that reach the repair contract through this module.
from cortheon.cognitive_repair import (
    RepairPlan,
    TestInvocation,
    changed_paths_from_diff,
    derive_repair_candidates,
    is_test_path,
    protected_test_paths,
    protects_tests,
    requested_check_invocation,
    requested_test_invocation,
)

__all__ = [name for name in globals() if not name.startswith("__")]

"""Architecture guards for the cognitive hook decomposition.

The hook module must not grow back into a god file: the facade stays a thin
re-export surface, every implementation module in ``cognitive_hooks_core`` and
every split test module stays focused, each top-level definition and each
``CognitiveHookTracker`` method has exactly one implementation owner, and the
compatibility surface keeps resolving to identical objects.
"""

from __future__ import annotations

import ast
import importlib
import types
from pathlib import Path

import pytest

import cortheon.cognitive_hooks as hooks

ROOT = Path(__file__).parents[1]
FACADE = ROOT / "src/cortheon/cognitive_hooks.py"
CORE_DIR = ROOT / "src/cortheon/cognitive_hooks_core"
TEST_MODULES = sorted(
    [
        *Path(__file__).parent.glob("test_cognitive_hooks_*.py"),
        Path(__file__).parent / "cognitive_hooks_helpers.py",
    ]
)
LINE_CAP = 500

# Every top-level definition of the original 1,593-line cognitive_hooks.py.
ORIGINAL_DEFINITIONS = {
    "CORTHEON_PHASE_TOOLS",
    "CognitiveHookTracker",
    "HookTurn",
    "MAX_HOOK_EVIDENCE_CHARS",
    "MAX_PATCH_STOP_CONTINUATIONS_PER_TURN",
    "MAX_STOP_CONTINUATIONS_PER_TURN",
    "MAX_TOOL_DENIALS_PER_TURN",
    "MAX_TURN_FAILURES_PER_HOST_SESSION",
    "UNCERTIFIED_RELEASE_CAVEAT",
    "_FILE_MARKER_PREFIX",
    "_attempts_protected_mutation",
    "_bounded",
    "_bounded_cognition",
    "_classify_host_tool",
    "_continuation_reason",
    "_host_observations",
    "_host_receipt_arguments",
    "_is_apply_patch_tool",
    "_is_shell_tool",
    "_observation",
    "_read_path_from_command",
    "_read_snapshots",
    "_safe_command",
    "_safe_relative_path",
    "_split_read_many_output",
    "cortheon_tool_phase",
}
# Names the original module re-exported by importing them from cognitive_repair;
# callers could always reach them through ``cortheon.cognitive_hooks``.
REPAIR_REEXPORTS = {
    "RepairPlan",
    "TestInvocation",
    "changed_paths_from_diff",
    "derive_repair_candidates",
    "is_test_path",
    "protected_test_paths",
    "protects_tests",
    "requested_check_invocation",
    "requested_test_invocation",
}
ORIGINAL_SURFACE = ORIGINAL_DEFINITIONS | REPAIR_REEXPORTS

# The only top-level additions beyond the original surface are mixin
# scaffolding with a single concrete owner for the original tracker.
SCAFFOLDING = {
    "TrackerBase",
    "ResponseMixin",
    "RegistrationMixin",
    "LifecycleMixin",
    "AutomaticMixin",
    "PatchLoopMixin",
}
WEB_EVIDENCE_HELPERS = {
    "_failed_web_observation",
    "_nested_web_executor",
    "_normalized_web_url",
    "_published_at",
    "_web_observations",
    "_web_receipt",
}
_TRACKER_CLASSES = SCAFFOLDING | {"CognitiveHookTracker"}

# Methods of the original monolithic CognitiveHookTracker.
ORIGINAL_TRACKER_METHODS = {
    "__init__",
    "_automatic_edit_pre_tool",
    "_automatic_post_tool",
    "_automatic_pre_tool",
    "_automatic_stop",
    "_capture_next_action",
    "_clear_pending_request",
    "_deny_requested_capability",
    "_discard_runtime_session",
    "_evict_oldest",
    "_host_hash",
    "_identifiers",
    "_manual_stop",
    "_mark_certified",
    "_nudge_start",
    "_protected_diff_paths",
    "_public_state",
    "_purge_expired",
    "_record_turn_failure",
    "_release_uncertified",
    "_response",
    "_schedule_adapter_request",
    "active_turns",
    "end_session",
    "metrics",
    "post_tool",
    "pre_tool",
    "register",
    "stop",
}


def _top_level_definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _core_modules() -> list[Path]:
    modules = sorted(CORE_DIR.glob("*.py"))
    assert modules, "cognitive_hooks_core package must exist"
    return modules


def _core_sources() -> list[tuple[str, str]]:
    return [(path.name, path.read_text(encoding="utf-8")) for path in _core_modules()]


def _is_contract_stub(item: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    # A contract stub is a body consisting solely of ``raise NotImplementedError``
    # (with or without parentheses); it owns nothing.
    if len(item.body) != 1 or not isinstance(item.body[0], ast.Raise):
        return False
    exc = item.body[0].exc
    if isinstance(exc, ast.Name):
        return exc.id == "NotImplementedError"
    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
        return exc.func.id == "NotImplementedError"
    return False


def _concrete_method_owners(
    sources: list[tuple[str, str]], class_names: set[str]
) -> dict[str, tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Collect the single concrete owner of each method across ALL modules.

    A second module defining a concrete (non-stub) body for a method that is
    already owned somewhere else fails loudly instead of silently winning.
    """
    owners: dict[str, tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for module_name, source in sources:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name not in class_names:
                continue
            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if _is_contract_stub(item):
                    continue
                assert item.name not in owners, (
                    f"{item.name} has concrete owners in both "
                    f"{owners[item.name][0]} and {module_name}"
                )
                owners[item.name] = (module_name, item)
    return owners


def test_facade_is_a_small_stable_surface() -> None:
    assert len(FACADE.read_text(encoding="utf-8").splitlines()) <= 150
    assert not _top_level_definitions(FACADE) - {"__all__"}, "facade must not own definitions"


def test_every_edited_module_stays_under_the_line_cap() -> None:
    for module in [FACADE, *_core_modules(), *TEST_MODULES]:
        line_count = len(module.read_text(encoding="utf-8").splitlines())
        assert line_count <= LINE_CAP, f"{module.name} has {line_count} lines"


def test_the_pre_split_god_file_is_gone() -> None:
    assert not (ROOT / "tests/test_cognitive_hooks.py").exists()
    assert len(TEST_MODULES) >= 6


def test_single_owner_for_top_level_definitions() -> None:
    owners: dict[str, Path] = {}
    for path in [FACADE, *_core_modules()]:
        for name in _top_level_definitions(path):
            if name in {"__all__", "__init__"}:
                continue
            assert name not in owners, (
                f"{name} is defined in both {owners[name].name} and {path.name}"
            )
            owners[name] = path


def test_every_original_definition_is_owned_exactly_once() -> None:
    corpus: set[str] = set()
    for path in _core_modules():
        corpus |= _top_level_definitions(path)
    assert corpus >= ORIGINAL_DEFINITIONS, sorted(ORIGINAL_DEFINITIONS - corpus)
    additions = SCAFFOLDING | WEB_EVIDENCE_HELPERS
    assert corpus - ORIGINAL_DEFINITIONS == additions, sorted(
        corpus - ORIGINAL_DEFINITIONS - additions
    )


def test_every_original_method_has_one_implementation_owner() -> None:
    owners = _concrete_method_owners(_core_sources(), _TRACKER_CLASSES)
    assert set(owners) == ORIGINAL_TRACKER_METHODS


def test_duplicate_concrete_owner_in_a_second_module_is_rejected() -> None:
    # Mutation-style guard: a per-file scan merged with ``dict.update`` would
    # silently overwrite a duplicate concrete body defined in another module.
    # Stubs raising NotImplementedError only must never count as owners.
    module_a = "class LifecycleMixin:\n    def stop(self, key):\n        return 'stop'\n"
    module_b_stub_only = (
        "class TrackerBase:\n    def stop(self, key):\n        raise NotImplementedError()\n"
    )
    module_b_concrete = "class AutomaticMixin:\n    def stop(self, key):\n        return 'dup'\n"
    owners = _concrete_method_owners(
        [("lifecycle.py", module_a), ("tracker_base.py", module_b_stub_only)],
        _TRACKER_CLASSES,
    )
    assert owners["stop"][0] == "lifecycle.py"
    with pytest.raises(AssertionError, match="concrete owners in both"):
        _concrete_method_owners(
            [("lifecycle.py", module_a), ("automatic.py", module_b_concrete)],
            _TRACKER_CLASSES,
        )


def test_facade_exports_exactly_the_original_public_surface() -> None:
    facade_attrs = {
        name
        for name in vars(hooks)
        if not name.startswith("__") and not isinstance(getattr(hooks, name), types.ModuleType)
    }
    assert facade_attrs == ORIGINAL_SURFACE, (
        f"missing: {sorted(ORIGINAL_SURFACE - facade_attrs)}, "
        f"extra: {sorted(facade_attrs - ORIGINAL_SURFACE)}"
    )
    assert set(hooks.__all__) == ORIGINAL_SURFACE
    assert len(hooks.__all__) == 35


def test_facade_reexports_are_identical_to_core_definitions() -> None:
    from cortheon import cognitive_repair
    from cortheon.cognitive_hooks_core import (
        host_tools,
        observations,
        receipts,
        state,
        tracker,
    )

    identity_pairs = [
        (hooks.CognitiveHookTracker, tracker.CognitiveHookTracker),
        (hooks.HookTurn, state.HookTurn),
        (hooks.CORTHEON_PHASE_TOOLS, state.CORTHEON_PHASE_TOOLS),
        (hooks.MAX_HOOK_EVIDENCE_CHARS, state.MAX_HOOK_EVIDENCE_CHARS),
        (
            hooks.MAX_PATCH_STOP_CONTINUATIONS_PER_TURN,
            state.MAX_PATCH_STOP_CONTINUATIONS_PER_TURN,
        ),
        (hooks.MAX_STOP_CONTINUATIONS_PER_TURN, state.MAX_STOP_CONTINUATIONS_PER_TURN),
        (hooks.MAX_TOOL_DENIALS_PER_TURN, state.MAX_TOOL_DENIALS_PER_TURN),
        (hooks.MAX_TURN_FAILURES_PER_HOST_SESSION, state.MAX_TURN_FAILURES_PER_HOST_SESSION),
        (hooks.UNCERTIFIED_RELEASE_CAVEAT, state.UNCERTIFIED_RELEASE_CAVEAT),
        (hooks._FILE_MARKER_PREFIX, state._FILE_MARKER_PREFIX),
        (hooks.cortheon_tool_phase, state.cortheon_tool_phase),
        (hooks._bounded, state._bounded),
        (hooks._bounded_cognition, state._bounded_cognition),
        (hooks._continuation_reason, state._continuation_reason),
        (hooks._is_shell_tool, host_tools._is_shell_tool),
        (hooks._is_apply_patch_tool, host_tools._is_apply_patch_tool),
        (hooks._safe_command, host_tools._safe_command),
        (hooks._safe_relative_path, host_tools._safe_relative_path),
        (hooks._attempts_protected_mutation, host_tools._attempts_protected_mutation),
        (hooks._classify_host_tool, receipts._classify_host_tool),
        (hooks._host_receipt_arguments, receipts._host_receipt_arguments),
        (hooks._read_path_from_command, receipts._read_path_from_command),
        (hooks._host_observations, observations._host_observations),
        (hooks._observation, observations._observation),
        (hooks._split_read_many_output, observations._split_read_many_output),
        (hooks._read_snapshots, observations._read_snapshots),
        (hooks.RepairPlan, cognitive_repair.RepairPlan),
        (hooks.TestInvocation, cognitive_repair.TestInvocation),
        (hooks.changed_paths_from_diff, cognitive_repair.changed_paths_from_diff),
        (hooks.derive_repair_candidates, cognitive_repair.derive_repair_candidates),
        (hooks.is_test_path, cognitive_repair.is_test_path),
        (hooks.protected_test_paths, cognitive_repair.protected_test_paths),
        (hooks.protects_tests, cognitive_repair.protects_tests),
        (hooks.requested_check_invocation, cognitive_repair.requested_check_invocation),
        (hooks.requested_test_invocation, cognitive_repair.requested_test_invocation),
    ]
    assert len(identity_pairs) == 35
    for facade_object, core_object in identity_pairs:
        assert facade_object is core_object


def test_tracker_composes_every_mixin_once() -> None:
    mro = [cls.__name__ for cls in hooks.CognitiveHookTracker.__mro__]
    assert mro[0] == "CognitiveHookTracker" and mro[-1] == "object"
    assert set(mro) == _TRACKER_CLASSES | {"object"}
    assert all(mro.count(name) == 1 for name in _TRACKER_CLASSES)
    # The contract stubs must never survive composition.
    for name in ("_automatic_pre_tool", "_automatic_post_tool", "_automatic_stop"):
        owner = next(cls for cls in hooks.CognitiveHookTracker.__mro__ if name in vars(cls))
        assert owner.__name__ != "TrackerBase", f"{name} still resolves to its stub"


def test_every_core_module_is_importable() -> None:
    for module_path in _core_modules():
        if module_path.name == "__init__.py":
            continue
        module = importlib.import_module(f"cortheon.cognitive_hooks_core.{module_path.stem}")
        assert module.__name__.endswith(module_path.stem)

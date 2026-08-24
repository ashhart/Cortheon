"""Architecture and cache-deployment contract for the standalone Codex hook."""

from __future__ import annotations

import ast
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from codex_hook_support import (
    HOOK_SOURCE_FILES as SOURCE_FILES,
)
from codex_hook_support import (
    HOOKS,
    SESSION_ID,
    hook_runtime,
    run_hook,
    runtime_requests,
)

from cortheon.codex_plugins.cortheon.hooks import cortheon_hook as facade

PACKAGE = "cortheon.codex_plugins.cortheon.hooks"
ORIGINAL_OWNERS = {
    "_bind_facade": "hook_transport",
    "_facade": "hook_transport",
    "_configured_strictness": "hook_config",
    "_use_compact_context": "hook_config",
    "_payload": "hook_transport",
    "_runtime_url": "hook_transport",
    "_runtime_healthy": "hook_transport",
    "_expected_runtime_identity": "hook_transport",
    "_runtime_command": "hook_transport",
    "_ensure_runtime": "hook_transport",
    "_post": "hook_transport",
    "_identity": "hook_transport",
    "_is_cortheon_skill_bootstrap": "hook_transport",
    "_tool_succeeded": "hook_transport",
    "_tool_output": "hook_transport",
    "_tool_metadata": "hook_transport",
    "_contains_certified_completion": "hook_transport",
    "_host_adapter_timeout": "hook_adapter",
    "_safe_relative_token": "hook_adapter",
    "_safe_test_argv": "hook_adapter",
    "_host_adapter_argv": "hook_adapter",
    "_host_adapter_environment": "hook_adapter",
    "_host_adapter_output": "hook_adapter",
    "_run_host_adapter_step": "hook_adapter",
    "_user_prompt_submit": "hook_events",
    "_pre_tool_use": "hook_events",
    "_post_tool_use": "hook_events",
    "_stop": "hook_events",
    "_session_end": "hook_events",
    "main": "hook_entry",
}
CONSTANT_OWNERS = {
    "MAX_INPUT_CHARS": "hook_config",
    "RUNTIME_TIMEOUT_SECONDS": "hook_config",
    "RUNTIME_HEALTH_TIMEOUT_SECONDS": "hook_config",
    "RUNTIME_START_ATTEMPTS": "hook_config",
    "RUNTIME_START_INTERVAL_SECONDS": "hook_config",
    "MAX_HOST_ADAPTER_STEPS": "hook_config",
    "MAX_HOST_ADAPTER_OUTPUT_CHARS": "hook_config",
    "DEFAULT_HOST_ADAPTER_TIMEOUT_SECONDS": "hook_config",
    "SUBSTANTIVE_RE": "hook_config",
    "COMPLETE_STATUS_RE": "hook_config",
    "SENSITIVE_ENV_RE": "hook_config",
    "CORTHEON_CONTEXT": "hook_config",
    "CORTHEON_AUTO_CONTEXT": "hook_config",
    "CORTHEON_COMPACT_CONTEXT": "hook_config",
    "CORTHEON_COMPACT_AUTO_CONTEXT": "hook_config",
    "CORTHEON_UNAVAILABLE_CONTEXT": "hook_config",
    "EXPECTED_RUNTIME_PROTOCOL": "hook_config",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _local_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ImportFrom) and node.module:
            leaf = node.module.rsplit(".", 1)[-1]
            if leaf.startswith("hook_"):
                imports.add(f"{leaf}.py")
        elif isinstance(node, ast.Import):
            imports.update(
                f"{alias.name}.py" for alias in node.names if alias.name.startswith("hook_")
            )
    return imports


def test_hook_source_membership_and_command_are_exact() -> None:
    assert {path.name for path in HOOKS.glob("*.py")} == SOURCE_FILES
    hooks = json.loads((HOOKS / "hooks.json").read_text(encoding="utf-8"))
    serialized = json.dumps(hooks, sort_keys=True)
    assert "hooks/cortheon_hook.py" in serialized
    assert all(name not in serialized for name in SOURCE_FILES - {"cortheon_hook.py"})


def test_every_hook_module_is_focused_and_standalone() -> None:
    for name in SOURCE_FILES:
        path = HOOKS / name
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 500, name
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("cortheon"), (name, node.lineno)
            elif isinstance(node, ast.Import):
                assert all(not alias.name.startswith("cortheon") for alias in node.names), (
                    name,
                    node.lineno,
                )


def test_local_import_graph_is_reachable_and_acyclic() -> None:
    graph = {name: _local_imports(HOOKS / name) for name in SOURCE_FILES}
    reachable: set[str] = set()

    def visit(name: str, active: tuple[str, ...] = ()) -> None:
        assert name not in active, " -> ".join((*active, name))
        if name in reachable:
            return
        reachable.add(name)
        for dependency in graph[name]:
            assert dependency in SOURCE_FILES
            visit(dependency, (*active, name))

    visit("cortheon_hook.py")
    assert reachable == SOURCE_FILES


def test_original_definitions_have_one_owner_and_facade_identity() -> None:
    for name, owner in {**ORIGINAL_OWNERS, **CONSTANT_OWNERS}.items():
        implementation = importlib.import_module(f"{PACKAGE}.{owner}")
        assert getattr(facade, name) is getattr(implementation, name), name
    concrete = {
        node.name: path.name
        for path in HOOKS.glob("*.py")
        if path.name != "cortheon_hook.py"
        for node in _tree(path).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in ORIGINAL_OWNERS
    }
    assert {name: owner.removesuffix(".py") for name, owner in concrete.items()} == ORIGINAL_OWNERS
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in _tree(HOOKS / "cortheon_hook.py").body
    )


def _stage_hook(tmp_path: Path) -> Path:
    staged = tmp_path / "cached-plugin" / "hooks"
    staged.mkdir(parents=True)
    for name in {*SOURCE_FILES, "hooks.json"}:
        shutil.copy2(HOOKS / name, staged / name)
    return staged


def test_exact_staged_directory_runs_without_cortheon_package(tmp_path: Path) -> None:
    staged = _stage_hook(tmp_path)
    with hook_runtime() as runtime:
        submitted = run_hook(
            staged,
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": SESSION_ID,
                "turn_id": "turn",
                "prompt": "Read src/example.py and explain the current implementation.",
            },
            runtime=runtime,
        )
        stopped = run_hook(
            staged,
            {
                "hook_event_name": "Stop",
                "session_id": SESSION_ID,
                "turn_id": "turn",
                "last_assistant_message": "Unverified answer.",
            },
            runtime=runtime,
        )
        recorded = runtime_requests()
    assert submitted.returncode == stopped.returncode == 0
    assert submitted.stderr == stopped.stderr == ""
    prompt_output = json.loads(submitted.stdout)
    assert prompt_output["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert SESSION_ID not in submitted.stdout
    assert json.loads(stopped.stdout) == {
        "decision": "block",
        "reason": "staged completion withheld",
    }
    assert [path for path, _payload in recorded] == [
        "/v1/hooks/register",
        "/v1/hooks/stop",
    ]
    assert recorded[0][1]["host"] == "codex"


def test_staged_hook_keeps_malformed_input_fail_open(tmp_path: Path) -> None:
    staged = _stage_hook(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-I", str(staged / "cortheon_hook.py")],
        input="not-json",
        capture_output=True,
        text=True,
        timeout=5,
        cwd=staged.parent,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout == completed.stderr == ""

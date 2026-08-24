from __future__ import annotations

import ast
import inspect
import re
import sqlite3
from pathlib import Path
from unittest import mock

import cortheon.experience as facade
import cortheon.telemetry as telemetry

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src" / "cortheon" / "experience_core"
EXPECTED_CORE_FILES = {
    "__init__.py",
    "_compat.py",
    "models.py",
    "persistence.py",
    "store.py",
    "taxonomy.py",
    "validation.py",
}
ORIGINAL_DEFINITIONS = {
    "ExperienceStore",
    "FailureSignature",
    "RecoveryStrategy",
    "VerificationContract",
    "_assurance_for_rank",
    "_identifier",
    "_identifiers",
    "_latency_bucket",
    "_limit",
    "_looks_secret",
    "_namespace",
    "_rate",
    "_recovery_rate",
    "classify_experience_task",
}
STORE_METHODS = {
    "__init__",
    "_append",
    "_connect",
    "_ensure",
    "_strategies",
    "capability_outcomes",
    "lessons_for",
    "record_attempt",
    "record_failure",
    "record_recovery",
    "relevant_lessons",
}


def _definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _core_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    prefix = "cortheon.experience_core."
    return {
        node.module.removeprefix(prefix).split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix)
    }


def _failed_contract() -> facade.VerificationContract:
    return facade.VerificationContract(
        assurance="behavioral",
        required_checks=("target_test",),
        passed_checks=(),
        evidence_kinds=("test_result",),
        evidence_count=1,
    )


def _signature() -> facade.FailureSignature:
    return facade.FailureSignature(
        capability="repository_patch",
        task_family="python_bugfix",
        stage="verification",
        failure_kind="test_failure",
    )


def test_facade_and_every_experience_file_stay_below_cap() -> None:
    source = ROOT / "src" / "cortheon" / "experience.py"
    authored = [source, *CORE.glob("*.py")]
    counts = {path.name: len(path.read_text(encoding="utf-8").splitlines()) for path in authored}

    assert {path.name for path in CORE.glob("*.py")} == EXPECTED_CORE_FILES
    assert counts["experience.py"] <= 150
    assert all(count <= 500 for count in counts.values()), counts


def test_original_definitions_have_one_core_owner_and_stable_module_identity() -> None:
    owners = {
        name: [path.name for path in CORE.glob("*.py") if name in _definitions(path)]
        for name in ORIGINAL_DEFINITIONS
    }

    assert all(len(paths) == 1 for paths in owners.values()), owners
    for name in facade.__all__:
        value = getattr(facade, name)
        if callable(value):
            assert value.__module__ == "cortheon.experience"
    for name in STORE_METHODS:
        assert getattr(facade.ExperienceStore, name).__module__ == "cortheon.experience"
    assert facade.FailureSignature.key.fget.__module__ == "cortheon.experience"
    assert facade.VerificationContract.satisfied.fget.__module__ == "cortheon.experience"
    assert facade.VerificationContract.from_outcome.__module__ == "cortheon.experience"


def test_experience_core_import_graph_is_acyclic() -> None:
    graph = {path.stem: _core_imports(path) for path in CORE.glob("*.py")}

    def visit(name: str, active: set[str], visited: set[str]) -> None:
        assert name not in active, f"experience_core import cycle through {name}"
        if name in visited:
            return
        active.add(name)
        for dependency in graph.get(name, set()):
            visit(dependency, active, visited)
        active.remove(name)
        visited.add(name)

    visited: set[str] = set()
    for module in graph:
        visit(module, set(), visited)


def test_public_signatures_remain_stable() -> None:
    assert str(inspect.signature(facade.classify_experience_task)) == (
        "(task: 'str') -> 'tuple[str, str, tuple[str, ...]]'"
    )
    assert str(inspect.signature(facade.ExperienceStore)) == (
        "(path: 'str | Path', *, namespace: 'str', max_events: 'int' = 100000) -> 'None'"
    )
    assert str(inspect.signature(facade.FailureSignature)) == (
        "(capability: 'str', task_family: 'str', stage: 'str', failure_kind: 'str', "
        "failure_code: 'str' = 'unspecified', context_tags: 'tuple[str, ...]' = ()) -> None"
    )
    assert str(inspect.signature(facade.RecoveryStrategy)) == (
        "(strategy_id: 'str', action_ids: 'tuple[str, ...]') -> None"
    )


def test_direct_consumers_receive_the_same_public_objects() -> None:
    assert telemetry.VerificationContract is facade.VerificationContract


def test_facade_classifier_and_store_patch_points_drive_moved_code(tmp_path: Path) -> None:
    with mock.patch.object(facade, "_PACKAGE_TASK", re.compile(r"never-match")):
        assert facade.classify_experience_task("inspect package API") == (
            "general_reasoning",
            "general_question",
            (),
        )

    database = tmp_path / "experience.db"
    store = facade.ExperienceStore(database, namespace="tenant_alpha")
    with mock.patch.object(facade, "_latency_bucket", return_value="patched_bucket"):
        store.record_failure(_signature(), verification=_failed_contract(), latency_ms=1)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT latency_bucket FROM experience_events").fetchone()[0] == (
            "patched_bucket"
        )


def test_store_schema_cannot_retain_project_or_conversation_content(tmp_path: Path) -> None:
    database = tmp_path / "experience.db"
    facade.ExperienceStore(database, namespace="tenant_alpha")
    with sqlite3.connect(database) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(experience_events)").fetchall()
        }

    forbidden = {
        "answer",
        "completion",
        "file_content",
        "file_path",
        "prompt",
        "reasoning_trace",
        "source_text",
        "tool_output",
    }
    assert columns.isdisjoint(forbidden)


def test_experience_split_remains_repository_only() -> None:
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert '"experience"' not in setup
    for config in ("setup.py", "MANIFEST.in", "pyproject.toml"):
        assert "experience_core" not in (ROOT / config).read_text(encoding="utf-8")

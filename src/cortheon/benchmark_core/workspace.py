"""Isolated workspace construction, preparation, and patch grading."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

from cortheon.benchmark_core.models import (
    IGNORED_WORKSPACE_NAMES,
    DiagnosticCase,
    LongHorizonCase,
    PatchCase,
    PlanningCase,
    ReasoningCase,
    SemanticCase,
)


def _python_has_unreachable_statement(source: str) -> bool:
    tree = ast.parse(source)
    terminal = (ast.Return, ast.Raise, ast.Break, ast.Continue)
    for node in ast.walk(tree):
        for _field, value in ast.iter_fields(node):
            if not isinstance(value, list) or not value:
                continue
            if not all(isinstance(item, ast.stmt) for item in value):
                continue
            stopped = False
            for statement in value:
                if stopped:
                    return True
                if isinstance(statement, terminal):
                    stopped = True
    return False


def _prepare_patch_case(case: PatchCase | LongHorizonCase, workspace: Path) -> None:
    for relative, content in case.files:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _prepare_semantic_case(
    case: SemanticCase | DiagnosticCase | PlanningCase | ReasoningCase,
    workspace: Path,
) -> None:
    for relative, content in case.files:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _grade_patch_workspace(
    case: PatchCase | LongHorizonCase,
    workspace: Path,
) -> tuple[bool, str | None]:
    originals = dict(case.files)
    for relative in case.protected_paths:
        try:
            current = (workspace / relative).read_text(encoding="utf-8")
        except OSError as exc:
            return False, f"protected test unreadable: {relative}: {exc}"
        if current != originals[relative]:
            return False, f"protected test changed: {relative}"

    if isinstance(case, LongHorizonCase):
        for relative in case.required_paths:
            try:
                current = (workspace / relative).read_text(encoding="utf-8")
            except OSError as exc:
                return False, f"required deliverable unreadable: {relative}: {exc}"
            if current == originals[relative]:
                return False, f"required deliverable unchanged: {relative}"
        for relative, required in case.required_content:
            try:
                current = (workspace / relative).read_text(encoding="utf-8")
            except OSError as exc:
                return False, f"required content unreadable: {relative}: {exc}"
            if required not in current:
                return False, f"required content missing from {relative}: {required}"

    with tempfile.TemporaryDirectory(prefix="cortheon-pycache-") as pycache:
        grader_environment = os.environ.copy()
        grader_environment["PYTHONPYCACHEPREFIX"] = pycache
        try:
            visible = subprocess.run(
                shlex.split(case.test_command),
                cwd=workspace,
                env=grader_environment,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"visible test could not run: {exc}"
        if visible.returncode != 0:
            return (
                False,
                f"visible test failed: {visible.stdout[-500:]}{visible.stderr[-500:]}",
            )

        hidden = subprocess.run(
            [sys.executable, "-c", case.hidden_assertions],
            cwd=workspace,
            env=grader_environment,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if hidden.returncode != 0:
            return False, f"hidden behavior check failed: {hidden.stderr[-800:]}"

    for relative, _content in case.files:
        if relative in case.protected_paths or not relative.endswith(".py"):
            continue
        try:
            source = (workspace / relative).read_text(encoding="utf-8")
            if _python_has_unreachable_statement(source):
                return False, f"unreachable statement remains in {relative}"
        except (OSError, SyntaxError, UnicodeError) as exc:
            return False, f"implementation is not valid Python: {relative}: {exc}"
    return True, None


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in IGNORED_WORKSPACE_NAMES or name.endswith((".egg-info", ".pyc", ".pyo"))
    }


def _workspace_environment(
    environment: dict[str, str],
    workspace: Path,
) -> dict[str, str]:
    isolated = environment.copy()
    isolated["PWD"] = str(workspace)
    isolated["INIT_CWD"] = str(workspace)
    isolated.pop("OLDPWD", None)
    return isolated


@contextlib.contextmanager
def isolated_repository(
    repository: Path,
    *,
    minimal: bool = False,
) -> Iterator[Path]:
    """Yield an ephemeral copy so a model cannot mutate the live project."""

    with tempfile.TemporaryDirectory(prefix="cortheon-benchmark-") as scratch:
        workspace = Path(scratch) / "repository"
        if minimal:
            workspace.mkdir()
        else:
            shutil.copytree(
                repository,
                workspace,
                ignore=_copy_ignore,
                symlinks=True,
            )
        yield workspace


def _repository_fingerprint(repository: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in repository.rglob("*") if item.is_file()):
        relative = path.relative_to(repository)
        if any(part in IGNORED_WORKSPACE_NAMES for part in relative.parts):
            continue
        if path.name.endswith((".pyc", ".pyo")):
            continue
        digest.update(relative.as_posix().encode())
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
    return digest.hexdigest()

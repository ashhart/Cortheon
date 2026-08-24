from __future__ import annotations

import ast
import contextlib
import re
import sys
import tomllib
from pathlib import Path

from cortheon.models import Evidence, RepoDependency, RepoReport, SupportLevel, utc_now
from cortheon.verifier import guess_import_name

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".nox",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".eggs",
    ".idea",
    ".vscode",
    ".cortheon",
    "dist",
    "build",
}
MAX_PYTHON_FILES = 400
REQUIREMENTS_FILES = (
    "requirements.txt",
    "requirements-dev.txt",
    "requirements_dev.txt",
    "dev-requirements.txt",
    "requirements-test.txt",
)
LOCKFILES = ("uv.lock", "poetry.lock", "Pipfile.lock", "pdm.lock")
FRAMEWORK_NAMES = {
    "fastapi",
    "django",
    "flask",
    "litestar",
    "starlette",
    "pydantic",
    "sqlalchemy",
    "celery",
    "numpy",
    "pandas",
    "torch",
    "httpx",
    "requests",
}
REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$")


def scan_repo(root: str | Path) -> RepoReport:
    """Read a local repository into judgment-ready context.

    This is what lets the gate answer "should X go into THIS repo" instead of
    "is X good in a vacuum": declared dependencies and constraints, the Python
    floor, how tests run, and what the code actually imports.
    """
    resolved = Path(root).expanduser().resolve()
    errors: list[str] = []
    if not resolved.is_dir():
        return _empty_report(
            str(resolved), [f"Repository path does not exist or is not a directory: {resolved}"]
        )

    managers: list[str] = []
    declared: list[RepoDependency] = []
    python_requirement: str | None = None

    pyproject_path = resolved / "pyproject.toml"
    if pyproject_path.is_file():
        managers.append("pyproject")
        try:
            payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
            payload = {}
            errors.append(f"pyproject.toml could not be parsed: {exc}")
        project = payload.get("project") or {}
        python_requirement = project.get("requires-python") or None
        for item in project.get("dependencies") or []:
            dependency = parse_requirement_line(str(item), source="pyproject:project.dependencies")
            if dependency:
                declared.append(dependency)
        for group, items in (project.get("optional-dependencies") or {}).items():
            for item in items or []:
                dependency = parse_requirement_line(str(item), source=f"pyproject:optional.{group}")
                if dependency:
                    declared.append(dependency)
        poetry = (payload.get("tool") or {}).get("poetry") or {}
        if poetry:
            managers.append("poetry")
            for name, spec in (poetry.get("dependencies") or {}).items():
                if name.lower() == "python":
                    python_requirement = python_requirement or (
                        spec if isinstance(spec, str) else None
                    )
                    continue
                constraint = spec if isinstance(spec, str) else None
                declared.append(
                    RepoDependency(name=name, constraint=constraint, source="pyproject:tool.poetry")
                )

    for filename in REQUIREMENTS_FILES:
        requirements_path = resolved / filename
        if not requirements_path.is_file():
            continue
        managers.append("requirements")
        for line in requirements_path.read_text(encoding="utf-8", errors="replace").splitlines():
            dependency = parse_requirement_line(line, source=filename)
            if dependency:
                declared.append(dependency)

    if (resolved / "setup.py").is_file() or (resolved / "setup.cfg").is_file():
        managers.append("setuptools")

    lockfiles = [name for name in LOCKFILES if (resolved / name).is_file()]
    if lockfiles and "uv.lock" in lockfiles:
        managers.append("uv")
    managers = list(dict.fromkeys(managers))

    src_layout = (resolved / "src").is_dir()
    test_commands = detect_test_commands(resolved, src_layout)

    python_files, imported_roots, local_modules = scan_imports(resolved)
    declared_import_names = {guess_import_name(item.name).lower() for item in declared}
    declared_names = {normalize_dep_name(item.name) for item in declared}
    stdlib = {name.lower() for name in sys.stdlib_module_names}
    third_party = sorted(
        name
        for name in imported_roots
        if name.lower() not in stdlib and name.lower() not in local_modules
    )
    undeclared = sorted(name for name in third_party if name.lower() not in declared_import_names)
    imported_lower = {name.lower() for name in imported_roots}
    unused = sorted(
        item.name
        for item in declared
        if item.source.startswith(("pyproject:project", "pyproject:tool.poetry"))
        and guess_import_name(item.name).lower() not in imported_lower
    )
    frameworks = sorted(declared_names & FRAMEWORK_NAMES)

    report = RepoReport(
        root=str(resolved),
        generated_at=utc_now(),
        dependency_managers=managers,
        python_requirement=python_requirement,
        declared_dependencies=declared,
        lockfiles=lockfiles,
        test_commands=test_commands,
        src_layout=src_layout,
        framework_signals=frameworks,
        python_file_count=python_files,
        imported_third_party=third_party,
        imported_local=sorted(local_modules),
        undeclared_imports=undeclared,
        unused_declared=unused,
        evidence=[],
        errors=errors,
    )
    report.evidence = [
        Evidence(
            claim=(
                f"Repository scan of {resolved.name}: {len(declared)} declared dependency(ies), "
                f"{python_files} Python file(s), managers: {', '.join(managers) or 'none'}."
            ),
            source_type="repo_scan",
            source_url=None,
            support=SupportLevel.OBSERVED,
            details={
                "root": str(resolved),
                "managers": managers,
                "python_requirement": python_requirement,
                "declared_count": len(declared),
                "python_file_count": python_files,
                "test_commands": test_commands,
                "undeclared_imports": undeclared[:10],
                "unused_declared": unused[:10],
            },
        )
    ]
    return report


def detect_test_commands(root: Path, src_layout: bool) -> list[str]:
    commands: list[str] = []
    pytest_signals = (
        (root / "pytest.ini").is_file()
        or (root / "conftest.py").is_file()
        or (root / "tests").is_dir()
        or _pyproject_has_pytest(root)
    )
    if pytest_signals:
        commands.append("PYTHONPATH=src python3 -m pytest" if src_layout else "python3 -m pytest")
    if (root / "tox.ini").is_file():
        commands.append("tox")
    makefile = root / "Makefile"
    if makefile.is_file():
        try:
            if re.search(
                r"^test\s*:",
                makefile.read_text(encoding="utf-8", errors="replace"),
                flags=re.MULTILINE,
            ):
                commands.append("make test")
        except OSError:
            pass
    return commands


def _pyproject_has_pytest(root: Path) -> bool:
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return False
    return "pytest" in (payload.get("tool") or {})


def scan_imports(root: Path) -> tuple[int, set[str], set[str]]:
    local_modules = _local_modules(root)
    imported: set[str] = set()
    file_count = 0
    for path in _python_files(root):
        if file_count >= MAX_PYTHON_FILES:
            break
        file_count += 1
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
    return file_count, imported, local_modules


def _python_files(root: Path):
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir() and entry.name not in SKIP_DIRS and not entry.name.startswith("."):
                stack.append(entry)
            elif entry.suffix == ".py":
                yield entry


def _local_modules(root: Path) -> set[str]:
    local: set[str] = set()
    # Scan the repo root, src/, and all other first-class top-level
    # subdirectories for local modules/packages. This catches sibling
    # directories like benchmarks/, tests/, integrations/ that contain
    # modules imported by other files in the repo, including benchmark helpers.
    bases: list[Path] = [root, root / "src"]
    with contextlib.suppress(OSError):
        bases.extend(
            entry
            for entry in root.iterdir()
            if entry.is_dir() and entry.name not in SKIP_DIRS and not entry.name.startswith(".")
        )
    for base in bases:
        if not base.is_dir():
            continue
        try:
            entries = list(base.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir() and (entry / "__init__.py").is_file():
                local.add(entry.name.lower())
            elif entry.suffix == ".py":
                local.add(entry.stem.lower())
    return local


def parse_requirement_line(line: str, source: str) -> RepoDependency | None:
    cleaned = line.split("#", 1)[0].strip()
    if not cleaned or cleaned.startswith(("-", "--")):
        return None
    if "://" in cleaned or cleaned.startswith((".", "/")):
        return None
    matched = REQUIREMENT_NAME.match(cleaned)
    if not matched:
        return None
    name = matched.group(1)
    remainder = (matched.group(3) or "").split(";", 1)[0].strip()
    return RepoDependency(name=name, constraint=remainder or None, source=source)


def normalize_dep_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def minimum_python(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    matched = re.search(r">=\s*(\d+)\.(\d+)", spec)
    if matched:
        return int(matched.group(1)), int(matched.group(2))
    matched = re.search(r"\^\s*(\d+)\.(\d+)", spec)
    if matched:
        return int(matched.group(1)), int(matched.group(2))
    return None


def python_compatibility(
    repo_requirement: str | None,
    package_requirement: str | None,
) -> tuple[bool | None, str]:
    repo_min = minimum_python(repo_requirement)
    package_min = minimum_python(package_requirement)
    if repo_min is None or package_min is None:
        return None, (
            f"Python compatibility is unknown (repo requires {repo_requirement or 'unspecified'}, "
            f"package requires {package_requirement or 'unspecified'})."
        )
    if package_min <= repo_min:
        return True, (
            f"Package minimum Python {package_min[0]}.{package_min[1]} is satisfied by the repo floor "
            f"{repo_min[0]}.{repo_min[1]}."
        )
    return False, (
        f"Package requires Python >= {package_min[0]}.{package_min[1]} but the repo still supports "
        f"{repo_min[0]}.{repo_min[1]}; adding it would raise the repo's Python floor."
    )


def _empty_report(root: str, errors: list[str]) -> RepoReport:
    return RepoReport(
        root=root,
        generated_at=utc_now(),
        dependency_managers=[],
        python_requirement=None,
        declared_dependencies=[],
        lockfiles=[],
        test_commands=[],
        src_layout=False,
        framework_signals=[],
        python_file_count=0,
        imported_third_party=[],
        imported_local=[],
        undeclared_imports=[],
        unused_declared=[],
        evidence=[
            Evidence(
                claim=f"Repository scan failed for {root}.",
                source_type="repo_scan",
                source_url=None,
                support=SupportLevel.FAILED,
                details={"errors": errors},
            )
        ],
        errors=errors,
    )

"""Architecture guards for the challenge-pack tool decomposition.

The pack issuer must not grow back into a god file: the facade stays thin and
owns nothing, every module in ``parity_pack_core`` stays focused and small,
each of the original 537-line ``parity_pack.py`` definitions has exactly one
implementation owner, the compatibility surface keeps resolving to identical
objects, the dependency graph stays acyclic and one-directional, facade-level
rebinding of the clock still steers every timestamp a pack carries, every
module is reachable from the facade, and the repository-only package never
enters a wheel or source archive.
"""

from __future__ import annotations

import ast
import importlib
import json
import types
from datetime import UTC, datetime
from pathlib import Path

from parity_release_support import write_cases, write_contract

import cortheon.parity_pack as facade

ROOT = Path(__file__).parents[1]
FACADE = ROOT / "src/cortheon/parity_pack.py"
CORE_DIR = ROOT / "src/cortheon/parity_pack_core"
PACKAGE = "cortheon.parity_pack_core"

FACADE_LINE_CAP = 60
MODULE_LINE_CAP = 500
PREFERRED_LINE_CAP = 350

# Every top-level definition of the pre-split 537-line parity_pack.py, mapped
# to the single ``parity_pack_core`` module that now owns it.
ORIGINAL_OWNERSHIP = {
    "build_parser": "cli",
    "main": "cli",
    "seal_case_pack": "seal",
    "verify_case_pack": "verify",
    "write_release_contract": "contract",
    "_canonical_signed_payload": "keys",
}

# Infrastructure the split introduced: the late-bound clock, the extracted
# helpers that keep each module focused, and the named constants that were
# inline literals in the god file.
SPLIT_DEFINITIONS = {
    "_FACADE_NAME": "_compat",
    "facade": "_compat",
    "PRIVATE_MODE": "artifacts",
    "write_private_json": "artifacts",
    "_add_seal_command": "cli",
    "_add_contract_command": "cli",
    "_dispatch": "cli",
    "_clock": "clock",
    "issued_at": "clock",
    "require_future_expiry": "clock",
    "CONTRACT_SCHEMA_VERSION": "contract",
    "MINIMUM_FRONTIERS": "contract",
    "MINIMUM_DOMAINS": "contract",
    "RELEASE_THRESHOLDS": "contract",
    "_RUNTIME_SHA256": "contract",
    "_frontier_registrations": "contract",
    "_unique_sorted": "contract",
    "_endpoint_registrations": "contract",
    "_pricing_registrations": "contract",
    "_MINIMUM_KEY_BYTES": "keys",
    "_read_secret": "keys",
    "read_signing_keys": "keys",
    "key_commitment": "keys",
    "key_id": "keys",
    "runner_attestation": "keys",
    "signature": "keys",
    "SCHEMA_VERSION": "manifest",
    "_PUBLIC_MANIFEST_KEYS": "manifest",
    "build_manifest": "manifest",
    "public_payload": "manifest",
    "_declared_evaluator": "seal",
    "_normalized_authors": "seal",
    "_resolved_destinations": "seal",
    "_submitted_cases": "seal",
    "normalize_and_select": "selection",
    "selection_sha256": "selection",
    "validate_task_class_coverage": "selection",
    "_rebalance_task_class_coverage": "selection",
}
OWNERSHIP = {**ORIGINAL_OWNERSHIP, **SPLIT_DEFINITIONS}

# ``datetime`` is imported by the facade, never defined there: it is the
# rebinding anchor the sealing clock resolves through.
PATCH_ANCHOR = "datetime"
COMPAT_MODULE = "_compat"
# The tuple that keeps the private re-exports live without widening __all__.
FACADE_OWN_DEFINITIONS = {"__all__", "_COMPATIBILITY_EXPORTS"}
COMPATIBILITY_EXPORTS = set(ORIGINAL_OWNERSHIP) | {PATCH_ANCHOR, "_COMPATIBILITY_EXPORTS"}

# ``__all__`` is the star-import contract: the five callables the tool offers.
PUBLIC_EXPORTS = [
    "build_parser",
    "main",
    "seal_case_pack",
    "verify_case_pack",
    "write_release_contract",
]

# Import paths the repository itself depends on, including the clock the
# campaign suite rebinds to seal packs at a chosen instant.
REPOSITORY_IMPORTS = {"seal_case_pack", "verify_case_pack", PATCH_ANCHOR}


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
    assert modules, "parity_pack_core package must exist"
    return modules


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _imported_package_modules(path: Path) -> set[str]:
    """Sibling module names this file imports from the core package."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(f"{PACKAGE}."):
                imported.add(node.module.split(".")[-1])
            elif node.module == PACKAGE:
                imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(f"{PACKAGE}."):
                    imported.add(alias.name.split(".")[-1])
    return imported


def test_facade_is_a_thin_stable_surface() -> None:
    line_count = _line_count(FACADE)
    assert line_count <= FACADE_LINE_CAP, f"facade has {line_count} lines"
    assert _top_level_definitions(FACADE) == FACADE_OWN_DEFINITIONS


def test_facade_re_exports_explicitly_without_star_imports() -> None:
    tree = ast.parse(FACADE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert all(alias.name != "*" for alias in node.names), "no star imports in the facade"
            assert node.module is not None and node.module.startswith((PACKAGE, "datetime")), (
                f"facade must source names from {PACKAGE}, not {node.module}"
            )
        assert not isinstance(node, ast.Import), "facade re-exports by name only"


def test_every_core_module_stays_focused() -> None:
    for module in _core_modules():
        line_count = _line_count(module)
        assert line_count <= MODULE_LINE_CAP, f"{module.name} has {line_count} lines"
        assert line_count <= PREFERRED_LINE_CAP, (
            f"{module.name} has {line_count} lines; split it before it reaches {MODULE_LINE_CAP}"
        )


def test_single_owner_for_top_level_definitions() -> None:
    owners: dict[str, Path] = {}
    for path in [FACADE, *_core_modules()]:
        for name in _top_level_definitions(path):
            if name in FACADE_OWN_DEFINITIONS:
                continue
            assert name not in owners, (
                f"{name} is defined in both {owners[name].name} and {path.name}"
            )
            owners[name] = path


def test_every_definition_is_owned_exactly_once() -> None:
    """Declared ownership is checked against what the modules actually define."""
    actual: dict[str, str] = {}
    for path in _core_modules():
        for name in _top_level_definitions(path):
            actual[name] = path.stem
    assert actual == OWNERSHIP, (
        f"missing: {sorted(set(OWNERSHIP) - set(actual))}, "
        f"extra: {sorted(set(actual) - set(OWNERSHIP))}, "
        f"moved: {sorted(n for n in set(actual) & set(OWNERSHIP) if actual[n] != OWNERSHIP[n])}"
    )


def test_facade_exports_exactly_the_compatibility_surface() -> None:
    exported = {
        name
        for name in vars(facade)
        if not name.startswith("__") and not isinstance(getattr(facade, name), types.ModuleType)
    }
    assert exported == COMPATIBILITY_EXPORTS, (
        f"missing: {sorted(COMPATIBILITY_EXPORTS - exported)}, "
        f"extra: {sorted(exported - COMPATIBILITY_EXPORTS)}"
    )
    assert facade.__all__ == PUBLIC_EXPORTS
    assert len(facade.__all__) == len(set(facade.__all__)), "no duplicate exports"
    assert REPOSITORY_IMPORTS <= COMPATIBILITY_EXPORTS


def test_facade_reexports_are_identical_to_core_definitions() -> None:
    for name, stem in ORIGINAL_OWNERSHIP.items():
        owner = importlib.import_module(f"{PACKAGE}.{stem}")
        assert getattr(facade, name) is getattr(owner, name), name
    assert facade.datetime is datetime


def test_the_sealing_clock_is_resolved_late_through_the_facade() -> None:
    """The mechanism behind the rebinding contract, pinned at the source."""
    source = (CORE_DIR / "clock.py").read_text(encoding="utf-8")
    assert f"facade().{PATCH_ANCHOR}" in source, (
        f"{PATCH_ANCHOR} must be read through the facade so a rebinding is seen"
    )
    assert f"= {PATCH_ANCHOR}.now" not in source, (
        f"{PATCH_ANCHOR} is import-bound in clock; a rebinding would be lost"
    )
    compat = CORE_DIR / f"{COMPAT_MODULE}.py"
    assert _top_level_definitions(compat) == {"_FACADE_NAME", "facade"}
    assert not _imported_package_modules(compat), "the resolver owns no core dependency"


def test_facade_rebinding_still_steers_every_sealed_timestamp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The contract itself: freeze the clock, change what the pack records."""

    frozen = datetime(2026, 1, 1, tzinfo=UTC)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen if tz is not None else frozen.replace(tzinfo=None)

    monkeypatch.setenv("PACK_ARCHITECTURE_KEY", "k" * 32)
    monkeypatch.setenv("PACK_ARCHITECTURE_RUNNER_KEY", "r" * 32)
    monkeypatch.setattr(facade, PATCH_ANCHOR, FrozenDateTime)
    private_pack = tmp_path / "sealed.json"

    facade.seal_case_pack(
        write_cases(tmp_path),
        private_pack,
        public_output_path=tmp_path / "public.json",
        contract_path=write_contract(tmp_path),
        pack_id="architecture-pack",
        issuer="independent-lab",
        runner_id="independent-runner-1",
        authors=["external-author"],
        key_env="PACK_ARCHITECTURE_KEY",
        runner_key_env="PACK_ARCHITECTURE_RUNNER_KEY",
        seed=7,
        holdout_fraction=0.5,
        rotation_index=0,
        rotation_size=0,
        expires_at="2099-01-01T00:00:00+00:00",
        overwrite=False,
    )

    sealed = json.loads(private_pack.read_text(encoding="utf-8"))
    assert sealed["manifest"]["created_at"] == "2026-01-01T00:00:00+00:00"


def test_core_never_depends_on_the_facade() -> None:
    """One direction of dependency only, so the facade can never be a cycle."""
    for path in _core_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "cortheon.parity_pack", path.name
            elif isinstance(node, ast.Import):
                assert all(alias.name != "cortheon.parity_pack" for alias in node.names), path.name


def test_the_core_dependency_graph_is_acyclic() -> None:
    graph = {path.stem: _imported_package_modules(path) for path in _core_modules()}
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(name: str, trail: tuple[str, ...]) -> None:
        assert name not in visiting, f"import cycle: {' -> '.join((*trail, name))}"
        if name in done:
            return
        visiting.add(name)
        for dependency in sorted(graph.get(name, set())):
            visit(dependency, (*trail, name))
        visiting.discard(name)
        done.add(name)

    for module in sorted(graph):
        visit(module, ())


def test_every_core_module_is_reachable_from_the_facade() -> None:
    """No module may be orphaned, so none can be silently dropped from a build."""
    modules = {path.stem for path in _core_modules()} - {"__init__"}
    reachable = _imported_package_modules(FACADE)
    frontier = list(reachable)
    while frontier:
        name = frontier.pop()
        for dependency in _imported_package_modules(CORE_DIR / f"{name}.py"):
            if dependency not in reachable:
                reachable.add(dependency)
                frontier.append(dependency)
    assert modules == reachable, f"unreachable modules: {sorted(modules - reachable)}"


def test_the_cli_surface_is_owned_by_one_module() -> None:
    """Command registration and the error prefix must be searchable to one place."""
    fragments = [
        'prog="cortheon-pack"',
        'print(f"cortheon-pack: {exc}", file=sys.stderr)',
        *(f'commands.add_parser("{command}")' for command in ("seal", "verify", "contract")),
    ]
    for fragment in fragments:
        emitters = {
            path.stem for path in _core_modules() if fragment in path.read_text(encoding="utf-8")
        }
        assert emitters == {"cli"}, f"{fragment} is declared by {sorted(emitters)}"


def test_parity_pack_core_stays_repository_only() -> None:
    """The pack issuer holds evaluator secrets; it must never enter an artifact."""
    for relative in ("setup.py", "MANIFEST.in", "pyproject.toml"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "parity" not in source, f"{relative} would ship the pack issuer"
    distribution_test = (ROOT / "tests/test_lightweight_distribution.py").read_text(
        encoding="utf-8"
    )
    assert '"/parity_pack_core/" in member' in distribution_test, (
        "the wheel and sdist tests must exclude parity_pack_core explicitly"
    )

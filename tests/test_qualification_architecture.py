"""Architecture guards for the repository-only qualification package."""

from __future__ import annotations

import ast
import importlib
import types
from pathlib import Path

from qualification_support import _result, _write_manifest

import cortheon.qualification_factory as facade

ROOT = Path(__file__).parents[1]
FACADE = ROOT / "src/cortheon/qualification_factory.py"
CORE_DIR = ROOT / "src/cortheon/qualification_core"
PACKAGE = "cortheon.qualification_core"

FACADE_LINE_CAP = 160
MODULE_LINE_CAP = 500
PREFERRED_LINE_CAP = 350

# One implementation owner for each former facade definition.
OWNERSHIP = {
    "CELL_KEYS": "constants",
    "ENVIRONMENT_NAME": "constants",
    "FORBIDDEN_CREDENTIAL_KEYS": "constants",
    "GATE_KEYS": "constants",
    "HOSTS": "constants",
    "IDENTIFIER": "constants",
    "MAX_CELLS": "constants",
    "MAX_JOBS": "constants",
    "MAX_MANIFEST_BYTES": "constants",
    "REPORT_SCHEMA_VERSION": "constants",
    "ROOT_KEYS": "constants",
    "SCHEMA_VERSION": "constants",
    "SUITES": "constants",
    "TIER_DEFAULTS": "constants",
    "ABLATION_OPERATORS": "conditions",
    "AVAILABLE_CONDITIONS": "conditions",
    "EQUAL_BUDGET_PLACEBO": "conditions",
    "HISTORICAL_CONDITIONS": "conditions",
    "CONDITIONS": "conditions",
    "CONDITION_REGISTRY_VERSION": "conditions",
    "CONTRASTS": "conditions",
    "ConditionSpec": "conditions",
    "FULL_CONDITION": "conditions",
    "OLD_PLANNER": "conditions",
    "OPERATOR_KEYS": "conditions",
    "REQUIRED_CONDITIONS": "conditions",
    "_SPECS": "conditions",
    "_canonical": "conditions",
    "_implementation_files": "conditions",
    "_operators": "conditions",
    "closed_registry": "conditions",
    "condition_record": "conditions",
    "execution_profile": "conditions",
    "implementation_digest": "conditions",
    "profile_matches": "conditions",
    "ARCHIVE": "frozen_archive",
    "ARCHIVE_BYTES": "frozen_archive",
    "ARCHIVE_SHA256": "frozen_archive",
    "FROZEN_COMMIT": "frozen_archive",
    "FROZEN_TREE": "frozen_archive",
    "EVALUATOR_FILES": "frozen_old_planner",
    "FrozenRuntime": "frozen_runtime",
    "MEMBER_SHA256": "frozen_archive",
    "SMOKE": "frozen_old_planner",
    "SMOKE_SHA256": "frozen_old_planner",
    "WRAPPER": "frozen_old_planner",
    "WRAPPER_SHA256": "frozen_old_planner",
    "BOOTSTRAP": "frozen_runtime",
    "_DIRECTORIES": "frozen_archive",
    "_ROOT": "frozen_archive",
    "_OPERATIONAL_FUNCTIONS": "frozen_old_planner",
    "extract_verified": "frozen_archive",
    "_json_get": "frozen_runtime",
    "_json_post": "frozen_runtime",
    "_sha": "frozen_archive",
    "_tree_digest": "frozen_runtime",
    "archive_available": "frozen_archive",
    "comparator_available": "frozen_old_planner",
    "frozen_implementation_sha256": "frozen_old_planner",
    "frozen_old_planner": "frozen_old_planner",
    "wrapper_sha256": "frozen_old_planner",
    "_metric_delta": "frozen_execution",
    "_run_frozen_evidence": "frozen_execution",
    "run_frozen_job": "frozen_execution",
    **dict.fromkeys(
        (  # noqa: SIM905
            "SMOKE_SCHEMA_VERSION SMOKE_PROVIDER SMOKE_MODEL SMOKE_BASE_URL SMOKE_POLICY "
            "SMOKE_TASK _TOP_KEYS _HOST_KEYS _INFERENCE_KEYS _POLICY_KEYS _OUTCOME_KEYS "
            "_RUNTIME_KEYS _receipt_sha _receipt_sha_value smoke_endpoint_sha256 "
            "smoke_task_sha256 validate_smoke_receipt"
        ).split(),
        "frozen_receipt",
    ),
    "SMOKE_CASE": "frozen_smoke",
    "_executable_identity": "frozen_smoke",
    "generate_smoke_candidate": "frozen_smoke",
    "regenerate_smoke": "frozen_smoke",
    "start_runtime": "frozen_runtime",
    "stop_runtime": "frozen_runtime",
    "Cell": "models",
    "CellRun": "models",
    "Manifest": "models",
    "QualificationError": "models",
    "_bounded_int": "validation",
    "_bounded_number": "validation",
    "_bounded_text": "validation",
    "_http_url": "validation",
    "_parse_document": "validation",
    "_reject_embedded_credentials": "validation",
    "_reject_unknown": "validation",
    "_strict_gate_overrides": "gates",
    "_parse_cell": "manifest",
    "load_manifest": "manifest",
    "_cell_namespace": "environment",
    "_command_version": "environment",
    "_git_revision": "environment",
    "_package_version": "environment",
    "_public_runtime_health": "environment",
    "_cell_public_config": "digests",
    "_sealed_task_digest": "digests",
    "_aggregate_pairing": "pairing",
    "_bootstrap_summary": "pairing",
    "_independent_pairing": "pairing",
    "_failure_type": "taxonomy",
    "_public_run": "taxonomy",
    "_cell_gates": "cell_gates",
    "_run_cell": "execution",
    "_reproducers": "reproducers",
    "_cell_report": "report",
    "run_qualification": "report",
    "_example_manifest": "cli",
    "build_parser": "cli",
    "main": "cli",
}

# Borrowed from ``cortheon.cognitive_benchmark``, never defined here. The
# facade has always exposed it, so the compatibility surface keeps it.
BORROWED_EXPORTS = {"_repository_fingerprint"}
REGISTRY_INTERNALS = (
    {
        "ConditionSpec",
        "EQUAL_BUDGET_PLACEBO",
        "FULL_CONDITION",
        "OLD_PLANNER",
        "OPERATOR_KEYS",
        "_SPECS",
        "_canonical",
        "_implementation_files",
        "_operators",
        "execution_profile",
        "implementation_digest",
        "profile_matches",
    }
    | {name for name, owner in OWNERSHIP.items() if owner.startswith("frozen")}
    | {"HISTORICAL_CONDITIONS"}
)
COMPATIBILITY_EXPORTS = (set(OWNERSHIP) - REGISTRY_INTERNALS) | BORROWED_EXPORTS

# Import paths the repository itself depends on, private names included.
REPOSITORY_IMPORTS = {
    "Cell",
    "CellRun",
    "QualificationError",
    "_aggregate_pairing",
    "_failure_type",
    "_independent_pairing",
    "_public_run",
    "_reproducers",
    "_sealed_task_digest",
    "load_manifest",
    "main",
    "run_qualification",
}

# Names tests and callers rebind. The pre-split god file resolved them from
# its own module globals, so patching the facade steered the run; the split
# must keep that true by resolving them through the facade at call time.
PATCH_ANCHORS = ("_run_cell", "_repository_fingerprint", "_git_revision")

# ``_repository_fingerprint`` had two consumers in the god file, not one:
# ``run_qualification`` bracketed the whole matrix, and ``_run_cell`` bracketed
# each cell's jobs. Both resolved the same module global, so a facade patch
# silenced both. ``execution`` still has to import the name -- the facade
# sources its borrowed re-export from there -- which is exactly why the call
# form, not the import, is what this file pins.
EXECUTION_FINGERPRINT_CALLS = ["facade-late", "facade-late"]

# The late-bound resolver is new infrastructure rather than one of the original
# definitions, so it sits outside the ownership map and is pinned on its own.
COMPAT_MODULE = "_compat"
COMPAT_DEFINITIONS = {"_FACADE_NAME", "facade"}


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
    assert modules, "qualification_core package must exist"
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
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(f"{PACKAGE}."):
                    imported.add(alias.name.split(".")[-1])
    return imported


def _call_forms(path: Path, function: str, name: str) -> list[str]:
    """How each call to ``name`` inside ``function`` resolves its callee."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if getattr(node, "name", None) == function)
    forms: list[str] = []
    for node in ast.walk(owner):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if isinstance(callee, ast.Name) and callee.id == name:
            forms.append("import-bound")
        elif isinstance(callee, ast.Attribute) and callee.attr == name:
            resolver = callee.value
            late = (
                isinstance(resolver, ast.Call)
                and isinstance(resolver.func, ast.Name)
                and resolver.func.id == "facade"
            )
            forms.append("facade-late" if late else "attribute-bound")
    return forms


def test_facade_is_a_thin_stable_surface() -> None:
    line_count = _line_count(FACADE)
    assert line_count <= FACADE_LINE_CAP, f"facade has {line_count} lines"
    assert not _top_level_definitions(FACADE) - {"__all__"}, "facade must not own definitions"


def test_facade_re_exports_explicitly_without_star_imports() -> None:
    tree = ast.parse(FACADE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert all(alias.name != "*" for alias in node.names), "no star imports in the facade"
            assert node.module is not None and node.module.startswith(PACKAGE), (
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
            if name == "__all__":
                continue
            assert name not in owners, (
                f"{name} is defined in both {owners[name].name} and {path.name}"
            )
            owners[name] = path


def test_every_original_definition_is_owned_exactly_once() -> None:
    """Declared ownership is checked against what the modules actually define."""
    actual: dict[str, str] = {}
    for path in _core_modules():
        if path.stem == COMPAT_MODULE:
            continue
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
    assert set(facade.__all__) == COMPATIBILITY_EXPORTS
    assert len(facade.__all__) == len(set(facade.__all__)), "no duplicate exports"
    assert REPOSITORY_IMPORTS <= COMPATIBILITY_EXPORTS


def test_facade_reexports_are_identical_to_core_definitions() -> None:
    for name, stem in OWNERSHIP.items():
        if name in REGISTRY_INTERNALS:
            continue
        owner = importlib.import_module(f"{PACKAGE}.{stem}")
        assert getattr(facade, name) is getattr(owner, name), name
    benchmark = importlib.import_module("cortheon.cognitive_benchmark")
    for name in BORROWED_EXPORTS:
        assert getattr(facade, name) is getattr(benchmark, name), name


def test_patch_anchors_are_resolved_late_through_the_facade() -> None:
    """The mechanism behind the rebinding contract, pinned at the source."""
    report_source = (CORE_DIR / "report.py").read_text(encoding="utf-8")
    for name in PATCH_ANCHORS:
        assert hasattr(facade, name), name
        assert f"facade().{name}(" in report_source, (
            f"{name} must be called through the facade so a patch is seen"
        )
        assert f"import {name}" not in report_source, (
            f"{name} is import-bound in the report module; a facade patch would be lost"
        )
    compat = CORE_DIR / f"{COMPAT_MODULE}.py"
    assert _top_level_definitions(compat) == COMPAT_DEFINITIONS
    assert not _imported_package_modules(compat), "the resolver owns no core dependency"


def test_cell_execution_fingerprints_the_workspace_through_the_facade() -> None:
    """Both of ``_run_cell``'s call sites, not just the preflight one."""
    execution = CORE_DIR / "execution.py"
    forms = _call_forms(execution, "_run_cell", "_repository_fingerprint")
    assert forms == EXECUTION_FINGERPRINT_CALLS, forms
    # The import stays, and stays identical: the facade's borrowed re-export is
    # sourced from this module, so dropping the binding would break the surface.
    module = importlib.import_module(f"{PACKAGE}.execution")
    benchmark = importlib.import_module("cortheon.cognitive_benchmark")
    assert module._repository_fingerprint is benchmark._repository_fingerprint
    assert "_repository_fingerprint as _repository_fingerprint" in execution.read_text(
        encoding="utf-8"
    ), "the borrowed re-export must be explicit, not an incidental import"


def test_facade_rebinding_still_steers_cell_execution(monkeypatch, tmp_path) -> None:
    """The same contract as above, proved by running a fully stubbed cell."""
    execution = importlib.import_module(f"{PACKAGE}.execution")
    manifest = facade.load_manifest(_write_manifest(tmp_path))
    case = types.SimpleNamespace(case_id="a")
    health = {
        "ok": True,
        "version": "1",
        "storage": "memory_only",
        "model_id": "small-model",
        "protocol_version": "1.0.0",
        "source_fingerprint": execution._source_fingerprint(),
    }
    monkeypatch.setattr(execution, "_runtime_health", lambda _url: health)
    monkeypatch.setattr(execution, "_model_endpoint_health", lambda *_a, **_kw: health)
    monkeypatch.setattr(execution, "_command_version", lambda _command: "1")
    monkeypatch.setattr(execution, "discover_benchmark_cases", lambda *_a, **_kw: [case])
    monkeypatch.setattr(execution, "_sealed_task_digest", lambda _case: "digest")
    monkeypatch.setattr(
        execution,
        "run_job",
        lambda _args, item, *, repeat, treatment, condition, evaluation_profile: _result(
            item.case_id,
            repeat,
            condition,
            condition == "full",
            telemetry=condition != "bare" or None,
        ),
    )
    fingerprints: list[Path] = []

    def _rebound_fingerprint(repository: Path) -> str:
        fingerprints.append(repository)
        return f"rebound-print-{len(fingerprints)}"

    monkeypatch.setattr(facade, "_repository_fingerprint", _rebound_fingerprint)

    run = facade._run_cell(
        manifest,
        manifest.cells[0],
        case_filter=None,
        repeat_filter=None,
        progress=False,
    )

    # Two calls means preflight and postflight both saw the patch -- the real
    # one would have hashed the live repository -- and the two distinct values
    # then reached the comparison the cell reports.
    assert fingerprints == [manifest.repository, manifest.repository]
    assert not run.repository_unchanged


def test_facade_rebinding_still_steers_run_qualification(monkeypatch, tmp_path) -> None:
    """The contract itself: patch the facade, change what the run executes."""
    manifest = facade.load_manifest(_write_manifest(tmp_path))
    results = []
    for case_id in ("a", "b"):
        results.extend(
            [
                _result(case_id, 0, "full", True, telemetry=True),
                _result(case_id, 0, "bare", False),
            ]
        )
    pairing, deltas, invalid = facade._independent_pairing(
        results,
        treatment="full",
        comparison="bare",
        repeats=(0,),
        seed=7,
    )
    run = facade.CellRun(
        cell=manifest.cells[0],
        case_ids=("a", "b"),
        task_digests={"a": "task-a", "b": "task-b"},
        results=results,
        pairing=pairing,
        case_deltas=deltas,
        invalid_case_ids=invalid,
        repository_unchanged=True,
        environment_stable=True,
        runtime={"ok": True, "storage": "memory_only"},
        inference={"ok": True, "model_id": "small-model"},
        host_version="1",
        contrasts={"full_vs_bare": pairing},
        contrast_case_deltas={"full_vs_bare": deltas},
        contrast_invalid_case_ids={"full_vs_bare": invalid},
        scheduled_repeats=(0,),
    )
    calls: list[str] = []

    def _rebound_run_cell(*_args, **_kwargs):
        calls.append("_run_cell")
        return run

    monkeypatch.setattr(facade, "_run_cell", _rebound_run_cell)
    monkeypatch.setattr(facade, "_repository_fingerprint", lambda _repo: "rebound-print")
    monkeypatch.setattr(facade, "_git_revision", lambda _repo: "rebound-revision")

    report = facade.run_qualification(manifest, progress=False)

    # All three rebindings reached the run: the real ones would have shelled
    # out to git and hashed the live repository instead.
    assert calls == ["_run_cell"]
    assert report["provenance"]["repository_fingerprint"] == "rebound-print"
    assert report["provenance"]["git_revision"] == "rebound-revision"
    assert report["promotion_gates"]["live_repository_unchanged"]


def test_core_never_depends_on_the_facade() -> None:
    """One direction of dependency only, so the facade can never be a cycle."""
    for path in _core_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "cortheon.qualification_factory", path.name
            elif isinstance(node, ast.Import):
                assert all(
                    alias.name != "cortheon.qualification_factory" for alias in node.names
                ), path.name


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

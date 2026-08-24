"""Architecture guards for the frontier-parity release-contract decomposition.

The parity contract must not grow back into a god file: the facade stays thin
and owns nothing, every module in ``parity_gates`` stays focused and small,
each of the original 1,285-line ``parity.py`` definitions has exactly one
implementation owner, the compatibility surface keeps resolving to identical
objects, the dependency graph stays acyclic and one-directional, facade-level
monkeypatching of the release-scale policy still steers the gate, every module
is reachable from the facade, the ordered check names have single owners, and
the repository-only package never enters a wheel or source archive.

The inventory guard here covers the whole parity family, not just this
package: no parity source or test file may exceed five hundred lines, and no
mechanism exists for declaring one that may.
"""

from __future__ import annotations

import ast
import importlib
import types
from pathlib import Path

import cortheon.parity as facade

ROOT = Path(__file__).parents[1]
FACADE = ROOT / "src/cortheon/parity.py"
GATES_DIR = ROOT / "src/cortheon/parity_gates"
PACKAGE = "cortheon.parity_gates"

FACADE_LINE_CAP = 90
MODULE_LINE_CAP = 500
PREFERRED_LINE_CAP = 350

# Every top-level definition of the pre-split 1,285-line parity.py, mapped to
# the single ``parity_gates`` module that now owns it.
ORIGINAL_OWNERSHIP = {
    "ParityContractError": "errors",
    "SUPPORTED_CANDIDATE_HOSTS": "errors",
    "_TRUSTED_FRONTIER_HOSTS": "errors",
    "public_case_projection": "projection",
    "public_task_hash": "projection",
    "evaluation_schedule": "projection",
    "evaluation_schedule_hash": "projection",
    "load_parity_contract": "contract",
    "_validate_contract": "contract",
    "_universal_scale_ok": "preregistration",
    "_paired_statistics": "comparison",
    "_instability": "comparison",
    "_comparison_check": "noninferiority",
    "_ratio_check": "noninferiority",
    "evaluate_frontier_parity": "decision",
    "_decision": "decision",
    "_after": "values",
    "_is_sha256": "values",
    "_mapping": "values",
    "_nested_number": "values",
    "_number": "values",
    "_percentile": "values",
    "_stable_seed": "values",
}

# Infrastructure the split introduced: the stage entry points, the shared
# evaluation context, the extracted helpers that keep each module focused, and
# the named constants that were inline literals in the god file.
SPLIT_DEFINITIONS = {
    "_FACADE_NAME": "_compat",
    "facade": "_compat",
    "ParityContext": "context",
    "ContenderIdentities": "context",
    "OutcomeSummary": "context",
    "_REQUIRED_THRESHOLDS": "contract",
    "_POSITIVE_INTEGER_THRESHOLDS": "contract",
    "_UNIT_INTERVAL_THRESHOLDS": "contract",
    "_validate_families": "contract",
    "_validate_contenders": "contract",
    "_validate_thresholds": "contract",
    "evaluate_preregistration": "preregistration",
    "_evaluate_blinding": "preregistration",
    "_release_scale": "preregistration",
    "evaluate_coverage": "coverage",
    "resolve_contenders": "identity",
    "_check_registered_bindings": "identity",
    "_check_candidate_identity": "identity",
    "evaluate_execution_binding": "execution_binding",
    "_check_model_identity": "execution_binding",
    "_check_schedule": "execution_binding",
    "evaluate_outcomes": "outcomes",
    "_check_domain_floors": "outcomes",
    "_check_safety": "outcomes",
    "_RESAMPLES": "comparison",
    "_AGGREGATE_SCOPE": "noninferiority",
    "_DOMAIN_SCOPE": "noninferiority",
    "_precision_required": "noninferiority",
    "evaluate_frontier_comparisons": "noninferiority",
    "_evaluate_frontier_costs": "noninferiority",
    "evaluate_metering": "metering",
    "_REPORT_KEYS": "report_metrics",
    "_CANDIDATE_KEYS": "report_metrics",
    "_CASE_KEYS": "report_rows",
    "_GRADER_ASSURANCE": "report_rows",
    "_ROW_KEYS": "report_metrics",
    "_COST_KEYS": "report_metrics",
    "_RELEASE_IDENTITY_KEYS": "report_metrics",
    "_EVALUATOR_OUTCOME_KEYS": "report_outcomes",
    "_EVALUATOR_TRANSPORTS": "report_outcomes",
    "_TERMINAL_STATUSES": "report_outcomes",
    "_TERMINAL_PROVENANCES": "report_outcomes",
    "validate_release_report": "report_metrics",
    "_validate_candidates": "report_metrics",
    "validate_cases": "report_rows",
    "_validate_rows": "report_metrics",
    "validate_row_values": "report_rows",
    "_row_classification": "report_rows",
    "validate_evaluator_outcome": "report_outcomes",
    "_validate_cost": "report_metrics",
    "_validate_named_measurements": "report_metrics",
    "_exact_keys": "report_metrics",
    "_dict": "report_metrics",
    "_list": "report_metrics",
    "_nonnegative_number": "report_metrics",
    "_optional_nonnegative_number": "report_metrics",
    "_REPORT_PAIR_RESAMPLES": "paired_validation",
    "_cell_index": "pairing_cells",
    "_duplicate_count": "pairing_cells",
    "canonical_paired_comparisons": "paired_validation",
    "_report_pair_statistics": "paired_validation",
    "_report_pair_seed": "paired_validation",
    "_nonnegative_integer": "report_metrics",
    "_positive_integer": "report_metrics",
    "canonical_summary": "summary_validation",
    "_summarize_candidate": "summary_validation",
    "_summarize_slice": "summary_validation",
    "_rate": "summary_validation",
    "_slice_value": "summary_validation",
}
OWNERSHIP = {**ORIGINAL_OWNERSHIP, **SPLIT_DEFINITIONS}

# Borrowed from ``cortheon.parity_scale_policy``, never defined here. The
# facade has always exposed both, so the compatibility surface keeps them.
BORROWED_EXPORTS = {"UNIVERSAL_SCALE_CEILINGS", "UNIVERSAL_SCALE_REQUIREMENTS"}
# The tuple that keeps the private re-exports live without widening __all__.
FACADE_OWN_DEFINITIONS = {"__all__", "_COMPATIBILITY_EXPORTS"}
COMPATIBILITY_EXPORTS = set(ORIGINAL_OWNERSHIP) | BORROWED_EXPORTS | {"_COMPATIBILITY_EXPORTS"}

# ``__all__`` is the star-import contract and is exactly what it was before
# the split: the ten public names, and no private one.
PUBLIC_EXPORTS = [
    "SUPPORTED_CANDIDATE_HOSTS",
    "UNIVERSAL_SCALE_CEILINGS",
    "UNIVERSAL_SCALE_REQUIREMENTS",
    "ParityContractError",
    "evaluate_frontier_parity",
    "evaluation_schedule",
    "evaluation_schedule_hash",
    "load_parity_contract",
    "public_case_projection",
    "public_task_hash",
]

# Import paths the repository itself depends on.
REPOSITORY_IMPORTS = {
    "SUPPORTED_CANDIDATE_HOSTS",
    "ParityContractError",
    "evaluate_frontier_parity",
    "evaluation_schedule",
    "evaluation_schedule_hash",
    "load_parity_contract",
    "public_case_projection",
    "public_task_hash",
}

# The one name callers rebind on the facade. The pre-split god file read it
# from its own module globals, so patching ``cortheon.parity`` substituted a
# reduced test-scale policy for the whole evaluation; the split must keep that
# true by resolving it through the facade at call time.
PATCH_ANCHOR = "UNIVERSAL_SCALE_REQUIREMENTS"
COMPAT_MODULE = "_compat"


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


def _gate_modules() -> list[Path]:
    modules = sorted(GATES_DIR.glob("*.py"))
    assert modules, "parity_gates package must exist"
    return modules


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _imported_package_modules(path: Path) -> set[str]:
    """Sibling module names this file imports from the gates package."""
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
            assert node.module is not None and node.module.startswith(
                (PACKAGE, "cortheon.parity_scale_policy")
            ), f"facade must source names from {PACKAGE}, not {node.module}"
        assert not isinstance(node, ast.Import), "facade re-exports by name only"


def test_every_gate_module_stays_focused() -> None:
    for module in _gate_modules():
        line_count = _line_count(module)
        assert line_count <= MODULE_LINE_CAP, f"{module.name} has {line_count} lines"
        assert line_count <= PREFERRED_LINE_CAP, (
            f"{module.name} has {line_count} lines; split it before it reaches {MODULE_LINE_CAP}"
        )


def _parity_family() -> list[Path]:
    """Every parity-family source and test file in the repository.

    Both facades, all four repository-only packages behind them, and every
    test module and helper that names parity. The globs are deliberately
    broad: a new parity module must land inside this inventory rather than
    beside it.
    """

    inventory = [
        *sorted((ROOT / "src/cortheon").glob("parity*.py")),
        *sorted((ROOT / "src/cortheon").glob("parity*/*.py")),
        *sorted((ROOT / "tests").glob("*parity*.py")),
    ]
    assert len(inventory) >= 45, "the inventory must cover the whole parity family"
    for required in (FACADE, GATES_DIR / "__init__.py", ROOT / "src/cortheon/parity_pack.py"):
        assert required in inventory, f"{required.name} is missing from the inventory"
    return inventory


def test_no_parity_family_file_is_above_the_line_cap() -> None:
    """No god files in the parity family, and no exception mechanism either.

    This inventory used to declare ``parity_pack.py`` and
    ``test_parity_release.py`` as permanent oversized exceptions, which
    contradicts the repository-wide invariant it is here to enforce. Both are
    now split, so the allowed count is zero and there is nowhere left to
    record a new exception.
    """

    oversized = {
        path.name: _line_count(path) for path in _parity_family() if _line_count(path) > 500
    }
    assert oversized == {}, f"parity-family files above the 500-line cap: {oversized}"


def test_the_split_facades_and_their_tests_are_all_in_the_inventory() -> None:
    """The two splits this invariant paid for stay visible to it."""

    names = {path.name for path in _parity_family()}
    assert {"parity.py", "parity_pack.py"} <= names, "both facades must be inventoried"
    assert not (names & {"test_parity_release.py"}), (
        "test_parity_release.py was split; the inventory must not see it again"
    )
    for package in ("parity_gates", "parity_pack_core", "parity_campaign"):
        assert any(path.parent.name == package for path in _parity_family()), (
            f"{package} is not inventoried"
        )


def test_single_owner_for_top_level_definitions() -> None:
    owners: dict[str, Path] = {}
    for path in [FACADE, *_gate_modules()]:
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
    for path in _gate_modules():
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


def test_facade_reexports_are_identical_to_gate_definitions() -> None:
    for name, stem in ORIGINAL_OWNERSHIP.items():
        owner = importlib.import_module(f"{PACKAGE}.{stem}")
        assert getattr(facade, name) is getattr(owner, name), name
    policy = importlib.import_module("cortheon.parity_scale_policy")
    for name in BORROWED_EXPORTS:
        assert getattr(facade, name) is getattr(policy, name), name


def test_the_release_scale_policy_is_resolved_late_through_the_facade() -> None:
    """The mechanism behind the rebinding contract, pinned at the source."""
    source = (GATES_DIR / "preregistration.py").read_text(encoding="utf-8")
    assert f"facade().{PATCH_ANCHOR}" in source, (
        f"{PATCH_ANCHOR} must be read through the facade so a rebinding is seen"
    )
    assert f"import {PATCH_ANCHOR}" not in source, (
        f"{PATCH_ANCHOR} is import-bound in preregistration; a rebinding would be lost"
    )
    compat = GATES_DIR / f"{COMPAT_MODULE}.py"
    assert _top_level_definitions(compat) == {"_FACADE_NAME", "facade"}
    assert not _imported_package_modules(compat), "the resolver owns no gate dependency"


def test_facade_rebinding_still_steers_the_release_scale_gate(monkeypatch) -> None:
    """The contract itself: rebind the policy, change what the gate demands."""
    from parity_gates_support import build_report, full_scale_contract

    contract = full_scale_contract()
    report, contract, digest = build_report(contract=contract)
    scale = {**facade.UNIVERSAL_SCALE_REQUIREMENTS, "min_cases": 10_000}
    monkeypatch.setattr(facade, PATCH_ANCHOR, scale)

    decision = facade.evaluate_frontier_parity(report, contract, contract_sha256=digest)

    gate = next(
        check for check in decision["checks"] if check["name"] == "universal_scale_preregistered"
    )
    # The real policy asks for 320 cases and this report registers exactly that.
    assert gate["passed"] is False
    assert gate["required"]["min_cases"] == 10_000


def test_gates_never_depend_on_the_facade() -> None:
    """One direction of dependency only, so the facade can never be a cycle."""
    for path in _gate_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "cortheon.parity", path.name
            elif isinstance(node, ast.Import):
                assert all(alias.name != "cortheon.parity" for alias in node.names), path.name


def test_the_gate_dependency_graph_is_acyclic() -> None:
    graph = {path.stem: _imported_package_modules(path) for path in _gate_modules()}
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


def test_every_gate_module_is_reachable_from_the_facade() -> None:
    """No module may be orphaned, so none can be silently dropped from a build."""
    modules = {path.stem for path in _gate_modules()} - {"__init__"}
    reachable = _imported_package_modules(FACADE)
    frontier = list(reachable)
    while frontier:
        name = frontier.pop()
        for dependency in _imported_package_modules(GATES_DIR / f"{name}.py"):
            if dependency not in reachable:
                reachable.add(dependency)
                frontier.append(dependency)
    assert modules == reachable, f"unreachable modules: {sorted(modules - reachable)}"


def test_each_check_name_has_exactly_one_emitting_module() -> None:
    """A gate name must be searchable to one place, or review cannot trust it."""
    names = {
        "universal_scale_preregistered": "preregistration",
        "labels_withheld_from_contenders": "preregistration",
        "exact_precommitted_repetitions": "coverage",
        "declared_contenders_present": "identity",
        "release_identity_bound": "identity",
        "model_identity_bound_per_execution": "execution_binding",
        "schedule_matches_contract": "execution_binding",
        "absolute_completion_floor": "outcomes",
        "false_allow_ceiling": "outcomes",
        "repeated_case_stability": "outcomes",
        "aggregate_noninferiority:": "noninferiority",
        "domain_noninferiority:": "noninferiority",
        "no_safety_regression:": "noninferiority",
        "independently_metered_contender_costs": "metering",
    }
    for name, owner in names.items():
        emitters = {
            path.stem for path in _gate_modules() if f'"{name}' in path.read_text(encoding="utf-8")
        }
        assert emitters == {owner}, f"{name} is emitted by {sorted(emitters)}, not {owner}"


def test_parity_gates_stays_repository_only() -> None:
    """The release contract is a repository tool; it must never enter an artifact."""
    for relative in ("setup.py", "MANIFEST.in", "pyproject.toml"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "parity" not in source, f"{relative} would ship the parity contract"
    distribution_test = (ROOT / "tests/test_lightweight_distribution.py").read_text(
        encoding="utf-8"
    )
    assert '"/parity_gates/" in member' in distribution_test, (
        "the wheel and sdist tests must exclude parity_gates explicitly"
    )

"""Architecture and compatibility contract for the shipped installer facade."""

from __future__ import annotations

import ast
import inspect
import pickle
from pathlib import Path
from typing import get_type_hints

import cortheon.cognitive_install as facade

ROOT = Path(__file__).parents[1]
FACADE = ROOT / "src/cortheon/cognitive_install.py"
CORE = ROOT / "src/cortheon/cognitive_install_core"
CORE_MEMBERS = {"__init__.py", "config.py", "hosts.py", "lifecycle.py", "model.py", "omp.py"}
OWNERS = {
    "config.py": {
        "package_asset",
        "_is_packaged_adapter_reference",
        "_load_json_config",
        "_atomic_json",
        "_xdg_config_home",
        "_xdg_data_home",
        "_pi_config_home",
        "_omp_config_home",
        "_configured_codex_plugins",
        "_installed_mcp_command",
    },
    "hosts.py": {
        "install_hosts",
        "_preflight_hosts",
        "_preflight_json_string_list",
        "install_opencode",
        "install_pi",
        "install_codex",
        "generic_mcp_config",
        "_normalize_hosts",
        "_run",
        "_configured_codex_marketplaces",
    },
    "lifecycle.py": {
        "host_installation_status",
        "_uninstall_adapter",
        "_uninstall_codex",
        "uninstall_hosts",
    },
    "model.py": {"install_facade_patch_bridge"},
    "omp.py": {
        "_omp_targets",
        "_preflight_omp_config",
        "_preflight_omp_skill",
        "_atomic_text",
        "_install_omp_skill",
        "_restore_omp_skill",
        "install_omp",
        "_omp_skill_owned",
        "_omp_server_owned",
        "_omp_installation_status",
        "_quarantine_skill",
        "_uninstall_omp",
    },
}
EXPECTED_FACADE_NAMES = {
    "Any",
    "InstallError",
    "InstallResult",
    "Iterable",
    "LEGACY_PACKAGE_NAMES",
    "MARKETPLACE_NAME",
    "Path",
    "SUPPORTED_HOSTS",
    "_atomic_json",
    "_configured_codex_marketplaces",
    "_configured_codex_plugins",
    "_is_packaged_adapter_reference",
    "_load_json_config",
    "_normalize_hosts",
    "_pi_config_home",
    "_preflight_hosts",
    "_preflight_json_string_list",
    "_preflight_omp_config",
    "_installed_mcp_command",
    "_install_omp_skill",
    "_omp_config_home",
    "_run",
    "_xdg_config_home",
    "_xdg_data_home",
    "annotations",
    "asdict",
    "dataclass",
    "files",
    "generic_mcp_config",
    "host_installation_status",
    "install_codex",
    "install_hosts",
    "install_opencode",
    "install_pi",
    "install_omp",
    "uninstall_hosts",
    "_uninstall_adapter",
    "_uninstall_codex",
    "_uninstall_omp",
    "json",
    "os",
    "package_asset",
    "shlex",
    "shutil",
    "subprocess",
    "sys",
    "tempfile",
}
SIGNATURES = {
    "InstallResult": "(host: 'str', status: 'str', target: 'str | None', details: 'dict[str, Any]') -> None",
    "package_asset": "(name: 'str') -> 'Path'",
    "_is_packaged_adapter_reference": "(value: 'str', name: 'str') -> 'bool'",
    "install_hosts": "(hosts: 'Iterable[str]', *, scope: 'str' = 'user', project_dir: 'Path | None' = None, dry_run: 'bool' = False, run_codex_cli: 'bool' = True) -> 'list[InstallResult]'",
    "_preflight_hosts": "(hosts: 'list[str]', *, scope: 'str', project_dir: 'Path', dry_run: 'bool', run_codex_cli: 'bool') -> 'None'",
    "_preflight_json_string_list": "(path: 'Path', field: 'str') -> 'None'",
    "install_opencode": "(*, scope: 'str', project_dir: 'Path', dry_run: 'bool') -> 'InstallResult'",
    "install_pi": "(*, scope: 'str', project_dir: 'Path', dry_run: 'bool') -> 'InstallResult'",
    "install_codex": "(*, dry_run: 'bool', run_cli: 'bool' = True, install_root: 'Path | None' = None) -> 'InstallResult'",
    "generic_mcp_config": "() -> 'InstallResult'",
    "host_installation_status": "(*, scope: 'str' = 'user', project_dir: 'Path | None' = None) -> 'dict[str, dict[str, Any]]'",
    "_configured_codex_plugins": "(codex: 'str') -> 'dict[str, str]'",
    "_uninstall_adapter": "(host: 'str', *, scope: 'str', project_dir: 'Path', dry_run: 'bool') -> 'InstallResult'",
    "_uninstall_codex": "(*, dry_run: 'bool', run_cli: 'bool') -> 'InstallResult'",
    "uninstall_hosts": "(hosts: 'Iterable[str]', *, scope: 'str' = 'user', project_dir: 'Path | None' = None, dry_run: 'bool' = False, run_codex_cli: 'bool' = True) -> 'list[InstallResult]'",
    "_normalize_hosts": "(hosts: 'Iterable[str]') -> 'list[str]'",
    "_load_json_config": "(path: 'Path') -> 'dict[str, Any]'",
    "_atomic_json": "(path: 'Path', payload: 'dict[str, Any]', *, backup_existing: 'bool' = False, sort_keys: 'bool' = True) -> 'None'",
    "_run": "(command: 'list[str]') -> 'dict[str, Any]'",
    "_configured_codex_marketplaces": "(codex: 'str') -> 'dict[str, Path]'",
    "_xdg_config_home": "() -> 'Path'",
    "_xdg_data_home": "() -> 'Path'",
    "_pi_config_home": "() -> 'Path'",
    "_omp_config_home": "() -> 'Path'",
    "_installed_mcp_command": "() -> 'str'",
    "_preflight_omp_config": "(path: 'Path') -> 'None'",
    "install_omp": "(*, scope: 'str', project_dir: 'Path', dry_run: 'bool') -> 'InstallResult'",
    "_install_omp_skill": "(skill_root: 'Path', *, dry_run: 'bool') -> 'bool'",
    "_uninstall_omp": "(*, scope: 'str', project_dir: 'Path', dry_run: 'bool') -> 'InstallResult'",
}


def _imports(path: Path, module: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        (isinstance(node, ast.ImportFrom) and node.module == module)
        or (isinstance(node, ast.Import) and any(alias.name == module for alias in node.names))
        for node in ast.walk(tree)
    )


def _core_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.module.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("cortheon.cognitive_install_core.")
    }


def test_installer_files_and_ownership_stay_explicit_and_below_cap() -> None:
    assert {path.name for path in CORE.glob("*.py")} == CORE_MEMBERS
    for path in [FACADE, *sorted(CORE.glob("*.py")), Path(__file__)]:
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 500, path
    for filename, expected in OWNERS.items():
        tree = ast.parse((CORE / filename).read_text(encoding="utf-8"))
        actual = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert actual == expected
        assert not _imports(CORE / filename, "cortheon.cognitive_install")


def test_installer_core_import_graph_is_acyclic() -> None:
    graph = {path.stem: _core_imports(path) for path in CORE.glob("*.py")}

    def visit(name: str, active: set[str], visited: set[str]) -> None:
        assert name not in active, f"cognitive_install_core cycle through {name}"
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


def test_facade_surface_signatures_hints_and_identities_are_stable() -> None:
    assert {name for name in vars(facade) if not name.startswith("__")} == EXPECTED_FACADE_NAMES
    for name, signature in SIGNATURES.items():
        value = getattr(facade, name)
        assert str(inspect.signature(value)) == signature
        assert value.__module__ == "cortheon.cognitive_install"
        hinted = value.__init__ if name == "InstallResult" else value
        assert get_type_hints(hinted)
    assert facade.InstallError.__module__ == "cortheon.cognitive_install"
    assert facade.InstallResult.public.__module__ == "cortheon.cognitive_install"


def test_result_pickle_and_public_shape_stay_stable() -> None:
    result = facade.InstallResult("pi", "present", None, {"changed": False})
    restored = pickle.loads(pickle.dumps(result))
    assert type(restored) is facade.InstallResult
    assert restored == result
    assert restored.public() == {
        "host": "pi",
        "status": "present",
        "target": None,
        "details": {"changed": False},
    }


def test_facade_function_replacements_reach_host_owner(monkeypatch, tmp_path: Path) -> None:
    plugin = tmp_path / "opencode_plugin.js"
    plugin.write_text("export default {}\n", encoding="utf-8")
    writes: list[tuple[Path, dict]] = []
    monkeypatch.setattr(facade, "package_asset", lambda _name: plugin)
    monkeypatch.setattr(facade, "_load_json_config", lambda _path: {})
    monkeypatch.setattr(
        facade,
        "_atomic_json",
        lambda path, payload, **_kwargs: writes.append((path, payload)),
    )

    result = facade.install_opencode(scope="project", project_dir=tmp_path, dry_run=False)

    assert result.details["plugin"] == plugin.as_uri()
    assert writes == [(tmp_path / "opencode.json", {"plugin": [plugin.as_uri()]})]


def test_facade_asdict_replacement_reaches_model_owner(monkeypatch) -> None:
    marker = {"patched": True}
    monkeypatch.setattr(facade, "asdict", lambda _value: marker)
    assert facade.InstallResult("pi", "present", None, {}).public() is marker


def test_only_the_cli_consumes_the_installer_from_shipped_source() -> None:
    consumers = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*.py")
        if path != FACADE and _imports(path, "cortheon.cognitive_install")
    }
    assert consumers == {
        "src/cortheon/cognitive_cli_core/conformance.py",
        "src/cortheon/cognitive_cli_core/diagnostics.py",
        "src/cortheon/cognitive_cli_core/operations.py",
    }

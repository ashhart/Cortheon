"""Host installation status and fail-closed removal operations."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cortheon.cognitive_install_core.config import (
    _atomic_json,
    _configured_codex_plugins,
    _is_packaged_adapter_reference,
    _load_json_config,
    _pi_config_home,
    _xdg_config_home,
    _xdg_data_home,
    package_asset,
)
from cortheon.cognitive_install_core.hosts import (
    _configured_codex_marketplaces,
    _normalize_hosts,
    _preflight_json_string_list,
    _run,
)
from cortheon.cognitive_install_core.model import (
    MARKETPLACE_NAME,
    InstallError,
    InstallResult,
)
from cortheon.cognitive_install_core.omp import (
    _omp_installation_status,
    _omp_targets,
    _preflight_omp_config,
    _preflight_omp_skill,
    _uninstall_omp,
)


def host_installation_status(
    *,
    scope: str = "user",
    project_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Return content-free integration diagnostics for one explicit scope."""

    if scope not in {"user", "project"}:
        raise InstallError("scope must be user or project")
    root = (project_dir or Path.cwd()).resolve()
    checks = (
        (
            "opencode",
            root / "opencode.json"
            if scope == "project"
            else _xdg_config_home() / "opencode" / "opencode.json",
            "plugin",
            package_asset("opencode_plugin.js").as_uri(),
        ),
        (
            "pi",
            root / ".pi" / "settings.json"
            if scope == "project"
            else _pi_config_home() / "settings.json",
            "extensions",
            str(package_asset("pi_extension.ts")),
        ),
    )
    statuses: dict[str, dict[str, Any]] = {}
    for host, path, field, expected in checks:
        try:
            configured = _load_json_config(path).get(field, [])
            valid = isinstance(configured, list) and all(
                isinstance(item, str) for item in configured
            )
            matching = (
                [
                    item
                    for item in configured
                    if _is_packaged_adapter_reference(item, Path(expected).name)
                ]
                if valid
                else []
            )
            statuses[host] = {
                "configured": bool(valid and matching == [expected]),
                "valid": valid,
                "target": str(path),
                "adapter_references": len(matching),
                "scope": scope,
            }
        except InstallError as exc:
            statuses[host] = {
                "configured": False,
                "valid": False,
                "target": str(path),
                "scope": scope,
                "error": str(exc),
            }
    codex_root = (_xdg_data_home() / "cortheon" / "codex-marketplace").resolve()
    codex = shutil.which("codex")
    plugin_id = f"cortheon@{MARKETPLACE_NAME}"
    plugins = _configured_codex_plugins(codex) if codex else {}
    listed = plugin_id in plugins
    manifest = codex_root / "plugins" / "cortheon" / ".codex-plugin" / "plugin.json"
    try:
        expected_version = _load_json_config(manifest).get("version")
    except InstallError:
        expected_version = None
    installed_version = plugins.get(plugin_id)
    version_matches = bool(
        listed and isinstance(expected_version, str) and installed_version == expected_version
    )
    statuses["codex"] = {
        "configured": scope == "user" and listed,
        "valid": scope == "user" and manifest.is_file() and version_matches,
        "target": str(manifest),
        "plugin_listed": listed,
        "installed_version": installed_version,
        "expected_version": expected_version,
        "version_matches": version_matches,
        "scope": scope,
    }
    mcp = shutil.which("cortheon-mcp")
    statuses["generic"] = {
        "configured": mcp is not None,
        "valid": mcp is not None,
        "target": mcp,
        "scope": "process",
        "configuration_only": True,
    }
    statuses["omp"] = _omp_installation_status(scope=scope, project_dir=root)
    return statuses


def _uninstall_adapter(
    host: str,
    *,
    scope: str,
    project_dir: Path,
    dry_run: bool,
) -> InstallResult:
    if host == "opencode":
        path = (
            project_dir / "opencode.json"
            if scope == "project"
            else _xdg_config_home() / "opencode" / "opencode.json"
        )
        field, asset = "plugin", "opencode_plugin.js"
    else:
        path = (
            project_dir / ".pi" / "settings.json"
            if scope == "project"
            else _pi_config_home() / "settings.json"
        )
        field, asset = "extensions", "pi_extension.ts"
    payload = _load_json_config(path)
    configured = payload.get(field, [])
    if not isinstance(configured, list) or not all(isinstance(item, str) for item in configured):
        raise InstallError(f"{path} field {field!r} must be an array of strings")
    updated = [item for item in configured if not _is_packaged_adapter_reference(item, asset)]
    removed = len(configured) - len(updated)
    if removed and not dry_run:
        payload[field] = updated
        _atomic_json(path, payload, backup_existing=True, sort_keys=False)
    return InstallResult(
        host=host,
        status="planned" if dry_run and removed else ("removed" if removed else "absent"),
        target=str(path),
        details={"scope": scope, "removed_references": removed, "changed": bool(removed)},
    )


def _uninstall_codex(*, dry_run: bool, run_cli: bool) -> InstallResult:
    root = (_xdg_data_home() / "cortheon" / "codex-marketplace").resolve()
    commands: list[list[str]] = []
    codex = shutil.which("codex") if run_cli else None
    if run_cli and codex is None:
        raise InstallError("the codex CLI is not on PATH")
    if codex:
        plugin_id = f"cortheon@{MARKETPLACE_NAME}"
        if plugin_id in _configured_codex_plugins(codex):
            commands.append([codex, "plugin", "remove", plugin_id, "--json"])
        configured = _configured_codex_marketplaces(codex)
        if configured.get(MARKETPLACE_NAME) == root:
            commands.append([codex, "plugin", "marketplace", "remove", MARKETPLACE_NAME, "--json"])
    exists = root.exists()
    if exists:
        manifest = root / ".agents" / "plugins" / "marketplace.json"
        if root.is_symlink() or not root.is_dir():
            raise InstallError(f"refusing to remove unsafe Codex marketplace target: {root}")
        try:
            owned = json.loads(manifest.read_text(encoding="utf-8")).get("name") == MARKETPLACE_NAME
        except (OSError, ValueError, AttributeError):
            owned = False
        if not owned:
            raise InstallError(f"refusing to remove unverified Codex marketplace directory: {root}")
    if not dry_run:
        for command in commands:
            _run(command)
        if exists:
            shutil.rmtree(root)
    changed = bool(commands or exists)
    return InstallResult(
        host="codex",
        status="planned" if dry_run and changed else ("removed" if changed else "absent"),
        target=str(root),
        details={"commands": commands, "removed_marketplace": exists, "changed": changed},
    )


def uninstall_hosts(
    hosts: Iterable[str],
    *,
    scope: str = "user",
    project_dir: Path | None = None,
    dry_run: bool = False,
    run_codex_cli: bool = True,
) -> list[InstallResult]:
    normalized = _normalize_hosts(hosts)
    root = (project_dir or Path.cwd()).resolve()
    if scope not in {"user", "project"}:
        raise InstallError("scope must be user or project")
    if scope == "project" and "codex" in normalized:
        raise InstallError("Codex plugins are user-installed; use --scope user for host codex")
    if "codex" in normalized and run_codex_cli and shutil.which("codex") is None:
        raise InstallError("the codex CLI is not on PATH")
    if "codex" in normalized:
        _uninstall_codex(dry_run=True, run_cli=run_codex_cli)
    for host in normalized:
        if host in {"opencode", "pi"}:
            _preflight_json_string_list(
                root / ("opencode.json" if host == "opencode" else ".pi/settings.json")
                if scope == "project"
                else (
                    _xdg_config_home() / "opencode/opencode.json"
                    if host == "opencode"
                    else _pi_config_home() / "settings.json"
                ),
                "plugin" if host == "opencode" else "extensions",
            )
    if "omp" in normalized:
        config_path, skill_file = _omp_targets(scope, root)
        _preflight_omp_config(config_path)
        _preflight_omp_skill(skill_file)
    results: list[InstallResult] = []
    for host in normalized:
        if host in {"opencode", "pi"}:
            results.append(_uninstall_adapter(host, scope=scope, project_dir=root, dry_run=dry_run))
        elif host == "codex":
            results.append(_uninstall_codex(dry_run=dry_run, run_cli=run_codex_cli))
        elif host == "omp":
            results.append(_uninstall_omp(scope=scope, project_dir=root, dry_run=dry_run))
        else:
            results.append(
                InstallResult(
                    host="generic",
                    status="configuration_only",
                    target=None,
                    details={
                        "changed": False,
                        "note": "Remove the printed MCP entry from its host.",
                    },
                )
            )
    return results


for _definition in (
    host_installation_status,
    _uninstall_adapter,
    _uninstall_codex,
    uninstall_hosts,
):
    _definition.__module__ = "cortheon.cognitive_install"

del _definition

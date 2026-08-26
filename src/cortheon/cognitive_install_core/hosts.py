"""Host preflight, installation, and CLI process helpers."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cortheon.cognitive_install_core.config import (
    _atomic_json,
    _configured_codex_plugins,
    _installed_mcp_command,
    _is_packaged_adapter_reference,
    _load_json_config,
    _pi_config_home,
    _xdg_config_home,
    _xdg_data_home,
    package_asset,
)
from cortheon.cognitive_install_core.model import (
    MARKETPLACE_NAME,
    SUPPORTED_HOSTS,
    InstallError,
    InstallResult,
)
from cortheon.cognitive_install_core.omp import (
    _omp_targets,
    _preflight_omp_config,
    _preflight_omp_skill,
    install_omp,
)


def install_hosts(
    hosts: Iterable[str],
    *,
    scope: str = "user",
    project_dir: Path | None = None,
    dry_run: bool = False,
    run_codex_cli: bool = True,
) -> list[InstallResult]:
    requested = [host.strip().lower() for host in hosts]
    normalized = _normalize_hosts(requested)
    all_hosts = not requested or "all" in requested
    if all_hosts:
        normalized = [host for host in normalized if host != "generic"]
    root = (project_dir or Path.cwd()).resolve()
    if scope not in {"user", "project"}:
        raise InstallError("scope must be user or project")
    if scope == "project" and "codex" in normalized:
        raise InstallError("Codex plugins are user-installed; use --scope user for host codex")
    if not all_hosts and "generic" in normalized:
        raise InstallError(
            "generic MCP is configuration-only; use cortheon configure --host generic"
        )
    _preflight_hosts(
        normalized,
        scope=scope,
        project_dir=root,
        dry_run=dry_run,
        run_codex_cli=run_codex_cli,
    )
    results: list[InstallResult] = []
    for host in normalized:
        if host == "opencode":
            results.append(install_opencode(scope=scope, project_dir=root, dry_run=dry_run))
        elif host == "pi":
            results.append(install_pi(scope=scope, project_dir=root, dry_run=dry_run))
        elif host == "codex":
            results.append(install_codex(dry_run=dry_run, run_cli=run_codex_cli))
        elif host == "omp":
            results.append(install_omp(scope=scope, project_dir=root, dry_run=dry_run))
    return results


def _preflight_hosts(
    hosts: list[str],
    *,
    scope: str,
    project_dir: Path,
    dry_run: bool,
    run_codex_cli: bool,
) -> None:
    """Reject deterministic configuration failures before changing any host."""

    if "opencode" in hosts:
        path = (
            project_dir / "opencode.json"
            if scope == "project"
            else _xdg_config_home() / "opencode" / "opencode.json"
        )
        _preflight_json_string_list(path, "plugin")
    if "pi" in hosts:
        path = (
            project_dir / ".pi" / "settings.json"
            if scope == "project"
            else _pi_config_home() / "settings.json"
        )
        _preflight_json_string_list(path, "extensions")
    if "omp" in hosts:
        config_path, skill_file = _omp_targets(scope, project_dir)
        _preflight_omp_config(config_path)
        _preflight_omp_skill(skill_file)
    if "codex" in hosts and run_codex_cli and not dry_run:
        codex = shutil.which("codex")
        if codex is None:
            raise InstallError("the codex CLI is not on PATH")
        root = (_xdg_data_home() / "cortheon" / "codex-marketplace").resolve()
        conflicting = _configured_codex_marketplaces(codex).get(MARKETPLACE_NAME)
        if conflicting is not None and conflicting != root:
            raise InstallError(
                f"Codex marketplace name {MARKETPLACE_NAME!r} already points to "
                f"{conflicting}, not {root}"
            )


def _preflight_json_string_list(path: Path, field: str) -> None:
    if path.is_symlink():
        raise InstallError(f"refusing to rewrite symlinked configuration: {path}")
    payload = _load_json_config(path)
    configured = payload.get(field, [])
    if not isinstance(configured, list) or not all(isinstance(item, str) for item in configured):
        raise InstallError(f"{path} field {field!r} must be an array of strings")


def install_opencode(*, scope: str, project_dir: Path, dry_run: bool) -> InstallResult:
    plugin = package_asset("opencode_plugin.js")
    config_path = (
        project_dir / "opencode.json"
        if scope == "project"
        else _xdg_config_home() / "opencode" / "opencode.json"
    )
    plugin_uri = plugin.as_uri()
    payload = _load_json_config(config_path)
    configured = payload.get("plugin", [])
    if not isinstance(configured, list) or not all(isinstance(item, str) for item in configured):
        raise InstallError(f"{config_path} field 'plugin' must be an array of strings")
    existing_cortheon = [
        item for item in configured if _is_packaged_adapter_reference(item, "opencode_plugin.js")
    ]
    updated = [
        item
        for item in configured
        if not _is_packaged_adapter_reference(item, "opencode_plugin.js")
    ]
    updated.append(plugin_uri)
    changed = updated != configured
    if changed:
        payload["plugin"] = updated
        if not dry_run:
            _atomic_json(config_path, payload, backup_existing=True, sort_keys=False)
    return InstallResult(
        host="opencode",
        status="planned" if dry_run and changed else ("installed" if changed else "present"),
        target=str(config_path),
        details={
            "plugin": plugin_uri,
            "scope": scope,
            "runtime_url": "http://127.0.0.1:8743",
            "changed": changed,
            "replaced_stale_plugins": sum(item != plugin_uri for item in existing_cortheon)
            + max(0, existing_cortheon.count(plugin_uri) - 1),
        },
    )


def install_pi(*, scope: str, project_dir: Path, dry_run: bool) -> InstallResult:
    extension = str(package_asset("pi_extension.ts"))
    settings_path = (
        project_dir / ".pi" / "settings.json"
        if scope == "project"
        else _pi_config_home() / "settings.json"
    )
    payload = _load_json_config(settings_path)
    configured = payload.get("extensions", [])
    if not isinstance(configured, list) or not all(isinstance(item, str) for item in configured):
        raise InstallError(f"{settings_path} field 'extensions' must be an array of strings")
    disabled = f"!{extension}"
    existing_cortheon = [
        item for item in configured if _is_packaged_adapter_reference(item, "pi_extension.ts")
    ]
    enabled = [
        item for item in configured if not _is_packaged_adapter_reference(item, "pi_extension.ts")
    ]
    enabled.append(extension)
    changed = enabled != configured
    if changed:
        payload["extensions"] = enabled
        if not dry_run:
            _atomic_json(settings_path, payload, backup_existing=True, sort_keys=False)
    return InstallResult(
        host="pi",
        status="planned" if dry_run and changed else ("installed" if changed else "present"),
        target=str(settings_path),
        details={
            "extension": extension,
            "scope": scope,
            "runtime_url": "http://127.0.0.1:8743",
            "changed": changed,
            "replaced_stale_extensions": sum(
                item.lstrip("!") != extension for item in existing_cortheon
            )
            + max(0, sum(item == extension for item in existing_cortheon) - 1)
            + sum(item == disabled for item in existing_cortheon),
        },
    )


def install_codex(
    *,
    dry_run: bool,
    run_cli: bool = True,
    install_root: Path | None = None,
) -> InstallResult:
    plugin_source = package_asset("codex_plugins/cortheon")
    root = (install_root or _xdg_data_home() / "cortheon" / "codex-marketplace").resolve()
    plugin_target = root / "plugins" / "cortheon"
    marketplace_path = root / ".agents" / "plugins" / "marketplace.json"
    marketplace = {
        "name": MARKETPLACE_NAME,
        "interface": {"displayName": "Cortheon Local"},
        "plugins": [
            {
                "name": "cortheon",
                "source": {"source": "local", "path": "./plugins/cortheon"},
                "policy": {"installation": "AVAILABLE", "authentication": "ON_USE"},
                "category": "Developer Tools",
            }
        ],
    }
    commands = [
        ["codex", "plugin", "marketplace", "add", str(root), "--json"],
        ["codex", "plugin", "add", f"cortheon@{MARKETPLACE_NAME}", "--json"],
    ]
    if dry_run:
        return InstallResult(
            host="codex",
            status="planned",
            target=str(root),
            details={"plugin_source": str(plugin_source), "commands": commands, "changed": True},
        )
    plugin_target.parent.mkdir(parents=True, exist_ok=True)
    if plugin_target.is_symlink():
        raise InstallError(f"refusing to replace symlinked Codex plugin: {plugin_target}")
    if plugin_target.exists() and not plugin_target.is_dir():
        raise InstallError(f"Codex plugin target is not a directory: {plugin_target}")
    from cortheon import __version__
    from cortheon.cognitive_http import _SOURCE_FINGERPRINT
    from cortheon.cognitive_protocol import CORTHEON_PROTOCOL_VERSION

    with tempfile.TemporaryDirectory(dir=plugin_target.parent, prefix=".cortheon-") as temporary:
        workspace = Path(temporary)
        staged = workspace / "cortheon"
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
        shutil.copytree(plugin_source, staged, ignore=ignore)
        plugin_manifest = staged / ".codex-plugin" / "plugin.json"
        plugin_payload = json.loads(plugin_manifest.read_text(encoding="utf-8"))
        plugin_payload["version"] = f"{__version__}+codex.{_SOURCE_FINGERPRINT}"
        plugin_manifest.write_text(
            json.dumps(plugin_payload, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        scripts = staged / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        python = shlex.quote(sys.executable)
        launchers = {
            "cortheon-mcp": f"{python} -m cortheon.cognitive_mcp",
            "cortheon-runtime": f"{python} -m cortheon.cognitive_cli serve",
        }
        for name, command in launchers.items():
            launcher = scripts / name
            launcher.write_text(
                f'#!/bin/sh\nset -eu\n\nexec {command} "$@"\n',
                encoding="utf-8",
            )
            launcher.chmod(0o755)
        (scripts / "cortheon-runtime.json").write_text(
            json.dumps(
                {
                    "version": __version__,
                    "protocol_version": CORTHEON_PROTOCOL_VERSION,
                    "source_fingerprint": _SOURCE_FINGERPRINT,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        previous = workspace / "previous"
        if plugin_target.exists():
            plugin_target.rename(previous)
        try:
            staged.rename(plugin_target)
        except Exception:
            if previous.exists():
                previous.rename(plugin_target)
            raise
    scripts = plugin_target / "scripts"
    _atomic_json(marketplace_path, marketplace)

    command_results: list[dict[str, Any]] = []
    if run_cli:
        codex = shutil.which("codex")
        if codex is None:
            raise InstallError(f"Codex plugin copied to {root}, but the codex CLI is not on PATH")
        configured = _configured_codex_marketplaces(codex)
        conflicting = configured.get(MARKETPLACE_NAME)
        if conflicting is not None and conflicting != root:
            raise InstallError(
                f"Codex marketplace name {MARKETPLACE_NAME!r} already points to "
                f"{conflicting}, not {root}"
            )
        if root not in configured.values():
            command_results.append(_run([codex, *commands[0][1:]]))
        plugin_id = f"cortheon@{MARKETPLACE_NAME}"
        if plugin_id in _configured_codex_plugins(codex):
            command_results.append(_run([codex, "plugin", "remove", plugin_id, "--json"]))
        command_results.append(_run([codex, *commands[1][1:]]))
    return InstallResult(
        host="codex",
        status="installed",
        target=str(root),
        details={
            "plugin": str(plugin_target),
            "marketplace": MARKETPLACE_NAME,
            "commands": command_results,
            "runtime_launcher": str(scripts / "cortheon-runtime"),
            "changed": True,
        },
    )


def generic_mcp_config() -> InstallResult:
    command = _installed_mcp_command()
    return InstallResult(
        host="generic",
        status="configuration",
        target=None,
        details={
            "mcpServers": {"cortheon": {"command": command, "args": []}},
            "assurance": "cooperative",
            "writes_files": False,
            "note": (
                "Generic MCP cannot intercept host tools. OpenCode and Pi adapters "
                "provide enforced evidence capture; generic hosts must return real "
                "tool results through cortheon_observe."
            ),
        },
    )


def _normalize_hosts(hosts: Iterable[str]) -> list[str]:
    values = [item.strip().lower() for item in hosts]
    if not values or "all" in values:
        return list(SUPPORTED_HOSTS)
    unknown = sorted(set(values) - set(SUPPORTED_HOSTS))
    if unknown:
        raise InstallError(f"unsupported host(s): {', '.join(unknown)}")
    return list(dict.fromkeys(values))


def _run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise InstallError(
            f"{' '.join(command[:4])} failed with exit {completed.returncode}: {message}"
        )
    return {"command": command, "stdout": completed.stdout.strip()[:4_000]}


def _configured_codex_marketplaces(codex: str) -> dict[str, Path]:
    completed = subprocess.run(
        [codex, "plugin", "marketplace", "list", "--json"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        return {}
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {}
    values = payload.get("marketplaces", []) if isinstance(payload, dict) else []
    configured: dict[str, Path] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        root = value.get("root")
        name = value.get("name")
        if isinstance(name, str) and name and isinstance(root, str) and root:
            configured[name] = Path(root).expanduser().resolve()
    return configured


for _definition in (
    install_hosts,
    _preflight_hosts,
    _preflight_json_string_list,
    install_opencode,
    install_pi,
    install_codex,
    generic_mcp_config,
    _normalize_hosts,
    _run,
    _configured_codex_marketplaces,
):
    _definition.__module__ = "cortheon.cognitive_install"

del _definition

"""Safe OMP configuration and bundled-skill lifecycle."""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from cortheon.cognitive_install_core.config import (
    _atomic_json,
    _installed_mcp_command,
    _load_json_config,
    _omp_config_home,
    package_asset,
)
from cortheon.cognitive_install_core.model import InstallError, InstallResult


def _omp_targets(scope: str, project_dir: Path) -> tuple[Path, Path]:
    root = project_dir / ".omp" if scope == "project" else _omp_config_home()
    return root / "mcp.json", root / "skills" / "cortheon-runtime" / "SKILL.md"


def _preflight_omp_config(path: Path) -> None:
    """Reject an OMP config that cannot be updated without following a symlink."""

    if path.is_symlink() or path.parent.is_symlink():
        raise InstallError(f"refusing to rewrite symlinked OMP configuration: {path}")
    payload = _load_json_config(path)
    if not isinstance(payload.get("mcpServers", {}), dict):
        raise InstallError(f"{path} field 'mcpServers' must be an object")


def _preflight_omp_skill(path: Path) -> None:
    """Reject skill paths that could escape or overwrite another file type."""

    labels = (
        (path.parent.parent, "skills directory"),
        (path.parent, "skill directory"),
        (path, "skill file"),
    )
    for candidate, label in labels:
        if candidate.is_symlink():
            raise InstallError(f"refusing to use symlinked OMP skill {label}: {candidate}")
    for candidate, label, expected in (
        (path.parent.parent, "skills directory", "directory"),
        (path.parent, "skill directory", "directory"),
        (path, "skill file", "file"),
    ):
        if candidate.exists() and (
            (expected == "directory" and not candidate.is_dir())
            or (expected == "file" and not candidate.is_file())
        ):
            raise InstallError(f"OMP {label} is not a {expected}: {candidate}")
    package_asset("omp_skill/cortheon-runtime/SKILL.md")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else None
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            temporary.chmod(mode)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _install_omp_skill(skill_root: Path, *, dry_run: bool) -> bool:
    """Install the bundled OMP skill with an atomic file replacement."""

    source = package_asset("omp_skill/cortheon-runtime/SKILL.md")
    target = skill_root / "cortheon-runtime" / "SKILL.md"
    _preflight_omp_skill(target)
    previous = target.read_text(encoding="utf-8") if target.is_file() else None
    updated = source.read_text(encoding="utf-8")
    changed = previous != updated
    if changed and not dry_run:
        _atomic_text(target, updated)
    return changed


def _restore_omp_skill(path: Path, previous: str | None) -> None:
    try:
        if previous is None:
            path.unlink(missing_ok=True)
            with suppress(OSError):
                path.parent.rmdir()
        else:
            _atomic_text(path, previous)
    except OSError as exc:
        raise InstallError(f"failed to roll back OMP skill after config failure: {exc}") from exc


def install_omp(*, scope: str, project_dir: Path, dry_run: bool) -> InstallResult:
    """Register Cortheon's MCP server and skill in one OMP scope."""

    config_path, skill_file = _omp_targets(scope, project_dir)
    _preflight_omp_config(config_path)
    _preflight_omp_skill(skill_file)
    payload = _load_json_config(config_path)
    servers = dict(payload.get("mcpServers", {}))
    entry = {"command": _installed_mcp_command(), "args": []}
    config_changed = servers.get("cortheon") != entry
    skill_previous = skill_file.read_text(encoding="utf-8") if skill_file.is_file() else None
    skill_changed = _install_omp_skill(skill_file.parent.parent, dry_run=dry_run)
    if config_changed:
        servers["cortheon"] = entry
        payload["mcpServers"] = servers
        if not dry_run:
            try:
                _atomic_json(config_path, payload, backup_existing=True, sort_keys=False)
            except Exception:
                if skill_changed:
                    _restore_omp_skill(skill_file, skill_previous)
                raise
    changed = config_changed or skill_changed
    return InstallResult(
        host="omp",
        status="planned" if dry_run and changed else ("installed" if changed else "present"),
        target=str(config_path),
        details={
            "mcp_server": entry,
            "assurance": "cooperative",
            "scope": scope,
            "skills_root": str(skill_file.parent),
            "changed": changed,
            "skill_changed": skill_changed,
        },
    )


def _omp_skill_owned(path: Path) -> bool:
    bundled = package_asset("omp_skill/cortheon-runtime/SKILL.md")
    return bool(
        path.is_file() and path.read_text(encoding="utf-8") == bundled.read_text(encoding="utf-8")
    )


def _omp_server_owned(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value) == {"command", "args"}
        and isinstance(value.get("command"), str)
        and Path(value["command"]).name == "cortheon-mcp"
        and value.get("args") == []
    )


def _omp_installation_status(*, scope: str, project_dir: Path) -> dict[str, Any]:
    config_path, skill_file = _omp_targets(scope, project_dir)
    try:
        _preflight_omp_config(config_path)
        _preflight_omp_skill(skill_file)
        payload = _load_json_config(config_path)
        servers = payload.get("mcpServers", {})
        server = servers.get("cortheon") if isinstance(servers, dict) else None
        configured = isinstance(server, dict) and isinstance(server.get("command"), str)
        server_matches = bool(
            configured and server == {"command": _installed_mcp_command(), "args": []}
        )
        skill_matches = _omp_skill_owned(skill_file)
        return {
            "configured": configured,
            "valid": server_matches and skill_matches,
            "target": str(config_path),
            "skill_present": skill_file.is_file(),
            "skill_matches": skill_matches,
            "scope": scope,
            "assurance": "cooperative",
        }
    except (InstallError, OSError) as exc:
        return {
            "configured": False,
            "valid": False,
            "target": str(config_path),
            "skill_present": skill_file.is_file(),
            "skill_matches": False,
            "scope": scope,
            "error": str(exc),
        }


def _quarantine_skill(path: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".remove",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    os.replace(path, temporary)
    return temporary


def _uninstall_omp(*, scope: str, project_dir: Path, dry_run: bool) -> InstallResult:
    config_path, skill_file = _omp_targets(scope, project_dir)
    _preflight_omp_config(config_path)
    _preflight_omp_skill(skill_file)
    payload = _load_json_config(config_path)
    servers = payload.get("mcpServers", {})
    assert isinstance(servers, dict)
    existing = servers.get("cortheon")
    if existing is not None and not _omp_server_owned(existing):
        raise InstallError(f"refusing to remove an unrecognized OMP server entry: {config_path}")
    removed_server = _omp_server_owned(existing)
    removed_skill = _omp_skill_owned(skill_file)
    changed = removed_server or removed_skill
    if changed and not dry_run:
        quarantine = _quarantine_skill(skill_file) if removed_skill else None
        try:
            if removed_server:
                payload["mcpServers"] = {
                    name: value for name, value in servers.items() if name != "cortheon"
                }
                _atomic_json(config_path, payload, backup_existing=True, sort_keys=False)
        except Exception:
            if quarantine is not None:
                os.replace(quarantine, skill_file)
            raise
        if quarantine is not None:
            quarantine.unlink()
            with suppress(OSError):
                skill_file.parent.rmdir()
    return InstallResult(
        host="omp",
        status="planned" if dry_run and changed else ("removed" if changed else "absent"),
        target=str(config_path),
        details={
            "scope": scope,
            "removed_server": removed_server,
            "removed_skill": removed_skill,
            "changed": changed,
        },
    )


for _definition in (
    _preflight_omp_config,
    _install_omp_skill,
    install_omp,
    _uninstall_omp,
):
    _definition.__module__ = "cortheon.cognitive_install"

del _definition

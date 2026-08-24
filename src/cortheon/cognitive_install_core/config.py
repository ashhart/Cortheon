"""Bundled-asset lookup and atomic host configuration storage."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from cortheon.cognitive_install_core.model import (
    LEGACY_PACKAGE_NAMES,
    InstallError,
)


def package_asset(name: str) -> Path:
    """Resolve a bundled adapter from an unpacked Python installation."""

    candidate = Path(str(files("cortheon").joinpath(name))).resolve()
    if not candidate.is_file() and not candidate.is_dir():
        raise InstallError(f"bundled Cortheon asset is missing: {name}")
    return candidate


def _is_packaged_adapter_reference(value: str, name: str) -> bool:
    normalized = value.lstrip("!").split("?", 1)[0].replace("\\", "/")
    package_names = {"cortheon", *LEGACY_PACKAGE_NAMES}
    return any(normalized.endswith(f"/{package}/{name}") for package in package_names)


def _load_json_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InstallError(
            f"{path} is not strict JSON; Cortheon will not rewrite it: {exc}"
        ) from exc
    except OSError as exc:
        raise InstallError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InstallError(f"{path} must contain a JSON object")
    return value


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
    *,
    backup_existing: bool = False,
    sort_keys: bool = True,
) -> None:
    if path.is_symlink():
        raise InstallError(f"refusing to rewrite symlinked configuration: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = path.stat().st_mode & 0o777 if path.exists() else None
    if backup_existing and path.exists():
        backup = path.with_name(f"{path.name}.cortheon.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=sort_keys)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if existing_mode is not None:
            temporary.chmod(existing_mode)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _xdg_config_home() -> Path:
    configured = os.environ.get("XDG_CONFIG_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".config"


def _xdg_data_home() -> Path:
    configured = os.environ.get("XDG_DATA_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".local" / "share"


def _pi_config_home() -> Path:
    configured = os.environ.get("PI_CODING_AGENT_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".pi" / "agent"


def _configured_codex_plugins(codex: str) -> dict[str, str]:
    completed = subprocess.run(
        [codex, "plugin", "list", "--json"],
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
    values = payload.get("installed", []) if isinstance(payload, dict) else []
    return {
        value["pluginId"]: value["version"]
        for value in values
        if isinstance(value, dict)
        and value.get("installed") is True
        and value.get("enabled") is True
        and isinstance(value.get("pluginId"), str)
        and isinstance(value.get("version"), str)
    }


for _definition in (
    package_asset,
    _is_packaged_adapter_reference,
    _load_json_config,
    _atomic_json,
    _xdg_config_home,
    _xdg_data_home,
    _pi_config_home,
    _configured_codex_plugins,
):
    _definition.__module__ = "cortheon.cognitive_install"

del _definition

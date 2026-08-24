"""Local installation and runtime diagnostics."""

from __future__ import annotations

import json
import platform
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from cortheon import cognitive_cli as surface


def runtime_identity(runtime: dict[str, Any]) -> tuple[dict[str, str], bool]:
    from cortheon.cognitive_http import _SOURCE_FINGERPRINT

    expected = {
        "service": "cortheon-cognitive",
        "version": surface.__version__,
        "protocol_version": surface.protocol_capabilities()["protocol_version"],
        "source_fingerprint": _SOURCE_FINGERPRINT,
        "storage": "memory_only",
    }
    return expected, bool(
        runtime.get("ok") is True
        and all(runtime.get(key) == value for key, value in expected.items())
    )


def doctor(
    runtime_url: str,
    *,
    token: str,
    require_runtime: bool,
    hosts: list[str] | tuple[str, ...],
    scope: str,
    project_dir: str | None,
) -> dict[str, Any]:
    from cortheon.cognitive_install import host_installation_status

    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, *, required: bool = True, **details: Any) -> None:
        checks.append({"name": name, "ok": ok, "required": required, **details})

    python_ok = surface.sys.version_info >= (3, 11)
    record(
        "python",
        python_ok,
        version=platform.python_version(),
        executable=surface.sys.executable,
    )

    for name, path in surface._asset_paths().items():
        record(f"asset:{name}", Path(path).exists(), path=path)

    for host in ("opencode", "pi", "codex"):
        executable = shutil.which(host)
        record(
            f"host:{host}",
            executable is not None,
            required=False,
            executable=executable,
        )

    selected = list(dict.fromkeys(host.casefold() for host in hosts))
    if "all" in selected:
        selected = list(surface.SUPPORTED_HOSTS)
    installation = host_installation_status(
        scope=scope,
        project_dir=Path(project_dir).resolve() if project_dir else None,
    )
    for host, details in installation.items():
        record(
            f"integration:{host}",
            details["configured"] and details["valid"],
            required=host in selected,
            **details,
        )

    runtime = surface._runtime_health(runtime_url, token=token)
    expected_runtime, identity_matches = runtime_identity(runtime)
    reachable = runtime.get("ok") is True
    record(
        "runtime",
        identity_matches,
        required=require_runtime or reachable,
        url=runtime_url,
        details=runtime,
        expected_identity=expected_runtime,
        identity_matches=identity_matches,
    )
    required_failures = [item["name"] for item in checks if item["required"] and not item["ok"]]
    return {
        "ok": not required_failures,
        "version": surface.__version__,
        "protocol": surface.protocol_capabilities(),
        "selected_hosts": selected,
        "scope": scope,
        "required_failures": required_failures,
        "checks": checks,
    }


def runtime_health(url: str, *, token: str) -> dict[str, Any]:
    return _runtime_get(url, "/healthz", token=token)


def runtime_metrics(url: str, *, token: str) -> dict[str, Any]:
    return _runtime_get(url, "/metrics", token=token)


def _runtime_get(url: str, path: str, *, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url.rstrip("/") + path,
        headers={
            "Accept": "application/json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc)}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "runtime response was not an object"}
    return payload


def runtime_results(url: str, *, token: str) -> dict[str, Any]:
    health = surface._runtime_health(url, token=token)
    expected, identity_matches = runtime_identity(health)
    if not identity_matches:
        return {
            "ok": False,
            "runtime_identity_matches": False,
            "error": (
                "runtime identity mismatch" if health.get("ok") is True else "runtime unavailable"
            ),
            "runtime": health,
            "expected_runtime_identity": expected,
        }
    metrics = runtime_metrics(url, token=token)
    if metrics.get("ok") is not True:
        return {"ok": False, "runtime_identity_matches": True, "error": metrics.get("error")}
    return {
        **metrics,
        "runtime_identity_matches": True,
        "scope": "since_runtime_start",
        "runtime_identity": expected,
    }

"""Bounded HTTP and host-payload handling for the Codex hook."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

if __package__:
    from .hook_config import (
        COMPLETE_STATUS_RE,
        EXPECTED_RUNTIME_PROTOCOL,
        RUNTIME_HEALTH_TIMEOUT_SECONDS,
        RUNTIME_START_ATTEMPTS,
        RUNTIME_START_INTERVAL_SECONDS,
        RUNTIME_TIMEOUT_SECONDS,
    )
else:
    from hook_config import (
        COMPLETE_STATUS_RE,
        EXPECTED_RUNTIME_PROTOCOL,
        RUNTIME_HEALTH_TIMEOUT_SECONDS,
        RUNTIME_START_ATTEMPTS,
        RUNTIME_START_INTERVAL_SECONDS,
        RUNTIME_TIMEOUT_SECONDS,
    )

_BOUND_FACADE: ModuleType | None = None
_DEFAULT_RUNTIME_URL = "http://127.0.0.1:8743"
_ACTIVE_RUNTIME_URL: str | None = None


def _bind_facade(module: ModuleType) -> None:
    global _BOUND_FACADE
    _BOUND_FACADE = module


def _facade() -> ModuleType:
    if _BOUND_FACADE is None:
        raise RuntimeError("Cortheon hook facade is not bound")
    return _BOUND_FACADE


def _payload(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("hook input must be an object")
    return value


def _runtime_url() -> str:
    configured = os.environ.get("CORTHEON_RUNTIME_URL")
    if configured is not None:
        return configured.rstrip("/")
    return _ACTIVE_RUNTIME_URL or _DEFAULT_RUNTIME_URL


def _expected_runtime_identity() -> dict[str, str] | None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "cortheon-runtime.json"
    if not path.exists():
        return {"protocol_version": EXPECTED_RUNTIME_PROTOCOL}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    keys = {"version", "protocol_version", "source_fingerprint"}
    if set(value) != keys or not all(isinstance(value[key], str) and value[key] for key in keys):
        return None
    return value


def _runtime_health_payload(url: str) -> dict[str, Any] | None:
    request = urllib.request.Request(
        url.rstrip("/") + "/healthz",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=RUNTIME_HEALTH_TIMEOUT_SECONDS,
        ) as response:
            value = json.load(response)
    except (
        OSError,
        TimeoutError,
        ValueError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ):
        return None
    return value if isinstance(value, dict) else None


def _runtime_matches(value: dict[str, Any] | None, expected: dict[str, str]) -> bool:
    return bool(
        value
        and value.get("ok") is True
        and value.get("service") == "cortheon-cognitive"
        and value.get("storage") == "memory_only"
        and all(value.get(key) == expected_value for key, expected_value in expected.items())
    )


def _runtime_healthy_at(url: str, expected: dict[str, str]) -> bool:
    return _runtime_matches(_runtime_health_payload(url), expected)


def _runtime_healthy() -> bool:
    expected = _expected_runtime_identity()
    return expected is not None and _runtime_healthy_at(_runtime_url(), expected)


def _fallback_runtime_url(expected: dict[str, str]) -> str:
    identity = "\0".join(expected.get(key, "") for key in sorted(expected))
    import hashlib

    offset = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % 1_000
    return f"http://127.0.0.1:{18_000 + offset}"


def _valid_runtime_endpoint(url: str) -> tuple[urllib.parse.SplitResult, int] | None:
    try:
        endpoint = urllib.parse.urlsplit(url)
        port = endpoint.port
    except ValueError:
        return None
    if (
        endpoint.scheme != "http"
        or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
        or endpoint.username is not None
        or endpoint.password is not None
        or endpoint.path not in {"", "/"}
        or endpoint.query
        or endpoint.fragment
        or port is None
    ):
        return None
    return endpoint, port


def _runtime_start_lock(port: int) -> tuple[Path, int] | None:
    path = Path(tempfile.gettempdir()) / f"cortheon-runtime-{port}.lock"
    for _attempt in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(descriptor, str(os.getpid()).encode())
            return path, descriptor
        except FileExistsError:
            try:
                stale = time.time() - path.stat().st_mtime > 5.0
            except OSError:
                stale = False
            if not stale:
                return None
            try:
                path.unlink()
            except OSError:
                return None
    return None


def _wait_for_runtime(url: str, expected: dict[str, str]) -> bool:
    for _attempt in range(RUNTIME_START_ATTEMPTS):
        time.sleep(RUNTIME_START_INTERVAL_SECONDS)
        if _runtime_healthy_at(url, expected):
            return True
    return False


def _start_runtime(
    url: str,
    expected: dict[str, str],
    command: list[str],
) -> bool:
    global _ACTIVE_RUNTIME_URL
    validated = _valid_runtime_endpoint(url)
    if validated is None:
        return False
    _endpoint, port = validated
    lock = _runtime_start_lock(port)
    if lock is None:
        if _wait_for_runtime(url, expected):
            _ACTIVE_RUNTIME_URL = url
            return True
        return False
    lock_path, descriptor = lock
    environment = {
        **os.environ,
        "CORTHEON_COGNITIVE_BIND": "127.0.0.1",
        "CORTHEON_COGNITIVE_PORT": str(port),
    }
    try:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
            env=environment,
        )
        if _wait_for_runtime(url, expected):
            _ACTIVE_RUNTIME_URL = url
            return True
        return False
    except OSError:
        return False
    finally:
        os.close(descriptor)
        with contextlib.suppress(OSError):
            lock_path.unlink()


def _runtime_command() -> list[str] | None:
    configured = os.environ.get("CORTHEON_RUNTIME_COMMAND", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        return [str(candidate)] if candidate.is_absolute() and candidate.is_file() else None
    launcher = Path(__file__).resolve().parents[1] / "scripts" / "cortheon-runtime"
    if launcher.is_file() and os.access(launcher, os.X_OK):
        return [str(launcher)]
    executable = shutil.which("cortheon")
    return [executable, "serve"] if executable else None


def _ensure_runtime() -> bool:
    global _ACTIVE_RUNTIME_URL
    expected = _expected_runtime_identity()
    if expected is None:
        return False
    if _runtime_healthy():
        return True
    forced = os.environ.get("CORTHEON_RUNTIME_AUTOSTART", "").strip().casefold()
    configured = "CORTHEON_RUNTIME_URL" in os.environ
    if configured and forced not in {"1", "true", "yes", "on"}:
        return False
    command = _runtime_command()
    if command is None:
        return False
    primary = _runtime_url()
    if configured:
        return _start_runtime(primary, expected, command)
    primary_health = _runtime_health_payload(primary)
    if primary_health is None:
        if _start_runtime(primary, expected, command):
            return True
    elif _runtime_matches(primary_health, expected):
        return True
    fallback = _fallback_runtime_url(expected)
    if _runtime_healthy_at(fallback, expected):
        _ACTIVE_RUNTIME_URL = fallback
        return True
    return _start_runtime(fallback, expected, command)


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("CORTHEON_COGNITIVE_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        _runtime_url() + path,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=RUNTIME_TIMEOUT_SECONDS) as response:
            value = json.load(response)
    except (
        OSError,
        TimeoutError,
        ValueError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ):
        return None
    return value if isinstance(value, dict) else None


def _identity(payload: dict[str, Any], *, include_turn: bool = True) -> dict[str, str] | None:
    session_id = payload.get("session_id")
    turn_id = payload.get("turn_id")
    if not isinstance(session_id, str) or not session_id:
        return None
    identity = {"host": "codex", "host_session_id": session_id}
    if include_turn:
        if not isinstance(turn_id, str) or not turn_id:
            return None
        identity["turn_id"] = turn_id
    return identity


def _is_cortheon_skill_bootstrap(payload: dict[str, Any]) -> bool:
    tool_input = payload.get("tool_input")
    try:
        serialized = json.dumps(tool_input, separators=(",", ":")).lower()
    except (TypeError, ValueError):
        return False
    return "skill.md" in serialized and (
        "cortheon-runtime" in serialized or "plugins/cortheon" in serialized
    )


def _tool_succeeded(response: Any) -> bool:
    if isinstance(response, dict):
        if response.get("isError") is True or response.get("killed") is True:
            return False
        for key in ("exit_code", "exitCode", "code"):
            value = response.get(key)
            if isinstance(value, int) and value != 0:
                return False
        status = response.get("status")
        if isinstance(status, str) and status.lower() in {"error", "failed"}:
            return False
        if response.get("error"):
            return False
    return response is not None


def _tool_output(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response[:40_000]
    if isinstance(response, dict):
        for key in ("output", "stdout", "text", "content"):
            value = response.get(key)
            if isinstance(value, str):
                return value[:40_000]
            if isinstance(value, list):
                texts = [
                    str(item.get("text"))
                    for item in value
                    if isinstance(item, dict) and isinstance(item.get("text"), str)
                ]
                if texts:
                    return "\n".join(texts)[:40_000]
    try:
        serialized = json.dumps(response, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        serialized = str(response)
    return serialized[:40_000]


_WEB_FIELDS = (
    "url",
    "finalUrl",
    "final_url",
    "title",
    "snippet",
    "content",
    "text",
    "publishedAt",
    "published_at",
    "date",
    "provider",
    "sourceType",
    "source_type",
    "authority",
)


def _web_item(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    item = {
        key: candidate[:4_000] if isinstance(candidate, str) else candidate
        for key in _WEB_FIELDS
        if isinstance((candidate := value.get(key)), (str, int, float))
        and not isinstance(candidate, bool)
    }
    return item or None


def _tool_metadata(response: Any) -> dict[str, Any]:
    """Keep only direct, attributable web fields from a host response."""

    if not isinstance(response, dict):
        return {}
    containers = [response]
    for key in ("details", "structuredContent", "structured_content"):
        value = response.get(key)
        if isinstance(value, dict):
            containers.append(value)
    metadata: dict[str, Any] = {}
    for container in containers:
        direct = _web_item(container)
        if direct:
            metadata.update(direct)
        results = container.get("results")
        if isinstance(results, list) and 0 < len(results) <= 8:
            cleaned = [_web_item(item) for item in results]
            if all(item is not None for item in cleaned):
                metadata["results"] = cleaned
    return metadata


def _contains_certified_completion(value: Any) -> bool:
    if isinstance(value, dict):
        status = value.get("status")
        if isinstance(status, str) and status.lower() == "complete":
            return True
        return any(_contains_certified_completion(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_certified_completion(item) for item in value)
    if isinstance(value, str):
        if COMPLETE_STATUS_RE.search(value):
            return True
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return False
        return _contains_certified_completion(decoded)
    return False

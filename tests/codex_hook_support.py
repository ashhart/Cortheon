"""Shared harness for exercising the Codex hook exactly as Codex runs it.

Codex copies a plugin directory into its own cache and executes the facade
there with a plain interpreter, so every check that matters runs the file
from a copied directory under ``python -I``: isolated mode drops
``PYTHONPATH`` and the user site directory, which is what proves the hook
never depends on an importable ``cortheon`` package.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

ROOT = Path(__file__).parents[1]
HOOKS = ROOT / "src/cortheon/codex_plugins/cortheon/hooks"
HOOK_SOURCE_FILES = {
    "cortheon_hook.py",
    "hook_adapter.py",
    "hook_config.py",
    "hook_entry.py",
    "hook_events.py",
    "hook_transport.py",
}
HOOK_PLUGIN_FILES = {*HOOK_SOURCE_FILES, "hooks.json"}
# A value the hook must never echo back to the host transcript.
SESSION_ID = "private-session-value"
TOKEN = "private-token-value"


class _HookRuntime(BaseHTTPRequestHandler):
    """Minimal stand-in for the Cortheon runtime's hook endpoints."""

    requests: ClassVar[list[tuple[str, dict[str, Any]]]] = []
    authorizations: ClassVar[list[str | None]] = []

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self.send_error(404)
            return
        body = json.dumps(
            {
                "ok": True,
                "service": "cortheon-cognitive",
                "protocol_version": "1.0.0",
                "storage": "memory_only",
            },
            separators=(",", ":"),
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.requests.append((self.path, payload))
        self.authorizations.append(self.headers.get("Authorization"))
        response = (
            {"automatic": False}
            if self.path == "/v1/hooks/register"
            else {"allow": False, "reason": "staged completion withheld"}
        )
        body = json.dumps(response, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def hook_runtime() -> Iterator[str]:
    """Serve the hook endpoints on a loopback port for the block's duration."""

    _HookRuntime.requests = []
    _HookRuntime.authorizations = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HookRuntime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def runtime_requests() -> list[tuple[str, dict[str, Any]]]:
    return list(_HookRuntime.requests)


def runtime_authorizations() -> list[str | None]:
    return list(_HookRuntime.authorizations)


def stage_hook_directory(source: Path, target: Path) -> Path:
    """Copy a plugin's ``hooks`` directory the way Codex populates its cache."""

    shutil.copytree(source, target)
    return target


def run_hook(
    staged: Path,
    payload: object,
    *,
    runtime: str | None = None,
    raw: str | None = None,
    overrides: dict[str, str] | None = None,
    flags: tuple[str, ...] = ("-I",),
) -> subprocess.CompletedProcess[str]:
    """Run the staged facade under ``python -I`` with a scrubbed environment.

    ``flags`` may add ``-S`` to drop site-packages as well, which removes any
    installed ``cortheon`` from the interpreter the hook runs in.
    """

    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "CORTHEON_COGNITIVE_TOKEN"}
    }
    if runtime is not None:
        environment["CORTHEON_RUNTIME_URL"] = runtime
    environment.update(overrides or {})
    return subprocess.run(
        [sys.executable, *flags, str(staged / "cortheon_hook.py")],
        input=json.dumps(payload) if raw is None else raw,
        capture_output=True,
        text=True,
        timeout=10,
        cwd=staged.parent,
        env=environment,
        check=False,
    )

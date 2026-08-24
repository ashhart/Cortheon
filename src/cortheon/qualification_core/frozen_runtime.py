"""Isolated process and HTTP client for the frozen runtime."""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import subprocess
import sys
import urllib.request
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BOOTSTRAP = r"""
import json,os,sys
control_fd=int(os.environ.pop('CORTHEON_CONTROL_FD'))
status_fd=int(os.environ.pop('CORTHEON_STATUS_FD'))
with os.fdopen(control_fd,'rb',closefd=True) as stream:
    control=json.load(stream)
sys.path.insert(0,control['source_root'])
from cortheon.cognitive_http import build_server
import cortheon.cognitive_http as module
if not module.__file__.startswith(control['source_root']):
    raise RuntimeError('historical runtime import escaped its artifact')
escaped=[]
for name,loaded in sys.modules.items():
    if name != 'cortheon' and not name.startswith('cortheon.'):
        continue
    location=getattr(loaded,'__file__',None)
    if location and not location.startswith(control['source_root']):
        escaped.append(name)
if escaped:
    raise RuntimeError('current Cortheon modules entered historical runtime')
server=build_server('127.0.0.1',0,token=control['token'])
with os.fdopen(status_fd,'w',closefd=True) as stream:
    json.dump({'port':server.server_port,'module':module.__file__},stream)
    stream.flush()
server.serve_forever()
"""


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _json_get(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=2) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("frozen runtime response is not an object")
    return value


def _json_post(url: str, token: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("frozen runtime response is not an object")
    return value


@dataclass(slots=True)
class FrozenRuntime:
    root: Path
    adapter: Path
    url: str
    token: str
    process: subprocess.Popen[bytes]
    initial_digest: str

    def metrics(self) -> dict[str, int]:
        payload = _json_get(self.url + "/metrics", self.token)
        names = (
            "sessions_started",
            "observations_accepted",
            "sessions_completed",
            "completion_withheld",
            "sessions_evidence_closed",
            "sessions_abandoned",
            "controller_decisions",
            "controller_alternatives_considered",
        )
        return {name: int(payload.get(name, 0)) for name in names}

    def health(self) -> dict[str, Any]:
        return _json_get(self.url + "/healthz", self.token)

    def unchanged(self) -> bool:
        return _tree_digest(self.root) == self.initial_digest

    def abandon_active(self) -> int:
        resumed = _json_post(self.url + "/v1/resume", self.token, {"limit": 3})
        sessions = resumed.get("sessions")
        if not isinstance(sessions, list) or len(sessions) > 3:
            raise ValueError("frozen runtime returned an invalid session inventory")
        abandoned = 0
        for item in sessions:
            if not isinstance(item, dict) or not isinstance(item.get("session_id"), str):
                raise ValueError("frozen runtime returned an invalid session identity")
            if item.get("status") in {"complete", "abandoned"}:
                continue
            _json_post(
                self.url + "/v1/abandon",
                self.token,
                {"session_id": item["session_id"]},
            )
            abandoned += 1
        if self.health().get("active_sessions") != 0:
            raise ValueError("frozen runtime cleanup did not close every session")
        return abandoned

    def control_payload(self) -> bytes:
        return json.dumps(
            {
                "schema_version": 1,
                "cognitive_token": self.token,
                "runtime_url": self.url,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()


def stop_runtime(process: subprocess.Popen[bytes]) -> None:
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
    finally:
        if process.stderr is not None:
            process.stderr.close()


def start_runtime(root: Path, adapter: Path, token: str) -> FrozenRuntime:
    bootstrap = root / "runtime.py"
    bootstrap.write_text(BOOTSTRAP, encoding="utf-8")
    control_read, control_write = os.pipe()
    status_read, status_write = os.pipe()
    process: subprocess.Popen[bytes] | None = None
    open_fds = {control_read, control_write, status_read, status_write}
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "CORTHEON_COGNITIVE_TOKEN"}
    }
    environment.update(
        CORTHEON_CONTROL_FD=str(control_read),
        CORTHEON_STATUS_FD=str(status_write),
    )
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", str(bootstrap)],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            pass_fds=(control_read, status_write),
        )
        for descriptor in (control_read, status_write):
            os.close(descriptor)
            open_fds.remove(descriptor)
        payload = json.dumps(
            {"source_root": str(root / "src"), "token": token},
            separators=(",", ":"),
        ).encode()
        os.write(control_write, payload)
        os.close(control_write)
        open_fds.remove(control_write)
        with selectors.DefaultSelector() as selector:
            selector.register(status_read, selectors.EVENT_READ)
            if not selector.select(5):
                raise ValueError("frozen runtime did not report readiness")
        descriptor = status_read
        open_fds.remove(status_read)
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            status = json.load(stream)
        if (
            not isinstance(status, dict)
            or set(status) != {"port", "module"}
            or type(status["port"]) is not int
            or not 1 <= status["port"] <= 65_535
            or not isinstance(status["module"], str)
            or not status["module"].startswith(str(root / "src"))
        ):
            raise ValueError("frozen runtime readiness record is invalid")
        runtime = FrozenRuntime(
            root=root,
            adapter=adapter,
            url=f"http://127.0.0.1:{status['port']}",
            token=token,
            process=process,
            initial_digest=_tree_digest(root),
        )
        health = runtime.health()
        if (
            health.get("protocol_version") != "1.0.0"
            or health.get("storage") != "memory_only"
            or health.get("active_sessions") != 0
        ):
            raise ValueError("frozen runtime health contract is invalid")
        return runtime
    except BaseException:
        for descriptor in open_fds:
            with suppress(OSError):
                os.close(descriptor)
        if process is not None:
            stop_runtime(process)
        raise

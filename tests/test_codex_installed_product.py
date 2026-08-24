"""Installed-wheel proof for the Codex launchers and HTTP lifecycle."""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
from pathlib import Path
from typing import Any

from cognitive_mcp_packaging_support import ROOT, build_wheel


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _json_get(url: str) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=0.25) as response:
            value = json.load(response)
    except (OSError, TimeoutError, ValueError, urllib.error.URLError):
        return None
    return value if isinstance(value, dict) else None


def _wait_for_health(base: str, *, present: bool) -> dict[str, Any] | None:
    for _attempt in range(80):
        value = _json_get(base + "/healthz")
        if (value is not None) is present:
            return value
        time.sleep(0.05)
    raise AssertionError(f"runtime health did not become present={present}")


def _hook_process(
    python: Path,
    plugin: Path,
    environment: dict[str, str],
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [str(python), "-I", str(plugin / "hooks" / "cortheon_hook.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=plugin,
        env=environment,
    )


def _run_hook(
    python: Path,
    plugin: Path,
    payload: dict[str, Any],
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(python), "-I", str(plugin / "hooks" / "cortheon_hook.py")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=10,
        cwd=plugin,
        env=environment,
        check=False,
    )


def _terminate_recorded_processes(path: Path) -> None:
    if not path.exists():
        return
    for value in set(path.read_text(encoding="utf-8").splitlines()):
        with contextlib.suppress(ProcessLookupError, ValueError):
            os.kill(int(value), signal.SIGTERM)


def test_fresh_venv_codex_cache_launchers_and_runtime_lifecycle(tmp_path: Path) -> None:
    wheel = build_wheel(ROOT, tmp_path / "wheel")
    environment_root = tmp_path / "environment"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment_root)
    installed_python = environment_root / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    install_environment = os.environ.copy()
    install_environment.pop("PYTHONHOME", None)
    install_environment.pop("PYTHONPATH", None)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(installed_python),
            "install",
            "--no-deps",
            "--no-index",
            str(wheel),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=install_environment,
        check=True,
    )
    marketplace = tmp_path / "marketplace"
    installed = subprocess.run(
        [
            str(installed_python),
            "-I",
            "-c",
            (
                "import json,sys;from pathlib import Path;"
                "from cortheon.cognitive_install import install_codex;"
                "print(json.dumps(install_codex(dry_run=False,run_cli=False,"
                "install_root=Path(sys.argv[1])).details))"
            ),
            str(marketplace),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
        check=True,
    )
    plugin_source = Path(json.loads(installed.stdout)["plugin"])
    plugin = tmp_path / "codex-cache" / "cortheon"
    plugin.parent.mkdir()
    shutil.copytree(plugin_source, plugin)
    for launcher in ("cortheon-mcp", "cortheon-runtime"):
        text = (plugin / "scripts" / launcher).read_text(encoding="utf-8")
        assert shlex.quote(str(installed_python)) in text

    initialized = subprocess.run(
        [str(plugin / "scripts" / "cortheon-mcp")],
        input=json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            }
        )
        + "\n",
        capture_output=True,
        text=True,
        timeout=10,
        cwd=plugin,
        check=True,
    )
    assert json.loads(initialized.stdout)["result"]["serverInfo"]["name"] == "cortheon"

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    pids = tmp_path / "runtime-pids"
    owner = tmp_path / "runtime-owner"
    owner.write_text(
        "#!/bin/sh\nset -eu\n"
        + f"printf '%s\\n' \"$$\" >> {shlex.quote(str(pids))}\n"
        + f'exec {shlex.quote(str(plugin / "scripts" / "cortheon-runtime"))} "$@"\n',
        encoding="utf-8",
    )
    owner.chmod(0o755)
    hook_environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH", "CORTHEON_COGNITIVE_TOKEN"}
    }
    hook_environment.update(
        {
            "CORTHEON_RUNTIME_URL": base,
            "CORTHEON_RUNTIME_AUTOSTART": "1",
            "CORTHEON_RUNTIME_COMMAND": str(owner),
        }
    )
    prompt = "Research the current release from fresh web sources and cite it."
    processes = [
        _hook_process(
            Path(sys.executable),
            plugin,
            hook_environment,
        )
        for _index in range(4)
    ]
    try:
        outputs = []
        for process in processes:
            stdout, stderr = process.communicate(
                input=json.dumps(
                    {
                        "hook_event_name": "UserPromptSubmit",
                        "session_id": f"concurrent-{processes.index(process)}",
                        "turn_id": "turn",
                        "prompt": prompt,
                    }
                ),
                timeout=10,
            )
            assert process.returncode == 0 and stderr == ""
            outputs.append(json.loads(stdout))
        assert all(
            "CORTHEON AUTOMATIC SESSION IS ACTIVE"
            in item["hookSpecificOutput"]["additionalContext"]
            for item in outputs
        )
        health = _wait_for_health(base, present=True)
        assert health is not None and health["active_hook_turns"] == 4
        assert len(set(pids.read_text(encoding="utf-8").splitlines())) == 1

        for index in range(3):
            ended = _run_hook(
                Path(sys.executable),
                plugin,
                {"hook_event_name": "SessionEnd", "session_id": f"concurrent-{index}"},
                hook_environment,
            )
            assert ended.returncode == 0 and ended.stderr == ""
        health = _json_get(base + "/healthz")
        assert health is not None and health["active_hook_turns"] == 1

        _terminate_recorded_processes(pids)
        _wait_for_health(base, present=False)
        stopped = _run_hook(
            Path(sys.executable),
            plugin,
            {
                "hook_event_name": "Stop",
                "session_id": "concurrent-3",
                "turn_id": "turn",
                "last_assistant_message": "Unverified answer.",
            },
            hook_environment,
        )
        assert json.loads(stopped.stdout) == {
            "systemMessage": "Cortheon runtime became unavailable; this answer was not certified."
        }

        restarted = _run_hook(
            Path(sys.executable),
            plugin,
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "restarted",
                "turn_id": "turn",
                "prompt": prompt,
            },
            hook_environment,
        )
        assert "CORTHEON AUTOMATIC SESSION IS ACTIVE" in restarted.stdout
        _wait_for_health(base, present=True)
        _run_hook(
            Path(sys.executable),
            plugin,
            {"hook_event_name": "SessionEnd", "session_id": "restarted"},
            hook_environment,
        )
        health = _json_get(base + "/healthz")
        assert health is not None and health["active_hook_turns"] == 0
    finally:
        _terminate_recorded_processes(pids)

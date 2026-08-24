"""The Codex hook has to survive packaging, not just the repository.

Codex copies a plugin directory into its own cache and runs the facade there
with an isolated interpreter, so a wheel that shipped only the facade left an
installed plugin unable to import its own siblings. Every check below runs
the hook from a directory copied out of a real installed wheel, under
``python -I``, with no importable ``cortheon`` package anywhere on the path.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from codex_hook_support import (
    HOOK_PLUGIN_FILES,
    HOOK_SOURCE_FILES,
    HOOKS,
    SESSION_ID,
    TOKEN,
    hook_runtime,
    run_hook,
    runtime_authorizations,
    runtime_requests,
    stage_hook_directory,
)
from cognitive_mcp_packaging_support import ROOT, build_sdist, build_wheel, install

WHEEL_HOOK_DIR = "cortheon/codex_plugins/cortheon/hooks/"
SDIST_HOOK_DIR = "cortheon-0.1.0/src/cortheon/codex_plugins/cortheon/hooks/"
PROMPT = {
    "hook_event_name": "UserPromptSubmit",
    "session_id": SESSION_ID,
    "turn_id": "turn",
    "prompt": "Read src/example.py and explain the current implementation.",
}
STOP = {
    "hook_event_name": "Stop",
    "session_id": SESSION_ID,
    "turn_id": "turn",
    "last_assistant_message": "Unverified answer.",
}
PRE_TOOL = {
    "hook_event_name": "PreToolUse",
    "session_id": SESSION_ID,
    "turn_id": "turn",
    "tool_name": "Bash",
    "tool_input": {"command": "ls"},
}
POST_TOOL = {
    "hook_event_name": "PostToolUse",
    "session_id": SESSION_ID,
    "turn_id": "turn",
    "tool_name": "Bash",
    "tool_response": {"output": "example output", "exit_code": 0},
}
SESSION_END = {"hook_event_name": "SessionEnd", "session_id": SESSION_ID}


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build both packaging paths once and install each resulting wheel."""

    base = tmp_path_factory.mktemp("codex-hook-packaging")
    wheel = build_wheel(ROOT, base / "wheel")
    sdist = build_sdist(base / "sdist")
    sdist_wheel = build_wheel(sdist, base / "from-sdist")
    return {
        "sdist": sdist,
        "wheel": wheel,
        "sdist_wheel": sdist_wheel,
        "wheel_install": install(wheel, base / "install"),
        "sdist_wheel_install": install(sdist_wheel, base / "install-from-sdist"),
    }


def _hook_members(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    return {name.removeprefix(WHEEL_HOOK_DIR) for name in names if name.startswith(WHEEL_HOOK_DIR)}


def _staged(artifacts: dict[str, Path], key: str, tmp_path: Path) -> Path:
    installed = artifacts[f"{key}_install"] / "cortheon/codex_plugins/cortheon/hooks"
    return stage_hook_directory(installed, tmp_path / f"codex-cache-{key}" / "hooks")


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_source_names_are_the_six_the_packaging_contract_covers() -> None:
    assert {path.name for path in HOOKS.glob("*.py")} == HOOK_SOURCE_FILES
    assert len(HOOK_SOURCE_FILES) == 6


def test_both_wheels_and_the_sdist_ship_every_hook_file(artifacts: dict[str, Path]) -> None:
    assert _hook_members(artifacts["wheel"]) == HOOK_PLUGIN_FILES
    assert _hook_members(artifacts["sdist_wheel"]) == HOOK_PLUGIN_FILES

    with tarfile.open(artifacts["sdist"], "r:gz") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
    assert {
        name.removeprefix(SDIST_HOOK_DIR) for name in names if name.startswith(SDIST_HOOK_DIR)
    } == HOOK_PLUGIN_FILES


def test_fresh_venv_installer_launches_real_runtime_and_mcp(
    artifacts: dict[str, Path], tmp_path: Path
) -> None:
    environment = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True, timeout=30)
    python = environment / "bin" / "python"
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", "--no-index", artifacts["wheel"]],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    root = tmp_path / "marketplace"
    installed = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import json,pathlib;from cortheon.cognitive_install import install_codex;"
                f"r=install_codex(dry_run=False,run_cli=False,install_root=pathlib.Path({str(root)!r}));"
                "print(json.dumps(r.public()))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    plugin = Path(json.loads(installed.stdout)["details"]["plugin"])
    mcp = plugin / "scripts" / "cortheon-mcp"
    runtime = plugin / "scripts" / "cortheon-runtime"
    assert str(python) in mcp.read_text(encoding="utf-8")
    assert str(python) in runtime.read_text(encoding="utf-8")

    initialized = subprocess.run(
        [str(mcp)],
        input=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert json.loads(initialized.stdout)["result"]["serverInfo"]["name"] == "cortheon"

    port = _free_port()
    pid_path = tmp_path / "runtime.pid"
    wrapper = tmp_path / "runtime-wrapper"
    wrapper.write_text(
        "#!/bin/sh\nset -eu\nprintf '%s' \"$$\" > "
        + shlex.quote(str(pid_path))
        + "\nexec "
        + shlex.quote(str(runtime))
        + ' "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    prompt = subprocess.run(
        [str(python), "-I", str(plugin / "hooks" / "cortheon_hook.py")],
        input=json.dumps(PROMPT),
        capture_output=True,
        text=True,
        timeout=10,
        env={
            **os.environ,
            "CORTHEON_RUNTIME_URL": f"http://127.0.0.1:{port}",
            "CORTHEON_RUNTIME_AUTOSTART": "1",
            "CORTHEON_RUNTIME_COMMAND": str(wrapper),
        },
        check=True,
    )
    try:
        context = json.loads(prompt.stdout)["hookSpecificOutput"]["additionalContext"]
        assert "AUTOMATIC SESSION IS ACTIVE" in context
        assert pid_path.exists()
    finally:
        if pid_path.exists():
            os.kill(int(pid_path.read_text()), signal.SIGTERM)


def test_the_two_packaging_paths_produce_equivalent_wheels(artifacts: dict[str, Path]) -> None:
    """Membership and every member's bytes agree between both build paths."""

    with (
        zipfile.ZipFile(artifacts["wheel"]) as source_built,
        zipfile.ZipFile(artifacts["sdist_wheel"]) as sdist_built,
    ):
        names = set(source_built.namelist())
        assert names == set(sdist_built.namelist())
        differing = {
            name
            for name in names
            # RECORD lists the other members' hashes, so it follows them.
            if not name.endswith(".dist-info/RECORD")
            and source_built.read(name) != sdist_built.read(name)
        }
    assert differing == set()


@pytest.mark.parametrize("built_from", ["wheel", "sdist_wheel"])
def test_cache_copied_wheel_hook_runs_without_the_cortheon_package(
    artifacts: dict[str, Path], tmp_path: Path, built_from: str
) -> None:
    staged = _staged(artifacts, built_from, tmp_path)
    # pip byte-compiles what it installs, so the copy also carries a
    # __pycache__ directory; the shipped files themselves must be exact.
    assert {path.name for path in staged.iterdir() if path.is_file()} == HOOK_PLUGIN_FILES
    with hook_runtime() as runtime:
        submitted = run_hook(
            staged, PROMPT, runtime=runtime, overrides={"CORTHEON_COGNITIVE_TOKEN": TOKEN}
        )
        stopped = run_hook(
            staged, STOP, runtime=runtime, overrides={"CORTHEON_COGNITIVE_TOKEN": TOKEN}
        )
        paths = [path for path, _payload in runtime_requests()]
        authorizations = runtime_authorizations()

    assert submitted.returncode == stopped.returncode == 0
    assert submitted.stderr == stopped.stderr == ""
    context = json.loads(submitted.stdout)["hookSpecificOutput"]
    assert context["hookEventName"] == "UserPromptSubmit"
    assert "CORTHEON IS ACTIVE." in context["additionalContext"]
    assert json.loads(stopped.stdout) == {
        "decision": "block",
        "reason": "staged completion withheld",
    }
    assert paths == ["/v1/hooks/register", "/v1/hooks/stop"]
    # The identity travels to the runtime and the token travels as a header;
    # neither is ever echoed into the host transcript.
    assert all(payload["host_session_id"] == SESSION_ID for _path, payload in runtime_requests())
    assert authorizations == [f"Bearer {TOKEN}"] * 2
    for stream in (submitted.stdout, submitted.stderr, stopped.stdout, stopped.stderr):
        assert SESSION_ID not in stream
        assert TOKEN not in stream


@pytest.mark.parametrize("built_from", ["wheel", "sdist_wheel"])
def test_cache_copied_wheel_hook_needs_no_cortheon_package_at_all(
    artifacts: dict[str, Path], tmp_path: Path, built_from: str
) -> None:
    """Adding -S removes site-packages, so no installed cortheon can be reached."""

    staged = _staged(artifacts, built_from, tmp_path)
    unreachable = subprocess.run(
        [sys.executable, "-I", "-S", "-c", "import cortheon"],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=staged.parent,
        check=False,
    )
    assert unreachable.returncode != 0 and "ModuleNotFoundError" in unreachable.stderr

    for payload in (PROMPT, STOP):
        with hook_runtime() as runtime:
            isolated = run_hook(staged, payload, runtime=runtime, flags=("-I", "-S"))
        with hook_runtime() as runtime:
            ordinary = run_hook(staged, payload, runtime=runtime)
        assert isolated.returncode == 0, isolated.stderr
        assert isolated.stderr == ""
        assert isolated.stdout == ordinary.stdout


@pytest.mark.parametrize("built_from", ["wheel", "sdist_wheel"])
def test_cache_copied_wheel_hook_degrades_explicitly(
    artifacts: dict[str, Path], tmp_path: Path, built_from: str
) -> None:
    """Malformed input stays inert; an absent runtime is visible and bounded."""

    staged = _staged(artifacts, built_from, tmp_path)
    for raw in ("not-json", "", "[1, 2]", '"a string"', "null"):
        malformed = run_hook(staged, None, raw=raw)
        assert malformed.returncode == 0, raw
        assert malformed.stdout == malformed.stderr == "", raw

    # Past MAX_INPUT_CHARS the hook stops reading and releases the turn rather
    # than forwarding an unbounded prompt to the runtime.
    oversize = run_hook(
        staged,
        {**PROMPT, "prompt": "x" * 1_000_100},
        runtime="http://127.0.0.1:1",
    )
    assert oversize.returncode == 0
    assert oversize.stdout == oversize.stderr == ""

    # Port 1 on loopback refuses immediately, so this is the unavailable case
    # rather than a timeout: the turn still gets its context and is released.
    offline = "http://127.0.0.1:1"
    submitted = run_hook(staged, PROMPT, runtime=offline)
    stopped = run_hook(staged, STOP, runtime=offline)
    assert submitted.returncode == stopped.returncode == 0
    assert submitted.stderr == stopped.stderr == ""
    assert (
        "CORTHEON IS UNAVAILABLE."
        in json.loads(submitted.stdout)["hookSpecificOutput"]["additionalContext"]
    )
    assert json.loads(stopped.stdout) == {
        "systemMessage": "Cortheon runtime became unavailable; this answer was not certified."
    }


@pytest.mark.parametrize("built_from", ["wheel", "sdist_wheel"])
def test_packaged_hook_answers_exactly_as_the_repository_source_does(
    artifacts: dict[str, Path], tmp_path: Path, built_from: str
) -> None:
    """Compaction is formatting only: same inputs, same bytes out."""

    packaged = _staged(artifacts, built_from, tmp_path)
    source = tmp_path / "source-cache" / "hooks"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.mkdir()
    for name in HOOK_PLUGIN_FILES:
        shutil.copy2(HOOKS / name, source / name)

    # Every event hooks.json registers, so no packaged handler is left unproven.
    for payload in (PROMPT, PRE_TOOL, POST_TOOL, STOP, SESSION_END):
        with hook_runtime() as runtime:
            from_source = run_hook(source, payload, runtime=runtime)
            source_paths = [path for path, _payload in runtime_requests()]
        with hook_runtime() as runtime:
            from_package = run_hook(packaged, payload, runtime=runtime)
            package_paths = [path for path, _payload in runtime_requests()]
        assert source_paths == package_paths
        assert from_source.returncode == from_package.returncode == 0
        assert from_source.stdout == from_package.stdout
        assert from_source.stderr == from_package.stderr == ""


@pytest.mark.parametrize("built_from", ["wheel", "sdist_wheel"])
def test_every_shipped_hook_module_imports_under_isolated_python(
    artifacts: dict[str, Path], tmp_path: Path, built_from: str
) -> None:
    """All six modules are real, packaged code, not a bundle the facade
    carries: each one imports on its own from the cache copy, with no
    PYTHONPATH, no user site directory, and no site-packages at all."""

    staged = _staged(artifacts, built_from, tmp_path)
    for name in sorted(HOOK_SOURCE_FILES):
        imported = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                f"import sys;sys.path.insert(0, {str(staged)!r});import {name[:-3]}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=staged.parent,
            check=False,
        )
        assert imported.returncode == 0, f"{name}: {imported.stderr}"
        assert imported.stderr == ""


@pytest.mark.parametrize("missing", sorted(HOOK_SOURCE_FILES - {"cortheon_hook.py"}))
def test_a_cache_copy_missing_one_module_fails_loudly(
    artifacts: dict[str, Path], tmp_path: Path, missing: str
) -> None:
    """A half-copied plugin directory is not a working one.

    The facade cannot enforce a lifecycle whose code is absent, so the only
    honest answers are to work or to say so. Silently exiting zero would
    leave the host believing a turn was governed when nothing governed it."""

    staged = _staged(artifacts, "wheel", tmp_path / missing)
    (staged / missing).unlink()

    broken = run_hook(staged, PROMPT, runtime="http://127.0.0.1:1")
    assert broken.returncode != 0
    assert "ModuleNotFoundError" in broken.stderr
    assert broken.stdout == ""

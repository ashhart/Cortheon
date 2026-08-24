"""Runtime truth for generated code, without executing it.

Static source checks can be wrong in both directions: a re-export or C
extension the parser cannot see (false block), or a decorator-mangled
signature the source text misrepresents (false allow). This module asks the
live installed package instead: import the real module inside a cached,
isolated virtualenv, resolve the exact attribute paths the generated code
calls, and signature-bind the keywords it passes. No generated line ever
executes — only import, getattr, and ``inspect.signature().bind_partial``
against the real objects. That makes it the arbiter rung: cheap enough to run
on every block verdict, grounded enough to overrule the source parser.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from cortheon.verifier import _scrubbed_env, _venv_python

MAX_PROBES = 12

# Runs inside the cached venv. Reads {"import_name", "probes"} on stdin,
# writes one JSON entry per probe: resolved, echoed kwargs, and which of them
# the live signature rejects. bind_partial with one keyword at a time so blame
# lands on the exact argument, and never on positionals we do not have.
PROBE_SOURCE = """
import importlib, inspect, json, sys
spec = json.load(sys.stdin)
module = importlib.import_module(spec["import_name"])
out = []
for probe in spec["probes"]:
    entry = {
        "path": probe["path"],
        "kwargs": probe["kwargs"],
        "resolved": True,
        "signature_known": False,
        "unexpected_kwargs": [],
    }
    target = module
    for part in probe["path"].split("."):
        try:
            target = getattr(target, part)
        except AttributeError:
            entry["resolved"] = False
            break
    if entry["resolved"] and probe["kwargs"]:
        try:
            signature = inspect.signature(target)
        except (ValueError, TypeError):
            signature = None
        if signature is not None:
            entry["signature_known"] = True
            for name in probe["kwargs"]:
                try:
                    signature.bind_partial(**{name: None})
                except TypeError:
                    entry["unexpected_kwargs"].append(name)
    out.append(entry)
print(json.dumps(out))
"""


def extract_call_probes(code: str, import_root: str) -> list[dict[str, Any]]:
    """Module-rooted calls the code makes: attribute path + keyword names.

    Only paths provably reachable from the package root are probed: bare
    ``import root [as x]`` attribute chains and top-level
    ``from root import name [as x]`` bindings. Instance-method calls and
    submodule imports stay with the static checker — a getattr miss there
    would be an artifact of the probe, not of the code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    module_aliases: set[str] = set()
    from_bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == import_root:
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == import_root:
            for alias in node.names:
                if alias.name != "*":
                    from_bindings[alias.asname or alias.name] = alias.name

    probes: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        path: str | None = None
        func = node.func
        if isinstance(func, ast.Name) and func.id in from_bindings:
            path = from_bindings[func.id]
        elif isinstance(func, ast.Attribute):
            parts: list[str] = []
            inner: ast.expr = func
            while isinstance(inner, ast.Attribute):
                parts.append(inner.attr)
                inner = inner.value
            if isinstance(inner, ast.Name) and inner.id in module_aliases:
                path = ".".join(reversed(parts))
        if not path:
            continue
        kwargs = tuple(keyword.arg for keyword in node.keywords if keyword.arg)
        key = (path, kwargs)
        if key in seen:
            continue
        seen.add(key)
        probes.append({"path": path, "kwargs": list(kwargs)})
        if len(probes) >= MAX_PROBES:
            break
    return probes


def run_bind_probe(
    python: Path | str,
    import_name: str,
    probes: list[dict[str, Any]],
    timeout_seconds: int = 30,
) -> list[dict[str, Any]] | None:
    """Execute the probe in the given interpreter; None means no verdict.

    A failed import or crashed probe is infrastructure uncertainty (wrong
    import-name guess, native build quirk) — the caller must treat it as
    "runtime unavailable", never as evidence against the answer.
    """
    if not probes:
        return []
    payload = json.dumps({"import_name": import_name, "probes": probes})
    try:
        completed = subprocess.run(
            [str(python), "-c", PROBE_SOURCE],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=_scrubbed_env(Path(python).parent),
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None
    try:
        results = json.loads(completed.stdout.strip())
    except ValueError:
        return None
    return results if isinstance(results, list) else None


class RuntimeEnvPool:
    """Cached isolated virtualenvs, one per package==version.

    The first request for a package pays venv-create + pip-install once;
    every later probe is a subprocess import against the ready environment.
    Build failures are remembered for the process lifetime so a broken
    package cannot stall every request.

    Two usability additions over the naive pool:

    - envs persist on disk (the ``.ready`` marker), so a proxy restart does not
      pay rebuild costs — only the very first request for a package ever blocks.
    - ``prewarm`` builds a list of envs in background threads at startup, so the
      common packages are ready before the first request arrives; and
      ``bind_check`` never blocks on a cold build when ``build_lazily`` is set
      — it returns ``None`` ("runtime unavailable, try again") and triggers the
      build in the background, so the proxy can return the answer unverified
      with a banner instead of stalling the request path for tens of seconds.
    """

    def __init__(self, root: Path | str, build_timeout_seconds: int = 240) -> None:
        self.root = Path(root)
        self.build_timeout_seconds = build_timeout_seconds
        self._lock = threading.Lock()
        self._failed: set[str] = set()
        self._building: set[str] = set()

    def _key(self, package: str, version: str) -> str:
        return f"{package.lower().replace('/', '_')}-{version}"

    @staticmethod
    def _specs_key(specs: list[str]) -> str:
        """Deterministic cache key for a set of pip install specs.

        Sorted + normalized so ['httpx','rich'] and ['rich','httpx'] share an
        env. A short sha256 keeps the directory name filesystem-safe and stable
        across processes (the pool is disk-persistent across proxy restarts).
        """
        import hashlib

        normalized = ",".join(sorted(s.strip().lower() for s in specs if s.strip()))
        digest = hashlib.sha256(normalized.encode()).hexdigest()[:12]
        return f"specs-{digest}"

    def ready(self, package: str, version: str) -> bool:
        """Is a built, importable env already on disk for this package==version?"""
        env_dir = self.root / self._key(package, version)
        return (env_dir / ".ready").exists() and _venv_python(env_dir).exists()

    def _ready_key(self, key: str) -> bool:
        env_dir = self.root / key
        return (env_dir / ".ready").exists() and _venv_python(env_dir).exists()

    def is_building(self, package: str, version: str) -> bool:
        key = self._key(package, version)
        with self._lock:
            return key in self._building

    def _build_sync_specs(self, key: str, specs: list[str]) -> Path | None:
        """Build an env at ``root/key`` installing ``specs`` (pip install args).

        Empty ``specs`` builds a stdlib-only venv (no pip install step) — a
        program that imports only the standard library still gets isolation and
        a scrubbed env. Caller owns the lock semantics.
        """
        env_dir = self.root / key
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if env_dir.exists():
                shutil.rmtree(env_dir)
            subprocess.run(
                [sys.executable, "-m", "venv", str(env_dir)],
                check=True,
                capture_output=True,
                timeout=self.build_timeout_seconds,
            )
            python = _venv_python(env_dir)
            if specs:  # skip pip entirely for stdlib-only programs
                subprocess.run(
                    [str(python), "-m", "pip", "install", "--quiet", *specs],
                    check=True,
                    capture_output=True,
                    timeout=self.build_timeout_seconds,
                )
            (env_dir / ".ready").touch()
            return python
        except (subprocess.SubprocessError, OSError):
            with self._lock:
                self._failed.add(key)
            shutil.rmtree(env_dir, ignore_errors=True)
            return None

    def _build_sync(self, package: str, version: str) -> Path | None:
        """Build the env synchronously. Caller is responsible for the lock semantics.

        Delegates to the multi-spec builder so single- and multi-package envs
        share one code path. The key format is unchanged (``pkg-version``) so
        existing on-disk envs keep being reused.
        """
        return self._build_sync_specs(self._key(package, version), [f"{package}=={version}"])

    def _build_background(self, package: str, version: str) -> None:
        """Build in a daemon thread; returns immediately. Used by prewarm and
        the build-lazily path so the request thread is never blocked on a cold build."""
        key = self._key(package, version)
        with self._lock:
            if key in self._building or key in self._failed:
                return
            if self.ready(package, version):
                return
            self._building.add(key)

        def _work() -> None:
            try:
                self._build_sync(package, version)
            finally:
                with self._lock:
                    self._building.discard(key)

        threading.Thread(target=_work, daemon=True).start()

    def prewarm(self, packages: list[tuple[str, str]]) -> int:
        """Best-effort background build of envs for (package, version) pairs.

        Called at startup so common packages are ready before the first request.
        Non-blocking: returns the count of builds it kicked off; never raises.
        """
        kicked = 0
        for package, version in packages:
            if not self.ready(package, version):
                self._build_background(package, version)
                kicked += 1
        return kicked

    def python_for(self, package: str, version: str, *, wait: bool = True) -> Path | None:
        """Resolve a python for package==version, building it if needed.

        With ``wait=True`` (default, backward-compatible) a missing env blocks
        until built — the original behaviour, used by the offline CLI tools.
        With ``wait=False`` a not-yet-built env triggers a background build and
        returns ``None`` immediately ("unavailable, try again") so the proxy's
        request path is never stalled by a cold venv create.
        """
        key = self._key(package, version)
        with self._lock:
            if key in self._failed:
                return None
        if self.ready(package, version):
            return _venv_python(self.root / key)
        if not wait:
            self._build_background(package, version)
            return None
        # Synchronous build path. Serialize per-key via the lock so two requests
        # for the same cold package do not race the rmtree.
        with self._lock:
            if key in self._building:
                building = True
            else:
                self._building.add(key)
                building = False
        if building:
            # Another thread is building it; wait for the .ready marker.
            return self._await_ready(package, version)
        try:
            return self._build_sync(package, version)
        finally:
            with self._lock:
                self._building.discard(key)

    def _await_ready(self, package: str, version: str, timeout: int | None = None) -> Path | None:
        import time

        deadline = None if timeout is None else time.monotonic() + timeout
        env_dir = self.root / self._key(package, version)
        while True:
            with self._lock:
                if self._key(package, version) in self._failed:
                    return None
            if self.ready(package, version):
                return _venv_python(env_dir)
            if deadline and time.monotonic() > deadline:
                return None
            time.sleep(0.1)

    def _await_ready_key(self, key: str, timeout: int | None = None) -> Path | None:
        import time

        deadline = None if timeout is None else time.monotonic() + timeout
        env_dir = self.root / key
        while True:
            with self._lock:
                if key in self._failed:
                    return None
            if self._ready_key(key):
                return _venv_python(env_dir)
            if deadline and time.monotonic() > deadline:
                return None
            time.sleep(0.1)

    def python_for_specs(self, specs: list[str], *, wait: bool = True) -> Path | None:
        """Resolve a python for a multi-package env (used by the behavioral rung).

        ``specs`` are pip install arguments (e.g. ``["httpx==0.28.1", "rich"]``);
        an empty list yields a stdlib-only venv. One env serves all blocks of a
        program so import cost is paid once. Cached by a deterministic hash of
        the spec set, so two programs with the same dependencies share an env,
        and the cache persists across proxy restarts.

        Same ``wait`` contract as ``python_for``: ``wait=False`` triggers a
        background build and returns ``None`` immediately so the request path is
        never blocked by a cold venv create.
        """
        normalized = [s.strip() for s in specs if s and s.strip()]
        key = self._specs_key(normalized)
        with self._lock:
            if key in self._failed:
                return None
        if self._ready_key(key):
            return _venv_python(self.root / key)
        if not wait:
            self._build_background_specs(key, normalized)
            return None
        with self._lock:
            if key in self._building:
                building = True
            else:
                self._building.add(key)
                building = False
        if building:
            return self._await_ready_key(key)
        try:
            return self._build_sync_specs(key, normalized)
        finally:
            with self._lock:
                self._building.discard(key)

    def _build_background_specs(self, key: str, specs: list[str]) -> None:
        with self._lock:
            if key in self._building or key in self._failed:
                return
            if self._ready_key(key):
                return
            self._building.add(key)

        def _work() -> None:
            try:
                self._build_sync_specs(key, specs)
            finally:
                with self._lock:
                    self._building.discard(key)

        threading.Thread(target=_work, daemon=True).start()

    def bind_check(
        self,
        package: str,
        version: str,
        import_name: str,
        probes: list[dict[str, Any]],
        *,
        wait: bool = True,
    ) -> list[dict[str, Any]] | None:
        if not probes:
            return []
        python = self.python_for(package, version, wait=wait)
        if python is None:
            return None
        return run_bind_probe(python, import_name, probes)

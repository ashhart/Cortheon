"""Local runtime diagnostics for Cortheon (`cortheon doctor`).

Runs 5 fast, honest, non-destructive diagnostic checks:
  1. OS Sandbox Containment (live `run_python` execution + TCP/outside-write probes)
  2. Local Solver Stack (Python 3.11+, SymPy, Z3, NumPy, optional pytest)
  3. Coding Harness Integrations (OMP, Pi, OpenCode, Claude Code, Cursor, Codex)
  4. Backend Probe (`/v1/models` via `TensorFoldClient`, flavor & reasoning headroom)
  5. Live Fast-Route & Throughput Probe (optional tiny `enable_thinking=False` call)
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import httpx

from .client import CallSpec, TensorFoldClient
from .integrations import ALL_HARNESSES
from .sandbox import run_python, sandbox_backend

CheckStatus = Literal["pass", "warn", "fail", "skip"]

_FLAVOR_TENSORFOLD = "TensorFold Native"
_FLAVOR_GENERIC_OPENAI = "Generic OpenAI (oMLX / Ollama / llama.cpp / vLLM / LM Studio)"

_REQUIRED_SOLVERS: tuple[tuple[str, str], ...] = (
    ("sympy", "sympy"),
    ("z3", "z3-solver"),
    ("numpy", "numpy"),
)
_OPTIONAL_SOLVERS: tuple[tuple[str, str], ...] = (
    ("pytest", "pytest"),
)

_SANDBOX_FULL_PROBE_SCRIPT = """\
import socket
from pathlib import Path

print("EXEC=OK")

try:
    sock = socket.create_connection(("1.1.1.1", 80), timeout=0.4)
    sock.close()
    print("NET=ALLOWED")
except PermissionError as exc:
    print(f"NET=BLOCKED:PermissionError:{exc.errno}")
except TimeoutError:
    print("NET=TIMEOUT")
except OSError as exc:
    print(f"NET=BLOCKED:OSError:{exc.errno}")

probe_path = Path.home() / ".cortheon_sandbox_probe_never_created"
try:
    with open(probe_path, "w", encoding="utf-8") as fh:
        fh.write("probe")
    try:
        probe_path.unlink(missing_ok=True)
    except OSError:
        pass
    print("WRITE=ESCAPED")
except PermissionError as exc:
    print(f"WRITE=BLOCKED:PermissionError:{exc.errno}")
except OSError as exc:
    print(f"WRITE=BLOCKED:OSError:{exc.errno}")
"""

_SANDBOX_EXEC_ONLY_SCRIPT = 'print("EXEC=OK")\n'


@dataclass
class CheckResult:
    id: str
    title: str
    status: CheckStatus
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "summary": self.summary,
            "details": self.details,
            "hint": self.hint,
        }


@dataclass
class DoctorReport:
    url: str
    checks: list[CheckResult]
    recommended_streams: int = 4
    recommended_reason: str = "Balanced default (~4-7 calls with early-abort)"
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return all(c.status != "fail" for c in self.checks)

    def check_by_id(self, check_id: str) -> Optional[CheckResult]:
        for c in self.checks:
            if c.id == check_id:
                return c
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "url": self.url,
            "elapsed_s": round(self.elapsed_s, 3),
            "recommended_streams": self.recommended_streams,
            "recommended_reason": self.recommended_reason,
            "checks": [c.to_dict() for c in self.checks],
        }


def _parse_kv_stdout(stdout: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in (stdout or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            parsed[k.strip()] = v.strip()
    return parsed


def check_sandbox() -> CheckResult:
    """Live check of OS sandbox containment (`macos-seatbelt` / `linux-bwrap` / fallback)."""
    backend = sandbox_backend()
    os_contained = backend in ("macos-seatbelt", "linux-bwrap")
    script = _SANDBOX_FULL_PROBE_SCRIPT if os_contained else _SANDBOX_EXEC_ONLY_SCRIPT

    try:
        res = run_python(script, timeout_s=4.0)
    except Exception as exc:
        return CheckResult(
            id="sandbox",
            title="OS Sandbox Containment",
            status="fail",
            summary=f"{backend}: sandbox invocation error ({type(exc).__name__}: {exc})",
            details={
                "backend": backend,
                "os_containment": os_contained,
                "exec_ok": False,
                "network_blocked": None,
                "outside_write_blocked": None,
            },
            hint="Verify Python and OS sandbox binary availability on this host.",
        )

    kv = _parse_kv_stdout(res.stdout)
    exec_ok = bool(res.ok and kv.get("EXEC") == "OK")
    elapsed_ms = round(res.elapsed_s * 1000.0, 1)

    if os_contained:
        net_val = kv.get("NET", "")
        write_val = kv.get("WRITE", "")
        net_blocked = net_val.startswith("BLOCKED:")
        write_blocked = write_val.startswith("BLOCKED:")
        details = {
            "backend": backend,
            "os_containment": True,
            "exec_ok": exec_ok,
            "network_blocked": net_blocked,
            "network_probe": net_val or "unknown",
            "outside_write_blocked": write_blocked,
            "outside_write_probe": write_val or "unknown",
            "probe_elapsed_ms": elapsed_ms,
        }
        if exec_ok and net_blocked and write_blocked:
            return CheckResult(
                id="sandbox",
                title="OS Sandbox Containment",
                status="pass",
                summary=f"{backend} active (exec OK, outbound TCP blocked, outside write blocked; {elapsed_ms:.0f}ms)",
                details=details,
            )

        failures: list[str] = []
        if not exec_ok:
            err_tail = res.short_error(160) or f"rc={res.returncode}"
            failures.append(f"exec failed ({err_tail})")
        if not net_blocked:
            failures.append(f"outbound TCP not blocked ({net_val or 'unknown'})")
        if not write_blocked:
            failures.append(f"outside-tempdir write not blocked ({write_val or 'unknown'})")
        return CheckResult(
            id="sandbox",
            title="OS Sandbox Containment",
            status="fail",
            summary=f"{backend} containment check failed: {'; '.join(failures)}",
            details=details,
            hint="Inspect OS sandbox permissions or run pytest tests/test_sandbox_security.py.",
        )

    details = {
        "backend": backend,
        "os_containment": False,
        "exec_ok": exec_ok,
        "network_blocked": None,
        "outside_write_blocked": None,
        "probe_elapsed_ms": elapsed_ms,
    }
    if not exec_ok:
        err_tail = res.short_error(160) or f"rc={res.returncode}"
        return CheckResult(
            id="sandbox",
            title="OS Sandbox Containment",
            status="fail",
            summary=f"{backend}: basic Python execution failed ({err_tail})",
            details=details,
            hint="Verify the active Python interpreter works in subprocesses.",
        )

    if backend == "unsafe-disabled":
        return CheckResult(
            id="sandbox",
            title="OS Sandbox Containment",
            status="warn",
            summary=f"unsafe-disabled (exec OK in {elapsed_ms:.0f}ms; OS network/write probes skipped)",
            details=details,
            hint="Unset CORTHEON_UNSAFE_NO_SANDBOX to re-enable OS sandbox containment.",
        )

    return CheckResult(
        id="sandbox",
        title="OS Sandbox Containment",
        status="warn",
        summary=f"subprocess-rlimit fallback (exec OK in {elapsed_ms:.0f}ms; POSIX rlimit + scrubbed env, no OS network/fs jail)",
        details=details,
        hint="Install bubblewrap (`bwrap`) on Linux for full OS network & filesystem isolation.",
    )


def _probe_module(mod_name: str, dist_name: str) -> tuple[bool, Optional[str], Optional[str]]:
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"

    version: Optional[str] = getattr(mod, "__version__", None)
    if not version and hasattr(mod, "get_version_string"):
        try:
            version = str(mod.get_version_string())
        except Exception:
            version = None
    if not version:
        try:
            version = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            version = "installed"
    return True, str(version), None


def check_solver_stack() -> CheckResult:
    """Check Python >= 3.11 and required local math/verification libraries (plus optional pytest)."""
    py_ver = platform.python_version()
    py_ok = sys.version_info >= (3, 11)

    packages: dict[str, dict[str, Any]] = {}
    missing_required: list[str] = []
    missing_optional: list[str] = []
    required_dist_map = dict(_REQUIRED_SOLVERS)

    for mod_name, dist_name in _REQUIRED_SOLVERS:
        ok, ver, err = _probe_module(mod_name, dist_name)
        packages[mod_name] = {"installed": ok, "version": ver, "required": True, "error": err}
        if not ok:
            missing_required.append(mod_name)

    for mod_name, dist_name in _OPTIONAL_SOLVERS:
        ok, ver, err = _probe_module(mod_name, dist_name)
        packages[mod_name] = {"installed": ok, "version": ver, "required": False, "error": err}
        if not ok:
            missing_optional.append(mod_name)

    details = {
        "python_version": py_ver,
        "python_ok": py_ok,
        "packages": packages,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
    }

    parts: list[str] = [f"Python {py_ver}"]
    for mod_name, _ in _REQUIRED_SOLVERS:
        info = packages[mod_name]
        parts.append(f"{mod_name} {info['version']}" if info["installed"] else f"{mod_name} MISSING")
    for mod_name, _ in _OPTIONAL_SOLVERS:
        info = packages[mod_name]
        parts.append(f"{mod_name} {info['version']}" if info["installed"] else f"{mod_name} optional/absent")

    if not py_ok or missing_required:
        hints: list[str] = []
        if not py_ok:
            hints.append("Upgrade to Python 3.11 or newer.")
        if missing_required:
            dists = " ".join(required_dist_map[m] for m in missing_required)
            hints.append(f"Install missing required solvers: pip install {dists}")
        return CheckResult(
            id="solver_stack",
            title="Local Solver Stack",
            status="fail",
            summary=" | ".join(parts),
            details=details,
            hint=" ".join(hints),
        )

    return CheckResult(
        id="solver_stack",
        title="Local Solver Stack",
        status="pass",
        summary=" | ".join(parts),
        details=details,
    )


def check_harnesses(
    *,
    home_dir: Optional[Path] = None,
    project_dir: Optional[Path] = None,
) -> CheckResult:
    """Non-destructively report which coding harnesses are detected and where `/cortheon` is installed."""
    custom_home = home_dir is not None
    home = (home_dir or Path.home()).expanduser()
    proj = project_dir.expanduser() if project_dir is not None else (None if custom_home else Path.cwd())

    harnesses: dict[str, dict[str, Any]] = {}
    detected: list[str] = []
    installed: list[str] = []

    for name in ALL_HARNESSES:
        installed_paths: list[str] = []
        has_cli = bool(shutil.which(name))
        has_dir = False

        if name == "omp":
            omp_dir = home / ".omp"
            has_dir = omp_dir.is_dir()
            candidates = [omp_dir / "agent" / "extensions" / "cortheon.ts"]
            profiles_dir = omp_dir / "profiles"
            if profiles_dir.is_dir():
                candidates.extend(sorted(profiles_dir.glob("*/agent/extensions/cortheon.ts")))
            if proj is not None:
                candidates.append(proj / ".omp" / "extensions" / "cortheon.ts")
        elif name == "pi":
            pi_dir = home / ".pi"
            has_dir = pi_dir.is_dir()
            candidates = [pi_dir / "agent" / "extensions" / "cortheon.ts"]
            if proj is not None:
                candidates.append(proj / ".pi" / "extensions" / "cortheon.ts")
        elif name == "opencode":
            oc_dir = home / ".config" / "opencode"
            has_dir = oc_dir.is_dir()
            candidates = [oc_dir / "command" / "cortheon.md"]
            if proj is not None:
                candidates.append(proj / ".opencode" / "command" / "cortheon.md")
        elif name == "claude":
            cl_dir = home / ".claude"
            has_dir = cl_dir.is_dir()
            candidates = [cl_dir / "commands" / "cortheon.md"]
            if proj is not None:
                candidates.append(proj / ".claude" / "commands" / "cortheon.md")
        elif name == "cursor":
            cu_dir = home / ".cursor"
            has_dir = cu_dir.is_dir() or (
                not custom_home and sys.platform == "darwin" and Path("/Applications/Cursor.app").is_dir()
            )
            candidates = [cu_dir / "commands" / "cortheon.md"]
            if proj is not None:
                candidates.append(proj / ".cursor" / "commands" / "cortheon.md")
        else:  # codex
            has_dir = (home / ".codex").is_dir() or (home / ".agents").is_dir()
            candidates = [home / ".agents" / "skills" / "cortheon" / "SKILL.md"]
            if proj is not None:
                candidates.append(proj / ".agents" / "skills" / "cortheon" / "SKILL.md")

        seen: set[str] = set()
        for cand in candidates:
            if cand.is_file():
                s = str(cand)
                if s not in seen:
                    seen.add(s)
                    installed_paths.append(s)

        is_installed = bool(installed_paths)
        is_detected = bool(has_cli or has_dir or is_installed)
        harnesses[name] = {
            "detected": is_detected,
            "installed": is_installed,
            "installed_paths": installed_paths,
        }
        if is_detected:
            detected.append(name)
        if is_installed:
            installed.append(name)

    missing_on_detected = [h for h in detected if h not in installed]
    details = {
        "harnesses": harnesses,
        "detected": detected,
        "installed": installed,
        "missing_on_detected": missing_on_detected,
    }

    if installed:
        hint = (
            f"Install into remaining detected harness(es): cortheon install-commands --harness {','.join(missing_on_detected)}"
            if missing_on_detected
            else ""
        )
        return CheckResult(
            id="harnesses",
            title="Harness Integrations",
            status="pass",
            summary=f"/cortheon installed: {', '.join(installed)} | detected: {', '.join(detected)}",
            details=details,
            hint=hint,
        )

    if detected:
        return CheckResult(
            id="harnesses",
            title="Harness Integrations",
            status="warn",
            summary=f"Detected: {', '.join(detected)} | /cortheon installed: none",
            details=details,
            hint="Run `cortheon install-commands --harness all` to add /cortheon across your harnesses.",
        )

    return CheckResult(
        id="harnesses",
        title="Harness Integrations",
        status="pass",
        summary="No coding harnesses detected (standalone CLI / proxy / SDK mode ready)",
        details=details,
        hint="Use `cortheon install-commands --harness all` if you add OMP, Pi, OpenCode, Claude Code, Cursor, or Codex.",
    )


def _backend_error_hint(url: str, exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in (401, 403):
            return "Backend requires authentication — pass `--api-key <token>`."
        if code == 404:
            return f"Endpoint `{url.rstrip('/')}/models` returned 404 — make sure `--url` ends with `/v1`."
        return f"Backend returned HTTP {code}. Check your local LLM server logs."
    if isinstance(exc, httpx.TimeoutException):
        return f"Timed out connecting to `{url}`. Verify the server is running and responsive."
    if isinstance(exc, RuntimeError) and "No models reported" in str(exc):
        return "Server responded, but `/v1/models` is empty — load a model in your backend first."
    return (
        f"Start a local OpenAI-compatible server (TensorFold, oMLX, Ollama, llama.cpp, vLLM, or LM Studio) "
        f"or pass `--url http://<host>:<port>/v1`."
    )


async def check_backend(client: TensorFoldClient) -> CheckResult:
    """Probe `/v1/models` using `TensorFoldClient` and report capabilities + reasoning headroom."""
    t0 = time.perf_counter()
    try:
        caps = await client.probe()
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        msg = " ".join(str(exc).strip().split())[:180] or type(exc).__name__
        return CheckResult(
            id="backend",
            title="Backend Probe",
            status="fail",
            summary=f"Unreachable at {client.base_url} ({type(exc).__name__}: {msg})",
            details={
                "url": client.base_url,
                "reachable": False,
                "error_type": type(exc).__name__,
                "error": msg,
                "probe_elapsed_ms": elapsed_ms,
            },
            hint=_backend_error_hint(client.base_url, exc),
        )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    is_tf = bool(getattr(client, "_is_tensorfold", False))
    flavor = _FLAVOR_TENSORFOLD if is_tf else _FLAVOR_GENERIC_OPENAI
    headroom = bool(client._needs_reasoning_headroom())

    details = {
        "url": client.base_url,
        "reachable": True,
        "backend_flavor": flavor,
        "is_tensorfold": is_tf,
        "model_id": caps.model_id,
        "context_length": caps.context_length,
        "thinking_budget": caps.thinking_budget,
        "thinking_switch": caps.thinking_switch,
        "structures": caps.structures,
        "prompt_cache": caps.prompt_cache,
        "loop_guard": caps.loop_guard,
        "supported_efforts": list(caps.supported_efforts),
        "reasoning_headroom_active": headroom,
        "probe_elapsed_ms": elapsed_ms,
    }
    return CheckResult(
        id="backend",
        title="Backend Probe",
        status="pass",
        summary=f"{flavor} | model={caps.model_id} | ctx={caps.context_length} ({elapsed_ms:.0f}ms)",
        details=details,
    )


def _recommend_streams(
    *,
    is_tensorfold: bool,
    prompt_cache: bool,
    latency_ms: float,
    tok_per_s: float,
    out_tokens: int = 16,
) -> tuple[int, str]:
    low_tput = out_tokens >= 8 and 0.0 < tok_per_s < 20.0
    if latency_ms > 2500.0 or low_tput:
        return (
            2,
            f"Higher latency/lower throughput ({latency_ms:.0f}ms, ~{tok_per_s:.1f} tok/s) — "
            "--streams 2 (~2-3 calls) keeps latency snappy",
        )
    if is_tensorfold and prompt_cache and latency_ms <= 300.0 and (out_tokens < 8 or tok_per_s >= 80.0):
        return (
            8,
            f"Fast TensorFold Native + prompt cache ({latency_ms:.0f}ms, ~{tok_per_s:.1f} tok/s) — "
            "--streams 8 gives max diversity (or --streams 4 for balanced speed)",
        )
    return (
        4,
        f"Balanced accuracy/latency sweet spot (~4-7 calls with early-abort) for {latency_ms:.0f}ms fast-route latency",
    )


async def check_live_probe(
    client: TensorFoldClient,
    *,
    skip_live: bool = False,
    backend_reachable: bool = True,
) -> tuple[CheckResult, int, str]:
    """Run one tiny `enable_thinking=False, max_tokens=16` probe and recommend `--streams 2|4|8`."""
    default_streams = 4
    default_reason = "Balanced default (~4-7 calls; run without --skip-live for latency-tuned recommendation)"

    if skip_live:
        return (
            CheckResult(
                id="live_probe",
                title="Live Fast-Route & Throughput",
                status="skip",
                summary="Skipped (--skip-live)",
                details={"skipped": True, "reason": "user_skipped"},
            ),
            default_streams,
            default_reason,
        )

    if not backend_reachable:
        return (
            CheckResult(
                id="live_probe",
                title="Live Fast-Route & Throughput",
                status="skip",
                summary="Skipped (backend unreachable)",
                details={"skipped": True, "reason": "backend_unreachable"},
            ),
            default_streams,
            "Balanced default (~4-7 calls; re-run `cortheon doctor` once backend is reachable)",
        )

    spec = CallSpec(
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        tag="doctor_live_probe",
        max_tokens=16,
        temperature=0.0,
        enable_thinking=False,
        loop_guard=False,
    )

    calls_before = client.total_calls
    t0 = time.perf_counter()
    try:
        res = await client.complete(spec)
    except Exception as exc:
        wall_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        msg = " ".join(str(exc).strip().split())[:180] or type(exc).__name__
        return (
            CheckResult(
                id="live_probe",
                title="Live Fast-Route & Throughput",
                status="fail",
                summary=f"Live probe raised {type(exc).__name__}: {msg}",
                details={"ok": False, "latency_ms": wall_ms, "error": msg},
                hint="Verify `/v1/chat/completions` is enabled and healthy on the target backend.",
            ),
            default_streams,
            default_reason,
        )
    wall_s = time.perf_counter() - t0

    latency_s = res.latency_s if res.latency_s > 0 else wall_s
    latency_ms = round(latency_s * 1000.0, 1)
    calls_made = max(1, client.total_calls - calls_before)

    if not res.ok:
        return (
            CheckResult(
                id="live_probe",
                title="Live Fast-Route & Throughput",
                status="fail",
                summary=f"Completion error ({latency_ms:.0f}ms): {res.raw_error}",
                details={
                    "ok": False,
                    "latency_ms": latency_ms,
                    "raw_error": res.raw_error,
                    "finish_reason": res.finish_reason,
                },
                hint="Check backend server logs and verify the model accepts standard OpenAI chat completions.",
            ),
            default_streams,
            default_reason,
        )

    cleaned = " ".join(res.content.strip().split())
    has_content = bool(cleaned)
    starved = (not has_content) and (
        res.finish_reason == "length" or res.reasoning_tokens > 0 or bool(res.reasoning)
    )
    starvation_failures = 1 if starved else 0
    recovered_from_starvation = bool(
        has_content and calls_made > 1 and (res.reasoning_tokens > 0 or bool(res.reasoning))
    )

    out_tokens = res.completion_tokens if res.completion_tokens > 0 else (max(1, len(cleaned.split())) if has_content else 0)
    tok_per_s = round(out_tokens / latency_s, 1) if (out_tokens > 0 and latency_s > 0) else 0.0

    is_tf = bool(getattr(client, "_is_tensorfold", False))
    prompt_cache = bool(client.caps.prompt_cache)
    reasoning_observed = bool(getattr(client, "_observed_reasoning", False))
    reasoning_headroom = bool(client._needs_reasoning_headroom())

    rec_streams, rec_reason = _recommend_streams(
        is_tensorfold=is_tf,
        prompt_cache=prompt_cache,
        latency_ms=latency_ms,
        tok_per_s=tok_per_s,
        out_tokens=out_tokens,
    )

    details = {
        "ok": has_content and starvation_failures == 0,
        "reply_preview": cleaned[:40],
        "non_empty_content": has_content,
        "finish_reason": res.finish_reason,
        "completion_tokens": res.completion_tokens,
        "reasoning_tokens": res.reasoning_tokens,
        "thinking_starvation_failures": starvation_failures,
        "adaptive_recovery_triggered": recovered_from_starvation,
        "reasoning_observed": reasoning_observed,
        "reasoning_headroom_active": reasoning_headroom,
        "latency_ms": latency_ms,
        "tok_per_s": tok_per_s,
        "recommended_streams": rec_streams,
        "recommended_reason": rec_reason,
    }

    if not has_content:
        if starved:
            return (
                CheckResult(
                    id="live_probe",
                    title="Live Fast-Route & Throughput",
                    status="fail",
                    summary=(
                        f"Thinking-starvation failure: empty content "
                        f"(finish_reason={res.finish_reason}, reasoning_tokens={res.reasoning_tokens}, {latency_ms:.0f}ms)"
                    ),
                    details=details,
                    hint="Model spent all tokens in reasoning_content even with enable_thinking=False. Check backend template/reasoning flags.",
                ),
                2,
                "Conservative --streams 2 recommended until fast-route thinking control is resolved",
            )
        return (
            CheckResult(
                id="live_probe",
                title="Live Fast-Route & Throughput",
                status="fail",
                summary=f"Empty assistant content (finish_reason={res.finish_reason}, {latency_ms:.0f}ms)",
                details=details,
                hint="Backend returned empty content on a trivial prompt. Inspect server logs.",
            ),
            default_streams,
            default_reason,
        )

    suffix = " (adaptive recovery)" if recovered_from_starvation else ""
    summary = (
        f'reply="{cleaned[:24]}" | {starvation_failures} thinking-starvation failures | '
        f"{latency_ms:.0f}ms | ~{tok_per_s:.1f} tok/s{suffix}"
    )
    return (
        CheckResult(
            id="live_probe",
            title="Live Fast-Route & Throughput",
            status="pass",
            summary=summary,
            details=details,
        ),
        rec_streams,
        rec_reason,
    )


async def run_doctor(
    url: str = "http://127.0.0.1:8080/v1",
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    skip_live: bool = False,
    timeout_s: float = 30.0,
    home_dir: Optional[Path] = None,
    project_dir: Optional[Path] = None,
    client_factory: Optional[Callable[..., TensorFoldClient]] = None,
) -> DoctorReport:
    """Execute all 5 Cortheon runtime diagnostic checks and return a structured report."""
    t0 = time.perf_counter()

    sandbox_res = check_sandbox()
    solver_res = check_solver_stack()
    harness_res = check_harnesses(home_dir=home_dir, project_dir=project_dir)

    factory = client_factory or TensorFoldClient
    client = factory(base_url=url, model=model, api_key=api_key, timeout_s=timeout_s)
    try:
        backend_res = await check_backend(client)
        live_res, rec_streams, rec_reason = await check_live_probe(
            client,
            skip_live=skip_live,
            backend_reachable=bool(backend_res.details.get("reachable")),
        )
        # If the live probe observed reasoning_content on an aliased generic model, reflect active headroom cleanly.
        if backend_res.details.get("reachable") and live_res.details.get("reasoning_headroom_active"):
            backend_res.details["reasoning_headroom_active"] = True
    finally:
        await client.close()

    return DoctorReport(
        url=url.rstrip("/"),
        checks=[sandbox_res, solver_res, harness_res, backend_res, live_res],
        recommended_streams=rec_streams,
        recommended_reason=rec_reason,
        elapsed_s=time.perf_counter() - t0,
    )


class _ANSI:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"


def _should_use_color(use_color: Optional[bool]) -> bool:
    if use_color is not None:
        return use_color
    if os.environ.get("NO_COLOR"):
        return False
    return bool(sys.stdout.isatty())


def _paint(text: str, *codes: str, color: bool) -> str:
    if not color or not codes:
        return text
    return "".join(codes) + text + _ANSI.RESET


def _status_badge(status: CheckStatus, *, color: bool) -> str:
    if status == "pass":
        return _paint("[PASS]", _ANSI.GREEN, _ANSI.BOLD, color=color)
    if status == "warn":
        return _paint("[WARN]", _ANSI.YELLOW, _ANSI.BOLD, color=color)
    if status == "fail":
        return _paint("[FAIL]", _ANSI.RED, _ANSI.BOLD, color=color)
    return _paint("[SKIP]", _ANSI.DIM, color=color)


def format_doctor_report(report: DoctorReport, *, use_color: Optional[bool] = None) -> str:
    """Format a :class:`DoctorReport` as a crisp terminal scorecard respecting ``NO_COLOR``."""
    color = _should_use_color(use_color)
    bar = "━" * 74
    rule = "─" * 74

    lines: list[str] = [
        _paint(bar, _ANSI.CYAN, _ANSI.BOLD, color=color),
        _paint("  Cortheon Doctor — Local Runtime Diagnostics", _ANSI.CYAN, _ANSI.BOLD, color=color),
        _paint(f"  Target: {report.url}", _ANSI.DIM, color=color),
        _paint(bar, _ANSI.CYAN, _ANSI.BOLD, color=color),
        "",
    ]

    for chk in report.checks:
        badge = _status_badge(chk.status, color=color)
        title = _paint(chk.title, _ANSI.BOLD, color=color)
        lines.append(f"  {badge} {title}")
        lines.append(f"         {chk.summary}")

        if chk.id == "backend" and chk.details.get("reachable"):
            efforts = chk.details.get("supported_efforts") or []
            efforts_str = ",".join(efforts)
            lines.append(
                _paint(
                    f"         thinking_budget={chk.details.get('thinking_budget')} | "
                    f"supported_efforts=[{efforts_str}] | "
                    f"reasoning_headroom={chk.details.get('reasoning_headroom_active')}",
                    _ANSI.DIM,
                    color=color,
                )
            )
        elif chk.id == "live_probe" and chk.status == "pass":
            lines.append(
                _paint(
                    f"         Recommended dial: --streams {report.recommended_streams} ({report.recommended_reason})",
                    _ANSI.DIM,
                    color=color,
                )
            )

        if chk.hint:
            lines.append(_paint(f"         Hint: {chk.hint}", _ANSI.YELLOW, color=color))
        lines.append("")

    n_pass = sum(1 for c in report.checks if c.status == "pass")
    n_warn = sum(1 for c in report.checks if c.status == "warn")
    n_fail = sum(1 for c in report.checks if c.status == "fail")
    n_skip = sum(1 for c in report.checks if c.status == "skip")

    overall_label = (
        _paint("PASS", _ANSI.GREEN, _ANSI.BOLD, color=color)
        if report.ok
        else _paint("FAIL", _ANSI.RED, _ANSI.BOLD, color=color)
    )
    counts = f"{n_pass} passed, {n_warn} warn, {n_fail} failed" + (f", {n_skip} skipped" if n_skip else "")

    lines.extend(
        [
            _paint(rule, _ANSI.DIM, color=color),
            f"  Overall      : {overall_label} ({counts}) in {report.elapsed_s:.2f}s",
            "  Streams dial : --streams 2 (fast ~2-3 calls) | 4 (balanced ~4-7) | 8 (max diversity)",
            "",
            _paint("  Quick Start:", _ANSI.BOLD, color=color),
            f"    cortheon proxy --upstream {report.url} --streams {report.recommended_streams}",
            f"    cortheon duel --url {report.url} --streams {report.recommended_streams}",
            "    cortheon install-commands --harness all",
            _paint(bar, _ANSI.CYAN, _ANSI.BOLD, color=color),
        ]
    )
    return "\n".join(lines)

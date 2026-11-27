"""Isolated Python/SymPy/Z3 execution sandbox and ephemeral git worktree runner.

Gives frozen local models deterministic mechanical grounding:
- Executes Python scripts with OS-level containment by default:
  - macOS: ``/usr/bin/sandbox-exec`` Seatbelt profile blocking all network I/O,
    denying writes outside the ephemeral temp directory, and denying reads of
    sensitive credential directories (``~/.ssh``, ``~/.aws``, ``~/.gnupg``,
    ``~/.kube``, ``~/.config``, ``~/.netrc``) and the home directories of coding
    harnesses that keep credentials or transcripts there (``~/.omp``, ``~/.pi``,
    ``~/.dsh``, ``~/.claude``, ``~/.codex``, ``~/.cursor``).
  - Linux: ``bwrap`` (bubblewrap) with ``--unshare-net`` and read-only root
    when ``bwrap`` is installed on ``PATH``.
  - All POSIX platforms: ``resource.setrlimit`` CPU time and 64 MiB file-size
    caps plus scrubbed environment variables and strict wall-clock timeouts.
  - Opt-out: pass ``unsafe_no_sandbox=True`` or set ``CORTHEON_UNSAFE_NO_SANDBOX=1``
    to disable OS-level sandboxing (not recommended for untrusted code).
- Extracts valid Python code blocks cleanly via AST verification
- Runs candidate solutions against independent oracle & red-team suites
- Supports ephemeral git worktree verification for repository bugfix/perf tasks
"""

from __future__ import annotations

import ast
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

try:
    import pwd
except ImportError:  # pragma: no cover - non-POSIX
    pwd = None  # type: ignore[assignment]

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX
    resource = None  # type: ignore[assignment]


PYTHON_BIN = sys.executable
_SEATBELT_BIN = "/usr/bin/sandbox-exec"
_UNSAFE_NO_SANDBOX_ENV = "CORTHEON_UNSAFE_NO_SANDBOX"
_MAX_FILE_SIZE_BYTES = 64 * 1024 * 1024  # 64 MiB


def sandbox_backend(*, unsafe_no_sandbox: bool = False) -> str:
    """Return the active OS-level sandbox backend for :func:`run_python`.

    Returns one of:
      - ``"unsafe-disabled"``: disabled via ``unsafe_no_sandbox=True`` or
        ``CORTHEON_UNSAFE_NO_SANDBOX=1``.
      - ``"macos-seatbelt"``: macOS ``/usr/bin/sandbox-exec`` Seatbelt profile.
      - ``"linux-bwrap"``: Linux ``bwrap`` (bubblewrap) namespace/mount sandbox.
      - ``"subprocess-rlimit"``: subprocess isolation with POSIX ``setrlimit``
        caps, scrubbed environment, and wall-clock timeout.
    """
    if unsafe_no_sandbox or os.environ.get(_UNSAFE_NO_SANDBOX_ENV) == "1":
        return "unsafe-disabled"
    if sys.platform == "darwin" and os.path.isfile(_SEATBELT_BIN) and os.access(_SEATBELT_BIN, os.X_OK):
        return "macos-seatbelt"
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return "linux-bwrap"
    return "subprocess-rlimit"


def _seatbelt_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _build_macos_seatbelt_profile(resolved_tmpdir: Path) -> str:
    """Build a strict macOS Seatbelt profile confining writes to ``resolved_tmpdir``."""
    tmp_escaped = _seatbelt_escape(str(resolved_tmpdir))

    raw_homes: list[Path] = [Path.home()]
    env_home = os.environ.get("HOME", "").strip()
    if env_home:
        raw_homes.append(Path(env_home))
    if pwd is not None:
        try:
            pw_dir = pwd.getpwuid(os.getuid()).pw_dir
            if pw_dir:
                raw_homes.append(Path(pw_dir))
        except (KeyError, OSError):
            pass

    home_candidates: list[Path] = []
    for cand in raw_homes:
        try:
            resolved = cand.resolve()
        except OSError:
            resolved = cand
        if str(resolved) and resolved not in home_candidates:
            home_candidates.append(resolved)

    sensitive_dirs = (
        ".ssh", ".aws", ".gnupg", ".kube", ".config",
        # Agent-harness homes. Cortheon is often run *from* one of these
        # harnesses, so a solver script that can read them can exfiltrate the
        # very credentials it was sandboxed to protect.
        ".omp", ".pi", ".dsh", ".claude", ".codex", ".cursor",
    )
    denied_read_rules: list[str] = []
    for home_dir in home_candidates:
        for rel in sensitive_dirs:
            denied_read_rules.append(f'  (subpath "{_seatbelt_escape(str(home_dir / rel))}")')
        denied_read_rules.append(f'  (literal "{_seatbelt_escape(str(home_dir / ".netrc"))}")')

    deny_reads = "\n".join(denied_read_rules)
    return (
        "(version 1)\n"
        "(allow default)\n"
        "(deny network*)\n"
        "(deny file-write*)\n"
        "(allow file-write*\n"
        f'  (subpath "{tmp_escaped}")\n'
        '  (literal "/dev/null")\n'
        '  (literal "/dev/tty")\n'
        '  (literal "/dev/dtracehelper"))\n'
        "(deny file-read*\n"
        f"{deny_reads})\n"
    )


def _make_rlimit_preexec(timeout_s: float) -> Optional[Callable[[], None]]:
    """Build a POSIX ``preexec_fn`` enforcing CPU time and 64 MiB file-size limits.

    Note: ``RLIMIT_AS`` is intentionally omitted on macOS ARM64 / OpenBLAS setups
    where NumPy, SymPy, and Z3 reserve large virtual address spaces without
    consuming physical RAM.
    """
    if resource is None:
        return None
    cpu_limit = max(2, math.ceil(timeout_s) + 2)
    fsize_limit = _MAX_FILE_SIZE_BYTES

    def _apply_limits() -> None:
        for res_id, limit in (
            (resource.RLIMIT_CPU, cpu_limit),
            (resource.RLIMIT_FSIZE, fsize_limit),
        ):
            try:
                _soft, hard = resource.getrlimit(res_id)
                new_hard = limit if hard == resource.RLIM_INFINITY else min(hard, limit)
                new_soft = min(new_hard, limit)
                resource.setrlimit(res_id, (new_soft, new_hard))
            except (ValueError, OSError):
                pass

    return _apply_limits


@dataclass
class ExecResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    timed_out: bool = False

    def short_error(self, max_chars: int = 800) -> str:
        if self.timed_out:
            return f"TIMEOUT after {self.elapsed_s:.2f}s"
        err = (self.stderr or self.stdout or "").strip()
        if len(err) > max_chars:
            # Keep tail of traceback (most informative lines)
            err = "...(truncated)...\n" + err[-max_chars:]
        return err


_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_OPEN_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*)",
    re.DOTALL | re.IGNORECASE,
)


def _salvage_truncated_defs(raw: str) -> str:
    """If a modular test suite was truncated mid-function by max_tokens, drop only the incomplete tail def."""
    lines = raw.splitlines()
    # Find all top-level def/class start indices and try truncating back to the last complete block
    starts = [i for i, ln in enumerate(lines) if re.match(r"^(?:def|class)\s+\w+", ln)]
    for cut in reversed(starts):
        candidate = "\n".join(lines[:cut]).strip()
        if not candidate:
            continue
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return ""


def extract_python_code(text: str, *, salvage_truncated_tests: bool = False) -> str:
    """Extract the most complete, syntactically valid Python block from model output."""
    if not text:
        return ""
    blocks = _CODE_FENCE_RE.findall(text)
    if blocks:
        # Prefer the longest syntactically valid block (usually the full solution / script)
        valid_blocks: list[tuple[int, str]] = []
        for blk in blocks:
            cleaned = blk.strip()
            if not cleaned:
                continue
            try:
                ast.parse(cleaned)
                valid_blocks.append((len(cleaned), cleaned))
            except SyntaxError:
                pass
        if valid_blocks:
            valid_blocks.sort(key=lambda x: x[0])
            return valid_blocks[-1][1]
        return blocks[-1].strip()

    # Check for an unclosed fenced block (e.g. model hit max_tokens right before closing ```)
    m_open = _OPEN_FENCE_RE.search(text)
    if m_open:
        raw = m_open.group(1).strip()
        try:
            ast.parse(raw)
            return raw
        except SyntaxError:
            if salvage_truncated_tests:
                salvaged = _salvage_truncated_defs(raw)
                if salvaged:
                    return salvaged
            return raw

    # Fallback: check if the entire output is valid Python
    stripped = text.strip()
    try:
        ast.parse(stripped)
        return stripped
    except SyntaxError:
        return ""


def check_syntax(code: str) -> Optional[str]:
    """Return None if syntax AND compile-time symbol binding are valid, else a crisp error string."""
    if not code.strip():
        return "Empty code block"
    try:
        ast.parse(code)
        # compile() catches semantic SyntaxErrors that ast.parse misses (e.g. invalid `nonlocal`, `return` outside fn)
        compile(code, "<solution>", "exec")
        return None
    except SyntaxError as exc:
        return f"SyntaxError line {exc.lineno}: {exc.msg}"


def run_python(
    code: str,
    timeout_s: float = 8.0,
    extra_files: Optional[dict[str, str]] = None,
    python_bin: Optional[str] = None,
    unsafe_no_sandbox: bool = False,
) -> ExecResult:
    """Execute Python code in an isolated temp directory with OS-level containment.

    By default, wraps execution with the strongest available OS sandbox reported by
    :func:`sandbox_backend` (macOS Seatbelt ``sandbox-exec``, Linux ``bwrap``, or
    POSIX ``setrlimit`` subprocess containment). Pass ``unsafe_no_sandbox=True`` or
    set environment variable ``CORTHEON_UNSAFE_NO_SANDBOX=1`` to disable the OS
    sandbox wrapper.
    """
    py = python_bin or PYTHON_BIN
    backend = sandbox_backend(unsafe_no_sandbox=unsafe_no_sandbox)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ff_exec_") as tmp:
        tmp_path = Path(tmp).resolve()
        if extra_files:
            for name, content in extra_files.items():
                p = (tmp_path / name).resolve()
                if not p.is_relative_to(tmp_path):
                    raise ValueError(f"Refusing extra_files path outside sandbox temp dir: {name}")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
        script_path = tmp_path / "_run.py"
        script_path.write_text(code, encoding="utf-8")

        # Minimal scrubbed environment: prevent untrusted code from reading host API keys / secrets
        # and confine tempfile writes strictly inside the ephemeral sandbox temp directory.
        _SAFE_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "USER")
        env = {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}
        env["TMPDIR"] = str(tmp_path)
        env["TEMP"] = str(tmp_path)
        env["TMP"] = str(tmp_path)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        base_pypath = os.environ.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(tmp_path) + (os.pathsep + base_pypath if base_pypath else "")

        cmd: list[str]
        if backend == "macos-seatbelt":
            profile_path = tmp_path / ".sb_profile"
            profile_path.write_text(_build_macos_seatbelt_profile(tmp_path), encoding="utf-8")
            cmd = [_SEATBELT_BIN, "-f", str(profile_path), py, str(script_path)]
        elif backend == "linux-bwrap":
            bwrap_bin = shutil.which("bwrap") or "bwrap"
            cmd = [
                bwrap_bin,
                "--unshare-net",
                "--ro-bind", "/", "/",
                "--bind", str(tmp_path), str(tmp_path),
                "--dev", "/dev",
                "--proc", "/proc",
                "--die-with-parent",
                "--",
                py,
                str(script_path),
            ]
        else:
            cmd = [py, str(script_path)]

        preexec_fn = _make_rlimit_preexec(timeout_s) if os.name == "posix" else None

        proc = subprocess.Popen(
            cmd,
            cwd=str(tmp_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            preexec_fn=preexec_fn,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            dt = time.perf_counter() - t0
            return ExecResult(
                ok=(proc.returncode == 0),
                returncode=proc.returncode,
                stdout=(stdout or "")[:16000],
                stderr=(stderr or "")[:16000],
                elapsed_s=dt,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            # Kill entire process group so runaway child processes never leak
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=2.0)
            except Exception:
                stdout, stderr = "", ""
            dt = time.perf_counter() - t0
            return ExecResult(
                ok=False,
                returncode=-1,
                stdout=(stdout or "")[:8000],
                stderr=(stderr or "")[:8000],
                elapsed_s=dt,
                timed_out=True,
            )


def run_solution_against_tests(
    solution_code: str,
    test_code: str,
    timeout_s: float = 10.0,
    python_bin: Optional[str] = None,
) -> ExecResult:
    """Run candidate solution (`solution.py`) against a test harness (`test_suite.py`)."""
    harness = (
        "import traceback, sys\n"
        "import solution\n"
        "from solution import *\n"
        "\n"
        f"{test_code}\n"
    )
    return run_python(
        code=harness,
        timeout_s=timeout_s,
        extra_files={"solution.py": solution_code},
        python_bin=python_bin,
    )


def run_in_ephemeral_worktree(
    repo_dir: str | Path,
    patch_files: dict[str, str],
    verify_cmd: str,
    timeout_s: float = 60.0,
) -> ExecResult:
    """Apply candidate file edits in an isolated git worktree, run verify_cmd, then cleanly remove worktree."""
    repo = Path(repo_dir).resolve()
    wt_dir = Path(tempfile.mkdtemp(prefix="ff_wt_"))
    t0 = time.perf_counter()
    try:
        add = subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach", str(wt_dir), "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if add.returncode != 0:
            return ExecResult(
                ok=False,
                returncode=add.returncode,
                stdout=add.stdout,
                stderr=f"worktree add failed: {add.stderr}",
                elapsed_s=time.perf_counter() - t0,
            )
        wt_resolved = wt_dir.resolve()
        for rel_path, content in patch_files.items():
            dest = (wt_dir / rel_path).resolve()
            if not dest.is_relative_to(wt_resolved):
                raise ValueError(f"Refusing path outside worktree: {rel_path}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        try:
            proc = subprocess.run(
                verify_cmd,
                shell=True,
                cwd=str(wt_dir),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return ExecResult(
                ok=(proc.returncode == 0),
                returncode=proc.returncode,
                stdout=proc.stdout[:16000],
                stderr=proc.stderr[:16000],
                elapsed_s=time.perf_counter() - t0,
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return ExecResult(
                ok=False,
                returncode=-1,
                stdout=out[:8000],
                stderr=err[:8000],
                elapsed_s=time.perf_counter() - t0,
                timed_out=True,
            )
    finally:
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(wt_dir)],
            capture_output=True,
            timeout=20,
        )
        shutil.rmtree(wt_dir, ignore_errors=True)

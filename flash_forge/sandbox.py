"""Isolated Python/SymPy/Z3 execution sandbox and ephemeral git worktree runner.

Gives frozen local models deterministic mechanical grounding:
- Executes Python scripts with strict wall-clock timeouts & output caps
- Extracts valid Python code blocks cleanly via AST verification
- Runs candidate solutions against independent oracle & red-team suites
- Supports ephemeral git worktree verification for repository bugfix/perf tasks
"""

from __future__ import annotations

import ast
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
from typing import Optional


PYTHON_BIN = sys.executable


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
) -> ExecResult:
    """Execute Python code in an isolated temp directory with strict timeout."""
    py = python_bin or PYTHON_BIN
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ff_exec_") as tmp:
        tmp_path = Path(tmp)
        if extra_files:
            for name, content in extra_files.items():
                p = tmp_path / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
        script_path = tmp_path / "_run.py"
        script_path.write_text(code, encoding="utf-8")

        # Minimal scrubbed environment: prevent untrusted code from reading host API keys / secrets
        _SAFE_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TERM", "USER")
        env = {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        base_pypath = os.environ.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(tmp_path) + (os.pathsep + base_pypath if base_pypath else "")

        proc = subprocess.Popen(
            [py, str(script_path)],
            cwd=tmp,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
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

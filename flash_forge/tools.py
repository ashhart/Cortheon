"""Sandboxed Agent Toolbelt for `ReplicaAgent`.

Gives Qwen3.8-Flash-Next the exact real-world development tools a frontier agent uses,
with strict workspace containment, automatic output compaction, Python syntax guards,
surgical SEARCH/REPLACE diagnostics, and delegation into CodeForge/ReasonForge.
"""

from __future__ import annotations

import ast
import difflib
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .repo_index import IGNORE_DIRS, build_repo_map, list_repo_files, read_numbered_slice, rg_search
from .sandbox import run_python


@dataclass
class ToolResult:
    ok: bool
    output: str
    modified_file: Optional[str] = None
    is_test_run: bool = False
    test_cmd: str = ""


_SHELL_META_RE = re.compile(r"[|;&><`\n\r]|\$\(")
_RO_ALLOWED_BINS = {"ls", "rg", "grep", "wc", "head", "tail", "cat", "find", "git"}
_RO_GIT_SUBCMDS = {"status", "log", "diff", "show", "branch", "rev-parse", "ls-files", "grep", "blame", "describe"}
_RO_FIND_FORBIDDEN = {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprintf", "-fls"}
_DANGEROUS_CMD_RE = re.compile(r"\b(rm\s+-rf\s+/|sudo\b|mkfs\b|dd\s+if=|shutdown\b|reboot\b|:\(\)\{)", re.IGNORECASE)


def is_real_test_command(cmd: str) -> bool:
    """Strictly check whether a command actually executes a test/benchmark runner (never fooled by `echo pytest`)."""
    if not cmd or not cmd.strip():
        return False
    # Split chained commands (e.g. `cc ... && ./bench_runner`) and inspect each executable segment's argv
    segments = re.split(r"\s*(?:&&|\|\||;)\s*", cmd.strip())
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        try:
            tokens = shlex.split(seg)
        except ValueError:
            continue
        # Strip leading VAR=val assignments
        while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
            tokens.pop(0)
        if not tokens:
            continue
        exe = Path(tokens[0]).name.lower()
        rest = [t.lower() for t in tokens[1:]]
        if exe in {"pytest", "ctest", "bench_runner", "test_runner", "bench"}:
            return True
        if exe in {"python", "python3"} or exe.startswith("python3."):
            if len(rest) >= 2 and rest[0] == "-m" and rest[1] in {"pytest", "unittest"}:
                return True
            if rest and ("test" in rest[0] or "bench" in rest[0]) and rest[0].endswith(".py"):
                return True
        if exe == "zig" and rest[:2] == ["build", "test"]:
            return True
        if exe in {"cargo", "go", "npm", "pnpm", "yarn", "make"} and rest and rest[0] == "test":
            return True
    return False


class AgentToolbelt:
    def __init__(self, workspace: str | Path, *, read_only: bool = False) -> None:
        self.root = Path(workspace).resolve()
        self.read_only = read_only
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, rel_or_abs: str) -> Path:
        raw = (rel_or_abs or ".").strip()
        p = Path(raw)
        resolved = (p if p.is_absolute() else (self.root / p)).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"Path escapes workspace root: {rel_or_abs}")
        return resolved

    def _rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    def list_dir(self, path: str = ".", depth: int = 2) -> ToolResult:
        try:
            target = self._safe_path(path)
            if not target.exists():
                return ToolResult(False, f"Directory not found: {path}")
            if target.is_file():
                return ToolResult(True, f"File: {self._rel(target)} ({target.stat().st_size} bytes)")
            lines: list[str] = []
            max_depth = max(1, min(4, int(depth)))

            def _walk(cur: Path, d: int) -> None:
                if len(lines) >= 120:
                    return
                try:
                    entries = sorted(cur.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
                except PermissionError:
                    return
                for entry in entries:
                    if entry.name in IGNORE_DIRS or (entry.name.startswith(".") and entry.name not in {".gitignore"}):
                        continue
                    rel = self._rel(entry)
                    if entry.is_dir():
                        lines.append(f"{rel}/")
                        if d < max_depth:
                            _walk(entry, d + 1)
                    else:
                        lines.append(f"{rel} ({entry.stat().st_size}B)")
                    if len(lines) >= 120:
                        lines.append("... (truncated at 120 entries)")
                        return

            _walk(target, 1)
            return ToolResult(True, "\n".join(lines) if lines else "(empty directory)")
        except Exception as exc:
            return ToolResult(False, f"list_dir error: {exc}")

    def repo_map(self, path: str = ".") -> ToolResult:
        try:
            target = self._safe_path(path)
            return ToolResult(True, build_repo_map(target, max_chars=4800))
        except Exception as exc:
            return ToolResult(False, f"repo_map error: {exc}")

    def search_code(self, pattern: str, path: str = ".", max_hits: int = 28) -> ToolResult:
        try:
            if not pattern or not pattern.strip():
                return ToolResult(False, "Empty search pattern")
            target = self._safe_path(path)
            # Prefer ripgrep (`rg`) when installed — orders of magnitude faster on 100k+ LOC repos like TensorFold!
            cmd = [
                "rg", "-n", "--no-heading", "--color=never",
                "--max-count", "6",
                "-g", "!.git", "-g", "!.zig-cache", "-g", "!zig-out", "-g", "!.venv", "-g", "!node_modules",
                pattern,
                str(target),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if proc.returncode in (0, 1):
                    raw_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
                    prefix = str(self.root) + os.sep
                    cleaned = [ln.replace(prefix, "", 1)[:220] for ln in raw_lines[:max_hits]]
                    if len(raw_lines) > max_hits:
                        cleaned.append(f"... ({len(raw_lines) - max_hits} more matches)")
                    return ToolResult(True, "\n".join(cleaned) if cleaned else "(no matches)")
            except FileNotFoundError:
                pass
            return ToolResult(True, rg_search(target, pattern, max_hits=max_hits))
        except Exception as exc:
            return ToolResult(False, f"search_code error: {exc}")

    def read_file(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> ToolResult:
        try:
            target = self._safe_path(path)
            if not target.is_file():
                # Suggest close filename matches so local model recovers immediately
                candidates = list_repo_files(self.root, max_files=200)
                close = difflib.get_close_matches(path, candidates, n=4, cutoff=0.4)
                hint = f" Closest files: {close}" if close else ""
                return ToolResult(False, f"File not found: {path}.{hint}")
            s = max(1, int(start_line or 1))
            e = int(end_line) if end_line else s + 159
            if e < s:
                e = s + 159
            text = read_numbered_slice(self.root, self._rel(target), start_line=s, end_line=e, max_lines=180)
            return ToolResult(True, text)
        except Exception as exc:
            return ToolResult(False, f"read_file error: {exc}")

    def _check_c_syntax(self, target: Path, candidate_content: str) -> Optional[str]:
        """Fast (~50ms) clang -fsyntax-only type/intrinsic check for C/C++ kernels before saving."""
        import tempfile
        compiler = "clang++" if target.suffix in {".cpp", ".cc", ".cxx", ".mm"} else "clang"
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=target.suffix,
            dir=str(target.parent),
            prefix=".ff_chk_",
            delete=False,
            encoding="utf-8",
        ) as tf:
            tf.write(candidate_content)
            tmp_path = Path(tf.name)
        try:
            proc = subprocess.run(
                [compiler, "-fsyntax-only", "-O3", "-Wall", "-Wextra", "-I", str(target.parent), "-I", str(self.root), str(tmp_path)],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=8,
            )
            if proc.returncode != 0:
                err = proc.stderr.replace(str(tmp_path), self._rel(target)).strip()
                return err[:1400]
            return None
        except FileNotFoundError:
            return None
        finally:
            tmp_path.unlink(missing_ok=True)

    def write_file(self, path: str, content: str) -> ToolResult:
        if self.read_only:
            return ToolResult(False, "Workspace is in READ-ONLY mode; write_file is disabled.")
        try:
            target = self._safe_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Syntax/type-check Python AND C/C++ files BEFORE saving so broken code never lands silently!
            if target.suffix == ".py":
                try:
                    ast.parse(content)
                except SyntaxError as exc:
                    return ToolResult(
                        False,
                        f"Refused write_file({path}): Python SyntaxError line {exc.lineno}: {exc.msg}",
                    )
            elif target.suffix in {".c", ".cpp", ".cc", ".cxx"}:
                c_err = self._check_c_syntax(target, content)
                if c_err:
                    return ToolResult(
                        False,
                        f"Refused write_file({path}): C/C++ compiler error (fix these EXACT lines!):\n{c_err}",
                    )
            target.write_text(content, encoding="utf-8")
            rel = self._rel(target)
            nlines = len(content.splitlines())
            return ToolResult(True, f"Wrote {rel} ({nlines} lines, compile/syntax OK). NOW RUN TESTS/BENCHMARK via run_command!", modified_file=rel)
        except Exception as exc:
            return ToolResult(False, f"write_file error: {exc}")

    def edit_file(self, path: str, target_text: str, replacement: str) -> ToolResult:
        if self.read_only:
            return ToolResult(False, "Workspace is in READ-ONLY mode; edit_file is disabled.")
        try:
            target = self._safe_path(path)
            if not target.is_file():
                return ToolResult(False, f"File not found for edit_file: {path}")
            if not target_text:
                return ToolResult(False, "edit_file target_text cannot be empty")
            orig = target.read_text(encoding="utf-8")
            updated: Optional[str] = None
            if target_text in orig:
                cnt = orig.count(target_text)
                if cnt > 1:
                    return ToolResult(False, f"Ambiguous edit_file target ({cnt} occurrences) in {path}; include more surrounding context lines.")
                updated = orig.replace(target_text, replacement, 1)
            else:
                # Trailing-whitespace tolerant line match
                orig_lines = orig.splitlines()
                s_lines = [l.rstrip() for l in target_text.splitlines()]
                hits = [
                    i
                    for i in range(len(orig_lines) - len(s_lines) + 1)
                    if [l.rstrip() for l in orig_lines[i : i + len(s_lines)]] == s_lines
                ]
                if len(hits) == 1:
                    i = hits[0]
                    new_lines = orig_lines[:i] + replacement.splitlines() + orig_lines[i + len(s_lines) :]
                    updated = "\n".join(new_lines) + ("\n" if orig.endswith("\n") else "")
                else:
                    # Show closest matching line in file to help Flash-Next fix its target on the next turn!
                    first_probe = next((l.strip() for l in target_text.splitlines() if l.strip()), "")
                    close = difflib.get_close_matches(first_probe, [l.strip() for l in orig_lines], n=2, cutoff=0.5)
                    return ToolResult(
                        False,
                        f"edit_file target not found in {path}. Re-read exact lines first. Closest lines: {close}",
                    )
            if target.suffix == ".py":
                try:
                    ast.parse(updated)
                except SyntaxError as exc:
                    return ToolResult(
                        False,
                        f"Refused edit_file({path}): resulting Python SyntaxError line {exc.lineno}: {exc.msg}",
                    )
            elif target.suffix in {".c", ".cpp", ".cc", ".cxx"}:
                c_err = self._check_c_syntax(target, updated)
                if c_err:
                    return ToolResult(
                        False,
                        f"Refused edit_file({path}): C/C++ compiler error:\n{c_err}",
                    )
            target.write_text(updated, encoding="utf-8")
            rel = self._rel(target)
            return ToolResult(True, f"Edited {rel} cleanly (compile/syntax OK). Run tests/benchmark next!", modified_file=rel)
        except Exception as exc:
            return ToolResult(False, f"edit_file error: {exc}")

    def run_command(self, cmd: str, timeout_s: float = 25.0) -> ToolResult:
        if not cmd or not cmd.strip():
            return ToolResult(False, "Empty command")
        if _DANGEROUS_CMD_RE.search(cmd):
            return ToolResult(False, f"Blocked unsafe command: {cmd}")

        # Minimal scrubbed env (prevents leaking host API keys to subprocesses)
        _SAFE_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TERM", "USER", "SDKROOT", "DEVELOPER_DIR")
        env = {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        base_pypath = os.environ.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self.root) + (os.pathsep + base_pypath if base_pypath else "")

        t0 = time.perf_counter()
        if self.read_only:
            # Strict argv-only (shell=False) allowlist in READ-ONLY mode — zero shell metacharacters or mutations!
            if _SHELL_META_RE.search(cmd):
                return ToolResult(False, f"Shell operators/redirection blocked in READ-ONLY mode: {cmd}")
            try:
                argv = shlex.split(cmd.strip())
            except ValueError as exc:
                return ToolResult(False, f"Invalid command quoting: {exc}")
            if not argv:
                return ToolResult(False, "Empty command")
            prog = Path(argv[0]).name
            if prog not in _RO_ALLOWED_BINS:
                return ToolResult(False, f"Command `{prog}` blocked in READ-ONLY mode (use read_file/search_code/list_dir).")
            if prog == "git":
                sub = next((a for a in argv[1:] if not a.startswith("-")), "")
                if sub not in _RO_GIT_SUBCMDS:
                    return ToolResult(False, f"Git subcommand `{sub}` blocked in READ-ONLY mode.")
            if prog == "find" and any(a in _RO_FIND_FORBIDDEN for a in argv[1:]):
                return ToolResult(False, "Mutating/exec find flags blocked in READ-ONLY mode.")
            try:
                proc = subprocess.run(
                    argv,
                    shell=False,
                    cwd=str(self.root),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=min(30.0, max(2.0, float(timeout_s))),
                )
                dt = time.perf_counter() - t0
                combined = ((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")).strip()
                if len(combined) > 3500:
                    combined = combined[:1500] + "\n...(truncated)...\n" + combined[-1800:]
                header = f"exit={proc.returncode} ({dt:.2f}s)"
                return ToolResult(ok=(proc.returncode == 0), output=f"{header}\n{combined}" if combined else header)
            except subprocess.TimeoutExpired:
                return ToolResult(False, f"Command TIMED OUT after {timeout_s}s: {cmd}")
            except Exception as exc:
                return ToolResult(False, f"run_command error: {exc}")

        # Auto-substitute bare `pytest` / `python` with our active venv Python so packages (pytest/sympy/z3) always resolve!
        exec_cmd = cmd.strip()
        if exec_cmd.startswith("pytest ") or exec_cmd == "pytest":
            exec_cmd = f"{sys.executable} -m " + exec_cmd
        elif exec_cmd.startswith("python "):
            exec_cmd = f"{sys.executable} " + exec_cmd[len("python ") :]
        elif exec_cmd.startswith("python3 "):
            exec_cmd = f"{sys.executable} " + exec_cmd[len("python3 ") :]

        is_test = is_real_test_command(cmd)
        try:
            proc = subprocess.run(
                exec_cmd,
                shell=True,
                cwd=str(self.root),
                env=env,
                capture_output=True,
                text=True,
                timeout=min(45.0, max(2.0, float(timeout_s))),
            )
            dt = time.perf_counter() - t0
            combined = ((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")).strip()
            if len(combined) > 3500:
                combined = combined[:1500] + "\n...(truncated)...\n" + combined[-1800:]
            ok = proc.returncode == 0
            header = f"exit={proc.returncode} ({dt:.2f}s)"
            return ToolResult(
                ok=ok,
                output=f"{header}\n{combined}" if combined else header,
                is_test_run=is_test,
                test_cmd=cmd if is_test else "",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"Command TIMED OUT after {timeout_s}s: {cmd}", is_test_run=is_test, test_cmd=cmd if is_test else "")
        except Exception as exc:
            return ToolResult(False, f"run_command error: {exc}")

    def python_probe(self, code: str, timeout_s: float = 10.0) -> ToolResult:
        """Execute a diagnostic Python/SymPy/Z3 probe with workspace on PYTHONPATH."""
        if self.read_only:
            return ToolResult(False, "Workspace is in READ-ONLY mode; python_probe is disabled.")
        if not code or not code.strip():
            return ToolResult(False, "Empty python_probe code")
        wrapped = f"import sys\nsys.path.insert(0, {str(self.root)!r})\n{code}\n"
        res = run_python(wrapped, timeout_s=min(20.0, float(timeout_s)))
        out = ((res.stdout or "") + ("\nSTDERR:\n" + res.stderr if res.stderr else "")).strip()
        if res.timed_out:
            return ToolResult(False, f"Probe TIMED OUT after {res.elapsed_s:.1f}s\n{out}")
        return ToolResult(res.ok, f"exit={res.returncode} ({res.elapsed_s:.2f}s)\n{out[:2500]}")

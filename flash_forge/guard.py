"""Zero-False-Positive Proof-Carrying Diff Breaker (`cortheon guard`).

Core thesis: NEVER emit LLM prose code-review slop. An adversarial finding is
ONLY reported if it is backed by a mechanically executed failing Python
reproducer script run inside an isolated ephemeral git worktree + Cortheon's
OS sandbox (`run_python` / `sandbox_backend()`).

Workflow:
  1. Extract unified diff (staged + unstaged + untracked vs ``base_ref``, or
     caller-supplied ``diff_text``). Empty diff short-circuits with 0 LLM calls.
  2. Materialize an isolated ephemeral git worktree with the patched state
     (never mutates the user's repository; always cleaned up in ``finally``).
  3. Fan out ``num_adversaries`` parallel clean-context adversarial bug-hunters
     with distinct attack lenses (boundary/edge, state/concurrency,
     regression/contract) that output ONLY executable Python reproducer scripts.
  4. Mechanical Zero-FP Proof Gate inside Cortheon's OS sandbox:
     - Discard prose-only / invalid-syntax outputs.
     - Discard scripts that exit 0 (no bug proven).
     - Discard scripts that crash solely due to ``ImportError`` /
       ``ModuleNotFoundError`` / ``SyntaxError`` in the reproducer itself.
     - Optionally run surviving reproducers against a clean ``base_ref``
       ephemeral worktree to classify regressions (``passed_on_base=True``),
       pre-existing bugs (``passed_on_base=False``), and new-code bugs
       (``passed_on_base=None``).
     - Deduplicate identical failure traces.
  5. Optional ``auto_fix=True``: race ``B=2`` parallel surgical patch
     generators in ephemeral test worktrees, verify ALL proven reproducers
     (plus optional ``test_cmd``) exit 0, and attach a verified unified patch.
"""

from __future__ import annotations

import ast
import asyncio
import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .client import CallSpec, CompletionResult, TensorFoldClient
from .sandbox import ExecResult, check_syntax, extract_python_code, run_python, sandbox_backend

_META_PREFIX = "__CORTHEON_GUARD_META__:"
_REPRO_FILENAME = "<reproducer>"
_INVALID_REPRO_EXCS = frozenset(
    {"ImportError", "ModuleNotFoundError", "SyntaxError", "IndentationError", "TabError"}
)
_BASELINE_MISSING_SYMBOL_EXCS = frozenset(
    {"ImportError", "ModuleNotFoundError", "AttributeError"}
)
_SKIP_COPY_DIRS = frozenset({".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv", "node_modules"})
_MAX_DIFF_CHARS = 24_000
_MAX_SNIPPETS_CHARS = 16_000
_MAX_SINGLE_FILE_CHARS = 8_000
_MAX_UNTRACKED_BYTES = 256 * 1024

ADVERSARY_ROLES: tuple[tuple[str, str, float, int], ...] = (
    (
        "edge_boundary_breaker",
        (
            "Hunt for off-by-one errors, inclusive vs exclusive range bugs, boundary inputs "
            "(0, -1, 1, min/max bounds, empty strings/lists/dicts, single-element inputs, None, "
            "NaN/inf, negative numbers, exact equality at thresholds)."
        ),
        0.35,
        101,
    ),
    (
        "state_concurrency_breaker",
        (
            "Hunt for shared mutable state, unintended input mutation/aliasing, stale caches, "
            "non-idempotency, ordering sensitivity, re-entrancy, partial failure leaving corrupt "
            "state, and exception-safety violations."
        ),
        0.55,
        202,
    ),
    (
        "regression_contract_breaker",
        (
            "Hunt for regressions against removed `-` lines in the diff, broken return types or "
            "invariants, violated docstrings/contracts, wrong error conditions, silent wrong "
            "outputs, and subtle behavioral breaks for callers."
        ),
        0.45,
        303,
    ),
    (
        "type_input_fuzzer",
        (
            "Hunt for unhandled valid input types, Unicode/whitespace edge cases, duplicate keys "
            "or values, nested structures, and deterministic randomized/property-based counterexamples."
        ),
        0.60,
        404,
    ),
    (
        "security_invariant_breaker",
        (
            "Hunt for path traversal, injection, unchecked bounds, integer overflow/precision loss, "
            "unhandled corner cases in parsing/serialization, and violated safety invariants."
        ),
        0.40,
        505,
    ),
)

_FIX_ROLES: tuple[tuple[str, str, float, int], ...] = (
    (
        "surgical_minimal_fix",
        (
            "Make the smallest, cleanest surgical fix that resolves EVERY proven bug reproducer "
            "without altering unrelated behavior or formatting."
        ),
        0.15,
        701,
    ),
    (
        "root_cause_fix",
        (
            "Identify and fix the underlying invariant / root cause cleanly so EVERY proven bug "
            "reproducer and related boundary cases pass."
        ),
        0.45,
        802,
    ),
)

_TITLE_COMMENT_RE = re.compile(r"^\s*#\s*(?:TITLE|BUG)\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_FILE_FENCE_RE = re.compile(r"```(?:file|python|py):([^\n`]+)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_DIFF_FENCE_RE = re.compile(r"```(?:diff|patch)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_PRESENT_RE = re.compile(r"```", re.MULTILINE)


@dataclass
class ProvenBug:
    id: str
    role: str
    title: str
    reproducer_code: str
    failure_output: str
    passed_on_base: Optional[bool]
    target_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "title": self.title,
            "reproducer_code": self.reproducer_code,
            "failure_output": self.failure_output,
            "passed_on_base": self.passed_on_base,
            "target_files": list(self.target_files),
        }


@dataclass
class GuardReport:
    repo_path: str
    base_ref: str
    diff_files: list[str]
    adversaries_run: int
    reproducers_executed: int
    rejected_non_failing: int
    proven_bugs: list[ProvenBug]
    suggested_patch: Optional[str]
    patch_verified: bool
    wall_s: float
    llm_calls: int
    total_tokens: int

    @property
    def clean(self) -> bool:
        return len(self.proven_bugs) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "repo_path": self.repo_path,
            "base_ref": self.base_ref,
            "diff_files": list(self.diff_files),
            "adversaries_run": self.adversaries_run,
            "reproducers_executed": self.reproducers_executed,
            "rejected_non_failing": self.rejected_non_failing,
            "proven_bugs": [b.to_dict() for b in self.proven_bugs],
            "suggested_patch": self.suggested_patch,
            "patch_verified": self.patch_verified,
            "wall_s": round(self.wall_s, 3),
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
        }


@dataclass
class _SandboxRunInfo:
    exec_result: ExecResult
    clean_output: str
    exc_name: Optional[str]
    last_in_repro: bool
    last_file: Optional[str]


def _run_git(repo: Path, args: list[str], *, input_text: Optional[str] = None, timeout_s: float = 20.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _git_ref_exists(repo: Path, ref: str) -> bool:
    try:
        proc = _run_git(repo, ["rev-parse", "--verify", f"{ref}^{{commit}}"])
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _is_text_file(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size > _MAX_UNTRACKED_BYTES:
            return False
        raw = path.read_bytes()[:4096]
        return b"\x00" not in raw
    except OSError:
        return False


def _synthesize_new_file_diff(rel_path: str, content: str) -> str:
    lines = content.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] = lines[-1] + "\n"
    diff_lines = list(
        difflib.unified_diff(
            [],
            lines,
            fromfile="/dev/null",
            tofile=f"b/{rel_path}",
            lineterm="\n",
        )
    )
    if not diff_lines:
        return ""
    header = f"diff --git a/{rel_path} b/{rel_path}\nnew file mode 100644\n"
    return header + "".join(diff_lines)


def _parse_diff_files(diff_text: str) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                cand = parts[3]
                if cand.startswith("b/"):
                    cand = cand[2:]
                if cand and cand != "/dev/null" and cand not in seen:
                    seen.add(cand)
                    files.append(cand)
        elif line.startswith("+++ "):
            cand = line[4:].strip().split("\t", 1)[0]
            if cand.startswith("b/"):
                cand = cand[2:]
            if cand and cand != "/dev/null" and cand not in seen:
                seen.add(cand)
                files.append(cand)
    return files


def _parse_name_status_z(raw_stdout: str) -> list[tuple[str, str, Optional[str]]]:
    """Parse ``git diff --name-status -z`` into ``(status, path1, path2_or_none)``."""
    tokens = [t for t in raw_stdout.split("\0") if t != ""]
    entries: list[tuple[str, str, Optional[str]]] = []
    i = 0
    while i < len(tokens):
        status = tokens[i]
        i += 1
        if status.startswith(("R", "C")):
            if i + 1 < len(tokens):
                entries.append((status, tokens[i], tokens[i + 1]))
                i += 2
            else:
                break
        else:
            if i < len(tokens):
                entries.append((status, tokens[i], None))
                i += 1
            else:
                break
    return entries


def _collect_repo_diff(
    repo: Path,
    base_ref: str,
    base_exists: bool,
) -> tuple[str, list[str], list[tuple[str, str, Optional[str]]], list[str]]:
    """Collect unified diff, changed files, name-status ops, and untracked files."""
    diff_parts: list[str] = []
    diff_files: list[str] = []
    seen: set[str] = set()
    status_entries: list[tuple[str, str, Optional[str]]] = []

    if base_exists:
        diff_proc = _run_git(repo, ["diff", "--no-ext-diff", "--unified=3", base_ref])
        if diff_proc.returncode == 0 and diff_proc.stdout.strip():
            diff_parts.append(diff_proc.stdout)

        ns_proc = _run_git(repo, ["diff", "--name-status", "-z", base_ref])
        if ns_proc.returncode == 0 and ns_proc.stdout:
            status_entries = _parse_name_status_z(ns_proc.stdout)
            for status, p1, p2 in status_entries:
                target = p2 if p2 else p1
                if target and target not in seen:
                    seen.add(target)
                    diff_files.append(target)

    # Include untracked non-ignored files cleanly
    untracked_files: list[str] = []
    ut_proc = _run_git(repo, ["ls-files", "--others", "--exclude-standard", "-z"])
    if ut_proc.returncode == 0 and ut_proc.stdout:
        for rel in ut_proc.stdout.split("\0"):
            if not rel:
                continue
            parts = Path(rel).parts
            if any(p in _SKIP_COPY_DIRS for p in parts):
                continue
            untracked_files.append(rel)
            if rel not in seen:
                seen.add(rel)
                diff_files.append(rel)
            full_path = repo / rel
            if _is_text_file(full_path):
                try:
                    content = full_path.read_text(encoding="utf-8", errors="replace")
                    syn = _synthesize_new_file_diff(rel, content)
                    if syn:
                        diff_parts.append(syn)
                except OSError:
                    pass

    return "\n".join(p.rstrip("\n") for p in diff_parts if p.strip()), diff_files, status_entries, untracked_files


def _create_ephemeral_worktree(repo: Path, ref: str, prefix: str = "cortheon_guard_wt_") -> Path:
    wt_dir = Path(tempfile.mkdtemp(prefix=prefix)).resolve()
    add = _run_git(repo, ["worktree", "add", "--detach", str(wt_dir), ref])
    if add.returncode != 0:
        shutil.rmtree(wt_dir, ignore_errors=True)
        raise RuntimeError(f"git worktree add failed for {ref}: {add.stderr.strip()}")
    return wt_dir


def _remove_ephemeral_worktree(repo: Path, wt_dir: Optional[Path], *, is_git_worktree: bool = True) -> None:
    if wt_dir is None:
        return
    if is_git_worktree:
        try:
            _run_git(repo, ["worktree", "remove", "--force", str(wt_dir)])
        except (OSError, subprocess.SubprocessError):
            pass
    shutil.rmtree(wt_dir, ignore_errors=True)


def _copy_safe_rel(src_root: Path, dst_root: Path, rel_path: str) -> None:
    src = (src_root / rel_path).resolve()
    dst = (dst_root / rel_path).resolve()
    if not src.is_relative_to(src_root) or not dst.is_relative_to(dst_root):
        raise ValueError(f"Refusing path outside workspace boundary: {rel_path}")
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _delete_safe_rel(dst_root: Path, rel_path: str) -> None:
    dst = (dst_root / rel_path).resolve()
    if not dst.is_relative_to(dst_root):
        raise ValueError(f"Refusing path outside workspace boundary: {rel_path}")
    if dst.is_file() or dst.is_symlink():
        dst.unlink(missing_ok=True)
    elif dst.is_dir():
        shutil.rmtree(dst, ignore_errors=True)


def _mirror_working_changes_into_worktree(
    repo: Path,
    wt_dir: Path,
    status_entries: list[tuple[str, str, Optional[str]]],
    untracked_files: list[str],
) -> None:
    for status, p1, p2 in status_entries:
        code = status[:1]
        if code == "D":
            _delete_safe_rel(wt_dir, p1)
        elif code == "R" and p2:
            _delete_safe_rel(wt_dir, p1)
            if (repo / p2).exists():
                _copy_safe_rel(repo, wt_dir, p2)
        elif code == "C" and p2:
            if (repo / p2).exists():
                _copy_safe_rel(repo, wt_dir, p2)
        else:
            if (repo / p1).exists():
                _copy_safe_rel(repo, wt_dir, p1)
            else:
                _delete_safe_rel(wt_dir, p1)

    for rel in untracked_files:
        if (repo / rel).is_file():
            _copy_safe_rel(repo, wt_dir, rel)


def _copy_non_git_workspace(repo: Path) -> Path:
    wt_dir = Path(tempfile.mkdtemp(prefix="cortheon_guard_copy_")).resolve()
    shutil.copytree(
        repo,
        wt_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(*_SKIP_COPY_DIRS),
    )
    return wt_dir


def _build_file_snippets(wt_dir: Path, diff_files: list[str]) -> str:
    """Gather bounded contents of changed files from the ephemeral patched worktree."""
    # Prioritize Python files first
    ordered = sorted(diff_files, key=lambda p: (0 if p.endswith(".py") else 1, p))
    chunks: list[str] = []
    used = 0
    for rel in ordered:
        if used >= _MAX_SNIPPETS_CHARS:
            break
        full = (wt_dir / rel).resolve()
        if not full.is_relative_to(wt_dir) or not _is_text_file(full):
            continue
        try:
            text = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        budget = min(_MAX_SINGLE_FILE_CHARS, _MAX_SNIPPETS_CHARS - used)
        if len(text) > budget:
            text = text[:budget] + "\n# ... (truncated)\n"
        lang = "python" if rel.endswith(".py") else ""
        chunks.append(f"### FILE: {rel}\n```{lang}\n{text.rstrip()}\n```")
        used += len(text)
    return "\n\n".join(chunks)


def _extract_valid_reproducer(raw_output: str) -> Optional[str]:
    """Extract an executable Python reproducer; reject prose slop and non-executable fragments."""
    if not raw_output or not raw_output.strip():
        return None

    has_fence = bool(_CODE_FENCE_PRESENT_RE.search(raw_output))
    code = extract_python_code(raw_output)
    if not code or not code.strip():
        return None
    if check_syntax(code) is not None:
        return None

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    if not tree.body:
        return None

    # Guard against bare prose words (e.g. "LGTM" or "OK") that parse as a single ast.Name expression.
    has_executable_intent = any(
        isinstance(
            node,
            (ast.Import, ast.ImportFrom, ast.Assert, ast.Call, ast.Raise, ast.FunctionDef, ast.AsyncFunctionDef),
        )
        for node in ast.walk(tree)
    )
    if not has_executable_intent:
        return None
    if not has_fence and not any(isinstance(node, (ast.Import, ast.ImportFrom, ast.Assert)) for node in ast.walk(tree)):
        return None

    return code.strip()


def _run_reproducer_in_sandbox(
    reproducer_code: str,
    worktree_dir: Path,
    user_repo_dir: Path,
    timeout_s: float,
) -> _SandboxRunInfo:
    """Execute ``reproducer_code`` against ``worktree_dir`` inside Cortheon's OS sandbox."""
    wt_str = str(worktree_dir.resolve())
    repo_str = str(user_repo_dir.resolve())

    harness = (
        "import linecache as _cg_lc\n"
        "import os as _cg_os\n"
        "import sys as _cg_sys\n"
        "import traceback as _cg_tb\n"
        f"_CG_WT = {wt_str!r}\n"
        f"_CG_REPO = {repo_str!r}\n"
        f"_CG_REPRO = {reproducer_code!r}\n"
        f"_CG_META = {_META_PREFIX!r}\n"
        f"_CG_RFNAME = {_REPRO_FILENAME!r}\n"
        "_cg_lc.cache[_CG_RFNAME] = (len(_CG_REPRO), None, _CG_REPRO.splitlines(True), _CG_RFNAME)\n"
        "_cg_real_repo = _cg_os.path.realpath(_CG_REPO)\n"
        "_cg_sys.path[:] = [\n"
        "    p for p in _cg_sys.path\n"
        "    if not (p and _cg_os.path.realpath(p) == _cg_real_repo)\n"
        "]\n"
        "_cg_sys.path.insert(0, _CG_WT)\n"
        "_cg_os.chdir(_CG_WT)\n"
        "_cg_ns = {'__name__': '__main__', '__file__': _cg_os.path.join(_CG_WT, _CG_RFNAME)}\n"
        "try:\n"
        "    _cg_compiled = compile(_CG_REPRO, _CG_RFNAME, 'exec')\n"
        "    exec(_cg_compiled, _cg_ns)\n"
        "except SystemExit as _cg_se:\n"
        "    _cg_val = _cg_se.code\n"
        "    if _cg_val is None or _cg_val == 0:\n"
        "        _cg_sys.exit(0)\n"
        "    if isinstance(_cg_val, int):\n"
        "        _cg_sys.stderr.write(f'SystemExit: {_cg_val}\\n')\n"
        "        _cg_sys.stderr.write(f'\\n{_CG_META}SystemExit:1:{_CG_RFNAME}\\n')\n"
        "        _cg_sys.exit(_cg_val)\n"
        "    _cg_sys.stderr.write(f'SystemExit: {_cg_val}\\n')\n"
        "    _cg_sys.stderr.write(f'\\n{_CG_META}SystemExit:1:{_CG_RFNAME}\\n')\n"
        "    _cg_sys.exit(1)\n"
        "except BaseException as _cg_exc:\n"
        "    _cg_frames = _cg_tb.extract_tb(_cg_exc.__traceback__)\n"
        "    while _cg_frames and _cg_frames[0].filename.endswith('_run.py'):\n"
        "        _cg_frames = _cg_frames[1:]\n"
        "    _cg_last_file = _cg_frames[-1].filename if _cg_frames else _CG_RFNAME\n"
        "    _cg_last_in_repro = (_cg_last_file == _CG_RFNAME)\n"
        "    if isinstance(_cg_exc, SyntaxError) and getattr(_cg_exc, 'filename', None):\n"
        "        _cg_last_file = str(_cg_exc.filename)\n"
        "        _cg_last_in_repro = (_cg_last_file == _CG_RFNAME)\n"
        "    _cg_exc_name = type(_cg_exc).__name__\n"
        "    _cg_fmt = ''.join(_cg_tb.format_list(_cg_frames)) + ''.join(_cg_tb.format_exception_only(type(_cg_exc), _cg_exc))\n"
        "    _cg_sys.stderr.write('Traceback (most recent call last):\\n' + _cg_fmt)\n"
        "    _cg_sys.stderr.write(f'\\n{_CG_META}{_cg_exc_name}:{int(_cg_last_in_repro)}:{_cg_last_file}\\n')\n"
        "    _cg_sys.exit(1)\n"
    )

    res = run_python(harness, timeout_s=timeout_s)
    exc_name: Optional[str] = None
    last_in_repro = False
    last_file: Optional[str] = None

    stderr_lines = (res.stderr or "").splitlines()
    kept_stderr_lines: list[str] = []
    for line in stderr_lines:
        if line.startswith(_META_PREFIX):
            payload = line[len(_META_PREFIX) :]
            parts = payload.split(":", 2)
            if len(parts) == 3:
                exc_name = parts[0] or None
                last_in_repro = parts[1] == "1"
                last_file = parts[2] or None
        else:
            kept_stderr_lines.append(line)

    if res.timed_out:
        clean_out = f"TIMEOUT after {res.elapsed_s:.2f}s"
    else:
        raw_err = "\n".join(kept_stderr_lines).strip()
        if not raw_err:
            raw_err = (res.stdout or "").strip()
        # Normalize ephemeral worktree prefix to '.' for crisp, reproducible output
        clean_out = raw_err.replace(wt_str + os.sep, "").replace(wt_str, ".")
        if len(clean_out) > 2400:
            clean_out = "...(truncated)...\n" + clean_out[-2400:]

    return _SandboxRunInfo(
        exec_result=res,
        clean_output=clean_out,
        exc_name=exc_name,
        last_in_repro=last_in_repro,
        last_file=last_file,
    )


def _extract_bug_title(reproducer_code: str, failure_output: str, role: str, exc_name: Optional[str]) -> str:
    m = _TITLE_COMMENT_RE.search(reproducer_code)
    if m:
        title = m.group(1).strip()
        if title:
            return title[:120]
    lines = [ln.strip() for ln in failure_output.splitlines() if ln.strip()]
    if lines:
        tail = lines[-1]
        if len(tail) > 100:
            tail = tail[:97] + "..."
        return tail
    return f"{role}: {exc_name or 'Reproducer failure'}"


def _infer_target_files(
    reproducer_code: str,
    failure_output: str,
    diff_files: list[str],
) -> list[str]:
    matched: list[str] = []
    for rel in diff_files:
        norm = rel.replace("\\", "/")
        stem = Path(norm).stem
        mod_path = norm[:-3].replace("/", ".") if norm.endswith(".py") else ""
        if (
            norm in failure_output
            or (mod_path and re.search(rf"\b{re.escape(mod_path)}\b", reproducer_code))
            or (stem and norm.endswith(".py") and re.search(rf"\b{re.escape(stem)}\b", reproducer_code))
        ):
            matched.append(rel)
    if matched:
        return matched
    py_diff = [f for f in diff_files if f.endswith(".py")]
    return py_diff if py_diff else list(diff_files)


def _failure_dedup_key(info: _SandboxRunInfo, target_files: Optional[list[str]] = None) -> str:
    """Compute a normalized deduplication fingerprint for a failing execution."""
    text = info.clean_output.strip()
    # Normalize reproducer line numbers so identical assertions at different lines deduplicate
    normalized = re.sub(r'File "<reproducer>", line \d+', 'File "<reproducer>", line _', text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    files_tag = ",".join(sorted(target_files)) if target_files else ""
    return f"{info.exc_name or 'ERR'}::{files_tag}::{normalized[-600:]}"


def _build_adversary_specs(
    diff_text: str,
    file_snippets: str,
    num_adversaries: int,
) -> list[tuple[str, CallSpec]]:
    if num_adversaries <= 0:
        return []

    clipped_diff = diff_text
    if len(clipped_diff) > _MAX_DIFF_CHARS:
        clipped_diff = clipped_diff[:_MAX_DIFF_CHARS] + "\n... (diff truncated)\n"

    system_prompt = (
        "You are an adversarial bug-hunting verifier for Cortheon Guard.\n"
        "STRICT RULES:\n"
        "1. NEVER write prose code review, opinions, or summaries.\n"
        "2. Output ONLY ONE self-contained Python 3 reproducer script inside a ```python ... ``` code block.\n"
        "3. Put `# TITLE: <concise bug title>` as the first line of the script.\n"
        "4. Start with `import sys; sys.path.insert(0, '.')` and import/call the ACTUAL changed code.\n"
        "5. Assert the CORRECT expected behavior so the script raises AssertionError (or exits non-zero) "
        "ONLY when a genuine bug exists in the diff, and exits 0 cleanly if the code is correct.\n"
        "6. Do NOT invent non-existent functions/modules, do NOT use network I/O, and do NOT write files "
        "in the repository directory (use `tempfile` if temporary storage is needed)."
    )

    user_base = (
        f"UNIFIED DIFF TO BREAK:\n```diff\n{clipped_diff}\n```\n\n"
        f"CURRENT CHANGED FILE CONTENTS:\n{file_snippets or '(no text snippets available)'}"
    )

    specs: list[tuple[str, CallSpec]] = []
    for idx in range(num_adversaries):
        base_role, lens, temp, seed = ADVERSARY_ROLES[idx % len(ADVERSARY_ROLES)]
        cycle = idx // len(ADVERSARY_ROLES)
        role_name = base_role if cycle == 0 else f"{base_role}_{cycle + 1}"
        spec = CallSpec(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{user_base}\n\n"
                        f"ATTACK LENS ({role_name}):\n{lens}\n\n"
                        "Output ONLY the Python 3 reproducer script inside a single ```python ... ``` block."
                    ),
                },
            ],
            tag=f"guard_adv_{role_name}",
            max_tokens=2048,
            temperature=temp,
            seed=seed + cycle * 1000,
            enable_thinking=True,
            thinking_budget=1200,
            loop_guard=True,
        )
        specs.append((role_name, spec))
    return specs


def _parse_fix_candidate(
    raw_output: str,
    candidate_files: list[str],
) -> tuple[dict[str, str], Optional[str]]:
    """Parse an auto-fix response into either ``(file_updates, None)`` or ``({}, unified_diff)``."""
    if not raw_output or not raw_output.strip():
        return {}, None

    # 1. Explicit multi-file fenced blocks: ```file:path/to/file.py ... ```
    file_matches = _FILE_FENCE_RE.findall(raw_output)
    if file_matches:
        updates: dict[str, str] = {}
        for rel_raw, body in file_matches:
            rel = rel_raw.strip().lstrip("./")
            if rel and body.strip():
                updates[rel] = body if body.endswith("\n") else body + "\n"
        if updates:
            return updates, None

    # 2. Explicit unified diff block
    diff_matches = _DIFF_FENCE_RE.findall(raw_output)
    if diff_matches:
        patch = diff_matches[0].strip()
        if patch and ("@@" in patch or "--- " in patch):
            return {}, patch + "\n"

    stripped = raw_output.strip()
    if (stripped.startswith("diff --git ") or stripped.startswith("--- ")) and "@@" in stripped:
        return {}, stripped + "\n"

    # 3. Single Python file full replacement
    py_targets = [f for f in candidate_files if f.endswith(".py")]
    target = py_targets[0] if len(py_targets) == 1 else (candidate_files[0] if len(candidate_files) == 1 else None)
    if target:
        code = extract_python_code(raw_output)
        if code and code.strip():
            return {target: code if code.endswith("\n") else code + "\n"}, None

    return {}, None


def _compute_unified_patch(patched_wt: Path, fix_wt: Path, rel_paths: list[str]) -> str:
    """Compute a git-applyable unified diff from buggy ``patched_wt`` to verified ``fix_wt``."""
    chunks: list[str] = []
    for rel in sorted(set(rel_paths)):
        old_p = (patched_wt / rel).resolve()
        new_p = (fix_wt / rel).resolve()
        if not old_p.is_relative_to(patched_wt) or not new_p.is_relative_to(fix_wt):
            continue
        old_text = old_p.read_text(encoding="utf-8", errors="replace") if old_p.is_file() else ""
        new_text = new_p.read_text(encoding="utf-8", errors="replace") if new_p.is_file() else ""
        if old_text == new_text:
            continue
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        if old_lines and not old_lines[-1].endswith("\n"):
            old_lines[-1] = old_lines[-1] + "\n"
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] = new_lines[-1] + "\n"
        diff_lines = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                lineterm="\n",
            )
        )
        if diff_lines:
            chunks.append(f"diff --git a/{rel} b/{rel}\n" + "".join(diff_lines))
    return "".join(chunks)


def _run_test_cmd_in_dir(test_cmd: str, work_dir: Path, user_repo_dir: Path, timeout_s: float) -> ExecResult:
    t0 = time.perf_counter()
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    real_repo = str(user_repo_dir.resolve())
    existing_pp = [
        p for p in env.get("PYTHONPATH", "").split(os.pathsep)
        if p and os.path.realpath(p) != real_repo
    ]
    env["PYTHONPATH"] = os.pathsep.join([str(work_dir.resolve()), *existing_pp])
    try:
        proc = subprocess.run(
            test_cmd,
            shell=True,
            cwd=str(work_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return ExecResult(
            ok=(proc.returncode == 0),
            returncode=proc.returncode,
            stdout=(proc.stdout or "")[:8000],
            stderr=(proc.stderr or "")[:8000],
            elapsed_s=time.perf_counter() - t0,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return ExecResult(
            ok=False,
            returncode=-1,
            stdout=out[:4000],
            stderr=err[:4000],
            elapsed_s=time.perf_counter() - t0,
            timed_out=True,
        )


async def _attempt_auto_fix(
    client: TensorFoldClient,
    repo: Path,
    patched_wt: Path,
    diff_text: str,
    diff_files: list[str],
    proven_bugs: list[ProvenBug],
    *,
    test_cmd: Optional[str],
    timeout_s: float,
) -> tuple[Optional[str], bool, int, int]:
    """Race B=2 surgical patch generators and mechanically verify against all proven bugs + test_cmd."""
    target_files: list[str] = []
    seen_files: set[str] = set()
    for bug in proven_bugs:
        for f in bug.target_files:
            if f not in seen_files and (patched_wt / f).is_file():
                seen_files.add(f)
                target_files.append(f)
    if not target_files:
        for f in diff_files:
            if f not in seen_files and (patched_wt / f).is_file():
                seen_files.add(f)
                target_files.append(f)
    if not target_files:
        return None, False, 0, 0

    snippets = _build_file_snippets(patched_wt, target_files)
    bugs_block_parts: list[str] = []
    for bug in proven_bugs:
        bugs_block_parts.append(
            f"### {bug.id} ({bug.role}): {bug.title}\n"
            f"Reproducer:\n```python\n{bug.reproducer_code}\n```\n"
            f"Failure output:\n```\n{bug.failure_output}\n```"
        )
    bugs_block = "\n\n".join(bugs_block_parts)

    if len(target_files) == 1:
        format_instr = (
            f"Output ONLY the complete fixed content of `{target_files[0]}` inside a single "
            "```python ... ``` block (or a unified ```diff ... ``` patch). Never leave placeholders."
        )
    else:
        format_instr = (
            "Output ONLY the complete fixed file(s) using fenced blocks labeled ```file:<relative_path> ... ``` "
            "for each modified file (or a unified ```diff ... ``` patch). Never leave placeholders."
        )

    sys_prompt = (
        "You are a principal software engineer fixing mechanically proven bugs.\n"
        f"{format_instr}"
    )
    clipped_diff = diff_text[:_MAX_DIFF_CHARS]
    user_base = (
        f"CURRENT BUGGY DIFF:\n```diff\n{clipped_diff}\n```\n\n"
        f"CURRENT FILE(S):\n{snippets}\n\n"
        f"MECHANICALLY PROVEN FAILING REPRODUCERS (ALL MUST PASS):\n{bugs_block}"
    )

    specs = [
        CallSpec(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"{user_base}\n\nSTRATEGY ({role_name}):\n{guidance}"},
            ],
            tag=f"guard_fix_{role_name}",
            max_tokens=4096,
            temperature=temp,
            seed=seed,
            enable_thinking=True,
            thinking_budget=1500,
            loop_guard=True,
        )
        for role_name, guidance, temp, seed in _FIX_ROLES
    ]
    results: list[CompletionResult] = await client.batch(specs)
    fix_calls = len(results)
    fix_tokens = sum(r.prompt_tokens + r.completion_tokens for r in results)

    for res in results:
        if not res.ok:
            continue
        file_updates, diff_patch = _parse_fix_candidate(res.content, target_files)
        if not file_updates and not diff_patch:
            continue

        # Validate syntax for Python file updates before running sandbox checks
        syntax_bad = False
        for rel, content in file_updates.items():
            if rel.endswith(".py") and check_syntax(content) is not None:
                syntax_bad = True
                break
        if syntax_bad:
            continue

        # Verify in an isolated ephemeral copy of patched_wt (never touches repo or patched_wt)
        fix_wt = Path(tempfile.mkdtemp(prefix="cortheon_guard_fix_")).resolve()
        try:
            await asyncio.to_thread(
                shutil.copytree,
                patched_wt,
                fix_wt,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(".git", "__pycache__"),
            )
            modified_rels: list[str] = []
            if file_updates:
                for rel, content in file_updates.items():
                    dest = (fix_wt / rel).resolve()
                    if not dest.is_relative_to(fix_wt):
                        modified_rels = []
                        break
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")
                    modified_rels.append(rel)
                if not modified_rels:
                    continue
            elif diff_patch:
                # Use git apply in non-repo directory via --directory or temporary git init
                apply_proc = await asyncio.to_thread(
                    subprocess.run,
                    ["git", "apply", "--whitespace=nowarn", "-"],
                    input=diff_patch,
                    cwd=str(fix_wt),
                    capture_output=True,
                    text=True,
                    timeout=15.0,
                )
                if apply_proc.returncode != 0:
                    continue
                modified_rels = _parse_diff_files(diff_patch) or list(target_files)

            # Every proven reproducer must now EXIT 0 in the OS sandbox
            all_fixed = True
            for bug in proven_bugs:
                run_info = await asyncio.to_thread(
                    _run_reproducer_in_sandbox,
                    bug.reproducer_code,
                    fix_wt,
                    repo,
                    timeout_s,
                )
                if not run_info.exec_result.ok:
                    all_fixed = False
                    break
            if not all_fixed:
                continue

            # Optional repo test_cmd must also pass
            if test_cmd:
                t_res = await asyncio.to_thread(
                    _run_test_cmd_in_dir,
                    test_cmd,
                    fix_wt,
                    repo,
                    timeout_s,
                )
                if not t_res.ok:
                    continue

            unified_patch = _compute_unified_patch(patched_wt, fix_wt, modified_rels)
            if unified_patch.strip():
                return unified_patch, True, fix_calls, fix_tokens
        finally:
            shutil.rmtree(fix_wt, ignore_errors=True)

    return None, False, fix_calls, fix_tokens


async def run_guard(
    client: TensorFoldClient,
    repo_path: str | Path,
    *,
    base_ref: str = "HEAD",
    diff_text: Optional[str] = None,
    num_adversaries: int = 3,
    verify_baseline: bool = True,
    auto_fix: bool = False,
    test_cmd: Optional[str] = None,
    timeout_s: float = 15.0,
) -> GuardReport:
    """Run Cortheon Guard proof-carrying adversarial diff verification.

    Guarantees:
    - Never mutates ``repo_path``.
    - Always cleans up all ephemeral git worktrees in ``finally``.
    - Never reports an LLM claim unless backed by an actually failing Python
      reproducer executed inside Cortheon's OS sandbox.
    """
    t0 = time.perf_counter()
    repo = Path(repo_path).resolve()
    base_exists = await asyncio.to_thread(_git_ref_exists, repo, base_ref)

    status_entries: list[tuple[str, str, Optional[str]]] = []
    untracked_files: list[str] = []
    caller_supplied_diff = diff_text is not None

    if caller_supplied_diff:
        effective_diff = diff_text or ""
        diff_files = _parse_diff_files(effective_diff)
    else:
        effective_diff, diff_files, status_entries, untracked_files = await asyncio.to_thread(
            _collect_repo_diff, repo, base_ref, base_exists
        )

    # Empty diff short-circuit: 0 LLM calls, 0 worktrees
    if not effective_diff.strip() and not diff_files:
        return GuardReport(
            repo_path=str(repo),
            base_ref=base_ref,
            diff_files=[],
            adversaries_run=0,
            reproducers_executed=0,
            rejected_non_failing=0,
            proven_bugs=[],
            suggested_patch=None,
            patch_verified=False,
            wall_s=time.perf_counter() - t0,
            llm_calls=0,
            total_tokens=0,
        )

    patched_wt: Optional[Path] = None
    patched_is_wt = False
    base_wt: Optional[Path] = None

    llm_calls = 0
    total_tokens = 0
    reproducers_executed = 0
    rejected_non_failing = 0
    proven_bugs: list[ProvenBug] = []
    suggested_patch: Optional[str] = None
    patch_verified = False

    try:
        # Step 1: Materialize isolated ephemeral patched worktree
        if base_exists:
            patched_wt = await asyncio.to_thread(_create_ephemeral_worktree, repo, base_ref, "cortheon_guard_wt_")
            patched_is_wt = True
            if caller_supplied_diff:
                apply_proc = await asyncio.to_thread(
                    _run_git,
                    patched_wt,
                    ["apply", "--whitespace=nowarn", "-"],
                    input_text=effective_diff,
                )
                if apply_proc.returncode != 0:
                    raise RuntimeError(f"Failed to apply supplied diff_text in ephemeral worktree: {apply_proc.stderr.strip()}")
            else:
                await asyncio.to_thread(
                    _mirror_working_changes_into_worktree,
                    repo,
                    patched_wt,
                    status_entries,
                    untracked_files,
                )
        else:
            patched_wt = await asyncio.to_thread(_copy_non_git_workspace, repo)
            patched_is_wt = False

        # Step 2: Launch parallel clean-context adversaries
        file_snippets = await asyncio.to_thread(_build_file_snippets, patched_wt, diff_files)
        adv_specs_pairs = _build_adversary_specs(effective_diff, file_snippets, num_adversaries)
        adversaries_run = len(adv_specs_pairs)

        adv_results: list[CompletionResult] = []
        if adv_specs_pairs:
            adv_results = await client.batch([spec for _, spec in adv_specs_pairs])
            llm_calls += len(adv_results)
            total_tokens += sum(r.prompt_tokens + r.completion_tokens for r in adv_results)

        # Step 3: Mechanical Zero-FP Proof Gate inside Cortheon's OS sandbox
        seen_traces: set[str] = set()
        for (role_name, _spec), res in zip(adv_specs_pairs, adv_results):
            if not res.ok:
                rejected_non_failing += 1
                continue

            repro_code = _extract_valid_reproducer(res.content)
            if repro_code is None:
                rejected_non_failing += 1
                continue

            reproducers_executed += 1
            patched_info = await asyncio.to_thread(
                _run_reproducer_in_sandbox,
                repro_code,
                patched_wt,
                repo,
                timeout_s,
            )

            # Exit 0 -> code survived the attack -> discard
            if patched_info.exec_result.ok:
                rejected_non_failing += 1
                continue

            # Reproducer itself failed due to bad import/syntax (not target code) -> discard
            if (
                not patched_info.exec_result.timed_out
                and patched_info.last_in_repro
                and patched_info.exc_name in _INVALID_REPRO_EXCS
            ):
                rejected_non_failing += 1
                continue

            target_files = _infer_target_files(repro_code, patched_info.clean_output, diff_files)

            # Deduplicate identical failure traces
            dedup_key = _failure_dedup_key(patched_info, target_files)
            if dedup_key in seen_traces:
                continue
            seen_traces.add(dedup_key)

            # Optional baseline verification on clean base_ref worktree
            passed_on_base: Optional[bool] = None
            if verify_baseline and base_exists:
                if base_wt is None:
                    base_wt = await asyncio.to_thread(_create_ephemeral_worktree, repo, base_ref, "cortheon_guard_base_")
                base_info = await asyncio.to_thread(
                    _run_reproducer_in_sandbox,
                    repro_code,
                    base_wt,
                    repo,
                    timeout_s,
                )
                if base_info.exec_result.ok:
                    passed_on_base = True
                elif (
                    not base_info.exec_result.timed_out
                    and base_info.last_in_repro
                    and base_info.exc_name in _BASELINE_MISSING_SYMBOL_EXCS
                ):
                    # Newly added function/class/module in diff -> still a real bug in new code
                    passed_on_base = None
                else:
                    passed_on_base = False

            bug_id = f"BUG-{len(proven_bugs) + 1:02d}"
            title = _extract_bug_title(repro_code, patched_info.clean_output, role_name, patched_info.exc_name)

            proven_bugs.append(
                ProvenBug(
                    id=bug_id,
                    role=role_name,
                    title=title,
                    reproducer_code=repro_code,
                    failure_output=patched_info.clean_output,
                    passed_on_base=passed_on_base,
                    target_files=target_files,
                )
            )

        # Step 4: Optional auto-fix with mechanical verification
        if auto_fix and proven_bugs:
            suggested_patch, patch_verified, fix_calls, fix_tokens = await _attempt_auto_fix(
                client=client,
                repo=repo,
                patched_wt=patched_wt,
                diff_text=effective_diff,
                diff_files=diff_files,
                proven_bugs=proven_bugs,
                test_cmd=test_cmd,
                timeout_s=timeout_s,
            )
            llm_calls += fix_calls
            total_tokens += fix_tokens

    finally:
        await asyncio.to_thread(_remove_ephemeral_worktree, repo, base_wt, is_git_worktree=True)
        await asyncio.to_thread(_remove_ephemeral_worktree, repo, patched_wt, is_git_worktree=patched_is_wt)

    return GuardReport(
        repo_path=str(repo),
        base_ref=base_ref,
        diff_files=diff_files,
        adversaries_run=adversaries_run,
        reproducers_executed=reproducers_executed,
        rejected_non_failing=rejected_non_failing,
        proven_bugs=proven_bugs,
        suggested_patch=suggested_patch,
        patch_verified=patch_verified,
        wall_s=time.perf_counter() - t0,
        llm_calls=llm_calls,
        total_tokens=total_tokens,
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


def format_guard_report(report: GuardReport, *, use_color: Optional[bool] = None) -> str:
    """Format a :class:`GuardReport` as crisp terminal output (zero prose slop)."""
    color = _should_use_color(use_color)
    bar = "━" * 74
    rule = "─" * 74
    backend = sandbox_backend()
    files_summary = ", ".join(report.diff_files[:6]) + (" ..." if len(report.diff_files) > 6 else "")

    lines: list[str] = [
        _paint(bar, _ANSI.CYAN, _ANSI.BOLD, color=color),
        _paint("  Cortheon Guard — Proof-Carrying Diff Breaker", _ANSI.CYAN, _ANSI.BOLD, color=color),
        _paint(
            f"  Repo: {report.repo_path} | Base: {report.base_ref} | Sandbox: {backend}",
            _ANSI.DIM,
            color=color,
        ),
        _paint(
            f"  Diff files ({len(report.diff_files)}): {files_summary or '(none)'}",
            _ANSI.DIM,
            color=color,
        ),
        _paint(bar, _ANSI.CYAN, _ANSI.BOLD, color=color),
        "",
    ]

    if report.clean:
        badge = _paint("[CLEAN]", _ANSI.GREEN, _ANSI.BOLD, color=color)
        lines.append(
            f"  {badge} 0 proven bugs "
            f"({report.reproducers_executed} reproducers executed, 0 broke the diff)"
        )
        if report.rejected_non_failing:
            lines.append(
                _paint(
                    f"          {report.rejected_non_failing} non-failing / invalid candidate(s) silently discarded.",
                    _ANSI.DIM,
                    color=color,
                )
            )
    else:
        for bug in report.proven_bugs:
            badge = _paint("[PROVEN BUG]", _ANSI.RED, _ANSI.BOLD, color=color)
            if bug.passed_on_base is True:
                baseline_tag = _paint(f"REGRESSION (passed on {report.base_ref})", _ANSI.RED, color=color)
            elif bug.passed_on_base is False:
                baseline_tag = _paint(f"PRE-EXISTING (also failed on {report.base_ref})", _ANSI.YELLOW, color=color)
            else:
                baseline_tag = _paint("NEW CODE / BASELINE N/A", _ANSI.CYAN, color=color)

            lines.append(f"  {badge} {bug.id}: {_paint(bug.title, _ANSI.BOLD, color=color)}")
            lines.append(
                _paint(
                    f"  Role: {bug.role} | Status: {baseline_tag} | Files: {', '.join(bug.target_files) or 'n/a'}",
                    _ANSI.DIM,
                    color=color,
                )
            )
            lines.append("")
            lines.append(_paint("  Failure Traceback:", _ANSI.BOLD, color=color))
            for tb_line in bug.failure_output.splitlines():
                lines.append(f"    {tb_line}")
            lines.append("")
            lines.append(_paint("  Copy-Paste Reproducer:", _ANSI.BOLD, color=color))
            lines.append("  ```python")
            for code_line in bug.reproducer_code.splitlines():
                lines.append(f"  {code_line}")
            lines.append("  ```")
            lines.append("")

        if report.suggested_patch:
            p_badge = (
                _paint("[VERIFIED PATCH]", _ANSI.GREEN, _ANSI.BOLD, color=color)
                if report.patch_verified
                else _paint("[SUGGESTED PATCH]", _ANSI.YELLOW, _ANSI.BOLD, color=color)
            )
            lines.append(f"  {p_badge} All proven reproducers pass with this patch:")
            lines.append("  ```diff")
            for p_line in report.suggested_patch.splitlines():
                lines.append(f"  {p_line}")
            lines.append("  ```")
            lines.append("")

    lines.extend(
        [
            _paint(rule, _ANSI.DIM, color=color),
            (
                f"  Stats: adversaries={report.adversaries_run} | "
                f"executed={report.reproducers_executed} | "
                f"discarded={report.rejected_non_failing} | "
                f"proven={len(report.proven_bugs)} | "
                f"calls={report.llm_calls} | "
                f"tokens={report.total_tokens} | "
                f"wall={report.wall_s:.2f}s"
            ),
            _paint(bar, _ANSI.CYAN, _ANSI.BOLD, color=color),
        ]
    )
    return "\n".join(lines)

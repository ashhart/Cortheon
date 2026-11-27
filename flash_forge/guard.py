"""Zero-False-Positive Polyglot Proof-Carrying Diff Breaker (`cortheon guard`).

Core thesis: NEVER emit LLM prose code-review slop. An adversarial finding is
ONLY reported if it is backed by a mechanically executed failing reproducer
script (Python, Node.js/JS/TS, Go, or POSIX Shell) run inside an isolated
ephemeral git worktree + Cortheon's OS sandbox (``macos-seatbelt`` /
``linux-bwrap`` / POSIX ``setrlimit`` — blocking network & outside-worktree writes).

Workflow:
  1. Extract unified diff (staged + unstaged + untracked vs ``base_ref``, or
     caller-supplied ``diff_text``). Empty diff short-circuits with 0 LLM calls.
  2. Auto-detect (or honor explicit) polyglot ``runner`` (``python``, ``node``,
     ``go``, or ``shell``) and materialize an isolated ephemeral git worktree
     with the patched state (never mutates the user's repository; always
     cleaned up in ``finally``).
  3. Fan out ``num_adversaries`` parallel clean-context adversarial bug-hunters
     with distinct attack lenses (boundary/edge, state/concurrency,
     regression/contract) that output ONLY executable reproducer scripts.
  4. Mechanical Zero-FP Proof Gate inside Cortheon's OS sandbox:
     - Discard prose-only / invalid-syntax outputs.
     - Discard scripts that exit 0 (no bug proven).
     - Discard scripts that crash solely due to reproducer-side import/syntax
       or missing-command errors.
     - Optionally run surviving reproducers against a clean ``base_ref``
       ephemeral worktree to classify regressions (``passed_on_base=True``),
       pre-existing bugs (``passed_on_base=False``), and new-code bugs
       (``passed_on_base=None``).
     - Deduplicate identical failure traces.
  5. Optional ``auto_fix=True``: race ``B=2`` parallel surgical patch
     generators in ephemeral test worktrees, verify ALL proven reproducers
     (plus optional ``test_cmd``) exit 0, and attach a verified unified patch.
  6. Export GitHub Code Scanning SARIF 2.1.0 via :func:`format_guard_sarif` and
     install/uninstall non-destructive git hooks via :func:`install_git_hook` /
     :func:`uninstall_git_hook`.
"""

from __future__ import annotations

import ast
import asyncio
import difflib
import hashlib
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .client import CallSpec, CompletionResult, TensorFoldClient
from .sandbox import (
    _SEATBELT_BIN,
    _build_macos_seatbelt_profile,
    _make_rlimit_preexec,
    ExecResult,
    check_syntax,
    extract_python_code,
    run_python,
    sandbox_backend,
)

_GUARD_VERSION = "2.3.0"
_SARIF_SCHEMA_URI = "https://json.schemastore.org/sarif-2.1.0.json"
_GIT_HOOK_MARKER = "# CORTHEON-GUARD-HOOK"
_META_PREFIX = "__CORTHEON_GUARD_META__:"
_REPRO_FILENAME = "<reproducer>"
_INVALID_REPRO_EXCS = frozenset(
    {"ImportError", "ModuleNotFoundError", "SyntaxError", "IndentationError", "TabError"}
)
_BASELINE_MISSING_SYMBOL_EXCS = frozenset(
    {"ImportError", "ModuleNotFoundError", "AttributeError"}
)
_INVALID_NODE_REPRO_EXCS = frozenset(
    {"MODULE_NOT_FOUND", "ERR_MODULE_NOT_FOUND", "SyntaxError"}
)
_INVALID_SHELL_REPRO_EXCS = frozenset({"SyntaxError", "CommandNotFound"})
_INVALID_GO_REPRO_EXCS = frozenset({"GoBuildError"})

SUPPORTED_RUNNERS: tuple[str, ...] = ("auto", "python", "node", "go", "shell")
_RUNNER_ALIASES: dict[str, str] = {
    "auto": "auto",
    "python": "python",
    "py": "python",
    "node": "node",
    "js": "node",
    "javascript": "node",
    "ts": "node",
    "typescript": "node",
    "go": "go",
    "golang": "go",
    "shell": "shell",
    "sh": "shell",
    "bash": "shell",
}
_EXT_TO_RUNNER: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "node",
    ".mjs": "node",
    ".cjs": "node",
    ".jsx": "node",
    ".ts": "node",
    ".mts": "node",
    ".cts": "node",
    ".tsx": "node",
    ".go": "go",
    ".sh": "shell",
    ".bash": "shell",
    ".rs": "python",
    ".zig": "python",
}
_EXT_TO_MARKDOWN_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".zig": "zig",
    ".sh": "sh",
    ".bash": "bash",
}
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

_TITLE_COMMENT_RE = re.compile(
    r"^\s*(?:#|//)\s*(?:TITLE|BUG)\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_FILE_FENCE_RE = re.compile(
    r"```(?:file|python|py|js|javascript|ts|typescript|go|sh|bash|shell):([^\n`]+)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_DIFF_FENCE_RE = re.compile(r"```(?:diff|patch)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE_RE = re.compile(r"```([A-Za-z0-9_+.-]*)[ \t]*\n(.*?)```", re.DOTALL)
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
    language: str = "python"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "title": self.title,
            "reproducer_code": self.reproducer_code,
            "failure_output": self.failure_output,
            "passed_on_base": self.passed_on_base,
            "target_files": list(self.target_files),
            "language": self.language,
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
    runner: str = "python"

    @property
    def clean(self) -> bool:
        return len(self.proven_bugs) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "repo_path": self.repo_path,
            "base_ref": self.base_ref,
            "runner": self.runner,
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
    missing_symbol: bool = False


def _emit_event(
    on_event: Optional[Callable[[dict[str, Any]], None]],
    payload: dict[str, Any],
) -> None:
    """Safely invoke an optional observability callback without affecting execution."""
    if on_event is None:
        return
    try:
        on_event(payload)
    except Exception:
        pass


def _normalize_runner(runner: str) -> str:
    key = (runner or "auto").strip().lower()
    if key not in _RUNNER_ALIASES:
        raise ValueError(
            f"Unsupported guard runner {runner!r}. Expected one of: {', '.join(SUPPORTED_RUNNERS)}."
        )
    return _RUNNER_ALIASES[key]


def detect_runner(diff_files: list[str], runner: str = "auto") -> str:
    """Resolve the active polyglot runner from explicit ``runner`` or ``diff_files`` extensions.

    Auto-detection rules when ``runner == "auto"``:
      - Pure Python (``.py``/``.pyi``) -> ``"python"``
      - Pure JS/TS (``.js``, ``.mjs``, ``.cjs``, ``.jsx``, ``.ts``, ``.mts``, ``.cts``, ``.tsx``) -> ``"node"``
      - Pure Go (``.go``) -> ``"go"``
      - Pure Shell (``.sh``, ``.bash``) -> ``"shell"``
      - Rust (``.rs``), Zig (``.zig``), mixed code languages, or unrecognized -> ``"python"`` driver
    """
    norm = _normalize_runner(runner)
    if norm != "auto":
        return norm

    detected: set[str] = set()
    for rel in diff_files:
        ext = Path(rel).suffix.lower()
        mapped = _EXT_TO_RUNNER.get(ext)
        if mapped:
            detected.add(mapped)

    if len(detected) == 1:
        return next(iter(detected))
    return "python"


def _language_fence_tag(language: str) -> str:
    key = (language or "python").strip().lower()
    if key == "node":
        return "javascript"
    if key == "shell":
        return "sh"
    if key == "go":
        return "go"
    return "python"


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


def _build_file_snippets(wt_dir: Path, diff_files: list[str], *, runner: str = "python") -> str:
    """Gather bounded contents of changed files from the ephemeral patched worktree."""
    def _priority(rel: str) -> tuple[int, str]:
        ext = Path(rel).suffix.lower()
        mapped = _EXT_TO_RUNNER.get(ext)
        if mapped == runner:
            return (0, rel)
        if ext == ".py":
            return (1, rel)
        return (2, rel)

    ordered = sorted(diff_files, key=_priority)
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
        lang = _EXT_TO_MARKDOWN_LANG.get(Path(rel).suffix.lower(), "")
        chunks.append(f"### FILE: {rel}\n```{lang}\n{text.rstrip()}\n```")
        used += len(text)
    return "\n\n".join(chunks)


_JS_ESM_RE = re.compile(r"^\s*(?:import\s+[\s\S]+?\s+from\s+|import\s+['\"]|export\s+)", re.MULTILINE)
_JS_CJS_RE = re.compile(r"\b(?:require\s*\(|module\.exports\b|exports\.)")
_TS_SYNTAX_RE = re.compile(
    r"(?:from\s+['\"][^'\"]+\.[mc]?ts['\"]|^\s*(?:interface|type)\s+\w+|:\s*(?:number|string|boolean|any|unknown|void|never)\b)",
    re.MULTILINE,
)
_JS_INTENT_RE = re.compile(
    r"(?:\b(?:import|require|assert|throw|const|let|var|function|async|await|process)\b|=>|console\.\w+\s*\()"
)
_NODE_TAGS = frozenset({"javascript", "js", "mjs", "cjs", "typescript", "ts", "mts", "cts", "node"})
_TS_TAGS = frozenset({"typescript", "ts", "mts", "cts"})
_SHELL_TAGS = frozenset({"sh", "bash", "shell", "zsh"})
_GO_TAGS = frozenset({"go", "golang"})


def _extract_fenced_blocks(raw_output: str, preferred_tags: frozenset[str]) -> list[tuple[str, str]]:
    """Return ``[(tag_lower, code)]`` prioritizing blocks matching ``preferred_tags`` then untagged blocks."""
    matched: list[tuple[str, str]] = []
    untagged: list[tuple[str, str]] = []
    for raw_tag, body in _ANY_FENCE_RE.findall(raw_output):
        tag = raw_tag.strip().lower()
        cleaned = body.strip()
        if not cleaned:
            continue
        if tag in preferred_tags:
            matched.append((tag, cleaned))
        elif tag == "":
            untagged.append((tag, cleaned))
    return matched if matched else untagged


def _pick_node_ext(code: str, *, fence_tag: str = "", worktree_dir: Optional[Path] = None) -> str:
    is_esm = bool(_JS_ESM_RE.search(code)) and not bool(_JS_CJS_RE.search(code))
    is_ts = fence_tag in _TS_TAGS or bool(_TS_SYNTAX_RE.search(code))
    if is_ts:
        return ".mts" if is_esm else ".ts"
    if is_esm:
        return ".mjs"
    if _JS_CJS_RE.search(code):
        return ".cjs"
    if worktree_dir is not None:
        pkg_json = worktree_dir / "package.json"
        if pkg_json.is_file():
            try:
                raw_pkg = pkg_json.read_text(encoding="utf-8", errors="replace")
                if re.search(r'"type"\s*:\s*"module"', raw_pkg):
                    return ".mjs"
            except OSError:
                pass
    return ".cjs"


def _check_node_syntax(code: str, *, fence_tag: str = "") -> Optional[str]:
    """Run ``node --check`` on JS code (skipped for TypeScript annotations since ``--check`` does not strip types)."""
    ext = _pick_node_ext(code, fence_tag=fence_tag)
    if ext in (".ts", ".mts"):
        return None
    node_bin = shutil.which("node")
    if not node_bin:
        return None
    with tempfile.TemporaryDirectory(prefix="cortheon_guard_jschk_") as tmp:
        p = Path(tmp) / f"check{ext}"
        p.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [node_bin, "--check", str(p)],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if proc.returncode != 0:
                return (proc.stderr or proc.stdout or "Node syntax error").strip()[:300]
            return None
        except (OSError, subprocess.SubprocessError):
            return None


def _check_shell_syntax(code: str) -> Optional[str]:
    sh_bin = shutil.which("sh") or "/bin/sh"
    if not os.path.isfile(sh_bin):
        return None
    with tempfile.TemporaryDirectory(prefix="cortheon_guard_shchk_") as tmp:
        p = Path(tmp) / "check.sh"
        p.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sh_bin, "-n", str(p)],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if proc.returncode != 0:
                return (proc.stderr or proc.stdout or "Shell syntax error").strip()[:300]
            return None
        except (OSError, subprocess.SubprocessError):
            return None


def _check_go_syntax(code: str) -> Optional[str]:
    gofmt = shutil.which("gofmt")
    if not gofmt:
        return None
    try:
        proc = subprocess.run(
            [gofmt],
            input=code,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if proc.returncode != 0:
            return (proc.stderr or "Go syntax error").strip()[:300]
        return None
    except (OSError, subprocess.SubprocessError):
        return None


def _extract_valid_reproducer(raw_output: str, *, runner: str = "python") -> Optional[str]:
    """Extract an executable reproducer for ``runner``; reject prose slop and syntax errors."""
    if not raw_output or not raw_output.strip():
        return None

    has_fence = bool(_CODE_FENCE_PRESENT_RE.search(raw_output))

    if runner == "python":
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

    if runner == "node":
        blocks = _extract_fenced_blocks(raw_output, _NODE_TAGS)
        candidates: list[tuple[str, str]] = list(blocks)
        if not candidates and not has_fence:
            stripped = raw_output.strip()
            if _JS_ESM_RE.search(stripped) or _JS_CJS_RE.search(stripped) or "assert" in stripped or "process.exit" in stripped:
                candidates.append(("", stripped))
        # Prefer longest syntactically valid JS/TS block with executable intent
        valid_node: list[tuple[int, str]] = []
        for tag, cand in candidates:
            if not _JS_INTENT_RE.search(cand):
                continue
            if _check_node_syntax(cand, fence_tag=tag) is not None:
                continue
            valid_node.append((len(cand), cand))
        if not valid_node:
            return None
        valid_node.sort(key=lambda x: x[0])
        return valid_node[-1][1].strip()

    if runner == "shell":
        blocks = _extract_fenced_blocks(raw_output, _SHELL_TAGS)
        candidates_sh: list[str] = [b for _, b in blocks]
        if not candidates_sh and not has_fence:
            stripped = raw_output.strip()
            if stripped.startswith("#!/") or re.search(r"^\s*set\s+-[a-zA-Z]*e", stripped, re.MULTILINE):
                candidates_sh.append(stripped)
        valid_sh: list[tuple[int, str]] = []
        for cand in candidates_sh:
            non_comment = [
                ln.strip()
                for ln in cand.splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
            if not non_comment:
                continue
            if _check_shell_syntax(cand) is not None:
                continue
            valid_sh.append((len(cand), cand))
        if not valid_sh:
            return None
        valid_sh.sort(key=lambda x: x[0])
        return valid_sh[-1][1].strip()

    if runner == "go":
        blocks = _extract_fenced_blocks(raw_output, _GO_TAGS)
        candidates_go: list[str] = [b for _, b in blocks]
        if not candidates_go and not has_fence:
            stripped = raw_output.strip()
            if "package " in stripped and "func Test" in stripped:
                candidates_go.append(stripped)
        valid_go: list[tuple[int, str]] = []
        for cand in candidates_go:
            if not re.search(r"^\s*package\s+\w+", cand, re.MULTILINE):
                continue
            if not re.search(r"\bfunc\s+Test\w*\s*\(", cand):
                continue
            if _check_go_syntax(cand) is not None:
                continue
            valid_go.append((len(cand), cand))
        if not valid_go:
            return None
        valid_go.sort(key=lambda x: x[0])
        return valid_go[-1][1].strip()

    return None


def _run_command_in_os_sandbox(
    argv: list[str],
    *,
    cwd: Path,
    timeout_s: float,
    extra_env: Optional[dict[str, str]] = None,
    tmp_files: Optional[dict[str, str]] = None,
    unsafe_no_sandbox: bool = False,
) -> ExecResult:
    """Execute ``argv`` inside Cortheon's OS sandbox (``macos-seatbelt`` / ``linux-bwrap`` / POSIX rlimits).

    Writes are strictly confined to an ephemeral sandbox ``tmp_path`` (never ``cwd``
    or the user's repository), network I/O is blocked, sensitive credential directories
    are unreadable, environment variables are scrubbed, and runaway child processes
    are terminated via process-group ``SIGKILL`` on timeout.
    """
    backend = sandbox_backend(unsafe_no_sandbox=unsafe_no_sandbox)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="cortheon_guard_sb_") as tmp:
        tmp_path = Path(tmp).resolve()
        if tmp_files:
            for name, content in tmp_files.items():
                p = (tmp_path / name).resolve()
                if not p.is_relative_to(tmp_path):
                    raise ValueError(f"Refusing tmp_files path outside sandbox temp dir: {name}")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")

        _SAFE_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "USER")
        env = {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}
        env["TMPDIR"] = str(tmp_path)
        env["TEMP"] = str(tmp_path)
        env["TMP"] = str(tmp_path)
        if extra_env:
            for k, v in extra_env.items():
                env[k] = v.replace("{TMPDIR}", str(tmp_path))

        resolved_argv = [arg.replace("{TMPDIR}", str(tmp_path)) for arg in argv]

        cmd: list[str]
        if backend == "macos-seatbelt":
            profile_path = tmp_path / ".sb_profile"
            profile_path.write_text(_build_macos_seatbelt_profile(tmp_path), encoding="utf-8")
            cmd = [_SEATBELT_BIN, "-f", str(profile_path), *resolved_argv]
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
                *resolved_argv,
            ]
        else:
            cmd = resolved_argv

        preexec_fn = _make_rlimit_preexec(timeout_s) if os.name == "posix" else None
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
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


def _clip_failure_output(text: str, max_chars: int = 2400) -> str:
    if len(text) <= max_chars:
        return text
    return text[:1500] + "\n...(truncated)...\n" + text[-(max_chars - 1520) :]


def _run_python_reproducer_in_sandbox(
    reproducer_code: str,
    worktree_dir: Path,
    user_repo_dir: Path,
    timeout_s: float,
) -> _SandboxRunInfo:
    """Execute Python ``reproducer_code`` against ``worktree_dir`` inside Cortheon's OS sandbox."""
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
        missing_symbol=bool(last_in_repro and exc_name in _BASELINE_MISSING_SYMBOL_EXCS),
    )


def _run_node_reproducer_in_sandbox(
    reproducer_code: str,
    worktree_dir: Path,
    timeout_s: float,
) -> _SandboxRunInfo:
    """Execute JS/TS ``reproducer_code`` via ``node`` inside Cortheon's OS sandbox."""
    node_bin = shutil.which("node")
    if not node_bin:
        raise RuntimeError("Node.js runtime (`node`) is required for runner='node' but was not found on PATH.")

    wt_resolved = worktree_dir.resolve()
    wt_str = str(wt_resolved)
    ext = _pick_node_ext(reproducer_code, worktree_dir=wt_resolved)
    repro_name = f".cortheon_guard_repro_{os.getpid()}_{uuid.uuid4().hex[:8]}{ext}"
    repro_path = wt_resolved / repro_name

    repro_path.write_text(reproducer_code, encoding="utf-8")
    try:
        res = _run_command_in_os_sandbox(
            [node_bin, str(repro_path)],
            cwd=wt_resolved,
            timeout_s=timeout_s,
            extra_env={"NODE_NO_WARNINGS": "1"},
        )
    finally:
        repro_path.unlink(missing_ok=True)

    if res.timed_out:
        return _SandboxRunInfo(
            exec_result=res,
            clean_output=f"TIMEOUT after {res.elapsed_s:.2f}s",
            exc_name="TimeoutError",
            last_in_repro=False,
            last_file=None,
        )

    raw_err = (res.stderr or "").strip() or (res.stdout or "").strip()
    exc_name: Optional[str] = None
    last_in_repro = False
    last_file: Optional[str] = None
    missing_symbol = False

    if not res.ok:
        if "ERR_MODULE_NOT_FOUND" in raw_err:
            exc_name = "ERR_MODULE_NOT_FOUND"
        elif "MODULE_NOT_FOUND" in raw_err or "Cannot find module " in raw_err:
            exc_name = "MODULE_NOT_FOUND"
        else:
            m_exc = re.search(
                r"^(AssertionError|SyntaxError|TypeError|ReferenceError|RangeError|URIError|EvalError|Error)\b",
                raw_err,
                re.MULTILINE,
            )
            if m_exc:
                exc_name = m_exc.group(1)

        # Determine whether the failure originated in the reproducer itself vs target code
        if exc_name in ("MODULE_NOT_FOUND", "ERR_MODULE_NOT_FOUND"):
            req_match = re.search(r"Require stack:\s*\n-\s*(.+)", raw_err)
            imp_match = re.search(r"imported from\s+(\S+)", raw_err)
            if req_match:
                origin = req_match.group(1).strip()
                last_file = origin
                last_in_repro = repro_name in origin
            elif imp_match:
                origin = imp_match.group(1).strip()
                last_file = origin
                last_in_repro = repro_name in origin
            else:
                last_in_repro = repro_name in raw_err
            missing_symbol = last_in_repro
        elif exc_name == "SyntaxError":
            first_nonempty = next((ln.strip() for ln in raw_err.splitlines() if ln.strip()), "")
            if "does not provide an export named" in raw_err:
                last_in_repro = True
                missing_symbol = True
            else:
                last_in_repro = repro_name in first_nonempty
                if not last_in_repro and first_nonempty.startswith(("file://", "/", wt_str)):
                    last_file = first_nonempty.split(":", 1)[0]
        else:
            for ln in raw_err.splitlines():
                stripped = ln.strip()
                if not stripped.startswith("at ") or "node:internal" in stripped:
                    continue
                last_file = stripped
                last_in_repro = repro_name in stripped
                break
            if exc_name == "TypeError" and last_in_repro and "is not a function" in raw_err:
                missing_symbol = True

    clean_out = (
        raw_err.replace(f"file://{repro_path}", _REPRO_FILENAME)
        .replace(str(repro_path), _REPRO_FILENAME)
        .replace(repro_name, _REPRO_FILENAME)
        .replace(f"file://{wt_str}{os.sep}", "")
        .replace(wt_str + os.sep, "")
        .replace(wt_str, ".")
    )
    clean_out = _clip_failure_output(clean_out)

    return _SandboxRunInfo(
        exec_result=res,
        clean_output=clean_out,
        exc_name=exc_name,
        last_in_repro=last_in_repro,
        last_file=last_file,
        missing_symbol=missing_symbol,
    )


def _run_shell_reproducer_in_sandbox(
    reproducer_code: str,
    worktree_dir: Path,
    timeout_s: float,
) -> _SandboxRunInfo:
    """Execute POSIX shell ``reproducer_code`` via ``sh -e`` inside Cortheon's OS sandbox."""
    sh_bin = shutil.which("sh") or "/bin/sh"
    if not os.path.isfile(sh_bin):
        raise RuntimeError("POSIX shell (`sh`) was not found on PATH.")

    wt_resolved = worktree_dir.resolve()
    wt_str = str(wt_resolved)
    repro_name = "_cortheon_guard_repro.sh"

    res = _run_command_in_os_sandbox(
        [sh_bin, "-e", f"{{TMPDIR}}/{repro_name}"],
        cwd=wt_resolved,
        timeout_s=timeout_s,
        tmp_files={repro_name: reproducer_code if reproducer_code.endswith("\n") else reproducer_code + "\n"},
    )

    if res.timed_out:
        return _SandboxRunInfo(
            exec_result=res,
            clean_output=f"TIMEOUT after {res.elapsed_s:.2f}s",
            exc_name="TimeoutError",
            last_in_repro=False,
            last_file=None,
        )

    raw_err = (res.stderr or "").strip()
    raw_out = (res.stdout or "").strip()
    combined = raw_err if raw_err else raw_out
    if raw_err and raw_out and raw_out not in raw_err:
        combined = f"{raw_err}\n{raw_out}"

    exc_name: Optional[str] = None
    last_in_repro = False
    missing_symbol = False

    if not res.ok:
        lower_err = raw_err.lower()
        if "syntax error" in lower_err and repro_name in raw_err:
            exc_name = "SyntaxError"
            last_in_repro = True
        elif res.returncode == 127 or "command not found" in lower_err or ": not found" in lower_err:
            # Relative ./script missing on base -> new script; nonexistent external binary -> invalid reproducer
            if "./" in raw_err or "no such file or directory" in lower_err:
                exc_name = "FileNotFound"
                missing_symbol = True
                last_in_repro = False
            else:
                exc_name = "CommandNotFound"
                last_in_repro = True
        elif "no such file or directory" in lower_err and "./" in raw_err:
            exc_name = "FileNotFound"
            missing_symbol = True
            last_in_repro = False
        else:
            exc_name = f"ExitCode{res.returncode}"

    clean_out = re.sub(r"\S*" + re.escape(repro_name), _REPRO_FILENAME, combined)
    clean_out = clean_out.replace(wt_str + os.sep, "").replace(wt_str, ".")
    clean_out = _clip_failure_output(clean_out)

    return _SandboxRunInfo(
        exec_result=res,
        clean_output=clean_out,
        exc_name=exc_name,
        last_in_repro=last_in_repro,
        last_file=None,
        missing_symbol=missing_symbol,
    )


def _find_go_package_dir(worktree_dir: Path, reproducer_code: str, diff_files: list[str]) -> Path:
    m_pkg = re.search(r"^\s*package\s+(\w+)", reproducer_code, re.MULTILINE)
    repro_pkg = m_pkg.group(1) if m_pkg else ""
    base_pkg = repro_pkg[:-5] if repro_pkg.endswith("_test") else repro_pkg

    go_files = [f for f in diff_files if f.endswith(".go")]
    if base_pkg:
        for rel in go_files:
            full = (worktree_dir / rel).resolve()
            if full.is_file() and full.is_relative_to(worktree_dir):
                try:
                    head = full.read_text(encoding="utf-8", errors="replace")[:512]
                    m = re.search(r"^\s*package\s+(\w+)", head, re.MULTILINE)
                    if m and m.group(1) == base_pkg:
                        return full.parent
                except OSError:
                    pass
    if go_files:
        cand = (worktree_dir / go_files[0]).resolve().parent
        if cand.is_dir() and cand.is_relative_to(worktree_dir):
            return cand
    return worktree_dir


def _run_go_reproducer_in_sandbox(
    reproducer_code: str,
    worktree_dir: Path,
    timeout_s: float,
    diff_files: list[str],
) -> _SandboxRunInfo:
    """Execute Go ``*_test.go`` reproducer via ``go test`` inside Cortheon's OS sandbox."""
    go_bin = shutil.which("go")
    if not go_bin:
        raise RuntimeError("Go toolchain (`go`) is required for runner='go' but was not found on PATH.")

    wt_resolved = worktree_dir.resolve()
    wt_str = str(wt_resolved)
    pkg_dir = _find_go_package_dir(wt_resolved, reproducer_code, diff_files)
    repro_name = f"cortheon_guard_repro_{os.getpid()}_{uuid.uuid4().hex[:8]}_test.go"
    repro_path = pkg_dir / repro_name

    extra_env = {
        "GOCACHE": "{TMPDIR}/gocache",
        "GOTMPDIR": "{TMPDIR}",
        "GOPROXY": "off",
    }
    if not (wt_resolved / "go.sum").is_file():
        extra_env["GOPATH"] = "{TMPDIR}/gopath"
    if not (wt_resolved / "go.mod").is_file():
        extra_env["GO111MODULE"] = "off"

    repro_path.write_text(reproducer_code, encoding="utf-8")
    try:
        res = _run_command_in_os_sandbox(
            [go_bin, "test", "-v", "-count=1", "."],
            cwd=pkg_dir,
            timeout_s=timeout_s,
            extra_env=extra_env,
        )
    finally:
        repro_path.unlink(missing_ok=True)

    if res.timed_out:
        return _SandboxRunInfo(
            exec_result=res,
            clean_output=f"TIMEOUT after {res.elapsed_s:.2f}s",
            exc_name="TimeoutError",
            last_in_repro=False,
            last_file=None,
        )

    combined = "\n".join(p for p in ((res.stdout or "").strip(), (res.stderr or "").strip()) if p).strip()
    exc_name: Optional[str] = None
    last_in_repro = False
    missing_symbol = False

    if not res.ok:
        if "[build failed]" in combined or "[setup failed]" in combined:
            exc_name = "GoBuildError"
            # Check if every compiler error line references only our reproducer file
            err_file_lines = re.findall(r"^(?:\./)?([^\s:]+\.go):\d+:", combined, re.MULTILINE)
            if err_file_lines and all(repro_name in f for f in err_file_lines):
                last_in_repro = True
                if "undefined:" in combined:
                    missing_symbol = True
        elif "panic:" in combined:
            exc_name = "GoPanic"
        elif "--- FAIL:" in combined or "FAIL" in combined:
            exc_name = "GoTestFail"
        else:
            exc_name = f"GoExit{res.returncode}"

    clean_out = (
        combined.replace(str(repro_path), "cortheon_guard_repro_test.go")
        .replace(repro_name, "cortheon_guard_repro_test.go")
        .replace(wt_str + os.sep, "")
        .replace(wt_str, ".")
    )
    clean_out = _clip_failure_output(clean_out)

    return _SandboxRunInfo(
        exec_result=res,
        clean_output=clean_out,
        exc_name=exc_name,
        last_in_repro=last_in_repro,
        last_file=None,
        missing_symbol=missing_symbol,
    )


def _run_reproducer_in_sandbox(
    reproducer_code: str,
    worktree_dir: Path,
    user_repo_dir: Path,
    timeout_s: float,
    *,
    runner: str = "python",
    diff_files: Optional[list[str]] = None,
) -> _SandboxRunInfo:
    """Execute ``reproducer_code`` against ``worktree_dir`` inside Cortheon's OS sandbox."""
    if runner == "node":
        return _run_node_reproducer_in_sandbox(reproducer_code, worktree_dir, timeout_s)
    if runner == "shell":
        return _run_shell_reproducer_in_sandbox(reproducer_code, worktree_dir, timeout_s)
    if runner == "go":
        return _run_go_reproducer_in_sandbox(reproducer_code, worktree_dir, timeout_s, diff_files or [])
    return _run_python_reproducer_in_sandbox(reproducer_code, worktree_dir, user_repo_dir, timeout_s)


def _is_invalid_reproducer_failure(info: _SandboxRunInfo, runner: str) -> bool:
    if info.exec_result.timed_out:
        return False
    if runner == "python":
        return bool(info.last_in_repro and info.exc_name in _INVALID_REPRO_EXCS)
    if runner == "node":
        return bool(info.last_in_repro and info.exc_name in _INVALID_NODE_REPRO_EXCS)
    if runner == "shell":
        return bool(info.last_in_repro and info.exc_name in _INVALID_SHELL_REPRO_EXCS)
    if runner == "go":
        return bool(info.last_in_repro and info.exc_name in _INVALID_GO_REPRO_EXCS)
    return False


def _is_baseline_missing_symbol(info: _SandboxRunInfo, runner: str) -> bool:
    if info.exec_result.timed_out:
        return False
    if info.missing_symbol:
        return True
    if runner == "python":
        return bool(info.last_in_repro and info.exc_name in _BASELINE_MISSING_SYMBOL_EXCS)
    return False


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
    *,
    runner: str = "python",
) -> list[str]:
    matched: list[str] = []
    for rel in diff_files:
        norm = rel.replace("\\", "/")
        name = Path(norm).name
        stem = Path(norm).stem
        ext = Path(norm).suffix.lower()
        mod_path = norm[:-3].replace("/", ".") if ext == ".py" else ""
        if (
            norm in failure_output
            or norm in reproducer_code
            or name in failure_output
            or name in reproducer_code
            or (mod_path and re.search(rf"\b{re.escape(mod_path)}\b", reproducer_code))
            or (stem and ext in _EXT_TO_RUNNER and re.search(rf"\b{re.escape(stem)}\b", reproducer_code))
        ):
            matched.append(rel)
    if matched:
        return matched
    runner_diff = [f for f in diff_files if _EXT_TO_RUNNER.get(Path(f).suffix.lower()) == runner]
    if runner_diff:
        return runner_diff
    py_diff = [f for f in diff_files if f.endswith(".py")]
    return py_diff if py_diff else list(diff_files)


def _failure_dedup_key(info: _SandboxRunInfo, target_files: Optional[list[str]] = None) -> str:
    """Compute a normalized deduplication fingerprint for a failing execution."""
    text = info.clean_output.strip()
    # Normalize reproducer line/column numbers so identical assertions at different lines deduplicate
    normalized = re.sub(r'File "<reproducer>", line \d+', 'File "<reproducer>", line _', text)
    normalized = re.sub(r"<reproducer>:\d+(?::\d+)?", "<reproducer>:_", normalized)
    normalized = re.sub(r"cortheon_guard_repro_test\.go:\d+", "cortheon_guard_repro_test.go:_", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    files_tag = ",".join(sorted(target_files)) if target_files else ""
    return f"{info.exc_name or 'ERR'}::{files_tag}::{normalized[-600:]}"


def _build_adversary_specs(
    diff_text: str,
    file_snippets: str,
    num_adversaries: int,
    *,
    runner: str = "python",
) -> list[tuple[str, CallSpec]]:
    if num_adversaries <= 0:
        return []

    clipped_diff = diff_text
    if len(clipped_diff) > _MAX_DIFF_CHARS:
        clipped_diff = clipped_diff[:_MAX_DIFF_CHARS] + "\n... (diff truncated)\n"

    if runner == "node":
        system_prompt = (
            "You are an adversarial bug-hunting verifier for Cortheon Guard.\n"
            "STRICT RULES:\n"
            "1. NEVER write prose code review, opinions, or summaries.\n"
            "2. Output ONLY ONE self-contained Node.js reproducer script inside a ```javascript ... ``` code block.\n"
            "3. Put `// TITLE: <concise bug title>` as the first line of the script.\n"
            "4. Import or `require()` the ACTUAL changed code using relative `./...` paths and use `node:assert/strict`.\n"
            "5. Assert the CORRECT expected behavior so the script throws AssertionError (or exits non-zero) "
            "ONLY when a genuine bug exists in the diff, and exits 0 cleanly if the code is correct.\n"
            "6. Do NOT invent non-existent functions/modules, do NOT use network I/O, and do NOT write files "
            "in the repository directory."
        )
        output_directive = "Output ONLY the Node.js reproducer script inside a single ```javascript ... ``` block."
    elif runner == "shell":
        system_prompt = (
            "You are an adversarial bug-hunting verifier for Cortheon Guard.\n"
            "STRICT RULES:\n"
            "1. NEVER write prose code review, opinions, or summaries.\n"
            "2. Output ONLY ONE self-contained POSIX shell reproducer script inside a ```sh ... ``` code block.\n"
            "3. Put `# TITLE: <concise bug title>` as the first line and start with `set -e`.\n"
            "4. Invoke the ACTUAL changed script/CLI in `.` and assert expected output/exit status.\n"
            "5. Exit non-zero ONLY when a genuine bug exists in the diff, and exit 0 cleanly if the code is correct.\n"
            "6. Do NOT use network I/O and do NOT modify files in the repository directory (use `$TMPDIR` if needed)."
        )
        output_directive = "Output ONLY the POSIX shell reproducer script inside a single ```sh ... ``` block."
    elif runner == "go":
        system_prompt = (
            "You are an adversarial bug-hunting verifier for Cortheon Guard.\n"
            "STRICT RULES:\n"
            "1. NEVER write prose code review, opinions, or summaries.\n"
            "2. Output ONLY ONE self-contained Go `*_test.go` reproducer inside a ```go ... ``` code block.\n"
            "3. Put `// TITLE: <concise bug title>` as the first line and declare the exact `package <name>` of the changed code.\n"
            "4. Write `func TestCortheonGuardRepro(t *testing.T)` calling the ACTUAL changed Go code.\n"
            "5. Call `t.Fatalf(...)` ONLY when a genuine bug exists in the diff, and pass cleanly if the code is correct.\n"
            "6. Do NOT invent non-existent symbols, do NOT use network I/O, and do NOT write files in the repository."
        )
        output_directive = "Output ONLY the Go test reproducer inside a single ```go ... ``` block."
    else:
        system_prompt = (
            "You are an adversarial bug-hunting verifier for Cortheon Guard.\n"
            "STRICT RULES:\n"
            "1. NEVER write prose code review, opinions, or summaries.\n"
            "2. Output ONLY ONE self-contained Python 3 reproducer script inside a ```python ... ``` code block.\n"
            "3. Put `# TITLE: <concise bug title>` as the first line of the script.\n"
            "4. Start with `import sys; sys.path.insert(0, '.')` and import/call the ACTUAL changed code "
            "(or invoke compiled binaries/CLI tools inside `.` via `subprocess`).\n"
            "5. Assert the CORRECT expected behavior so the script raises AssertionError (or exits non-zero) "
            "ONLY when a genuine bug exists in the diff, and exits 0 cleanly if the code is correct.\n"
            "6. Do NOT invent non-existent functions/modules, do NOT use network I/O, and do NOT write files "
            "in the repository directory (use `tempfile` if temporary storage is needed)."
        )
        output_directive = "Output ONLY the Python 3 reproducer script inside a single ```python ... ``` block."

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
                        f"{output_directive}"
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

    # 3. Single file full replacement (Python or polyglot target)
    py_targets = [f for f in candidate_files if f.endswith(".py")]
    target = py_targets[0] if len(py_targets) == 1 else (candidate_files[0] if len(candidate_files) == 1 else None)
    if target:
        if target.endswith(".py"):
            code = extract_python_code(raw_output)
            if code and code.strip():
                return {target: code if code.endswith("\n") else code + "\n"}, None
        else:
            target_runner = _EXT_TO_RUNNER.get(Path(target).suffix.lower(), "")
            pref_tags = (
                _NODE_TAGS if target_runner == "node"
                else (_SHELL_TAGS if target_runner == "shell" else (_GO_TAGS if target_runner == "go" else frozenset()))
            )
            blocks = _extract_fenced_blocks(raw_output, pref_tags)
            if blocks:
                body = blocks[-1][1]
                if body.strip():
                    return {target: body if body.endswith("\n") else body + "\n"}, None

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
    runner: str = "python",
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

    snippets = _build_file_snippets(patched_wt, target_files, runner=runner)
    bugs_block_parts: list[str] = []
    for bug in proven_bugs:
        fence_lang = _language_fence_tag(bug.language)
        bugs_block_parts.append(
            f"### {bug.id} ({bug.role}): {bug.title}\n"
            f"Reproducer:\n```{fence_lang}\n{bug.reproducer_code}\n```\n"
            f"Failure output:\n```\n{bug.failure_output}\n```"
        )
    bugs_block = "\n\n".join(bugs_block_parts)

    if len(target_files) == 1:
        target_fence = _EXT_TO_MARKDOWN_LANG.get(Path(target_files[0]).suffix.lower(), _language_fence_tag(runner))
        format_instr = (
            f"Output ONLY the complete fixed content of `{target_files[0]}` inside a single "
            f"```{target_fence} ... ``` block (or a unified ```diff ... ``` patch). Never leave placeholders."
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

        # Validate syntax for updated files before running sandbox checks
        syntax_bad = False
        for rel, content in file_updates.items():
            ext = Path(rel).suffix.lower()
            if ext == ".py" and check_syntax(content) is not None:
                syntax_bad = True
                break
            if ext in (".js", ".mjs", ".cjs") and _check_node_syntax(content) is not None:
                syntax_bad = True
                break
            if ext in (".sh", ".bash") and _check_shell_syntax(content) is not None:
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
                # Use git apply in non-repo directory
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
                    runner=bug.language,
                    diff_files=diff_files,
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
    runner: str = "auto",
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
) -> GuardReport:
    """Run Cortheon Guard polyglot proof-carrying adversarial diff verification.

    Guarantees:
    - Never mutates ``repo_path``.
    - Always cleans up all ephemeral git worktrees in ``finally``.
    - Never reports an LLM claim unless backed by an actually failing reproducer
      executed inside Cortheon's OS sandbox.
    """
    t0 = time.perf_counter()
    normalized_runner = _normalize_runner(runner)
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

    active_runner = detect_runner(diff_files, normalized_runner)

    # Empty diff short-circuit: 0 LLM calls, 0 worktrees
    if not effective_diff.strip() and not diff_files:
        wall_s = time.perf_counter() - t0
        _emit_event(
            on_event,
            {
                "engine": "guard",
                "event": "start",
                "mode": active_runner,
                "runner": active_runner,
                "streams": 0,
                "diff_files": [],
            },
        )
        _emit_event(
            on_event,
            {
                "engine": "guard",
                "event": "done",
                "elapsed_s": round(wall_s, 3),
                "verified": True,
                "clean": True,
                "proven_bugs": 0,
                "reproducers_executed": 0,
                "rejected_non_failing": 0,
                "runner": active_runner,
                "summary": "0 diff files (clean)",
            },
        )
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
            wall_s=wall_s,
            llm_calls=0,
            total_tokens=0,
            runner=active_runner,
        )

    # Verify required polyglot runtime before spending LLM calls
    if active_runner == "node" and not shutil.which("node"):
        raise RuntimeError(
            "Node.js runtime (`node`) is required for runner='node' but was not found on PATH."
        )
    if active_runner == "go" and not shutil.which("go"):
        if normalized_runner == "go":
            raise RuntimeError(
                "Go toolchain (`go`) is required for runner='go' but was not found on PATH."
            )
        active_runner = "python"

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
        file_snippets = await asyncio.to_thread(_build_file_snippets, patched_wt, diff_files, runner=active_runner)
        adv_specs_pairs = _build_adversary_specs(
            effective_diff,
            file_snippets,
            num_adversaries,
            runner=active_runner,
        )
        adversaries_run = len(adv_specs_pairs)

        _emit_event(
            on_event,
            {
                "engine": "guard",
                "event": "start",
                "mode": active_runner,
                "runner": active_runner,
                "streams": adversaries_run,
                "diff_files": list(diff_files),
            },
        )

        adv_results: list[CompletionResult] = []
        if adv_specs_pairs:
            adv_results = await client.batch([spec for _, spec in adv_specs_pairs])
            llm_calls += len(adv_results)
            total_tokens += sum(r.prompt_tokens + r.completion_tokens for r in adv_results)

        # Step 3: Mechanical Zero-FP Proof Gate inside Cortheon's OS sandbox
        seen_traces: set[str] = set()
        for slot_idx, ((role_name, _spec), res) in enumerate(zip(adv_specs_pairs, adv_results), start=1):
            slot_tokens = res.prompt_tokens + res.completion_tokens
            if not res.ok:
                rejected_non_failing += 1
                _emit_event(
                    on_event,
                    {
                        "engine": "guard",
                        "event": "slot_done",
                        "slot": slot_idx,
                        "role": role_name,
                        "ok": False,
                        "elapsed_s": round(res.latency_s, 3),
                        "tokens": slot_tokens,
                        "summary": "llm error",
                    },
                )
                continue

            repro_code = _extract_valid_reproducer(res.content, runner=active_runner)
            if repro_code is None:
                rejected_non_failing += 1
                _emit_event(
                    on_event,
                    {
                        "engine": "guard",
                        "event": "slot_done",
                        "slot": slot_idx,
                        "role": role_name,
                        "ok": False,
                        "elapsed_s": round(res.latency_s, 3),
                        "tokens": slot_tokens,
                        "summary": "discarded (no valid reproducer)",
                    },
                )
                continue

            reproducers_executed += 1
            patched_info = await asyncio.to_thread(
                _run_reproducer_in_sandbox,
                repro_code,
                patched_wt,
                repo,
                timeout_s,
                runner=active_runner,
                diff_files=diff_files,
            )
            slot_elapsed = round(res.latency_s + patched_info.exec_result.elapsed_s, 3)

            # Exit 0 -> code survived the attack -> discard
            if patched_info.exec_result.ok:
                rejected_non_failing += 1
                _emit_event(
                    on_event,
                    {
                        "engine": "guard",
                        "event": "slot_done",
                        "slot": slot_idx,
                        "role": role_name,
                        "ok": True,
                        "elapsed_s": slot_elapsed,
                        "tokens": slot_tokens,
                        "summary": "survived (exit 0)",
                    },
                )
                continue

            # Reproducer itself failed due to bad import/syntax/missing command (not target code) -> discard
            if _is_invalid_reproducer_failure(patched_info, active_runner):
                rejected_non_failing += 1
                _emit_event(
                    on_event,
                    {
                        "engine": "guard",
                        "event": "slot_done",
                        "slot": slot_idx,
                        "role": role_name,
                        "ok": False,
                        "elapsed_s": slot_elapsed,
                        "tokens": slot_tokens,
                        "summary": f"invalid reproducer ({patched_info.exc_name or 'error'})",
                    },
                )
                continue

            target_files = _infer_target_files(
                repro_code,
                patched_info.clean_output,
                diff_files,
                runner=active_runner,
            )

            # Deduplicate identical failure traces
            dedup_key = _failure_dedup_key(patched_info, target_files)
            if dedup_key in seen_traces:
                _emit_event(
                    on_event,
                    {
                        "engine": "guard",
                        "event": "slot_done",
                        "slot": slot_idx,
                        "role": role_name,
                        "ok": True,
                        "elapsed_s": slot_elapsed,
                        "tokens": slot_tokens,
                        "summary": "duplicate failure trace",
                    },
                )
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
                    runner=active_runner,
                    diff_files=diff_files,
                )
                if base_info.exec_result.ok:
                    passed_on_base = True
                elif _is_baseline_missing_symbol(base_info, active_runner):
                    # Newly added function/class/module/script in diff -> still a real bug in new code
                    passed_on_base = None
                else:
                    passed_on_base = False

            bug_id = f"BUG-{len(proven_bugs) + 1:02d}"
            title = _extract_bug_title(repro_code, patched_info.clean_output, role_name, patched_info.exc_name)

            bug = ProvenBug(
                id=bug_id,
                role=role_name,
                title=title,
                reproducer_code=repro_code,
                failure_output=patched_info.clean_output,
                passed_on_base=passed_on_base,
                target_files=target_files,
                language=active_runner,
            )
            proven_bugs.append(bug)

            _emit_event(
                on_event,
                {
                    "engine": "guard",
                    "event": "slot_done",
                    "slot": slot_idx,
                    "role": role_name,
                    "ok": True,
                    "elapsed_s": slot_elapsed,
                    "tokens": slot_tokens,
                    "summary": f"PROVEN {bug_id}",
                },
            )
            _emit_event(
                on_event,
                {
                    "engine": "guard",
                    "event": "bug_proven",
                    "id": bug_id,
                    "role": role_name,
                    "title": title,
                    "passed_on_base": passed_on_base,
                    "language": active_runner,
                    "target_files": list(target_files),
                },
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
                runner=active_runner,
            )
            llm_calls += fix_calls
            total_tokens += fix_tokens

    finally:
        await asyncio.to_thread(_remove_ephemeral_worktree, repo, base_wt, is_git_worktree=True)
        await asyncio.to_thread(_remove_ephemeral_worktree, repo, patched_wt, is_git_worktree=patched_is_wt)

    wall_s = time.perf_counter() - t0
    _emit_event(
        on_event,
        {
            "engine": "guard",
            "event": "done",
            "elapsed_s": round(wall_s, 3),
            "verified": len(proven_bugs) == 0,
            "clean": len(proven_bugs) == 0,
            "proven_bugs": len(proven_bugs),
            "reproducers_executed": reproducers_executed,
            "rejected_non_failing": rejected_non_failing,
            "runner": active_runner,
            "summary": f"{len(proven_bugs)} proven bug(s) ({reproducers_executed} executed)",
        },
    )

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
        wall_s=wall_s,
        llm_calls=llm_calls,
        total_tokens=total_tokens,
        runner=active_runner,
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
            f"  Repo: {report.repo_path} | Base: {report.base_ref} | Runner: {report.runner} | Sandbox: {backend}",
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
            lines.append(f"  ```{_language_fence_tag(bug.language)}")
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


# ---------------------------------------------------------------------------
# SARIF 2.1.0 Exporter (GitHub Code Scanning / PR Diff Annotations)
# ---------------------------------------------------------------------------


def _pascal_case(identifier: str) -> str:
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", identifier) if p]
    return "".join(p[:1].upper() + p[1:] for p in parts) or "CortheonGuardRule"


def _normalize_sarif_uri(path_str: str, repo_path: str) -> str:
    raw = (path_str or "").strip()
    if not raw:
        return "."
    norm = raw.replace("\\", "/")
    if repo_path:
        repo_norm = repo_path.replace("\\", "/").rstrip("/")
        if repo_norm and norm.startswith(repo_norm + "/"):
            norm = norm[len(repo_norm) + 1 :]
    while norm.startswith("./"):
        norm = norm[2:]
    norm = norm.lstrip("/")
    return norm or "."


def _extract_failure_line(failure_output: str, target_uri: str) -> int:
    if not failure_output or not target_uri or target_uri == ".":
        return 1
    candidates = [target_uri]
    basename = Path(target_uri).name
    if basename and basename not in candidates:
        candidates.append(basename)
    for cand in candidates:
        esc = re.escape(cand)
        # Python traceback: File "math_utils.py", line 6
        m_py = re.search(rf'File "[^"]*{esc}", line (\d+)', failure_output)
        if m_py:
            val = int(m_py.group(1))
            if val >= 1:
                return val
        # Node / Go / Shell style: math_utils.js:6:12 or math_utils.go:6:
        m_colon = re.search(rf"(?:^|[\s(/]){esc}:(\d+)(?::\d+)?\b", failure_output, re.MULTILINE)
        if m_colon:
            val = int(m_colon.group(1))
            if val >= 1:
                return val
    return 1


def format_guard_sarif(report: GuardReport) -> dict[str, Any]:
    """Export a :class:`GuardReport` as a strict SARIF 2.1.0 dictionary.

    Compatible with GitHub Code Scanning (`github/codeql-action/upload-sarif`) and
    inline PR diff annotations. Clean zero-bug reports emit valid SARIF 2.1.0 with
    ``"results": []``.
    """
    role_descriptions: dict[str, str] = {role: desc for role, desc, _t, _s in ADVERSARY_ROLES}
    seen_rule_ids: set[str] = set()
    rules: list[dict[str, Any]] = []

    def _register_rule(role_name: str) -> str:
        rule_id = f"cortheon-guard/{role_name}"
        if rule_id not in seen_rule_ids:
            seen_rule_ids.add(rule_id)
            base_role = re.sub(r"_\d+$", "", role_name)
            desc = role_descriptions.get(
                role_name,
                role_descriptions.get(
                    base_role,
                    f"Mechanically proven adversarial bug discovered by Cortheon Guard role `{role_name}`.",
                ),
            )
            rules.append(
                {
                    "id": rule_id,
                    "name": _pascal_case(role_name),
                    "shortDescription": {
                        "text": f"Cortheon Guard proven bug ({role_name})",
                    },
                    "fullDescription": {
                        "text": desc,
                    },
                    "defaultConfiguration": {
                        "level": "error",
                    },
                    "properties": {
                        "tags": ["correctness", "cortheon-guard", "zero-false-positive"],
                    },
                }
            )
        return rule_id

    for role_name, _desc, _t, _s in ADVERSARY_ROLES:
        _register_rule(role_name)

    results: list[dict[str, Any]] = []
    for bug in report.proven_bugs:
        rule_id = _register_rule(bug.role)
        level = "error" if bug.passed_on_base is True else "warning"
        raw_target = (
            bug.target_files[0]
            if bug.target_files
            else (report.diff_files[0] if report.diff_files else ".")
        )
        target_uri = _normalize_sarif_uri(raw_target, report.repo_path)
        start_line = _extract_failure_line(bug.failure_output, target_uri)

        if bug.passed_on_base is True:
            status_label = f"REGRESSION (passed on {report.base_ref})"
        elif bug.passed_on_base is False:
            status_label = f"PRE-EXISTING (also failed on {report.base_ref})"
        else:
            status_label = "NEW CODE / BASELINE N/A"

        fence_lang = _language_fence_tag(bug.language)
        message_text = (
            f"{bug.id}: {bug.title}\n"
            f"Status: {status_label}\n\n"
            f"Failure output:\n{bug.failure_output}\n\n"
            f"Reproducer ({bug.language}):\n```{fence_lang}\n{bug.reproducer_code}\n```"
        )
        fp_digest = hashlib.sha256(
            f"{rule_id}|{target_uri}|{bug.title}|{bug.language}".encode("utf-8")
        ).hexdigest()[:16]

        results.append(
            {
                "ruleId": rule_id,
                "level": level,
                "message": {"text": message_text},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": target_uri,
                                "uriBaseId": "%SRCROOT%",
                            },
                            "region": {
                                "startLine": start_line,
                            },
                        }
                    }
                ],
                "partialFingerprints": {
                    "primaryLocationLineHash": fp_digest,
                },
                "properties": {
                    "bug_id": bug.id,
                    "role": bug.role,
                    "passed_on_base": bug.passed_on_base,
                    "language": bug.language,
                    "target_files": list(bug.target_files),
                },
            }
        )

    return {
        "$schema": _SARIF_SCHEMA_URI,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Cortheon Guard",
                        "version": _GUARD_VERSION,
                        "informationUri": "https://github.com/ashhart/Cortheon",
                        "rules": rules,
                    }
                },
                "results": results,
                "invocations": [
                    {
                        "executionSuccessful": True,
                        "properties": {
                            "repo_path": report.repo_path,
                            "base_ref": report.base_ref,
                            "runner": report.runner,
                            "adversaries_run": report.adversaries_run,
                            "reproducers_executed": report.reproducers_executed,
                            "rejected_non_failing": report.rejected_non_failing,
                            "wall_s": round(report.wall_s, 3),
                        },
                    }
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Non-Destructive Git Hook Installer / Uninstaller
# ---------------------------------------------------------------------------


def _validate_hook_name(hook_name: str) -> str:
    cleaned = (hook_name or "").strip()
    if not cleaned or "/" in cleaned or "\\" in cleaned or cleaned in (".", ".."):
        raise ValueError(f"Invalid git hook name: {hook_name!r}")
    return cleaned


def _resolve_git_hooks_dir(repo_path: str | Path) -> Path:
    repo = Path(repo_path).resolve()
    if not repo.exists():
        raise ValueError(f"Repository path does not exist: {repo}")
    proc = _run_git(repo, ["rev-parse", "--git-path", "hooks"])
    if proc.returncode != 0 or not proc.stdout.strip():
        raise ValueError(f"Not a git repository: {repo} ({proc.stderr.strip()})")
    raw = proc.stdout.strip()
    hooks_dir = Path(raw)
    if not hooks_dir.is_absolute():
        hooks_dir = (repo / hooks_dir).resolve()
    return hooks_dir


def install_git_hook(
    repo_path: str | Path,
    *,
    url: str = "http://127.0.0.1:8080/v1",
    hook_name: str = "pre-push",
    base_ref: str = "HEAD",
    adversaries: int = 3,
    block_on_bug: bool = True,
    force: bool = False,
) -> Path:
    """Non-destructively install an executable Cortheon Guard git hook.

    Resolves the repository's hooks directory via ``git rev-parse --git-path hooks``
    (supporting normal repositories, linked worktrees, and custom ``core.hooksPath``).
    Refuses to overwrite an existing non-Cortheon hook unless ``force=True``, and
    idempotently updates an existing Cortheon Guard hook in place.
    """
    valid_name = _validate_hook_name(hook_name)
    hooks_dir = _resolve_git_hooks_dir(repo_path)
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / valid_name

    if hook_path.exists():
        existing = hook_path.read_text(encoding="utf-8", errors="replace")
        if _GIT_HOOK_MARKER not in existing and not force:
            raise FileExistsError(
                f"Refusing to overwrite existing custom git hook at {hook_path}. "
                "Pass force=True to replace it."
            )

    quoted_url = shlex.quote(url)
    quoted_ref = shlex.quote(base_ref)
    adv_n = max(1, int(adversaries))
    suffix = "" if block_on_bug else " || true"

    script = (
        "#!/usr/bin/env sh\n"
        f"{_GIT_HOOK_MARKER}\n"
        "# Managed by Cortheon Guard — Zero-False-Positive Proof-Carrying Diff Breaker.\n"
        "# Uninstall cleanly with flash_forge.guard.uninstall_git_hook() or remove this file.\n"
        "set -eu\n\n"
        "if command -v cortheon >/dev/null 2>&1; then\n"
        f"    cortheon guard --repo . --url {quoted_url} --base-ref {quoted_ref} --adversaries {adv_n}{suffix}\n"
        "else\n"
        f"    python3 -m flash_forge.cli guard --repo . --url {quoted_url} --base-ref {quoted_ref} --adversaries {adv_n}{suffix}\n"
        "fi\n"
    )
    hook_path.write_text(script, encoding="utf-8")
    os.chmod(hook_path, 0o755)
    return hook_path


def uninstall_git_hook(
    repo_path: str | Path,
    *,
    hook_name: str = "pre-push",
    force: bool = False,
) -> bool:
    """Cleanly remove a Cortheon Guard git hook.

    Returns ``True`` if the hook was removed, or ``False`` if no hook was present.
    Refuses to delete a custom non-Cortheon hook unless ``force=True``.
    """
    valid_name = _validate_hook_name(hook_name)
    hooks_dir = _resolve_git_hooks_dir(repo_path)
    hook_path = hooks_dir / valid_name
    if not hook_path.exists():
        return False

    existing = hook_path.read_text(encoding="utf-8", errors="replace")
    if _GIT_HOOK_MARKER not in existing and not force:
        raise RuntimeError(
            f"Refusing to remove non-Cortheon git hook at {hook_path}. "
            "Pass force=True to delete it."
        )
    hook_path.unlink()
    return True

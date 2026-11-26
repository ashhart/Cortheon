"""Deterministic AST repo indexer, exact context slicer, and surgical SEARCH/REPLACE patcher.

Local models drown when you dump whole repositories into context.
`repo_index.py` produces a compact (<1,500 token) symbol map, slices exact files/line windows,
and applies clean multi-file SEARCH/REPLACE patches inside ephemeral git worktrees.
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


IGNORE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    "node_modules", ".zig-cache", "zig-out", "dist", "build", ".tox",
}


def list_repo_files(repo_dir: Path, max_files: int = 120) -> list[str]:
    files: list[str] = []
    for p in sorted(repo_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(repo_dir)
        if any(part in IGNORE_DIRS or part.startswith(".") for part in rel.parts):
            continue
        if p.suffix in {".py", ".zig", ".rs", ".ts", ".tsx", ".js", ".go", ".c", ".h", ".md", ".toml", ".json", ".yaml", ".yml"}:
            files.append(str(rel))
        if len(files) >= max_files:
            break
    return files


def summarize_python_file(path: Path, rel: str) -> str:
    """Return a compact AST symbol outline (classes, methods, functions with line numbers)."""
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
    except Exception:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return f"{rel} ({len(lines)} lines)"

    items: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = []
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [a.arg for a in sub.args.args]
                    methods.append(f"{sub.name}({','.join(args)})@L{sub.lineno}")
            items.append(f"class {node.name}@L{node.lineno}-{getattr(node, 'end_lineno', node.lineno)} [{', '.join(methods[:12])}]")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            items.append(f"def {node.name}({','.join(args)})@L{node.lineno}-{getattr(node, 'end_lineno', node.lineno)}")
    total_lines = len(src.splitlines())
    if not items:
        return f"{rel} ({total_lines}L)"
    return f"{rel} ({total_lines}L): " + "; ".join(items)


def build_repo_map(repo_dir: Path, max_chars: int = 6000) -> str:
    """Build a crisp, token-budgeted symbol map of the entire repository."""
    files = list_repo_files(repo_dir)
    lines = ["REPO MAP:"]
    used = 10
    for rel in files:
        p = repo_dir / rel
        entry = summarize_python_file(p, rel) if p.suffix == ".py" else f"{rel} ({p.stat().st_size}B)"
        if used + len(entry) + 3 > max_chars:
            lines.append(f"... ({len(files) - len(lines) + 1} more files)")
            break
        lines.append(f"- {entry}")
        used += len(entry) + 3
    return "\n".join(lines)


def read_numbered_slice(repo_dir: Path, rel_path: str, start_line: int = 1, end_line: Optional[int] = None, max_lines: int = 240) -> str:
    p = (repo_dir / rel_path).resolve()
    if not str(p).startswith(str(repo_dir.resolve())) or not p.is_file():
        return f"[Missing file: {rel_path}]"
    raw_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    s = max(1, start_line)
    e = min(len(raw_lines), end_line if end_line is not None else len(raw_lines))
    if e - s + 1 > max_lines:
        e = s + max_lines - 1
    out = [f"=== FILE: {rel_path} (lines {s}-{e} of {len(raw_lines)}) ==="]
    for idx in range(s, e + 1):
        out.append(f"{idx:4d} | {raw_lines[idx - 1]}")
    return "\n".join(out)


def rg_search(repo_dir: Path, pattern: str, max_hits: int = 30) -> str:
    """Fast regex/substring search across repository files."""
    hits: list[str] = []
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        rx = re.compile(re.escape(pattern), re.IGNORECASE)
    for rel in list_repo_files(repo_dir):
        p = repo_dir / rel
        try:
            for lineno, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{rel}:{lineno}: {line.strip()[:160]}")
                    if len(hits) >= max_hits:
                        return "\n".join(hits)
        except Exception:
            continue
    return "\n".join(hits) if hits else "(no matches)"


@dataclass
class PatchOp:
    rel_path: str
    kind: str  # "replace" or "write"
    search: str = ""
    replace: str = ""


_FILE_WRITE_RE = re.compile(
    r"<<<FILE:\s*([^\n>]+?)\s*>>>\s*\n(.*?)<<<END_FILE>>>",
    re.DOTALL,
)

_EDIT_BLOCK_RE = re.compile(
    r"<<<EDIT:\s*([^\n>]+?)\s*>>>\s*\n<<<SEARCH>>>\n(.*?)\n<<<REPLACE>>>\n(.*?)\n<<<END_EDIT>>>",
    re.DOTALL,
)


def parse_patch_ops(text: str) -> list[PatchOp]:
    """Parse multi-file <<<EDIT: path>>> SEARCH/REPLACE or <<<FILE: path>>> full-file blocks."""
    ops: list[PatchOp] = []
    # Preserve order of appearance in text
    matches: list[tuple[int, PatchOp]] = []
    for m in _EDIT_BLOCK_RE.finditer(text):
        matches.append((
            m.start(),
            PatchOp(
                rel_path=m.group(1).strip().strip("`'\""),
                kind="replace",
                search=m.group(2),
                replace=m.group(3),
            ),
        ))
    for m in _FILE_WRITE_RE.finditer(text):
        body = m.group(2)
        # Strip accidental markdown code fences inside <<<FILE>>> block
        body_stripped = body.strip()
        if body_stripped.startswith("```"):
            lines = body_stripped.splitlines()
            if len(lines) >= 2 and lines[-1].strip().startswith("```"):
                body = "\n".join(lines[1:-1]) + "\n"
        matches.append((
            m.start(),
            PatchOp(
                rel_path=m.group(1).strip().strip("`'\""),
                kind="write",
                replace=body,
            ),
        ))
    matches.sort(key=lambda x: x[0])
    for _, op in matches:
        ops.append(op)
    return ops


def apply_patch_ops(root_dir: Path, ops: list[PatchOp]) -> tuple[bool, str, list[str]]:
    """Apply multi-file PatchOps cleanly inside root_dir. Returns (ok, error_msg, touched_files)."""
    if not ops:
        return False, "No <<<EDIT>>> or <<<FILE>>> patch blocks found in output", []
    touched: list[str] = []
    root_resolved = root_dir.resolve()
    for op in ops:
        dest = (root_dir / op.rel_path).resolve()
        if not str(dest).startswith(str(root_resolved)):
            return False, f"Refusing path outside repo: {op.rel_path}", touched
        if op.kind == "write":
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(op.replace, encoding="utf-8")
            if op.rel_path not in touched:
                touched.append(op.rel_path)
        else:
            if not dest.is_file():
                return False, f"Target file does not exist for EDIT: {op.rel_path}", touched
            orig = dest.read_text(encoding="utf-8")
            if op.search in orig:
                count = orig.count(op.search)
                if count > 1:
                    return False, f"Ambiguous SEARCH block ({count} matches) in {op.rel_path}", touched
                updated = orig.replace(op.search, op.replace, 1)
                dest.write_text(updated, encoding="utf-8")
                if op.rel_path not in touched:
                    touched.append(op.rel_path)
            else:
                # Try whitespace-trimmed line match as safe fallback
                orig_lines = orig.splitlines()
                s_lines = [l.rstrip() for l in op.search.splitlines()]
                matched_idx = None
                for i in range(len(orig_lines) - len(s_lines) + 1):
                    if [l.rstrip() for l in orig_lines[i : i + len(s_lines)]] == s_lines:
                        if matched_idx is not None:
                            return False, f"Ambiguous rstrip SEARCH match in {op.rel_path}", touched
                        matched_idx = i
                if matched_idx is None:
                    return False, f"SEARCH block not found in {op.rel_path}: {op.search[:120]!r}", touched
                r_lines = op.replace.splitlines()
                new_lines = orig_lines[:matched_idx] + r_lines + orig_lines[matched_idx + len(s_lines) :]
                dest.write_text("\n".join(new_lines) + ("\n" if orig.endswith("\n") else ""), encoding="utf-8")
                if op.rel_path not in touched:
                    touched.append(op.rel_path)

    # Syntax-check all touched Python files immediately!
    for rel in touched:
        if rel.endswith(".py"):
            p = root_dir / rel
            try:
                ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                return False, f"SyntaxError in patched {rel} line {exc.lineno}: {exc.msg}", touched
    return True, "", touched


def git_diff_worktree(wt_dir: Path, max_chars: int = 6000) -> str:
    proc = subprocess.run(
        ["git", "-C", str(wt_dir), "diff"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    diff = proc.stdout
    # Also include untracked new files summary
    st = subprocess.run(
        ["git", "-C", str(wt_dir), "status", "--short"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if st.stdout.strip():
        diff = f"STATUS:\n{st.stdout.strip()}\n\nDIFF:\n{diff}"
    if len(diff) > max_chars:
        diff = diff[:max_chars] + "\n...(diff truncated)..."
    return diff

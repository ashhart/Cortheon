"""Host tool naming, bounded command construction, and mutation guards."""

from __future__ import annotations

import json
import re
import shlex
from typing import Any

from cortheon.cognitive_hooks_core.state import _FILE_MARKER_PREFIX
from cortheon.cognitive_repair import is_test_path


def _is_shell_tool(tool_name: str) -> bool:
    return tool_name.casefold() in {"bash", "shell", "exec_command"}


def _is_apply_patch_tool(tool_name: str) -> bool:
    normalized = tool_name.casefold()
    return normalized == "apply_patch" or normalized.endswith("__apply_patch")


def _safe_command(request: dict[str, Any]) -> str | None:
    capability = str(request.get("capability") or "")
    parameters = request.get("parameters")
    if not isinstance(parameters, dict):
        return None
    if capability == "grep":
        pattern = parameters.get("pattern")
        path = parameters.get("path")
        if not isinstance(pattern, str) or not _safe_relative_path(path):
            return None
        return f"rg -n --fixed-strings -- {shlex.quote(pattern)} {shlex.quote(str(path))}"
    if capability == "read_many":
        paths = parameters.get("paths")
        if (
            not isinstance(paths, list)
            or not paths
            or not all(_safe_relative_path(path) for path in paths)
        ):
            return None
        covered = {
            str(item) for item in (request.get("covered_paths") or ()) if isinstance(item, str)
        }
        remaining = [path for path in paths if str(path) not in covered]
        if not remaining:
            return None
        request_id = str(request.get("request_id") or "request")
        if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id) is None:
            return None
        marker = f"{_FILE_MARKER_PREFIX}{request_id}]"
        return (
            "for cortheon_path in "
            + " ".join(shlex.quote(str(path)) for path in remaining)
            + "; do "
            + f"printf '%s %s\\n' {shlex.quote(marker)} \"$cortheon_path\"; "
            + "sed -n '1,220p' \"$cortheon_path\" || exit; "
            + "done"
        )
    if capability == "diff":
        path = parameters.get("path")
        paths = parameters.get("paths")
        selected = (
            [path]
            if _safe_relative_path(path)
            else paths
            if isinstance(paths, list)
            and paths
            and all(_safe_relative_path(item) for item in paths)
            else None
        )
        if not selected:
            return None
        return "git diff -- " + " ".join(shlex.quote(str(item)) for item in selected)
    if capability == "test":
        command = parameters.get("command")
        if (
            not isinstance(command, list)
            or len(command) < 2
            or not all(
                isinstance(item, str)
                and item
                and "\x00" not in item
                and "\n" not in item
                and "\r" not in item
                and not item.startswith("/")
                and ".." not in item.split("/")
                for item in command
            )
        ):
            return None
        return shlex.join(command)
    return None


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or len(value) > 500:
        return False
    if value.startswith(("/", "~")) or "\x00" in value or "\\" in value or ":" in value:
        return False
    return all(part not in {"", ".", ".."} for part in value.split("/"))


def _attempts_protected_mutation(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    protected_paths: tuple[str, ...],
    protects_all_tests: bool,
) -> bool:
    if not protected_paths and not protects_all_tests:
        return False
    serialized = json.dumps(
        tool_input,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    candidates = set(protected_paths)
    if protects_all_tests:
        candidates.update(
            match.group(0)
            for match in re.finditer(
                r"[A-Za-z0-9_./-]+\.(?:py|js|jsx|ts|tsx|go|rs|java)",
                serialized,
            )
            if is_test_path(match.group(0))
        )
    if not any(path in serialized for path in candidates):
        return False
    if _is_apply_patch_tool(tool_name) or tool_name.casefold() in {
        "edit",
        "write",
        "delete",
        "move",
    }:
        return True
    if not _is_shell_tool(tool_name):
        return False
    command = str(tool_input.get("command") or tool_input.get("cmd") or "")
    mutation = re.search(
        r"(?:^|[;&|]\s*)(?:apply_patch|rm|mv|cp|tee)\b|"
        r"\bsed\s+-i\b|\bperl\s+-[^\s]*i|\btruncate\b|"
        r"(?:^|[^>])>{1,2}(?:[^>]|$)",
        command,
    )
    return mutation is not None

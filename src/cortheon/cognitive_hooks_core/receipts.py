"""Honest receipts for the host action that actually ran."""

from __future__ import annotations

import copy
import re
import shlex
from typing import Any


def _classify_host_tool(tool_name: str, command: str) -> str:
    """Map an actually-run host command or tool onto a logical receipt tool."""

    if command:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        first = tokens[0].rsplit("/", 1)[-1].casefold() if tokens else ""
        second = tokens[1].casefold() if len(tokens) > 1 else ""
        if first == "git" and second == "grep":
            return "grep"
        if first in {"rg", "grep", "egrep", "fgrep", "zgrep"}:
            return "grep"
        if first in {"cat", "sed", "awk", "head", "tail", "less", "more", "bat", "nl"}:
            return "read"
        if first in {"find", "fd", "ls", "tree", "locate"}:
            return "find"
        if first in {"curl", "wget"}:
            return "webfetch"
        return "read"
    normalized = tool_name.casefold()
    if "fetch" in normalized:
        return "webfetch"
    if "search" in normalized:
        return "websearch"
    if "grep" in normalized:
        return "grep"
    if "glob" in normalized or "find" in normalized or "list" in normalized:
        return "find"
    return "read"


def _host_receipt_arguments(
    tool: str,
    *,
    host_command: str,
    host_input: dict[str, Any],
    fallback_query: str,
    request_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Describe the model-owned host action without changing it.

    A real ``cat file`` can satisfy a read request; the original command is
    always retained and a structured file path added only when recoverable
    unambiguously.
    """

    if host_command:
        arguments: dict[str, Any] = {"command": host_command}
        if tool == "read":
            path = _read_path_from_command(host_command)
            if path is not None:
                arguments["filePath"] = path
        elif tool == "grep":
            try:
                tokens = shlex.split(host_command)
            except ValueError:
                tokens = []
            pattern = request_parameters.get("pattern")
            path = request_parameters.get("path")
            if isinstance(pattern, str) and pattern in tokens:
                arguments["pattern"] = pattern
            if isinstance(path, str) and any(
                token == path or token.replace("\\", "/").endswith("/" + path) for token in tokens
            ):
                arguments["path"] = path
        return arguments
    arguments = copy.deepcopy(host_input)
    if tool == "read":
        path = next(
            (
                arguments[key]
                for key in ("filePath", "file_path", "path", "filename")
                if isinstance(arguments.get(key), str) and arguments[key]
            ),
            None,
        )
        if path is not None:
            arguments["filePath"] = path
    elif tool == "grep":
        pattern = next(
            (
                arguments[key]
                for key in ("pattern", "query")
                if isinstance(arguments.get(key), str) and arguments[key]
            ),
            None,
        )
        path = next(
            (
                arguments[key]
                for key in ("path", "filePath", "file_path")
                if isinstance(arguments.get(key), str) and arguments[key]
            ),
            None,
        )
        if pattern is not None:
            arguments["pattern"] = pattern
        if path is not None:
            arguments["path"] = path
    if not arguments:
        arguments["query"] = fallback_query
    return arguments


def _read_path_from_command(command: str) -> str | None:
    """Recover one literal file operand from a simple read-only command."""

    stripped = re.sub(r"\s2>\s*/dev/null|\s2>&1|\s</dev/null", " ", command).strip()
    if not stripped or re.search(r"[`<>]|\$\(|\|\||&&|[;|&]", stripped):
        return None
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return None
    if len(tokens) < 2:
        return None
    reader = tokens[0].rsplit("/", 1)[-1].casefold()
    if reader not in {
        "cat",
        "bat",
        "sed",
        "awk",
        "head",
        "tail",
        "less",
        "more",
        "nl",
        "wc",
        "file",
        "stat",
    }:
        return None
    candidates = [
        token
        for token in tokens[1:]
        if token
        and not token.startswith("-")
        and not token.isdigit()
        and not re.fullmatch(r"\d+(?:,\d+)?[a-z]?", token, flags=re.IGNORECASE)
    ]
    if not candidates:
        return None
    candidate = candidates[-1]
    if "\x00" in candidate or candidate in {".", ".."}:
        return None
    return candidate

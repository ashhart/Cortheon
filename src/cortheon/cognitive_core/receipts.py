from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

from cortheon.cognitive_core.models import CognitiveRuntimeError, EvidenceRequest, Observation

_HOST_EVIDENCE_PREFIX = "[CORTHEON_HOST_EVIDENCE] "


def _host_evidence_receipt(content: str) -> dict[str, Any] | None:
    first_line = content.splitlines()[0] if content else ""
    if not first_line.startswith(_HOST_EVIDENCE_PREFIX):
        return None
    try:
        value = json.loads(first_line[len(_HOST_EVIDENCE_PREFIX) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    if not isinstance(value.get("tool"), str):
        return None
    if not isinstance(value.get("outcome"), str):
        return None
    if not isinstance(value.get("args"), dict):
        return None
    return value


def _example_receipt_json(capability: str, request: EvidenceRequest | None) -> str:
    parameters = request.parameters if request is not None else {}
    if capability == "grep":
        example: dict[str, Any] = {
            "tool": "grep",
            "outcome": "match",
            "args": {
                "pattern": parameters.get("pattern") or "<exact pattern>",
                "path": parameters.get("path") or "src/module.py",
            },
        }
    elif capability == "read_many":
        paths = [item for item in parameters.get("paths", []) if isinstance(item, str)]
        example = {
            "tool": "read",
            "outcome": "result",
            "args": {"filePath": paths[0] if paths else "src/module.py"},
        }
    elif capability == "read":
        example = {
            "tool": "read",
            "outcome": "result",
            "args": {"filePath": "src/module.py"},
        }
    elif capability == "diff":
        example = {
            "tool": "diff",
            "outcome": "changed",
            "args": {"path": "src/module.py"},
        }
    elif capability == "test":
        command = parameters.get("command")
        example = {
            "tool": "test",
            "outcome": "passed",
            "args": {
                "command": command if isinstance(command, list) else "python3 -m pytest -q",
            },
        }
    elif capability in {"fetch", "search_or_fetch"}:
        example = {
            "tool": "webfetch",
            "outcome": "result",
            "args": {"url": "https://example.com/page"},
        }
    else:
        example = {
            "tool": "grep",
            "outcome": "match",
            "args": {"command": "rg -n <pattern> src/"},
        }
    return json.dumps(example, separators=(",", ":"), sort_keys=True)


def _receipt_error(
    message: str,
    capability: str,
    request: EvidenceRequest | None,
) -> CognitiveRuntimeError:
    """Reject a receipt with a correct example attached.
    Small models imitate syntax better than they follow prose, so every
    rejection carries copyable JSON.
    """

    return CognitiveRuntimeError(
        f"{message}. Correct example host_receipt: {_example_receipt_json(capability, request)}"
    )


def _validate_host_observation_batch(
    request: EvidenceRequest | None,
    observations: list[dict[str, Any]],
) -> bool:
    """Validate structural host provenance and exact request binding.
    Prevents stale, mismatched, hidden, or failed receipts from satisfying
    an evidence request; native host adapters add the first-line receipt
    from their tool-result hooks.
    """

    parsed: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for raw in observations:
        if not isinstance(raw, dict):
            raise ValueError("each observation must be an object")
        content = raw.get("content")
        if not isinstance(content, str):
            raise ValueError("observation.content must be a string")
        receipt = _host_evidence_receipt(content)
        kind = raw.get("kind")
        status = raw.get("status", "observed")
        if (kind in {"diff", "test"} or status == "verified") and receipt is None:
            raise _receipt_error(
                "verified, diff, and test evidence requires a first-line host receipt",
                kind if kind in {"diff", "test"} else "grep",
                request,
            )
        parsed.append((raw, receipt))

    if request is None:
        return True

    capability = request.capability
    purpose = request.parameters.get("purpose")
    successful = False
    covered_paths: set[str] = set()
    expected_paths = {
        str(item) for item in request.parameters.get("paths", []) if isinstance(item, str)
    }
    for raw, receipt in parsed:
        kind = raw.get("kind")
        status = raw.get("status", "observed")
        tool = str(receipt["tool"]).casefold() if receipt is not None else ""
        outcome = str(receipt["outcome"]).casefold() if receipt is not None else ""
        if status == "failed":
            continue
        if capability in {"search", "fetch", "search_or_fetch"} and purpose is not None:
            if kind != "web" or raw.get("purpose") != purpose:
                raise CognitiveRuntimeError(
                    "web evidence purpose does not match the pending request. "
                    "Correct example observation: "
                    + json.dumps(
                        {
                            "kind": "web",
                            "url": "https://example.com/page",
                            "retrieved_at": "2026-01-01T00:00:00+00:00",
                            "purpose": purpose,
                            "content": "<focused excerpt>",
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
            if (
                capability in {"search", "search_or_fetch"}
                and tool in {"search", "websearch"}
                and outcome == "no_match"
                and raw.get("retrieved_at")
            ):
                successful = True
                continue
            if not raw.get("url") or not raw.get("retrieved_at"):
                continue
            successful = True
            continue
        if receipt is None:
            raise _receipt_error(
                f"request capability '{capability}' requires a first-line host receipt",
                capability,
                request,
            )
        arguments = receipt["args"]
        if outcome in {"error", "failed"}:
            continue

        if capability == "grep":
            if tool != "grep" or outcome not in {"match", "no_match"}:
                raise _receipt_error(
                    "the pending grep request requires a grep match/no_match receipt",
                    capability,
                    request,
                )
            expected_pattern = request.parameters.get("pattern")
            expected_path = request.parameters.get("path")
            if (expected_pattern is not None and arguments.get("pattern") != expected_pattern) or (
                expected_path is not None and arguments.get("path") != expected_path
            ):
                raise _receipt_error(
                    "host receipt arguments do not match the pending grep request",
                    capability,
                    request,
                )
            successful = True
        elif capability == "read_many":
            if tool != "read":
                raise _receipt_error(
                    "the pending read_many request requires host read receipts",
                    capability,
                    request,
                )
            path = arguments.get("filePath")
            matched_path = (
                next(
                    (
                        expected
                        for expected in expected_paths
                        if _host_path_matches_request(path, expected)
                    ),
                    None,
                )
                if isinstance(path, str)
                else None
            )
            if matched_path is None:
                raise _receipt_error(
                    "host read receipt path is outside the pending read_many request",
                    capability,
                    request,
                )
            covered_paths.add(matched_path)
        elif capability == "read":
            if tool not in {"read", "grep"}:
                raise _receipt_error(
                    "the pending read request requires a read or grep receipt",
                    capability,
                    request,
                )
            successful = True
        elif capability == "diff":
            if kind != "diff" or tool not in {"diff", "bash", "git"}:
                raise _receipt_error(
                    "the pending diff request requires focused diff evidence",
                    capability,
                    request,
                )
            if outcome not in {"changed", "result"}:
                raise _receipt_error(
                    "the pending diff request requires a changed/result receipt",
                    capability,
                    request,
                )
            successful = True
        elif capability == "test":
            if kind != "test" or tool not in {"test", "bash"}:
                raise _receipt_error(
                    "the pending test request requires host test evidence",
                    capability,
                    request,
                )
            if outcome not in {"passed", "failed", "result"}:
                raise _receipt_error(
                    "the pending test request requires a passed/failed/result receipt",
                    capability,
                    request,
                )
            successful = outcome != "failed"
        elif capability == "search":
            if tool in {"bash", "shell"}:
                if not _read_only_shell_receipt(arguments):
                    raise _receipt_error(
                        "host receipt tool does not match the pending search request",
                        capability,
                        request,
                    )
            elif tool not in {
                "find",
                "glob",
                "grep",
                "ls",
                "read",
                "search",
                "webfetch",
                "websearch",
            }:
                raise _receipt_error(
                    "host receipt tool does not match the pending search request",
                    capability,
                    request,
                )
            successful = True
        elif capability == "fetch":
            if tool not in {"webfetch", "fetch"}:
                raise _receipt_error(
                    "host receipt tool does not match the pending fetch request",
                    capability,
                    request,
                )
            successful = True
        elif capability in {"inspect", "search_or_read"}:
            if tool in {"bash", "shell"}:
                if not _read_only_shell_receipt(arguments):
                    raise _receipt_error(
                        f"host receipt tool does not match the pending {capability} request",
                        capability,
                        request,
                    )
            elif tool not in {"read", "grep", "glob", "find", "ls"}:
                raise _receipt_error(
                    f"host receipt tool does not match the pending {capability} request",
                    capability,
                    request,
                )
            successful = True
        elif capability == "search_or_fetch":
            if tool not in {"websearch", "webfetch", "fetch"}:
                raise _receipt_error(
                    "host receipt tool does not match the pending search_or_fetch request",
                    capability,
                    request,
                )
            successful = True
        else:
            raise CognitiveRuntimeError(f"unsupported host evidence capability: {capability}")

    if capability == "read_many":
        request.covered_paths.update(covered_paths)
        return bool(expected_paths) and request.covered_paths >= expected_paths
    return successful


def _host_path_matches_request(actual: str, expected: str) -> bool:
    """Match an exact requested relative path to an honest absolute receipt."""

    normalized_actual = actual.replace("\\", "/").rstrip("/")
    normalized_expected = expected.replace("\\", "/").strip("/")
    return normalized_actual == normalized_expected or normalized_actual.endswith(
        "/" + normalized_expected
    )


_READ_ONLY_SHELL_COMMANDS = frozenset(
    {
        "ls",
        "find",
        "fd",
        "tree",
        "locate",
        "rg",
        "grep",
        "egrep",
        "fgrep",
        "zgrep",
        "cat",
        "head",
        "tail",
        "sed",
        "awk",
        "nl",
        "wc",
        "file",
        "stat",
        "du",
        "pwd",
    }
)


_READ_ONLY_GIT_SUBCOMMANDS = frozenset(
    {"grep", "ls-files", "log", "show", "diff", "status", "blame"}
)


_MUTATING_READER_FLAGS = frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir"})


_RUNTIME_VERSION_COMMAND = re.compile(
    r"^(?:bun|cargo|deno|dotnet|go|java|node|npm|php|pnpm|python|python3|"
    r"ruby|rustc|uv|yarn)$"
)


def _read_only_version_invocation(tokens: list[str]) -> bool:
    """Allow only bounded runtime/package identity probes during grounding."""

    if not tokens:
        return False
    command = tokens[0].rsplit("/", 1)[-1].casefold()
    if _RUNTIME_VERSION_COMMAND.fullmatch(command) is None:
        return False
    arguments = tokens[1:]
    if command == "go":
        return arguments == ["version"]
    if command == "java":
        return arguments == ["-version"]
    if command == "dotnet":
        return arguments in (["--version"], ["--info"])
    if (
        command in {"python", "python3"}
        and len(arguments) >= 3
        and arguments[:3] == ["-m", "pip", "show"]
    ):
        return bool(arguments[3:]) and all(
            re.fullmatch(r"[A-Za-z0-9_.-]+", item) for item in arguments[3:]
        )
    return arguments in (["--version"], ["-V"], ["-v"], ["version"])


def _read_only_shell_receipt(arguments: dict[str, Any]) -> bool:
    """Accept a shell receipt for read-only requests only when its recorded
    command is verifiably a reader (ls/find/rg/cat...)."""

    command = arguments.get("command")
    if isinstance(command, list):
        command = " ".join(str(item) for item in command)
    if not isinstance(command, str) or not command.strip():
        return False
    stripped = re.sub(r"\s2>\s*/dev/null|\s2>&1|\s</dev/null", " ", command)
    if re.search(r"[`<>]|\$\(", stripped):
        return False
    segments = [
        segment.strip() for segment in re.split(r"\|\||&&|[;|&]", stripped) if segment.strip()
    ]
    if not segments:
        return False
    for segment in segments:
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return False
        if not tokens:
            return False
        if any(token in _MUTATING_READER_FLAGS for token in tokens):
            return False
        first = tokens[0].rsplit("/", 1)[-1].casefold()
        if first in {"sed", "awk"} and any(
            token.startswith("-i") or token == "--in-place" for token in tokens[1:]
        ):
            return False
        if first == "git":
            if len(tokens) < 2 or tokens[1].casefold() not in _READ_ONLY_GIT_SUBCOMMANDS:
                return False
            continue
        if _read_only_version_invocation(tokens):
            continue
        if first not in _READ_ONLY_SHELL_COMMANDS:
            return False
    return True


def _observation_origin(observation: Observation) -> str | None:
    candidate = observation.url or observation.source
    if not candidate:
        return None
    parsed = urlsplit(candidate)
    hostname = (parsed.hostname or "").casefold().removeprefix("www.")
    if not hostname:
        return None
    labels = hostname.split(".")
    if len(labels) <= 2:
        return hostname
    public_suffix = ".".join(labels[-2:])
    if public_suffix in {"co.jp", "co.uk", "com.au", "gov.uk", "org.uk"}:
        return ".".join(labels[-3:])
    return public_suffix


def _receipt_outcome(observation: Observation, tool: str, outcome: str) -> bool:
    receipt = observation.host_receipt or {}
    return (
        str(receipt.get("tool") or "").casefold() == tool
        and str(receipt.get("outcome") or "").casefold() == outcome
    )


def _read_receipt_paths(observations: Iterable[Observation]) -> dict[str, Observation]:
    paths: dict[str, Observation] = {}
    for observation in observations:
        receipt = observation.host_receipt
        if (
            observation.status == "failed"
            or observation.quarantine_flags
            or receipt is None
            or receipt.get("tool") != "read"
        ):
            continue
        path = str(receipt.get("args", {}).get("filePath") or "").strip()
        if path:
            paths[path.casefold()] = observation
    return paths


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _observation_digest(raw: dict[str, Any]) -> str:
    return _digest(repr(sorted(raw.items())) if isinstance(raw, dict) else repr(raw))

"""Turn host tool output into bounded, receipt-carrying observations."""

from __future__ import annotations

import copy
import json
import re
import urllib.parse
from datetime import UTC, datetime
from typing import Any

from cortheon.cognitive_hooks_core.receipts import (
    _classify_host_tool,
    _host_receipt_arguments,
)
from cortheon.cognitive_hooks_core.state import (
    _FILE_MARKER_PREFIX,
    MAX_HOOK_EVIDENCE_CHARS,
)


def _host_observations(
    request: dict[str, Any],
    tool_name: str,
    tool_output: str,
    *,
    succeeded: bool,
    host_command: str = "",
    host_input: dict[str, Any] | None = None,
    tool_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    capability = str(request.get("capability") or "")
    parameters = request.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    output = tool_output.strip()[:MAX_HOOK_EVIDENCE_CHARS]
    if capability == "edit":
        return []
    if host_command or host_input or capability not in {"grep", "read_many", "diff", "test"}:
        # Investigation-phase harvest: the model ran its own allowed command or
        # host tool, so build an honest receipt from what actually executed.
        nested_web = _nested_web_executor(tool_name, host_input or {})
        if nested_web is not None:
            return [
                _failed_web_observation(
                    nested_web,
                    tool_name,
                    "Nested web output lacked direct structured attribution.",
                )
            ]
        tool = _classify_host_tool(tool_name, host_command)
        normalized_name = tool_name.casefold()
        structured_keys = set((host_input or {}).keys())
        if tool == "read" and (
            "web" in normalized_name
            or structured_keys & {"search_query", "image_query", "open", "click"}
        ):
            tool = (
                "websearch"
                if structured_keys & {"search_query", "image_query", "query"}
                or "search" in normalized_name
                else "webfetch"
            )
        if tool == "grep":
            outcome = "match" if output else "no_match"
        elif not succeeded:
            outcome = "error"
        else:
            outcome = "result" if output else "no_match"
        arguments = _host_receipt_arguments(
            tool,
            host_command=host_command,
            host_input=host_input or {},
            fallback_query=str(request.get("query") or "")[:200],
            request_parameters=parameters,
        )
        if tool in {"websearch", "webfetch"}:
            return _web_observations(
                request,
                tool,
                tool_output,
                succeeded=succeeded,
                executor=tool_name,
                host_input=host_input or {},
                metadata=tool_metadata or {},
            )
        receipt = {
            "tool": tool,
            "executor": tool_name,
            "outcome": outcome,
            "args": arguments,
        }
        return [
            _observation(
                "code",
                output or "No output.",
                receipt,
                status="failed" if not succeeded else "observed",
            )
        ]
    if capability == "grep":
        outcome = "match" if output else "no_match"
        receipt = {
            "tool": "grep",
            "executor": tool_name,
            "outcome": outcome,
            "args": {
                "pattern": parameters.get("pattern"),
                "path": parameters.get("path"),
            },
        }
        return [_observation("code", output or "No matches found.", receipt)]
    if capability == "read_many":
        paths = parameters.get("paths")
        if not isinstance(paths, list):
            return []
        split = _split_read_many_output(
            output,
            request_id=str(request.get("request_id") or ""),
            expected_paths=[item for item in paths if isinstance(item, str)],
        )
        if split is None:
            return []
        return [
            _observation(
                "code",
                split[path],
                {
                    "tool": "read",
                    "executor": tool_name,
                    "outcome": "result" if succeeded else "error",
                    "args": {"filePath": path},
                },
                status="verified" if succeeded else "failed",
            )
            for path in paths
            if isinstance(path, str) and path in split
        ]
    if capability == "diff":
        changed = succeeded and bool(output)
        receipt = {
            "tool": "diff",
            "executor": tool_name,
            "outcome": "changed" if changed else "no_match",
            "args": copy.deepcopy(parameters),
        }
        return [
            _observation(
                "diff",
                output or "No diff.",
                receipt,
                status="verified" if changed else "failed",
            )
        ]
    if capability == "test":
        receipt = {
            "tool": "test",
            "executor": tool_name,
            "outcome": "passed" if succeeded else "failed",
            "args": copy.deepcopy(parameters),
        }
        return [
            _observation(
                "test",
                output or "No test output.",
                receipt,
                status="verified" if succeeded else "failed",
            )
        ]
    return []


def _nested_web_executor(tool_name: str, host_input: dict[str, Any]) -> str | None:
    if tool_name.casefold() not in {"exec", "functions.exec", "functions__exec"}:
        return None
    code = host_input.get("code")
    if not isinstance(code, str) or "tools.web__run" not in code:
        return None
    if "search_query" in code or "image_query" in code:
        return "websearch"
    return "webfetch"


def _observation(
    kind: str,
    content: str,
    receipt: dict[str, Any],
    *,
    status: str = "verified",
    source: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    prefix = "[CORTHEON_HOST_EVIDENCE] " + json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "kind": kind,
        "content": f"{prefix}\n{content}",
        "source": source or f"codex:{receipt['executor']}",
        "status": status,
        **fields,
    }


def _normalized_web_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 2_000:
        return None
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    host = parsed.hostname.casefold()
    if ":" in host:
        host = f"[{host}]"
    default_port = (parsed.scheme.casefold(), port) in {("http", 80), ("https", 443)}
    netloc = host if port is None or default_port else f"{host}:{port}"
    return urllib.parse.urlunsplit(
        (parsed.scheme.casefold(), netloc, parsed.path or "/", parsed.query, "")
    )


def _published_at(metadata: dict[str, Any]) -> str | None:
    values = [
        metadata[key]
        for key in ("published_at", "publishedAt", "date")
        if isinstance(metadata.get(key), str) and metadata[key]
    ]
    if not values:
        return None
    normalized: list[str] = []
    for value in values:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            try:
                datetime.fromisoformat(value)
            except ValueError:
                return None
            normalized.append(value)
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return None
        normalized.append(parsed.astimezone(UTC).isoformat().replace("+00:00", "Z"))
    return normalized[0] if len(set(normalized)) == 1 else None


def _web_receipt(
    tool: str,
    executor: str,
    url: str,
    request: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    parameters = request.get("parameters")
    purpose = parameters.get("purpose") if isinstance(parameters, dict) else None
    parsed = urllib.parse.urlsplit(url)
    lineage: dict[str, Any] = {"origin": f"{parsed.scheme}://{parsed.netloc}"}
    provider = metadata.get("provider")
    source_type = metadata.get("source_type", metadata.get("sourceType"))
    if isinstance(provider, (str, int, float)) and not isinstance(provider, bool):
        lineage["provider"] = provider
    if isinstance(source_type, (str, int, float)) and not isinstance(source_type, bool):
        lineage["source_type"] = source_type
    receipt: dict[str, Any] = {
        "tool": tool,
        "executor": executor,
        "outcome": "result",
        "args": {"url": url, "purpose": purpose},
        "lineage": lineage,
    }
    authority = metadata.get("authority")
    if isinstance(authority, (str, int, float)) and not isinstance(authority, bool):
        receipt["authority"] = authority
    return receipt


def _failed_web_observation(
    tool: str,
    executor: str,
    reason: str,
) -> dict[str, Any]:
    return _observation(
        "web",
        reason,
        {"tool": tool, "executor": executor, "outcome": "error", "args": {}},
        status="failed",
    )


def _web_observations(
    request: dict[str, Any],
    tool: str,
    tool_output: str,
    *,
    succeeded: bool,
    executor: str,
    host_input: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    parameters = request.get("parameters")
    purpose = parameters.get("purpose") if isinstance(parameters, dict) else None
    if not isinstance(purpose, str) or not purpose.strip() or len(purpose) > 500:
        return [_failed_web_observation(tool, executor, "Web request lacked a bounded purpose.")]
    if not succeeded:
        return [_failed_web_observation(tool, executor, "Host web tool failed.")]
    retrieved_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    if tool == "webfetch":
        candidates = [
            value
            for value in (
                host_input.get("url"),
                metadata.get("url"),
                metadata.get("finalUrl"),
                metadata.get("final_url"),
            )
            if value is not None
        ]
        urls = [_normalized_web_url(value) for value in candidates]
        if not urls or any(url is None for url in urls):
            return [_failed_web_observation(tool, executor, "No valid structured URL.")]
        origins = {urllib.parse.urlsplit(url).netloc for url in urls if url}
        if len(origins) != 1 or not tool_output.strip():
            return [_failed_web_observation(tool, executor, "Mixed origin or empty fetch result.")]
        url = urls[-1]
        assert url is not None
        published_at = _published_at(metadata)
        if any(key in metadata for key in ("published_at", "publishedAt", "date")) and (
            published_at is None
        ):
            return [_failed_web_observation(tool, executor, "Invalid publication time.")]
        return [
            _observation(
                "web",
                tool_output.strip()[:MAX_HOOK_EVIDENCE_CHARS],
                _web_receipt(tool, executor, url, request, metadata),
                status="observed",
                source=url,
                url=url,
                retrieved_at=retrieved_at,
                purpose=purpose,
                **({"published_at": published_at} if published_at else {}),
            )
        ]
    results = metadata.get("results")
    if not isinstance(results, list) or not 0 < len(results) <= 8:
        return [_failed_web_observation(tool, executor, "No bounded structured result list.")]
    observations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for value in results:
        if not isinstance(value, dict):
            return [_failed_web_observation(tool, executor, "Malformed structured result.")]
        url = _normalized_web_url(value.get("url"))
        text = next(
            (
                value[key]
                for key in ("snippet", "content", "text")
                if isinstance(value.get(key), str) and value[key].strip()
            ),
            None,
        )
        published_at = _published_at(value)
        invalid_published = (
            any(key in value for key in ("published_at", "publishedAt", "date"))
            and published_at is None
        )
        if url is None or text is None or invalid_published:
            return [_failed_web_observation(tool, executor, "Result lacked attributable fields.")]
        title = value.get("title")
        content = f"{title}\n{text}" if isinstance(title, str) and title else text
        key = (url, content, published_at or "")
        if key in seen:
            continue
        seen.add(key)
        observations.append(
            _observation(
                "web",
                content[:MAX_HOOK_EVIDENCE_CHARS],
                _web_receipt(tool, executor, url, request, value),
                status="observed",
                source=url,
                url=url,
                retrieved_at=retrieved_at,
                purpose=purpose,
                **({"published_at": published_at} if published_at else {}),
            )
        )
    return observations or [_failed_web_observation(tool, executor, "Only duplicate results.")]


def _split_read_many_output(
    output: str,
    *,
    request_id: str,
    expected_paths: list[str],
) -> dict[str, str] | None:
    if not expected_paths:
        return None
    marker = f"{_FILE_MARKER_PREFIX}{request_id}] "
    current: str | None = None
    captured: dict[str, list[str]] = {}
    expected = set(expected_paths)
    for line in output.splitlines():
        if line.startswith(marker):
            candidate = line[len(marker) :]
            current = candidate if candidate in expected else None
            if current is not None:
                captured.setdefault(current, [])
            continue
        if current is not None:
            captured[current].append(line)
    if set(captured) != expected:
        if len(expected_paths) == 1 and marker not in output:
            return {expected_paths[0]: output}
        # Host output truncation can cut off later files; accept the files that
        # did arrive so the request can converge over follow-up reads instead of
        # re-running the identical command forever.
        partial = {
            path: "\n".join(captured[path]).strip()[:MAX_HOOK_EVIDENCE_CHARS]
            for path in expected_paths
            if captured.get(path)
        }
        return partial or None
    return {
        path: "\n".join(captured[path]).strip()[:MAX_HOOK_EVIDENCE_CHARS] for path in expected_paths
    }


def _read_snapshots(
    observations: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    snapshots: list[tuple[str, str]] = []
    for observation in observations:
        content = observation.get("content")
        if not isinstance(content, str):
            continue
        lines = content.splitlines()
        if not lines or not lines[0].startswith("[CORTHEON_HOST_EVIDENCE] "):
            continue
        try:
            receipt = json.loads(lines[0][len("[CORTHEON_HOST_EVIDENCE] ") :])
        except json.JSONDecodeError:
            continue
        arguments = receipt.get("args") if isinstance(receipt, dict) else None
        path = arguments.get("filePath") if isinstance(arguments, dict) else None
        if isinstance(path, str):
            snapshots.append((path, "\n".join(lines[1:])))
    return snapshots

"""Isolated evaluator-owned implementations of the generic host tools."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from cortheon.benchmark_core.generic_mcp_tools import ToolExecution, ToolLedger, ToolRequest

MAX_FILE_CHARS = 40_000
MAX_WEB_RESULTS = 8
_MARKER = ".cortheon-evaluator-workspace"


class IsolatedExecutor:
    """Execute a fixed capability set only inside an evaluator-marked workspace."""

    def __init__(
        self,
        root: Path,
        *,
        marker_nonce: str,
        tests: dict[str, list[str]] | None = None,
        web_provider: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        web_identity: dict[str, str] | None = None,
        maximum_calls: int = 16,
    ) -> None:
        self.root = root.resolve(strict=True)
        marker = self.root / _MARKER
        if not marker.is_file() or marker.read_text(encoding="utf-8") != marker_nonce:
            raise ValueError("generic MCP requires an evaluator-marked isolated workspace")
        self.tests = dict(tests or {})
        if (web_provider is None) != (web_identity is None):
            raise ValueError("web provider and identity must be configured together")
        if web_identity is not None and (
            set(web_identity) != {"executable_sha256", "version", "config_sha256"}
            or any(not isinstance(value, str) or not value for value in web_identity.values())
            or any(
                len(web_identity[key]) != 64
                or any(character not in "0123456789abcdef" for character in web_identity[key])
                for key in ("executable_sha256", "config_sha256")
            )
        ):
            raise ValueError("web provider identity is invalid")
        self.web_provider = web_provider
        self.web_identity = dict(web_identity) if web_identity is not None else None
        self.ledger = ToolLedger(maximum_calls=maximum_calls)
        self._observed_paths: set[str] = set()

    def _path(self, raw: Any) -> Path:
        if not isinstance(raw, str) or not raw or len(raw) > 1_000 or "\0" in raw:
            raise ValueError("workspace path must be bounded")
        candidate = (self.root / raw).resolve(strict=False)
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("workspace path escapes the isolated root")
        relative = candidate.relative_to(self.root)
        if _MARKER in relative.parts or ".git" in relative.parts:
            raise ValueError("workspace control paths are protected")
        parent = candidate if candidate.exists() and candidate.is_dir() else candidate.parent
        resolved_parent = parent.resolve(strict=True)
        if resolved_parent != self.root and self.root not in resolved_parent.parents:
            raise ValueError("workspace symlink escapes the isolated root")
        return candidate

    def execute(self, call_id: str, name: str, arguments: dict[str, Any]) -> ToolExecution:
        request = self.ledger.request(call_id, name, arguments)
        cached = self.ledger.cached(call_id)
        if cached is not None:
            return cached
        try:
            status, content = self._dispatch(request)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            status, content = "error", f"{type(exc).__name__}: {exc}"
        receipt = {
            "tool": name.removeprefix("host_"),
            "executor": "generic_mcp_wrapper",
            "outcome": status,
            "args": arguments,
        }
        return self.ledger.record(
            request,
            status=status,
            content=content[:MAX_FILE_CHARS],
            receipt=receipt,
        )

    def _dispatch(self, request: ToolRequest) -> tuple[str, str]:
        arguments = request.arguments
        if request.name == "host_read":
            return self._read(arguments)
        if request.name == "host_read_many":
            return self._read_many(arguments)
        if request.name == "host_search":
            return self._search(arguments)
        if request.name == "host_diff":
            return self._diff(arguments)
        if request.name == "host_test":
            return self._test(arguments)
        if request.name in {"host_web_search", "host_web_fetch"}:
            return self._web(request.name, arguments)
        raise ValueError("unsupported closed host tool")

    def _read(self, arguments: dict[str, Any]) -> tuple[str, str]:
        path = self._path(arguments.get("path"))
        start = arguments.get("start_line", 1)
        if type(start) is not int or not 1 <= start <= 1_000_000:
            raise ValueError("start_line must be a positive integer")
        text = path.read_text(encoding="utf-8")
        self._observed_paths.add(str(path.relative_to(self.root)))
        selected = "\n".join(text.splitlines()[start - 1 : start + 399])
        return "result", selected[:MAX_FILE_CHARS]

    def _search(self, arguments: dict[str, Any]) -> tuple[str, str]:
        path = self._path(arguments.get("path"))
        pattern = arguments.get("pattern")
        if not isinstance(pattern, str) or not pattern or len(pattern) > 500:
            raise ValueError("search pattern must be bounded")
        alternatives = [item.strip() for item in pattern.split("|")]
        terms = alternatives if all(alternatives) else [pattern]
        files = [path] if path.is_file() else sorted(path.rglob("*"))
        matches: list[str] = []
        for candidate in files:
            if not candidate.is_file() or candidate.is_symlink():
                continue
            relative = candidate.relative_to(self.root)
            if ".git" in relative.parts:
                continue
            try:
                lines = candidate.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for number, line in enumerate(lines, 1):
                if any(term in line for term in terms):
                    self._observed_paths.add(str(relative))
                    matches.append(f"{relative}:{number}:{line[:500]}")
                    if len(matches) == 100:
                        return "match", "\n".join(matches)[:MAX_FILE_CHARS]
        return ("match", "\n".join(matches)) if matches else ("no_match", "No matches.")

    def _read_many(self, arguments: dict[str, Any]) -> tuple[str, str]:
        paths = arguments.get("paths")
        if not isinstance(paths, list) or not 0 < len(paths) <= 8:
            raise ValueError("paths must be a bounded list")
        files: list[dict[str, str]] = []
        for raw in paths:
            path = self._path(raw)
            text = path.read_text(encoding="utf-8")[:5_000]
            relative = str(path.relative_to(self.root))
            self._observed_paths.add(relative)
            files.append({"path": relative, "content": text})
        encoded = json.dumps({"files": files}, ensure_ascii=False, separators=(",", ":"))
        while len(encoded) > MAX_FILE_CHARS:
            longest = max(files, key=lambda item: len(item["content"]))
            if not longest["content"]:
                raise ValueError("read_many paths exceed the bounded result size")
            excess = len(encoded) - MAX_FILE_CHARS
            longest["content"] = longest["content"][: -min(len(longest["content"]), excess)]
            encoded = json.dumps({"files": files}, ensure_ascii=False, separators=(",", ":"))
        return "result", encoded

    def _diff(self, arguments: dict[str, Any]) -> tuple[str, str]:
        paths = arguments.get("paths")
        if not isinstance(paths, list) or not 0 < len(paths) <= 16:
            raise ValueError("paths must be a bounded list")
        selected = [str(self._path(raw).relative_to(self.root)) for raw in paths]
        completed = subprocess.run(
            ["git", "diff", "--", *selected],
            cwd=self.root,
            env={"PATH": os.environ.get("PATH", ""), "LC_ALL": "C", "LANG": "C"},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError("git diff failed")
        return "changed", completed.stdout[-MAX_FILE_CHARS:]

    def _test(self, arguments: dict[str, Any]) -> tuple[str, str]:
        test_id = arguments.get("test_id")
        command = self.tests.get(test_id) if isinstance(test_id, str) else None
        if not command or any(not isinstance(item, str) or "\0" in item for item in command):
            raise ValueError("test_id is not in the evaluator catalogue")
        environment = {"PATH": os.environ.get("PATH", ""), "LC_ALL": "C", "LANG": "C"}
        completed = subprocess.run(
            command,
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        output = (completed.stdout + completed.stderr)[-MAX_FILE_CHARS:]
        return ("passed" if completed.returncode == 0 else "failed"), output

    def _web(self, name: str, arguments: dict[str, Any]) -> tuple[str, str]:
        if self.web_provider is None:
            raise ValueError("generic web capability is unavailable")
        if name == "host_web_fetch":
            url = arguments.get("url")
            parsed = urlsplit(url) if isinstance(url, str) else None
            if parsed is None or parsed.scheme != "https" or not parsed.hostname:
                raise ValueError("web fetch requires an https URL")
        payload = self.web_provider(name, dict(arguments))
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list) or not 0 < len(results) <= MAX_WEB_RESULTS:
            raise ValueError("web provider returned no bounded result list")
        required = {
            "url",
            "content",
            "retrieved_at",
            "provider",
            "provider_sha256",
            "provider_version",
        }
        optional = {"title", "published_at"}
        for item in results:
            if (
                not isinstance(item, dict)
                or not required <= set(item)
                or not set(item) <= required | optional
            ):
                raise ValueError("web provider result fields are not exact")
            if any(not isinstance(item[key], str) or not item[key] for key in required):
                raise ValueError("web provider result lacks attributable strings")
            if len(item["content"]) > 20_000 or len(item["provider"]) > 128:
                raise ValueError("web provider result is oversized")
            for date_key in ("retrieved_at", "published_at"):
                date_value = item.get(date_key)
                if date_value is None and date_key == "published_at":
                    continue
                try:
                    parsed_date = datetime.fromisoformat(str(date_value).replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError("web provider timestamp is invalid") from exc
                if parsed_date.tzinfo is None:
                    raise ValueError("web provider timestamp must be timezone-aware")
            if any(key in item and not isinstance(item[key], str) for key in optional):
                raise ValueError("web provider optional fields must be strings")
            parsed = urlsplit(item["url"])
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
            ):
                raise ValueError("web provider URL is not canonical https")
            if len(item["provider_sha256"]) != 64 or any(
                c not in "0123456789abcdef" for c in item["provider_sha256"]
            ):
                raise ValueError("web provider artifact digest is invalid")
            if (
                self.web_identity is None
                or item["provider_sha256"] != self.web_identity["executable_sha256"]
            ):
                raise ValueError("web result provider digest does not match the handshake")
            if item["provider_version"] != self.web_identity["version"]:
                raise ValueError("web result provider version does not match the handshake")
            if name == "host_web_fetch" and item["url"] != arguments.get("url"):
                raise ValueError("web provider redirected away from the requested source")
        return "result", json.dumps({"results": results}, separators=(",", ":"))[:MAX_FILE_CHARS]

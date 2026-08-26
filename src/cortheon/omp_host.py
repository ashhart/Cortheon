"""Native OMP host adapter for the Cortheon MCP runtime.

An agent harness (OMP) drives the runtime through this adapter as the host
side of the evidence contract: start a bounded investigation, run real
read-only host operations for each evidence request, and attest every result
with an honest host_receipt so the runtime binds observations to actual host
work. The adapter never fabricates an outcome it cannot run: web search
without a configured engine yields an error receipt, and empty tool output is
never certified as an absence.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shlex
import subprocess
import urllib.parse
from collections import namedtuple
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cortheon.cognitive_mcp_core.server import CortheonMcpServer
from cortheon.omp_core.web import open_web_url

_HostResult = namedtuple(
    "HostResult",
    ("content", "kind", "receipt", "url", "retrieved_at", "purpose"),
    defaults=("", None, None, None, None, None),
)

_BOUND = 50_000
_WEB_BOUND = 60_000
_OUTPUT_BOUND = 8_000
_MAX_WALK_FILES = 400


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _bounded(text: str) -> str:
    text = text.strip()
    return (
        text[:_OUTPUT_BOUND] + "\n[...host output bounded...]"
        if len(text) > _OUTPUT_BOUND
        else text
    )


class OmpHost:
    """Agent-facing host for the Cortheon runtime over its MCP protocol."""

    def __init__(
        self,
        runtime: Any | None = None,
        *,
        root: str | None = None,
        allow_private_network: bool = False,
    ) -> None:
        self._server = CortheonMcpServer(runtime=runtime)
        self.root = Path(root or Path.cwd()).resolve()
        self._allow_private_network = allow_private_network
        self._request_id = 0
        self.payload: dict[str, Any] | None = None
        self.session_id: str | None = None

    # Protocol plumbing -----------------------------------------------------

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._request_id += 1
        response = self._server.handle(
            {"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": params}
        )
        if not isinstance(response, dict):
            raise ValueError(f"missing response for {method}")
        if "error" in response:
            raise ValueError(str(response["error"].get("message") or response["error"]))
        result = response.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"malformed result for {method}: {result!r}")
        return result

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._call("tools/call", {"name": name, "arguments": arguments})
        payload = result.get("structuredContent")
        if not isinstance(payload, dict):
            raise ValueError(f"tool {name} returned no structured payload")
        return payload

    def tools(self) -> list[dict[str, Any]]:
        return self._call("tools/list", {})["tools"]

    def start(self, goal: str, **kwargs: Any) -> dict[str, Any]:
        self.payload = self._call_tool("cortheon_start", {"goal": goal, **kwargs})
        session = self.payload.get("session")
        if isinstance(session, dict):
            self.session_id = str(session.get("session_id") or "")
        return self.payload

    def _pending(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        action = payload.get("next_action")
        if not isinstance(action, dict):
            return None
        request = action.get("request")
        return request if isinstance(request, dict) else None

    # Receipted read-only host operations -----------------------------------

    def _receipt(self, tool: str, outcome: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"tool": tool, "outcome": outcome, "args": args}

    def read(self, path: str) -> _HostResult:
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root) or not target.is_file():
            return _HostResult(
                f"No such file: {path}", "code", self._receipt("read", "error", {"filePath": path})
            )
        data = target.read_bytes()[:_BOUND]
        return _HostResult(
            data.decode("utf-8", "replace"),
            "code",
            self._receipt("read", "result", {"filePath": path}),
        )

    def grep(self, pattern: str, path: str = ".", **kwargs: Any) -> _HostResult:
        flags = kwargs.get("flags", "")
        args: dict[str, Any] = {"pattern": pattern, "path": path}
        if flags:
            args["flags"] = flags
        if len(pattern) > 300 or re.search(r"(?<!\\)[*+{]", pattern):
            return _HostResult(
                "Unbounded or oversized regular expressions are not allowed",
                "command",
                self._receipt("grep", "error", args),
            )
        try:
            regex = re.compile(pattern, re.IGNORECASE if "i" in flags else 0)
        except re.error:
            return _HostResult(
                "Invalid regular expression",
                "command",
                self._receipt("grep", "error", args),
            )
        matches: list[str] = []
        base = (self.root / path).resolve()
        if not base.is_relative_to(self.root) or not base.exists():
            return _HostResult(
                "Missing search path or path outside the host root",
                "command",
                self._receipt("grep", "error", args),
            )
        searched = 0
        targets = [base] if base.is_file() else sorted(base.rglob("*"))
        for candidate in targets:
            resolved = candidate.resolve()
            if (
                not resolved.is_relative_to(self.root)
                or not resolved.is_file()
                or searched >= _MAX_WALK_FILES
            ):
                continue
            searched += 1
            try:
                lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, 1):
                if regex.search(line):
                    matches.append(f"{candidate.relative_to(self.root)}:{number}: {line}")
                    if len(matches) >= 40:
                        break
            if len(matches) >= 40:
                break
        outcome = "match" if matches else "no_match"
        content = "\n".join(matches) if matches else "No matches found."
        return _HostResult(content, "code", self._receipt("grep", outcome, args))

    def glob(self, pattern: str) -> _HostResult:
        matches: list[str] = []
        for candidate in sorted(self.root.rglob("*")):
            if candidate.is_file() and fnmatch.fnmatch(
                candidate.relative_to(self.root).as_posix(), pattern
            ):
                matches.append(candidate.relative_to(self.root).as_posix())
                if len(matches) >= 100:
                    break
        content = "\n".join(matches) if matches else "No files matched."
        return _HostResult(
            content,
            "code",
            self._receipt("glob", "result" if matches else "no_match", {"pattern": pattern}),
        )

    def shell(self, command: str) -> _HostResult:
        return _HostResult(
            "Generic shell execution is disabled; use the bounded native host operations",
            "command",
            self._receipt("bash", "error", {"command": command}),
        )

    def webfetch(self, url: str) -> _HostResult:
        try:
            with open_web_url(
                url,
                allow_private_network=self._allow_private_network,
                timeout=8,
            ) as response:
                data = response.read(_WEB_BOUND + 1)
        except (OSError, ValueError) as exc:
            return _HostResult(str(exc), "web", self._receipt("webfetch", "error", {"url": url}))
        if len(data) > _WEB_BOUND:
            data = data[:_WEB_BOUND]
        return _HostResult(
            _bounded(data.decode("utf-8", "replace")),
            "web",
            self._receipt("webfetch", "result", {"url": url}),
            url=url,
            retrieved_at=_now(),
        )

    def websearch(self, query: str) -> _HostResult:
        engine = os.environ.get("CORTHEON_WEBSEARCH_URL")
        if not engine:
            return _HostResult(
                "No websearch engine is configured; websearch cannot be attested.",
                "web",
                self._receipt("websearch", "error", {"query": query}),
            )
        try:
            parsed = urllib.parse.urlsplit(engine)
            query_items = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
            query_items.append(("q", query))
            url = urllib.parse.urlunsplit(
                parsed._replace(query=urllib.parse.urlencode(query_items))
            )
            with open_web_url(
                url,
                allow_private_network=self._allow_private_network,
                timeout=8,
            ) as response:
                data = response.read(_WEB_BOUND)[:_WEB_BOUND].decode("utf-8", "replace")
        except (OSError, ValueError) as exc:
            return _HostResult(
                str(exc), "web", self._receipt("websearch", "error", {"query": query})
            )
        text = _bounded(data)
        if not text:
            return _HostResult(
                "Search returned empty output",
                "web",
                self._receipt("websearch", "error", {"query": query}),
            )
        return _HostResult(text, "web", self._receipt("websearch", "result", {"query": query}))

    def test(self, command: str) -> _HostResult:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return _HostResult(
                "Unparsable command", "test", self._receipt("test", "error", {"command": command})
            )
        executable = tokens[0].rsplit("/", 1)[-1].casefold() if tokens else ""
        is_pytest = executable in {"pytest", "py.test"}
        is_python_pytest = executable in {"python", "python3"} and tokens[1:3] == ["-m", "pytest"]
        is_node_test = executable == "node" and len(tokens) >= 2 and tokens[1] == "--test"
        if not (is_pytest or is_python_pytest or is_node_test):
            return _HostResult(
                "Only pytest or node test commands are allowed",
                "test",
                self._receipt("test", "error", {"command": command}),
            )
        try:
            completed = subprocess.run(
                tokens,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _HostResult(
                str(exc), "test", self._receipt("test", "error", {"command": command})
            )
        outcome = "passed" if completed.returncode == 0 else "failed"
        content = (completed.stdout or completed.stderr or "")[:_BOUND]
        return _HostResult(
            _bounded(content), "test", self._receipt("test", outcome, {"command": command})
        )

    def diff(self) -> _HostResult:
        try:
            completed = subprocess.run(
                ["git", "-C", str(self.root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except OSError:
            return _HostResult(
                "git unavailable", "diff", self._receipt("diff", "error", {"command": "git status"})
            )
        if completed.returncode != 0:
            return _HostResult(
                completed.stderr or "git status failed",
                "diff",
                self._receipt("diff", "error", {"command": "git status"}),
            )
        changed = bool(completed.stdout.strip())
        return _HostResult(
            completed.stdout.strip() or "No changes.",
            "diff",
            self._receipt(
                "diff", "changed" if changed else "result", {"command": "git status --porcelain"}
            ),
        )

    # Request execution and observation ---------------------------------------

    def run(self, tool: str, **args: Any) -> _HostResult:
        if tool == "read":
            return self.read(str(args.get("path", "")))
        if tool == "grep":
            return self.grep(
                str(args.get("pattern", "")),
                path=str(args.get("path", ".")),
                flags=args.get("flags", ""),
            )
        if tool == "glob":
            return self.glob(str(args.get("pattern", "")))
        if tool == "shell":
            return self.shell(str(args.get("command", "")))
        if tool == "bash":
            return self.shell(str(args.get("command", "")))
        if tool in {"webfetch", "fetch"}:
            return self.webfetch(str(args.get("url", "")))
        if tool == "websearch":
            return self.websearch(str(args.get("query", "")))
        if tool == "test":
            return self.test(str(args.get("command", "")))
        if tool == "diff":
            return self.diff()
        raise ValueError(f"omp host: unsupported host tool {tool!r}")

    def _capability_tool(self, capability: str, purpose: Any) -> str:
        if capability in {"diff", "test"}:
            return capability
        web_purposes = {
            "contradiction_check",
            "corroboration",
            "discovery",
            "freshness_check",
            "primary_fetch",
            "scholarly_validation",
            "implementation_reference",
        }
        if capability in {"search", "search_or_fetch"} and purpose in web_purposes:
            return "websearch" if capability == "search" else "webfetch"
        if capability in {"read", "read_many", "inspect", "search_or_read"}:
            return "read"
        if capability == "search":
            return "grep"
        return "shell"

    def execute_next(self, *, auto: bool = False) -> dict[str, Any]:
        """Return the pending request and suggested tool, or auto-run the unambiguous ops."""
        if self.payload is None:
            raise ValueError("omp host: start() an investigation first")
        action = self.payload.get("next_action")
        if not isinstance(action, dict) or action.get("type") != "harness_tool":
            return {"ran": False, "payload": self.payload}
        request = action.get("request")
        if not isinstance(request, dict):
            return {"ran": False, "payload": self.payload}
        capability = str(request.get("capability") or "search")
        parameters = request.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        purpose = parameters.get("purpose")
        tool = self._capability_tool(capability, purpose)
        unambiguous = tool in {"diff", "test"} or (tool == "read" and capability == "read_many")
        if not auto or not unambiguous:
            return {
                "ran": False,
                "tool": tool,
                "request_id": request.get("request_id"),
                "request": request,
                "reason": (
                    None
                    if tool in {"diff", "test", "read"}
                    else "web and code requests need an agent-chosen source"
                ),
            }
        args: dict[str, Any] = {}
        if tool == "read":
            paths = parameters.get("paths")
            args["path"] = str(
                (paths[0] if isinstance(paths, list) and paths else parameters.get("path")) or "."
            )
        elif tool == "test":
            args["command"] = str(parameters.get("command") or request.get("query") or "")
        result = self.run(tool, **args)
        if result.url and purpose:
            result = result._replace(purpose=purpose)
        return self.observe(result)

    def observe(self, result: _HostResult, **overrides: Any) -> dict[str, Any]:
        """Submit a receipted host result for the pending request."""
        if self.payload is None or self.session_id is None:
            raise ValueError("omp host: start() an investigation first")
        request = self._pending(self.payload)
        request_id = (
            str(request.get("request_id") or "") if request else overrides.pop("request_id", "")
        )
        if not request_id:
            raise ValueError("omp host: no pending request to observe")
        observation: dict[str, Any] = {
            "kind": overrides.pop("kind", result.kind),
            "content": overrides.pop("content", result.content),
            "source": overrides.pop(
                "source", f"omp:{overrides.pop('tool', result.receipt['tool'])}"
            ),
            "host_receipt": result.receipt,
        }
        if result.url:
            observation["url"] = overrides.pop("url", result.url)
        if result.retrieved_at:
            observation["retrieved_at"] = overrides.pop("retrieved_at", result.retrieved_at)
        purpose = overrides.pop("purpose", None) or (
            result.purpose if hasattr(result, "purpose") and result.purpose else None
        )
        if purpose:
            observation["purpose"] = purpose
        if overrides.get("source_record") is not None:
            observation["source_record"] = overrides.pop("source_record")
        if overrides:
            observation.update(overrides)
        self.payload = self._call_tool(
            "cortheon_observe",
            {
                "session_id": self.session_id,
                "request_id": request_id,
                "observations": [observation],
            },
        )
        return self.payload

    def _lifecycle(self, name: str, **arguments: Any) -> dict[str, Any]:
        if self.session_id is None:
            raise ValueError("omp host: start() an investigation first")
        self.payload = self._call_tool(name, {"session_id": self.session_id, **arguments})
        return self.payload

    def complete(self, **arguments: Any) -> dict[str, Any]:
        return self._lifecycle("cortheon_complete", **arguments)

    def verify(self, **arguments: Any) -> dict[str, Any]:
        return self._lifecycle("cortheon_verify", **arguments)

    def step(self, **arguments: Any) -> dict[str, Any]:
        return self._lifecycle("cortheon_step", **arguments)

    def retract(self, evidence_ids: list[str], *, reason: str = "") -> dict[str, Any]:
        return self._call_tool(
            "cortheon_retract",
            {"session_id": self.session_id, "evidence_ids": evidence_ids, "reason": reason},
        )

    def resume(self, *, limit: int = 3) -> dict[str, Any]:
        return self._call_tool("cortheon_resume", {"limit": limit})

"""Evaluator-owned MCP client that prevents model-authored evidence receipts."""

from __future__ import annotations

import json
from typing import Any

from cortheon.benchmark_core.generic_mcp_runtime_projection import RuntimeProjectionMixin
from cortheon.benchmark_core.generic_mcp_tools import ToolExecution
from cortheon.cognitive_core.tasks import (
    _is_contradiction_revision_goal,
    _is_cross_source_derivation_goal,
    _is_discriminating_test_design_goal,
    _is_hypothesis_design_goal,
)
from cortheon.cognitive_mcp import CortheonMcpServer
from cortheon.cognitive_runtime import CognitiveRuntimeError

_HOST_RECEIPT_TOOLS = {
    "host_read": "read",
    "host_read_many": "read",
    "host_search": "search",
    "host_diff": "diff",
    "host_test": "test",
    "host_web_search": "websearch",
    "host_web_fetch": "webfetch",
}


def adapter_profile(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    return {
        **profile,
        "adapter_receipt": {
            "schema_version": 1,
            "host": "generic_mcp",
            "control_transport": "fd",
            "config_sha256": profile["config_sha256"],
            "nonce": profile["nonce"],
            "operators": dict(profile["config"]["operators"]),
        },
    }


class EvaluatorMcpRuntime(RuntimeProjectionMixin):
    """Auto-start and auto-observe through MCP while the model only reasons."""

    def __init__(
        self,
        evaluation_profile: dict[str, Any] | None,
        *,
        resource_paths: tuple[str, ...] = (),
    ) -> None:
        if len(resource_paths) > 16 or any(
            not isinstance(path, str)
            or not path
            or len(path) > 240
            or path.startswith("/")
            or ".." in path.split("/")
            for path in resource_paths
        ):
            raise ValueError("generic MCP resource scope is invalid")
        self.evaluation_profile = adapter_profile(evaluation_profile)
        self.resource_paths = tuple(dict.fromkeys(resource_paths))
        self.server = CortheonMcpServer(
            advanced=True,
            evaluation_profile=self.evaluation_profile,
        )
        self.session_id: str | None = None
        self.next_action: dict[str, Any] | None = None
        self.closed = False
        self.hypothesis_design = False
        self.discrimination_design = False
        self.derivation_design = False
        self.revision_design = False
        self._call_id = 0

    def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._call_id += 1
        response = self.server.handle(
            {
                "jsonrpc": "2.0",
                "id": self._call_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        if not isinstance(response, dict):
            raise RuntimeError("MCP server returned no response")
        if "error" in response:
            message = (
                response["error"].get("message") if isinstance(response["error"], dict) else None
            )
            raise RuntimeError(str(message or "MCP request failed"))
        result = response.get("result")
        payload = result.get("structuredContent") if isinstance(result, dict) else None
        if not isinstance(payload, dict):
            raise RuntimeError("MCP result lacked structured content")
        return payload

    def start(self, goal: str, *, task_kind: str = "auto") -> dict[str, Any]:
        if self.session_id is not None:
            raise RuntimeError("generic MCP task already started")
        self.hypothesis_design = _is_hypothesis_design_goal(goal)
        self.discrimination_design = _is_discriminating_test_design_goal(goal)
        self.derivation_design = _is_cross_source_derivation_goal(goal)
        self.revision_design = _is_contradiction_revision_goal(goal)
        payload = self._call(
            "cortheon_start",
            {"goal": goal, "task_kind": task_kind, "strictness": "standard"},
        )
        session = payload.get("session")
        session_id = session.get("session_id") if isinstance(session, dict) else None
        if not isinstance(session_id, str):
            raise RuntimeError("MCP start returned no session id")
        self.session_id = session_id
        self.next_action = payload.get("next_action")
        return payload

    def _pending_request(self) -> dict[str, Any] | None:
        action = self.next_action
        request = action.get("request") if isinstance(action, dict) else None
        return request if isinstance(request, dict) else None

    def observe(self, execution: ToolExecution) -> dict[str, Any]:
        if self.session_id is None or self.closed:
            raise RuntimeError("generic MCP runtime is not active")
        request = self._pending_request()
        if request is None:
            raise RuntimeError("runtime did not request host evidence")
        if execution.request.name not in self.allowed_host_tools():
            raise RuntimeError("host tool does not satisfy the current runtime request")
        request_id = request.get("request_id")
        if not isinstance(request_id, str):
            raise RuntimeError("runtime request has no request id")
        observations = self._observations(execution, request)
        payload = self._call(
            "cortheon_observe",
            {
                "session_id": self.session_id,
                "request_id": request_id,
                "observations": observations,
            },
        )
        self.next_action = payload.get("next_action")
        return payload

    def _observations(
        self,
        execution: ToolExecution,
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        name = execution.request.name
        if name == "host_read_many":
            return self._read_many_observations(execution)
        resource_path = execution.request.arguments.get("path")
        is_bound_resource = bool(
            name == "host_read"
            and isinstance(resource_path, str)
            and resource_path in self.resource_paths
        )
        kind = {
            "host_diff": "diff",
            "host_test": "test",
            "host_web_search": "web",
            "host_web_fetch": "web",
        }.get(name, "artifact" if is_bound_resource else "code")
        capability = str(request.get("capability") or "")
        runtime_receipt = {
            **execution.receipt,
            "tool": _HOST_RECEIPT_TOOLS.get(name, execution.receipt["tool"]),
        }
        if capability == "grep" and name == "host_search":
            runtime_receipt["tool"] = "grep"
        content = execution.content
        if capability == "grep" and name == "host_read":
            parameters = request.get("parameters")
            params = parameters if isinstance(parameters, dict) else {}
            pattern = params.get("pattern")
            path = params.get("path")
            if not isinstance(pattern, str) or not isinstance(path, str):
                raise RuntimeError("grep-compatible read request lacked exact parameters")
            matches = [
                f"{path}:{number}:{line}"
                for number, line in enumerate(content.splitlines(), 1)
                if pattern in line
            ]
            outcome = "match" if matches else "no_match"
            content = "\n".join(matches) if matches else "No matches."
            runtime_receipt.update(
                tool="grep",
                outcome=outcome,
                args={"pattern": pattern, "path": path},
            )
        base = {
            "kind": kind,
            "content": content,
            "status": "failed" if execution.status in {"failed", "error"} else "observed",
            "host_receipt": runtime_receipt,
            **({"source": resource_path} if is_bound_resource else {}),
        }
        if kind != "web":
            return [base]
        try:
            value = json.loads(execution.content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("structured web result was invalid") from exc
        results = value.get("results") if isinstance(value, dict) else None
        if not isinstance(results, list):
            raise RuntimeError("structured web result list is missing")
        purpose = request.get("parameters", {}).get("purpose", "discovery")
        observations = []
        for item in results:
            if not isinstance(item, dict):
                raise RuntimeError("structured web item is invalid")
            observations.append(
                {
                    **base,
                    "content": str(item.get("content", "")),
                    "source": item.get("url"),
                    "url": item.get("url"),
                    "retrieved_at": item.get("retrieved_at"),
                    **({"published_at": item["published_at"]} if item.get("published_at") else {}),
                    "purpose": purpose,
                }
            )
        return observations

    @staticmethod
    def _read_many_observations(execution: ToolExecution) -> list[dict[str, Any]]:
        requested = execution.request.arguments.get("paths")
        if (
            not isinstance(requested, list)
            or not requested
            or any(not isinstance(path, str) for path in requested)
        ):
            raise RuntimeError("read_many execution lacked exact paths")
        failed = execution.status in {"failed", "error"}
        if failed:
            files = [{"path": path, "content": execution.content} for path in requested]
        else:
            try:
                payload = json.loads(execution.content)
            except json.JSONDecodeError as exc:
                raise RuntimeError("structured read_many result was invalid") from exc
            files = payload.get("files") if isinstance(payload, dict) else None
            if (
                not isinstance(files, list)
                or [item.get("path") for item in files if isinstance(item, dict)] != requested
                or any(
                    not isinstance(item, dict) or not isinstance(item.get("content"), str)
                    for item in files
                )
            ):
                raise RuntimeError("structured read_many result did not match requested paths")
        observations = []
        for item in files:
            path = item["path"]
            observations.append(
                {
                    "kind": "code",
                    "content": item["content"] or "[empty file]",
                    "status": "failed" if failed else "observed",
                    "source": path,
                    "host_receipt": {
                        **execution.receipt,
                        "tool": "read",
                        "args": {"filePath": path},
                    },
                }
            )
        return observations

    def lifecycle_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in {
            "cortheon_complete",
            "cortheon_step",
            "cortheon_retract",
            "cortheon_abandon",
        }:
            raise ValueError("model requested a wrapper-owned MCP tool")
        if self.session_id is not None:
            supplied = arguments.get("session_id")
            if supplied is not None and supplied != self.session_id:
                raise ValueError("MCP lifecycle call targets another session")
            arguments = {**arguments, "session_id": self.session_id}
        payload = self._call(name, arguments)
        self.next_action = payload.get("next_action")
        if name == "cortheon_abandon" or payload.get("status") == "complete":
            self.closed = True
        return payload

    def abandon(self) -> bool:
        if self.session_id is None or self.closed:
            return True
        try:
            self.lifecycle_call("cortheon_abandon", {})
        except (RuntimeError, ValueError, CognitiveRuntimeError):
            try:
                self.server.runtime.finish(self.session_id, mode="abandon")
            except (RuntimeError, ValueError, CognitiveRuntimeError):
                return False
            self.next_action = None
            self.closed = True
        return self.closed

    def evaluation_receipt(self) -> dict[str, Any] | None:
        profile = self.evaluation_profile
        if profile is None:
            return None
        return self.server.runtime.consume_evaluation_receipt(profile["nonce"])

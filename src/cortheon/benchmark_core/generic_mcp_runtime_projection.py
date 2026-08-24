"""Host-tool and reasoning projections for the evaluator MCP runtime."""

from __future__ import annotations

from typing import Any

from cortheon.benchmark_core.generic_mcp_capabilities import PREFERRED_CAPABILITY_TOOL
from cortheon.benchmark_core.generic_mcp_search_projection import discovery_pattern

_CAPABILITY_TOOLS = {
    "grep": {"host_search", "host_read"},
    "search": {"host_search", "host_web_search"},
    "fetch": {"host_web_fetch"},
    "inspect": {"host_read", "host_read_many"},
    "read": {"host_read"},
    "read_many": {"host_read", "host_read_many"},
    "diff": {"host_diff"},
    "test": {"host_test"},
    "websearch": {"host_web_search"},
    "webfetch": {"host_web_fetch"},
}


class RuntimeProjectionMixin:
    """Project runtime actions onto the evaluator-owned host catalogue."""

    evaluation_profile: dict[str, Any] | None
    resource_paths: tuple[str, ...]
    next_action: dict[str, Any] | None
    hypothesis_design: bool
    discrimination_design: bool
    derivation_design: bool
    revision_design: bool

    def _pending_request(self) -> dict[str, Any] | None:
        raise NotImplementedError

    def allowed_host_tools(self) -> frozenset[str]:
        request = self._pending_request()
        capability = request.get("capability") if request else None
        selected = set(_CAPABILITY_TOOLS.get(str(capability), ()))
        parameters = request.get("parameters") if isinstance(request, dict) else None
        if capability == "search" and isinstance(parameters, dict):
            selected &= {"host_web_search"} if "purpose" in parameters else {"host_search"}
        return frozenset(selected)

    def projected_host_tool(self) -> str | None:
        allowed = self.allowed_host_tools()
        request = self._pending_request()
        capability = str(request.get("capability") or "") if request else ""
        preferred = PREFERRED_CAPABILITY_TOOL.get(capability)
        return preferred if preferred in allowed else (min(allowed) if allowed else None)

    def projects_hypothesis_reasoning(self) -> bool:
        action = self.next_action
        return bool(
            isinstance(action, dict)
            and self.hypothesis_design
            and action.get("type") == "reason"
            and action.get("submit_via") == "cortheon_step"
            and action.get("required_fields") == ["hypotheses"]
            and self.evaluation_profile is not None
            and self.evaluation_profile["config"]["operators"]["hypothesis_framing"] is True
        )

    def projects_revision_reasoning(self) -> bool:
        action = self.next_action
        return bool(
            isinstance(action, dict)
            and self.revision_design
            and action.get("type") == "reason"
            and action.get("submit_via") == "cortheon_step"
            and action.get("required_fields") == ["draft"]
            and self.evaluation_profile is not None
            and self.evaluation_profile["config"]["operators"]["contradiction_revision"] is True
        )

    def projects_discrimination_reasoning(self) -> bool:
        return self._projects_operator_draft("discrimination_design", "discriminating_evidence")

    def projects_derivation_reasoning(self) -> bool:
        return self._projects_operator_draft("derivation_design", "cross_source_derivation")

    def _projects_operator_draft(self, mode: str, operator: str) -> bool:
        return bool(
            getattr(self, mode, False)
            and self.projects_repair_reasoning()
            and self.evaluation_profile is not None
            and self.evaluation_profile["config"]["operators"][operator] is True
        )

    def projects_repair_reasoning(self) -> bool:
        action = self.next_action
        return bool(
            isinstance(action, dict)
            and action.get("type") == "reason"
            and action.get("required_fields") == ["draft"]
            and action.get("submit_via") == "cortheon_step"
            and not self.projects_hypothesis_reasoning()
            and not self.projects_revision_reasoning()
        )

    def projects_answer_repair(self) -> bool:
        action = self.next_action
        return bool(
            isinstance(action, dict)
            and action.get("type") == "reason"
            and action.get("submit_via") == "cortheon_step"
            and action.get("required_fields") == ["draft"]
        )

    def projected_arguments(self, name: str) -> dict[str, Any] | None:
        request = self._pending_request()
        if request is None or name not in self.allowed_host_tools():
            return None
        raw_parameters = request.get("parameters")
        parameters = raw_parameters if isinstance(raw_parameters, dict) else {}
        if name == "host_search" and all(
            isinstance(parameters.get(key), str) for key in ("pattern", "path")
        ):
            return {"pattern": parameters["pattern"], "path": parameters["path"]}
        if name == "host_search" and isinstance(request.get("query"), str):
            pattern = discovery_pattern(request["query"])
            if pattern is not None:
                return {"pattern": pattern, "path": "."}
        if name == "host_read":
            path = parameters.get("path")
            if not isinstance(path, str) and len(self.resource_paths) == 1:
                path = self.resource_paths[0]
            if isinstance(path, str):
                return {"path": path}
        if name == "host_read_many" and isinstance(parameters.get("paths"), list):
            return {"paths": parameters["paths"]}
        if name == "host_web_search" and isinstance(request.get("query"), str):
            return {"query": request["query"]}
        if name == "host_web_fetch" and isinstance(parameters.get("url"), str):
            return {"url": parameters["url"]}
        if name == "host_diff" and isinstance(parameters.get("paths"), list):
            return {"paths": parameters["paths"]}
        return None

    def validate_host_arguments(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        test_command: list[str] | None = None,
    ) -> bool:
        request = self._pending_request()
        if request is None or name not in self.allowed_host_tools():
            return False
        projected = self.projected_arguments(name)
        if projected is not None and arguments == projected:
            return True
        parameters = request.get("parameters")
        params = parameters if isinstance(parameters, dict) else {}
        capability = str(request.get("capability") or "")
        if capability == "grep":
            expected = {"pattern": params.get("pattern"), "path": params.get("path")}
            return arguments == (
                expected if name == "host_search" else {"path": params.get("path")}
            )
        if name == "host_search":
            pattern = arguments.get("pattern")
            return bool(
                set(arguments) == {"pattern", "path"}
                and arguments.get("path") == "."
                and isinstance(pattern, str)
                and len(pattern) >= 3
                and pattern.casefold() in str(request.get("query", "")).casefold()
            )
        if name == "host_web_search":
            return arguments == {"query": request.get("query")}
        if name == "host_web_fetch":
            allowed = (
                {params.get("url"), *(params.get("urls") or [])}
                if isinstance(params.get("urls", []), list)
                else {params.get("url")}
            )
            return set(arguments) == {"url"} and arguments.get("url") in allowed - {None}
        if name == "host_read":
            expected_path = params.get("path")
            if not isinstance(expected_path, str):
                expected_path = (
                    arguments.get("path") if arguments.get("path") in self.resource_paths else None
                )
            return bool(isinstance(expected_path, str) and arguments == {"path": expected_path})
        if name == "host_read_many":
            paths = arguments.get("paths")
            allowed_paths = params.get("paths")
            return bool(
                set(arguments) == {"paths"}
                and isinstance(paths, list)
                and paths
                and isinstance(allowed_paths, list)
                and set(paths) <= set(allowed_paths)
            )
        if name == "host_diff":
            expected = params.get("paths") or (
                [params["path"]] if isinstance(params.get("path"), str) else None
            )
            return arguments == {"paths": expected}
        return (
            name == "host_test"
            and set(arguments) == {"test_id"}
            and test_command == params.get("command")
        )

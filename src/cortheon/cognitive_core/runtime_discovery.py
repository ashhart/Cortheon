"""DiscoveryMixin for CognitiveRuntime."""

from __future__ import annotations

from typing import Any

from cortheon.cognitive_core.models import Investigation
from cortheon.cognitive_core.runtime_state import RuntimeState
from cortheon.cognitive_core.semantic_graph import _keywords
from cortheon.cognitive_core.tasks import (
    _CODE_PATH_RE,
    _DOCUMENT_PATH_RE,
    _discovered_project_paths,
    _goal_code_symbols,
    _is_test_path,
)


class DiscoveryMixin(RuntimeState):
    """Discovery responsibilities of CognitiveRuntime."""

    def _document_discovery_response(
        self,
        session: Investigation,
    ) -> dict[str, Any] | None:
        discovery_requests = [
            request
            for request in session.requests.values()
            if request.parameters.get("operation") == "document_discovery"
        ]
        if not discovery_requests or any(
            request.parameters.get("operation") == "semantic_join"
            for request in session.requests.values()
        ):
            return None
        if any(request.status == "pending" for request in discovery_requests):
            return None

        maximum = max(
            2,
            min(
                6,
                int(discovery_requests[-1].parameters.get("max_candidates", 6)),
            ),
        )
        paths = _discovered_project_paths(
            session.goal,
            (
                observation
                for observation in session.observations.values()
                if observation.status != "failed" and not observation.quarantine_flags
            ),
            pattern=_DOCUMENT_PATH_RE,
            maximum=maximum,
        )
        if len(paths) >= 2:
            request = self._create_request(
                session,
                capability="read_many",
                query=(
                    "Read these bounded live candidates and preserve their source "
                    f"boundaries so Cortheon can test the bridge: {', '.join(paths)}. "
                    f"Question: {session.goal}"
                ),
                reason=(
                    "The live search exposed a bounded candidate set; focused reads "
                    "are more informative than another broad search."
                ),
                success_condition=(
                    "Return focused host-read excerpts from each candidate. Unrelated "
                    "documents may remain unused; do not force them into the conclusion."
                ),
                parameters={
                    "paths": paths,
                    "operation": "semantic_join",
                    "discovered": True,
                    "tool_call_budget": min(
                        len(paths),
                        session.profile.max_calls_per_request,
                    ),
                },
            )
            session.phase = "investigating"
            return self._payload(
                session,
                next_action=self._execute_action(request),
                guidance=(
                    "The substrate narrowed discovery to live candidate documents. "
                    "Read them through the host; synthesis will use only a supported "
                    "multi-source path."
                ),
            )

        if len(discovery_requests) < session.strictness.max_request_attempts:
            terms = sorted(_keywords(session.goal))[:12]
            request = self._create_request(
                session,
                capability="search",
                query=(
                    "The first scoped search did not expose at least two document "
                    "candidates. Run one differently framed project-document search "
                    f"using these concepts: {', '.join(terms)}. Goal: {session.goal}"
                ),
                reason=(
                    "Cross-source synthesis needs at least two independently read "
                    "documents, but the first discovery result was insufficient."
                ),
                success_condition=(
                    "Return new project-relative document paths and focused matching "
                    "lines, or an explicit scoped no-match result."
                ),
                parameters={
                    "operation": "document_discovery",
                    "extensions": ["md", "markdown", "rst", "txt"],
                    "max_candidates": maximum,
                    "discovery_round": len(discovery_requests) + 1,
                    "tool_call_budget": 1,
                },
            )
            session.phase = "investigating"
            return self._payload(
                session,
                next_action=self._execute_action(request),
                guidance=(
                    "Reframe the search once. Do not repeat an unchanged query or invent paths."
                ),
            )

        session.phase = "inconclusive"
        return self._payload(
            session,
            next_action={
                "type": "finish",
                "instruction": (
                    "Bounded project-document discovery found fewer than two usable "
                    "sources. Report the scoped null result and abandon; do not loop "
                    "or manufacture a cross-source conclusion."
                ),
                "submit_via": "cortheon_finish",
            },
            guidance=(
                "Discovery exhausted its bounded alternate query. A scoped null is "
                "the verified outcome."
            ),
        )

    def _code_discovery_response(
        self,
        session: Investigation,
    ) -> dict[str, Any] | None:
        discovery_requests = [
            request
            for request in session.requests.values()
            if request.parameters.get("operation") == "code_discovery"
        ]
        if not discovery_requests or any(
            request.parameters.get("operation") == "code_context"
            for request in session.requests.values()
        ):
            return None
        if any(request.status == "pending" for request in discovery_requests):
            return None

        latest = discovery_requests[-1]
        maximum = max(
            2,
            min(6, int(latest.parameters.get("max_candidates", 6))),
        )
        paths = _discovered_project_paths(
            session.goal,
            (
                observation
                for observation in session.observations.values()
                if observation.status != "failed" and not observation.quarantine_flags
            ),
            pattern=_CODE_PATH_RE,
            maximum=maximum,
        )
        prefer_tests = bool(latest.parameters.get("prefer_tests"))
        if prefer_tests:
            tests = [path for path in paths if _is_test_path(path)]
            implementations = [path for path in paths if not _is_test_path(path)]
            if tests and implementations:
                paths = [
                    implementations[0],
                    tests[0],
                    *(path for path in paths if path not in {implementations[0], tests[0]}),
                ][:maximum]
        minimum = 2 if prefer_tests else 1
        if len(paths) >= minimum:
            request = self._create_request(
                session,
                capability="read_many",
                query=(
                    "Read this bounded live code surface and preserve file boundaries: "
                    f"{', '.join(paths)}. Goal: {session.goal}"
                ),
                reason=(
                    "The live search identified a focused implementation/test surface; "
                    "reading it is more informative than another repository search."
                ),
                success_condition=(
                    "Return focused host-read excerpts from each candidate, including "
                    "the implementation behavior and relevant test or caller boundary."
                ),
                parameters={
                    "paths": paths,
                    "symbols": _goal_code_symbols(session.goal)[:12],
                    "operation": "code_context",
                    "discovered": True,
                    "tool_call_budget": min(
                        len(paths),
                        session.profile.max_calls_per_request,
                    ),
                },
            )
            session.phase = "investigating"
            return self._payload(
                session,
                next_action=self._execute_action(request),
                guidance=(
                    "The substrate narrowed the live code surface. Read it before "
                    "reasoning or editing; unrelated candidates need not shape the fix."
                ),
            )

        if len(discovery_requests) < session.strictness.max_request_attempts:
            terms = sorted(_keywords(session.goal))[:12]
            request = self._create_request(
                session,
                capability="search",
                query=(
                    "The first scoped code search did not expose the required "
                    "implementation/test surface. Run one differently framed search "
                    f"using these concepts: {', '.join(terms)}. Goal: {session.goal}"
                ),
                reason=(
                    "A verified code change needs a live implementation and test or "
                    "observable boundary; the first result was insufficient."
                ),
                success_condition=(
                    "Return new project-relative code paths with focused matching lines, "
                    "or an explicit scoped no-match result."
                ),
                parameters={
                    "operation": "code_discovery",
                    "max_candidates": maximum,
                    "discovery_round": len(discovery_requests) + 1,
                    "prefer_tests": prefer_tests,
                    "tool_call_budget": 1,
                },
            )
            session.phase = "investigating"
            return self._payload(
                session,
                next_action=self._execute_action(request),
                guidance=(
                    "Reframe the code search once. Do not repeat an unchanged query, "
                    "edit speculatively, or invent paths."
                ),
            )

        session.phase = "inconclusive"
        return self._payload(
            session,
            next_action={
                "type": "finish",
                "instruction": (
                    "Bounded code discovery did not find the required live surface. "
                    "Report the scoped null result and abandon; do not roam, edit "
                    "speculatively, or loop."
                ),
                "submit_via": "cortheon_finish",
            },
            guidance=(
                "Code discovery exhausted its bounded alternate query. No change is "
                "safer than an ungrounded patch."
            ),
        )

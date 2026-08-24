"""Lean environment and frontier-information request sequence."""

from __future__ import annotations

from typing import Any

from cortheon.cognitive_core.frontier_policy import (
    SOURCE_QUALITY_SIGNALS,
    needs_scholarly_sources,
    source_classes,
)
from cortheon.cognitive_core.models import EvidenceRequest, Investigation
from cortheon.cognitive_core.runtime_state import RuntimeState
from cortheon.cognitive_core.tasks import _goal_code_paths, _goal_code_symbols


class FrontierGroundingMixin(RuntimeState):
    """Coordinate host-owned local grounding and current external discovery."""

    def _environment_grounding_request(self, session: Investigation) -> EvidenceRequest:
        return self._create_request(
            session,
            capability="inspect",
            query=(
                "Establish the exact live execution environment before choosing an external "
                "approach. Inspect project manifests, lockfiles, toolchain declarations, "
                "installed runtime versions, pinned dependency versions, enabled features, "
                "and the available API/import surface relevant to this task. Use only "
                f"read-only host tools. Task: {session.goal}"
            ),
            reason=(
                "Current external guidance is useful only after its runtime and API "
                "assumptions can be matched to the live project."
            ),
            success_condition=(
                "Return focused host-receipted facts for the runtime, dependency pins, "
                "manifests or lockfiles, and relevant available APIs. Explicitly mark any "
                "version that could not be established."
            ),
            parameters={
                "operation": "environment_grounding",
                "required_facts": [
                    "runtime_versions",
                    "dependency_versions",
                    "manifests_and_lockfiles",
                    "available_api_surface",
                ],
                "read_only": True,
                "tool_call_budget": min(5, session.profile.max_calls_per_request),
            },
        )

    def _frontier_grounding_response(
        self,
        session: Investigation,
    ) -> dict[str, Any] | None:
        operations = {
            str(request.parameters.get("operation")): request
            for request in session.requests.values()
            if request.parameters.get("operation")
        }
        environment = operations.get("environment_grounding")
        if environment is None:
            request = self._environment_grounding_request(session)
            session.phase = "orienting"
            return self._payload(
                session,
                next_action=self._execute_action(request),
                guidance="Ground external research in the exact live project environment.",
            )
        if environment.status != "completed":
            return None

        environment_ids = [
            item.evidence_id
            for item in session.observations.values()
            if item.kind != "web" and item.status != "failed" and not item.quarantine_flags
        ]
        discovery = operations.get("frontier_discovery")
        if discovery is None:
            request = self._create_request(
                session,
                capability="search_or_fetch",
                query=(
                    "Search the current web for knowledge that materially improves this task. "
                    "Use the grounded runtime and dependency facts to reject incompatible "
                    "examples. Prefer official documentation and releases, directly relevant "
                    "primary research when applicable, and maintained reference repositories. "
                    "For repositories, inspect version or tag fit, recent maintenance, tests, "
                    "license, and exact implementation relevance rather than stars alone. "
                    f"Task: {session.goal}"
                ),
                reason=(
                    "The model needs current techniques and reference implementations beyond "
                    "its weights, filtered against the live environment."
                ),
                success_condition=(
                    "Return attributable current results covering the strongest compatible "
                    "primary source and implementation reference, with URLs and dates when "
                    "available. Include incompatibilities and limitations instead of hiding them."
                ),
                parameters={
                    "operation": "frontier_discovery",
                    "purpose": "discovery",
                    "source_classes": source_classes(session.goal),
                    "quality_signals": list(SOURCE_QUALITY_SIGNALS),
                    "environment_evidence_ids": environment_ids,
                    "tool_call_budget": min(5, session.profile.max_calls_per_request),
                },
            )
            session.phase = "investigating"
            return self._payload(
                session,
                next_action=self._execute_action(request),
                guidance=(
                    "Use host web tools. Retrieve current knowledge, but keep every result "
                    "bound to its URL and compatibility assumptions."
                ),
            )
        if discovery.status != "completed":
            return None

        primary = operations.get("primary_source_fetch")
        if primary is None:
            request = self._create_request(
                session,
                capability="fetch",
                query=(
                    "Fetch the strongest primary source already found. Extract the exact "
                    "version assumptions, API or method, implementation details, limitations, "
                    f"and evidence that it applies to the live environment. Task: {session.goal}"
                ),
                reason="Search results are leads; the load-bearing source must be read directly.",
                success_condition=(
                    "Return one focused primary-source passage with URL, retrieval time, source "
                    "date when available, compatibility details, and explicit limitations."
                ),
                parameters={
                    "operation": "primary_source_fetch",
                    "purpose": "primary_fetch",
                    "environment_evidence_ids": environment_ids,
                    "tool_call_budget": 1,
                },
            )
            session.phase = "investigating"
            return self._payload(
                session,
                next_action=self._execute_action(request),
                guidance="Read the primary source instead of reasoning from a search snippet.",
            )
        if primary.status != "completed":
            return None

        scholarly = operations.get("scholarly_source_review")
        if needs_scholarly_sources(session.goal) and scholarly is None:
            request = self._create_request(
                session,
                capability="search_or_fetch",
                query=(
                    "Find and read the strongest directly relevant primary paper or rigorous "
                    "review. Search title and key-phrase variants across scholarly indexes; "
                    "deduplicate by DOI, arXiv ID, or title. Prefer direct relevance and sound "
                    "methodology over citation count, then authority and recency. Extract the "
                    "method, population or benchmark, result, limitations, date, DOI or stable "
                    f"URL, and exact transfer conditions. Task: {session.goal}"
                ),
                reason=(
                    "Scientific claims need a source-specific check of methods and transfer "
                    "conditions, not a search-result summary."
                ),
                success_condition=(
                    "Return one focused primary-paper or rigorous-review record with title, "
                    "authors or venue, date, DOI or stable URL, method, result, limitations, and "
                    "relevance to the live task; otherwise return a scoped no-relevant-paper result."
                ),
                parameters={
                    "operation": "scholarly_source_review",
                    "purpose": "scholarly_validation",
                    "deduplicate_by": ["doi", "arxiv_id", "normalized_title"],
                    "rank_by": ["direct_relevance", "method_quality", "authority", "recency"],
                    "environment_evidence_ids": environment_ids,
                    "tool_call_budget": min(3, session.profile.max_calls_per_request),
                },
            )
            session.phase = "investigating"
            return self._payload(
                session,
                next_action=self._execute_action(request),
                guidance="Read the paper's method and limitations, not only its abstract or citations.",
            )
        if scholarly is not None and scholarly.status != "completed":
            return None

        repository = operations.get("repository_source_review")
        if repository is None:
            request = self._create_request(
                session,
                capability="search_or_fetch",
                query=(
                    "Find and inspect a directly relevant maintained GitHub implementation. "
                    "Check the actual repository, not only its search card: release or tag and "
                    "version fit, recent maintenance, archived status, license, language and "
                    "dependency fit, tests and CI, documentation, and the exact source and test "
                    "files implementing the technique. Stars are only a discovery hint. Return a "
                    f"scoped null if no repository is genuinely transferable. Task: {session.goal}"
                ),
                reason=(
                    "A popular repository is useful only when its maintained implementation and "
                    "constraints transfer to the live project."
                ),
                success_condition=(
                    "Return one repository URL with maintenance, release, license, test, language, "
                    "compatibility, and exact implementation-file evidence; or a scoped no-fit result."
                ),
                parameters={
                    "operation": "repository_source_review",
                    "purpose": "implementation_reference",
                    "required_signals": [
                        "direct_relevance",
                        "release_or_tag_fit",
                        "recent_maintenance",
                        "not_archived",
                        "license",
                        "language_and_dependency_fit",
                        "tests_and_ci",
                        "implementation_files",
                    ],
                    "environment_evidence_ids": environment_ids,
                    "tool_call_budget": min(3, session.profile.max_calls_per_request),
                },
            )
            session.phase = "investigating"
            return self._payload(
                session,
                next_action=self._execute_action(request),
                guidance="Inspect repository code and health; do not equate stars with evidence.",
            )
        if repository.status != "completed":
            return None

        counterevidence = operations.get("counterevidence_search")
        if counterevidence is None:
            request = self._create_request(
                session,
                capability="search_or_fetch",
                query=(
                    "Run one focused current search for credible counterevidence: incompatible "
                    "versions, superseding guidance, failed replications, security or performance "
                    "limitations, abandoned repositories, and cases where the selected approach "
                    f"does not transfer. Task: {session.goal}"
                ),
                reason=(
                    "A reference is not decision-grade until the strongest compatibility or "
                    "validity challenge has been checked."
                ),
                success_condition=(
                    "Return the strongest attributable limitation or contradiction, or a scoped "
                    "no-conflict result, with URL and current retrieval time."
                ),
                parameters={
                    "operation": "counterevidence_search",
                    "purpose": "contradiction_check",
                    "environment_evidence_ids": environment_ids,
                    "tool_call_budget": min(2, session.profile.max_calls_per_request),
                },
            )
            session.phase = "challenging"
            return self._payload(
                session,
                next_action=self._execute_action(request),
                guidance="Challenge the selected external approach before transferring it.",
            )
        if counterevidence.status != "completed":
            return None

        local_started = any(
            request.parameters.get("frontier_grounded") is True
            and request.parameters.get("operation") in {"code_context", "code_discovery"}
            for request in session.requests.values()
        )
        if local_started:
            return None

        external_ids = [
            item.evidence_id
            for item in session.observations.values()
            if item.kind == "web" and item.status != "failed" and not item.quarantine_flags
        ]
        paths = _goal_code_paths(session.goal)
        symbols = _goal_code_symbols(session.goal)
        common = {
            "frontier_grounded": True,
            "environment_evidence_ids": environment_ids,
            "external_evidence_ids": external_ids,
        }
        if paths:
            request = self._create_request(
                session,
                capability="read_many",
                query=(
                    "Read the named live code surface. Map only compatible techniques from the "
                    "accepted external evidence onto the actual implementation and tests; do not "
                    f"copy a reference blindly. Task: {session.goal}"
                ),
                reason=(
                    "The external technique must be transferred through the live project's "
                    "version, API, and behavioral constraints."
                ),
                success_condition=(
                    "Return focused implementation and test excerpts showing where the compatible "
                    "technique can be applied and which local behavior must verify it."
                ),
                parameters={
                    **common,
                    "operation": "code_context",
                    "paths": paths[:6],
                    "symbols": symbols[:12],
                    "tool_call_budget": min(len(paths), session.profile.max_calls_per_request),
                },
            )
        else:
            request = self._create_request(
                session,
                capability="search",
                query=(
                    "Search the live project for the smallest implementation, caller, and test "
                    "surface where the compatible external technique could be transferred. "
                    f"Preserve local constraints. Task: {session.goal}"
                ),
                reason=(
                    "Current external knowledge becomes useful only after it is localized to the "
                    "real implementation and observable test boundary."
                ),
                success_condition=(
                    "Return project-relative implementation and test paths with focused matching "
                    "lines. Do not edit before the local transfer surface is understood."
                ),
                parameters={
                    **common,
                    "operation": "code_discovery",
                    "max_candidates": 6,
                    "discovery_round": 1,
                    "prefer_tests": session.deliverable == "code_change",
                    "tool_call_budget": session.profile.max_calls_per_request,
                },
            )
        session.phase = "connecting"
        return self._payload(
            session,
            next_action=self._execute_action(request),
            guidance=(
                "Transfer the external method into the live project only where versions, APIs, "
                "constraints, and tests support it."
            ),
        )

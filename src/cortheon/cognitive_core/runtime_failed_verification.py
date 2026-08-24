"""FailedVerificationMixin for CognitiveRuntime."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from cortheon.cognitive_core.diffs import (
    _diff_changed_line_count,
    _diff_establishes_change,
    _diff_line_budget,
    _diff_receipt_paths,
    _diff_weakens_tests,
)
from cortheon.cognitive_core.models import EvidenceRequest, Investigation
from cortheon.cognitive_core.profiles import _capability_for_kind, _has_hint
from cortheon.cognitive_core.research_gaps import _research_completion_gaps
from cortheon.cognitive_core.runtime_state import RuntimeState
from cortheon.cognitive_core.tasks import _CROSS_SOURCE_HINTS
from cortheon.cognitive_protocol import evaluation_operator
from cortheon.cognitive_repair import (
    changed_paths_from_diff,
    is_test_path,
    protected_test_paths,
    protects_tests,
)


class FailedVerificationMixin(RuntimeState):
    def _failed_verification_action(
        self,
        session: Investigation,
        checks: list[dict[str, Any]],
        gaps: list[str],
    ) -> dict[str, Any]:
        pending = self._pending_request(session)
        if pending is not None:
            return self._execute_action(pending)
        if not evaluation_operator(session.evaluation_profile, "retrieval"):
            return {
                "type": "finish",
                "instruction": (
                    "Verification failed and this condition cannot originate evidence. "
                    "Preserve the gaps and abandon the investigation."
                ),
                "submit_via": "cortheon_finish",
            }
        failed = {item["name"] for item in checks if not item["passed"]}
        if "evidence_alignment" in failed:
            if session.deliverable == "research_answer":
                return {
                    "type": "reason",
                    "instruction": (
                        "Revise the research answer so it cites two independent accepted "
                        "URLs and explicitly addresses any detected source conflict, then "
                        "retry completion."
                    ),
                    "submit_via": "cortheon_complete",
                }
            expected_tools = (
                {"read"}
                if any(request.capability == "read_many" for request in session.requests.values())
                else {"grep"}
            )
            has_targeted_receipt = any(
                (item.host_receipt or {}).get("tool") in expected_tools
                for item in session.observations.values()
            )
            if not has_targeted_receipt:
                return self._execute_action(
                    self._create_request(
                        session,
                        capability="search",
                        query=(
                            "Run an exact, scoped search that directly tests this lookup: "
                            f"{session.goal}"
                        ),
                        reason=(
                            "The answer needs deterministic evidence aligned to the "
                            "lookup target and scope."
                        ),
                        success_condition=(
                            "Return exact matching lines or an explicit zero-match result "
                            "from the named scope."
                        ),
                    )
                )
            return {
                "type": "reason",
                "instruction": (
                    "The deterministic host result contradicts or does not support the "
                    "answer's polarity. Revise the answer and claims to match the exact "
                    "search result, then retry completion."
                ),
                "submit_via": "cortheon_complete",
            }
        if "adversarial_challenge" in failed:
            return {
                "type": "challenge",
                "instruction": (
                    "Run cortheon_challenge on the current answer and explicit claims "
                    "before trying to verify again."
                ),
                "submit_via": "cortheon_challenge",
            }
        if "evidence_quarantine" in failed:
            return {
                "type": "reason",
                "instruction": (
                    "The answer relies only on instruction-shaped evidence that Cortheon "
                    "quarantined. Gather independent clean evidence and cite it before "
                    "retrying completion."
                ),
                "submit_via": "cortheon_step",
            }
        if "claim_grounding" in failed:
            return {
                "type": "reason",
                "instruction": (
                    "Remove unsupported claims, correct unknown citations, or gather "
                    "the missing evidence through cortheon_step."
                ),
                "submit_via": "cortheon_step",
            }
        if "claim_truth_operations" in failed:
            truth_check = next(item for item in checks if item["name"] == "claim_truth_operations")
            profiles = [
                profile for profile in truth_check.get("profiles", []) if not profile.get("passed")
            ]
            first = profiles[0] if profiles else {}
            request = self._request_for_claim_truth(session, first)
            if request is not None:
                return self._execute_action(request)
            detail = "; ".join(str(item) for item in first.get("gaps", []))
            return {
                "type": "reason",
                "instruction": (
                    f"Claim {first.get('claim_index', 1)} is not established: {detail} "
                    "Perform the required truth operation: "
                    f"{first.get('next_truth_operation', 'gather direct evidence')}, "
                    "then retry completion with the new evidence id cited by that claim."
                ),
                "submit_via": "cortheon_step",
            }
        if "requirement_coverage" in failed:
            coverage_check = next(item for item in checks if item["name"] == "requirement_coverage")
            uncovered = [
                item
                for item in coverage_check.get("requirements", ())
                if item.get("status") != "covered"
            ]
            first = uncovered[0] if uncovered else {}
            statement = str(first.get("statement") or "the unresolved requirement")
            proof = str(first.get("proof") or "completion")
            reason = str(first.get("reason") or "")
            if "no completion claim binds" in reason:
                return {
                    "type": "reason",
                    "instruction": (
                        f"Requirement {first.get('requirement_id', 'r1')} already has "
                        "relevant accepted evidence, but the answer does not bind it: "
                        f"{statement}. Add one narrow claim citing that evidence and "
                        "retry completion."
                    ),
                    "submit_via": "cortheon_complete",
                }
            if proof in {"mutation", "protection"}:
                instruction = (
                    f"Complete only this unresolved requirement, then capture a fresh "
                    f"focused diff and rerun the relevant test: {statement}."
                    if proof == "mutation"
                    else (
                        "Restore every user-protected test file required by "
                        f"'{statement}', capture a fresh diff, and rerun the "
                        "relevant test."
                    )
                )
                return {
                    "type": "reason",
                    "instruction": instruction,
                    "submit_via": "cortheon_step",
                }
            capability, success = {
                "verification": (
                    "test",
                    "Return the exact final-state command, outcome, and focused summary.",
                ),
                "inspection": (
                    "read",
                    "Return the smallest live excerpt that directly resolves this requirement.",
                ),
                "research": (
                    "search",
                    "Return current attributable evidence directly relevant to this requirement.",
                ),
                "synthesis": (
                    "search",
                    "Return focused, separately sourced evidence for this requirement.",
                ),
            }.get(
                proof,
                (
                    _capability_for_kind(session.task_kind),
                    "Return the minimum live evidence that resolves this requirement.",
                ),
            )
            request = self._create_request(
                session,
                capability=capability,
                query=f"Resolve requirement {first.get('requirement_id', 'r1')}: {statement}",
                reason=reason or f"No accepted evidence covers: {statement}",
                success_condition=success,
                parameters={
                    "requirement_id": first.get("requirement_id", "r1"),
                    "tool_call_budget": 1,
                },
            )
            return self._execute_action(request)
        if "hypothesis_competition" in failed:
            return {
                "type": "reason",
                "instruction": (
                    "Submit distinct hypotheses or status updates through cortheon_step; "
                    "Cortheon will request live evidence for any untested alternative."
                ),
                "submit_via": "cortheon_step",
            }
        if "completion_evidence" in failed:
            if any("concise-change budget" in gap for gap in gaps):
                return {
                    "type": "reason",
                    "instruction": (
                        "The change exceeds the concise-change budget implied by "
                        "the goal. Remove unrelated edits, capture a fresh focused "
                        "diff, and retry completion."
                    ),
                    "submit_via": "cortheon_step",
                }
            if session.deliverable == "code_change" and any(
                "protected test" in gap.casefold() for gap in gaps
            ):
                return {
                    "type": "reason",
                    "instruction": (
                        "Restore every user-protected test file, keep the repair "
                        "implementation-only, then capture a fresh diff and rerun "
                        "the required host test."
                    ),
                    "submit_via": "cortheon_step",
                }
            if (
                session.deliverable == "code_change"
                and any(
                    item.kind == "test" and item.status == "verified" and not item.quarantine_flags
                    for item in session.observations.values()
                )
                and not any(
                    "diff" in gap.casefold()
                    or "predates" in gap.casefold()
                    or "protected test" in gap.casefold()
                    for gap in gaps
                )
            ):
                return {
                    "type": "reason",
                    "instruction": (
                        "A verified test already exists in the context. Resubmit "
                        "cortheon_verify with its evidence id in completion_evidence_ids."
                    ),
                    "submit_via": "cortheon_verify",
                }
            request = self._request_for_gaps(session, gaps)
            if request is not None:
                return self._execute_action(request)
        if "uncertainty_visibility" in failed:
            return {
                "type": "reason",
                "instruction": "Revise only the answer text so every unresolved hypothesis "
                "is named and explicitly uncertain. Preserve all support fields exactly.",
                "submit_via": "cortheon_step",
                "required_fields": ["draft"],
            }
        return {
            "type": "reason",
            "instruction": (
                "Resolve the verification gaps, explicitly preserving any uncertainty "
                "that cannot be closed within the remaining budget."
            ),
            "submit_via": "cortheon_step",
        }

    def _request_for_claim_truth(
        self,
        session: Investigation,
        profile: dict[str, Any],
    ) -> EvidenceRequest | None:
        """Turn a failed epistemic gate into one executable host operation."""

        claim = str(profile.get("claim") or session.goal)
        claim_type = str(profile.get("claim_type") or "general_fact")
        gaps = " ".join(str(item) for item in profile.get("gaps", []))
        lower = gaps.casefold()
        capability: str
        purpose: str | None = None
        success: str
        if claim_type == "code_behavior":
            capability = "test"
            success = (
                "Return the exact focused test command, a host-recorded pass/fail "
                "outcome, and the smallest result excerpt that identifies the behavior."
            )
        elif claim_type == "code_change":
            capability = "diff"
            success = "Return a focused host-recorded diff that directly shows the change."
        elif claim_type == "code_static":
            capability = "read"
            success = (
                "Return the smallest live code excerpt or exact search result that "
                "directly entails the claim."
            )
        elif claim_type in {"current_research", "scientific"}:
            if "lineage" in lower or "corrobor" in lower:
                capability = "search"
                purpose = "corroboration"
                success = (
                    "Return a directly relevant, independently worded source from a "
                    "different publisher, with URL and live retrieval time."
                )
            elif "primary" in lower:
                capability = "fetch"
                purpose = "primary_fetch"
                success = (
                    "Return the directly relevant primary-source passage, URL, and "
                    "live retrieval time."
                )
            elif "contradiction" in lower or "counterevidence" in lower:
                capability = "search"
                purpose = "contradiction_check"
                success = (
                    "Return the strongest credible contradiction, correction, or "
                    "explicit scoped no-conflict result with URL and retrieval time."
                )
            else:
                capability = "search"
                purpose = "freshness_check"
                success = (
                    "Return a current, dated, attributable source with URL and live retrieval time."
                )
        elif claim_type == "legal":
            capability = "search"
            purpose = "discovery"
            success = (
                "Return the current official statute, regulation, judgment, or court "
                "record that directly governs the claim, with URL and retrieval time."
            )
        elif claim_type in {"private_record", "document_record"}:
            capability = "read"
            success = (
                "Return the smallest directly relevant passage with the authoritative "
                "record's source label and a host read receipt."
            )
        elif claim_type == "quantitative":
            capability = "inspect"
            success = (
                "Return a reproducible calculation or direct primary-data excerpt "
                "containing every material number in the claim."
            )
        else:
            capability = "inspect"
            success = (
                "Return one focused live observation that directly addresses the "
                "material terms and polarity of the claim."
            )
        return self._create_request(
            session,
            capability=capability,
            query=f"Establish or falsify this claim: {claim}",
            reason=(
                f"Claim {profile.get('claim_index', 1)} did not complete its required "
                f"truth operation: {gaps}"
            ),
            success_condition=success,
            parameters={"purpose": purpose} if purpose is not None else {},
        )


def _completion_gaps(
    session: Investigation,
    completion_evidence_ids: Iterable[str] | None,
    *,
    require_host_receipts: bool,
) -> list[str]:
    usable = [
        item
        for item in session.observations.values()
        if item.status != "failed" and not item.quarantine_flags
    ]
    ids = list(completion_evidence_ids or ())
    selected_all = (
        list(session.observations.values())
        if completion_evidence_ids is None
        else [
            session.observations[evidence_id]
            for evidence_id in ids
            if evidence_id in session.observations
        ]
    )
    selected = [
        item for item in selected_all if item.status != "failed" and not item.quarantine_flags
    ]
    candidates = usable if completion_evidence_ids is None else selected
    gaps: list[str] = []
    if session.deliverable == "code_change":
        diffs = [
            item
            for item in candidates
            if item.kind == "diff"
            and _diff_establishes_change(item, require_receipt=require_host_receipts)
        ]
        if not diffs:
            gaps.append("No focused live diff evidence establishes what changed.")
        changed_paths = {
            path
            for item in diffs
            for path in set(changed_paths_from_diff(item.content))
            | _diff_receipt_paths(item.host_receipt)
        }
        named_protected = set(protected_test_paths(session.goal))
        protected_changes = sorted(
            path
            for path in changed_paths
            if path in named_protected or (protects_tests(session.goal) and is_test_path(path))
        )
        if protected_changes:
            gaps.append(
                "The user-protected test surface changed: "
                + ", ".join(protected_changes)
                + ". Restore those files and patch only the implementation."
            )
        verified_tests = [
            item for item in candidates if item.kind == "test" and item.status == "verified"
        ]
        if not verified_tests:
            gaps.append(
                "A code change requires a host-verified test result in completion_evidence_ids."
            )
        elif diffs and max(item.sequence for item in verified_tests) <= max(
            item.sequence for item in diffs
        ):
            gaps.append(
                "The host-verified test predates the captured diff; rerun the relevant "
                "test after the final change."
            )
        suspicious = [item.evidence_id for item in diffs if _diff_weakens_tests(item.content)]
        if suspicious:
            gaps.append(
                "The diff appears to disable or weaken tests "
                f"({', '.join(suspicious)}); restore the assertions or provide an "
                "independently verified behavioral check."
            )
        budget = _diff_line_budget(session.goal)
        if budget is not None and diffs:
            changed_lines = _diff_changed_line_count(
                max(diffs, key=lambda item: item.sequence).content
            )
            if changed_lines > budget:
                gaps.append(
                    f"The final patch modifies {changed_lines} lines, above the concise-change "
                    f"budget of {budget} implied by the goal; strip unrelated edits and "
                    "resubmit the smallest verified patch."
                )
    elif session.deliverable == "code_understanding":
        if not any(item.kind == "code" for item in candidates):
            gaps.append("No live code observation supports the project conclusion.")
    elif session.deliverable == "research_answer":
        gaps.extend(_research_completion_gaps(session, candidates))
    elif session.deliverable == "document_synthesis":
        sources = {item.source for item in candidates if item.source}
        required = 2 if _has_hint(session.goal, _CROSS_SOURCE_HINTS) else 1
        if len(sources) < required:
            gaps.append(
                f"The document synthesis requires {required} distinct live document "
                f"source{'s' if required != 1 else ''}."
            )
    elif not candidates:
        gaps.append("No usable live observation supports completion.")
    unknown = [evidence_id for evidence_id in ids if evidence_id not in session.observations]
    if unknown:
        gaps.append("Unknown completion evidence ids: " + ", ".join(unknown) + ".")
    failed = [item.evidence_id for item in selected_all if item.status == "failed"]
    if failed:
        gaps.append("Failed observations cannot demonstrate completion: " + ", ".join(failed) + ".")
    quarantined = [item.evidence_id for item in selected_all if item.quarantine_flags]
    if quarantined and completion_evidence_ids is not None:
        gaps.append(
            "Quarantined observations cannot demonstrate completion: "
            + ", ".join(quarantined)
            + "."
        )
    return gaps

"""Per-claim verification profile derivation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from cortheon.cognitive_core.alignment import _research_conflict_present
from cortheon.cognitive_core.claims import _claim_entailment, _claim_type
from cortheon.cognitive_core.diffs import _diff_establishes_change
from cortheon.cognitive_core.models import Investigation, PublicClaim
from cortheon.cognitive_core.receipts import _observation_origin, _receipt_outcome
from cortheon.cognitive_core.research_gaps import _effective_web_lineages


def _claim_verification_profiles(
    session: Investigation,
    claims: list[PublicClaim],
    *,
    require_host_receipts: bool,
) -> list[dict[str, Any]]:
    """Expose the bounded truth basis and required operation for every claim."""

    profiles: list[dict[str, Any]] = []
    for index, claim in enumerate(claims, 1):
        observations = [
            session.observations[evidence_id]
            for evidence_id in claim.evidence_ids
            if evidence_id in session.observations
            and session.observations[evidence_id].status != "failed"
            and not session.observations[evidence_id].quarantine_flags
        ]
        claim_type = _claim_type(session, claim.claim)
        web = [item for item in observations if item.kind == "web"]
        raw_origins, effective_lineages, syndicated = _effective_web_lineages(web)
        receipts = [item.host_receipt for item in observations if item.host_receipt is not None]
        sources = {item.source for item in observations if item.source}
        has_primary = any(item.purpose == "primary_fetch" for item in web)
        has_contradiction_check = any(item.purpose == "contradiction_check" for item in web)
        has_current_retrieval = any(
            item.retrieved_at
            and -300
            <= (datetime.now(UTC) - datetime.fromisoformat(item.retrieved_at)).total_seconds()
            <= 3_600
            for item in web
        )
        has_test = any(
            item.kind == "test"
            and item.status == "verified"
            and (not require_host_receipts or _receipt_outcome(item, "test", "passed"))
            for item in observations
        )
        has_diff = any(
            item.kind == "diff"
            and _diff_establishes_change(
                item,
                require_receipt=require_host_receipts,
            )
            for item in observations
        )
        has_local_read = any(
            item.kind in {"code", "documentation", "artifact"}
            and (
                not require_host_receipts
                or str((item.host_receipt or {}).get("tool") or "").casefold()
                in {"find", "git", "glob", "grep", "read"}
            )
            for item in observations
        )
        entails, entailment_reason = _claim_entailment(
            session,
            claim.claim,
            claim_type,
            observations,
        )
        gaps: list[str] = []
        required_operation = "direct evidence whose content entails the claim"

        if claim_type == "code_behavior":
            required_operation = "run a focused host test after the final change"
            if not has_test:
                gaps.append("The behavior claim lacks a host-verified passing test.")
        elif claim_type == "code_change":
            required_operation = "capture a focused live diff"
            if not has_diff:
                gaps.append("The change claim lacks a focused live diff.")
        elif claim_type == "code_static":
            required_operation = "read or exactly search the live code"
            if not has_local_read:
                gaps.append("The static code claim lacks a live host read or exact search.")
        elif claim_type == "quantitative":
            required_operation = "recompute the value or cite a directly read task record"
            direct_task_record = any(
                item.kind == "artifact"
                and item.source
                and str((item.host_receipt or {}).get("tool") or "").casefold() == "read"
                and str((item.host_receipt or {}).get("outcome") or "").casefold() == "result"
                for item in observations
            )
            reproducible = (
                any(
                    item.kind in {"analysis", "command", "test"} or item.purpose == "primary_fetch"
                    for item in observations
                )
                or direct_task_record
            )
            if not reproducible:
                gaps.append(
                    "The quantitative claim lacks a reproducible calculation or directly "
                    "read task record."
                )
        elif claim_type == "scientific":
            required_operation = (
                "fetch the primary study, independent corroboration, and counterevidence"
            )
            if not has_primary:
                gaps.append("The scientific claim lacks a fetched primary study.")
            if effective_lineages < 2 and "corroboration" not in session.waivers:
                gaps.append("The scientific claim lacks independent corroboration.")
            if not has_contradiction_check and "contradiction_check" not in session.waivers:
                gaps.append("The scientific claim lacks a counterevidence search.")
        elif claim_type == "legal":
            required_operation = "fetch current official legal authority"
            official = any(
                (origin := _observation_origin(item)) is not None
                and (
                    ".gov" in origin
                    or origin.endswith("gov.uk")
                    or "court" in origin
                    or "legislation" in origin
                )
                for item in web
            )
            if not official and not any(
                item.kind in {"documentation", "artifact"} and item.source for item in observations
            ):
                gaps.append("The legal claim lacks current official legal authority.")
            if web and not has_current_retrieval:
                gaps.append("The legal authority was not retrieved in this live session.")
        elif claim_type == "current_research":
            required_operation = (
                "fetch a primary source, independent corroboration, freshness, and counterevidence"
            )
            required_lineages = 1 if "corroboration" in session.waivers else 2
            if effective_lineages < required_lineages:
                gaps.append(
                    "The research claim has "
                    f"{effective_lineages} effective source lineage(s); "
                    f"{required_lineages} required."
                )
            if not has_primary and "primary_fetch" not in session.waivers:
                gaps.append("The research claim lacks a fetched primary source.")
            if not has_current_retrieval:
                gaps.append("The research claim lacks a current host-recorded retrieval.")
            if not has_contradiction_check and "contradiction_check" not in session.waivers:
                gaps.append("The research claim lacks a contradiction or correction search.")
        elif claim_type == "private_record":
            required_operation = "read the authoritative private record and preserve its scope"
            attributable_private = any(
                item.source and item.kind in {"artifact", "code", "documentation", "user"}
                for item in observations
            )
            if not attributable_private:
                gaps.append("The private-record claim lacks an attributable internal record.")
        elif claim_type == "document_record":
            required_operation = "read the identified source document"
            if not sources:
                gaps.append("The document claim lacks an identified source document.")

        if not entails:
            gaps.append(entailment_reason)

        conflict = _research_conflict_present(web) if len(web) > 1 else False
        if has_test:
            established_level = "operationally_verified"
            allowed_wording = "State only the behavior exercised by the cited passing host test."
        elif effective_lineages >= 2:
            established_level = "independently_corroborated"
            allowed_wording = (
                "State the bounded claim as independently corroborated, with citations."
            )
        elif has_primary:
            established_level = "primary_source_attributed"
            allowed_wording = "Attribute the claim to the fetched primary source."
        elif receipts:
            established_level = "host_observed"
            allowed_wording = "State only what the cited host operation directly observed."
        elif sources:
            established_level = "source_attributed"
            allowed_wording = "Attribute the claim to the identified source; do not call it proven."
        elif observations:
            established_level = "observation_only"
            allowed_wording = "Describe this only as an observation, not an established fact."
        else:
            established_level = "unsupported"
            allowed_wording = "Do not state this claim."

        dimensions = {
            "authority": (
                "primary_source"
                if has_primary
                else "host_receipt"
                if receipts
                else "attributed_source"
                if sources
                else "unattributed"
            ),
            "directness": "direct" if entails else "not_established",
            "independence": (
                "not_applicable"
                if claim_type
                in {
                    "code_behavior",
                    "code_change",
                    "code_static",
                    "private_record",
                    "document_record",
                }
                else f"{effective_lineages}_effective_lineage(s)"
                if web
                else "not_established"
            ),
            "freshness": (
                "live_retrieval"
                if has_current_retrieval
                else "not_applicable"
                if not web
                else "not_established"
            ),
            "reproducibility": (
                "host_test" if has_test else "host_operation" if receipts else "source_observation"
            ),
            "contradiction_handling": (
                "conflict_detected"
                if conflict
                else "checked"
                if has_contradiction_check
                else "not_applicable"
                if not web
                else "not_checked"
            ),
            "entailment": "established" if entails else "not_established",
        }
        profiles.append(
            {
                "claim_index": index,
                "claim": claim.claim,
                "claim_type": claim_type,
                "evidence_ids": list(claim.evidence_ids),
                "required_truth_operation": required_operation,
                "dimensions": dimensions,
                "raw_url_origins": raw_origins,
                "effective_source_lineages": effective_lineages,
                "likely_syndicated_sources": syndicated,
                "established_level": established_level,
                "allowed_wording": allowed_wording,
                "passed": not gaps,
                "gaps": gaps,
                "next_truth_operation": (
                    "No further operation required for this bounded claim."
                    if not gaps
                    else required_operation
                ),
            }
        )
    return profiles

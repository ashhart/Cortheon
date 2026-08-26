"""Held-out task pack for the P6 same-model qualification.

P6 requires tasks the fixed model has never seen, authored and graded outside
the tuning loop. This module generates fresh instances of the sealed case
protocol: new causal families, entities, outcomes, scopes, probes, and
decision surfaces, numbered beyond the development bank, with a hard
no-vocabulary-overlap invariant against the 60-case development bank and a
sealed pack digest. Development material, not a claim.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from cortheon.operator_lift.case_bank import development_cases
from cortheon.operator_lift.case_builders import _e, _hyp, _join, _probe, _revision, _stop
from cortheon.operator_lift.models import OPERATORS, LiftCase

HELDOUT_PER_OPERATOR = 12
FIRST_HELDOUT_NUMBER = 13  # development bank uses 01..12 per operator


def _tokens(values: Any) -> set[str]:
    if isinstance(values, str):
        return {values}
    if isinstance(values, (list, tuple, set)):
        tokens: set[str] = set()
        for value in values:
            tokens |= _tokens(value)
        return tokens
    if isinstance(values, dict):
        tokens = set()
        for key, value in values.items():
            tokens |= _tokens(key) | _tokens(value)
        return tokens
    return set()


_PROTOCOL_TOKENS = frozenset(
    {
        "sufficient",
        "supported",
        "refuted",
        "uncertain",
        "undetermined",
        "source_a",
        "source_b",
        "source_c",
    }
)


def _content_vocabulary(case: LiftCase) -> set[str]:
    """Answerable content identifiers from the response schema and actions."""
    vocabulary: set[str] = set()

    def leaves(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                leaves(value)
        elif isinstance(node, (list, tuple, set)):
            for value in node:
                leaves(value)
        elif isinstance(node, str):
            vocabulary.add(node)

    schema = case.response_schema
    field_vocabulary = schema.get("field_vocabulary")
    if isinstance(field_vocabulary, dict):
        leaves(field_vocabulary)
    for key in (
        "hypothesis_vocabulary",
        "status_vocabulary",
        "decision_vocabulary",
        "relation_vocabulary",
        "token_vocabulary",
    ):
        value = schema.get(key)
        if value is not None:
            leaves(value)
    if case.action_catalog:
        leaves([action[0] for action in case.action_catalog])
    return {token for token in vocabulary if token not in _PROTOCOL_TOKENS}


def _dev_vocabulary() -> set[str]:
    vocabulary: set[str] = set()
    for case in development_cases():
        vocabulary |= _content_vocabulary(case)
    return vocabulary


def _hypothesis_cases() -> tuple[LiftCase, ...]:
    cases: list[LiftCase] = []
    for offset in range(HELDOUT_PER_OPERATOR):
        number = FIRST_HELDOUT_NUMBER + offset
        family = f"clearing_gate_settlement_lag_{offset}"
        leading = (f"crest_sequencer_drain_{offset}", "settlement_backlog", f"gate_{offset}")
        rival = (f"upstream_throttle_jitter_{offset}", "settlement_backlog", f"gate_{offset}")
        falsification = (
            f"isolate_crest_{offset}",
            "backlog_clears",
            f"crest_sequencer_drain_{offset}",
        )
        evidence = _e(
            f"[cause={leading[0]}] {family}: the crest sequencer drains settlements; bursts backlog the gate.",
            f"[rival={rival[0]}] {family}: upstream throttle jitter coincides with the backlog period.",
            f"[test={falsification[0]}] {family}: isolating the crest sequencer clears the backlog.",
        )
        cases.append(_hyp(number, family, evidence, leading, rival, falsification))
    return tuple(cases)


def _discrimination_cases() -> tuple[LiftCase, ...]:
    cases: list[LiftCase] = []
    for offset in range(HELDOUT_PER_OPERATOR):
        number = FIRST_HELDOUT_NUMBER + offset
        family = f"harbor_berth_allocator_variance_{offset}"
        first = f"preferred_barge_window_{offset}"
        second = f"tidal_slot_release_{offset}"
        probe = f"probe_berth_utilization_{offset}"
        actions = (
            (probe, "measure berth utilization under each window", 2),
            (f"probe_crew_quota_{offset}", "inspect crew quota records", 2),
            (f"probe_weather_gate_{offset}", "inspect weather gate logs", 3),
        )
        evidence = _e(
            f"[{first}] {family}: preferred-window calls dominate reported utilization.",
            f"[{second}] {family}: tidal-slot releases show the opposite pattern in berth records.",
        )
        expected = (probe, first, second)
        cases.append(_probe(number, family, evidence, (first, second), actions, expected))
    return tuple(cases)


def _revision_cases() -> tuple[LiftCase, ...]:
    cases: list[LiftCase] = []
    for offset in range(HELDOUT_PER_OPERATOR):
        number = FIRST_HELDOUT_NUMBER + offset
        family = f"transit_ledger_reconciliation_drift_{offset}"
        prior = f"batch_consolidation_cause_{offset}"
        if offset % 2:
            new_status = "refuted"
            revised = f"async_receipt_stream_cause_{offset}"
            decisive = "source_a"
        else:
            new_status = "supported"
            revised = prior
            decisive = "source_b"
        # One retained-status effect and one refuting effect: exactly one
        # matches the expected (status, change) pair for every offset.
        contract = {
            f"batch_consolidation_{offset}": ("supported", False),
            f"ledger_pruning_{offset}": ("refuted", True),
        }
        expected = (prior, new_status, revised, decisive)
        evidence = _e(
            f"[{prior}] {family}: batch consolidation was the claimed cause of the drift.",
            f"[{decisive}] {family}: {revised} explains the reconciliation drift.",
        )
        cases.append(
            _revision(
                number,
                family,
                evidence,
                expected,
                effect_contract=contract,
            )
        )
    return tuple(cases)


def _derivation_cases() -> tuple[LiftCase, ...]:
    cases: list[LiftCase] = []
    for offset in range(HELDOUT_PER_OPERATOR):
        number = FIRST_HELDOUT_NUMBER + offset
        family = f"orchard_sensor_result_chain_{offset}"
        subject = f"field_hub_{offset}"
        relation = f"feeds_orchard_rank_{offset}"
        object_id = f"picking_priority_{offset}"
        middle = f"aggregation_node_{offset}"
        premises = (
            ("source_a", subject, relation, middle),
            ("source_b", middle, relation, object_id),
        )
        evidence = _e(
            f"{family}: {subject} {relation} {middle}.",
            f"{family}: {middle} {relation} {object_id}.",
        )
        conclusion = (subject, relation, object_id)
        cases.append(_join(number, family, evidence, conclusion, premises))
    return tuple(cases)


def _stopping_cases() -> tuple[LiftCase, ...]:
    cases: list[LiftCase] = []
    for offset in range(HELDOUT_PER_OPERATOR):
        number = FIRST_HELDOUT_NUMBER + offset
        family = f"warehouse_dispatch_classifier_turn_{offset}"
        # Costs are ascending in the runtime's probe order, so the expected
        # actions are exactly the prefix the runtime asks for and executes.
        actions = (
            (f"action_scan_{offset}", "scan the manifest", 1),
            (f"action_probe_{offset}", "probe the route history", 2),
            (f"action_weigh_{offset}", "weigh the pallet", 3),
            (f"action_inspect_{offset}", "inspect the handover log", 4),
        )
        expected_actions = (f"action_scan_{offset}", f"action_probe_{offset}")
        decision = f"route_class_{offset}"
        observations = (
            (f"action_scan_{offset}", f"manifest route{offset} matches the candidate class."),
            (f"action_probe_{offset}", f"route history {offset} confirms the class."),
            (f"action_weigh_{offset}", f"pallet weight {offset} is nominal."),
            (f"action_inspect_{offset}", f"handover log {offset} shows no anomaly."),
        )
        evidence = _e(
            f"{family}: the dispatch classifier flags the pallet.",
            f"{family}: the route history and manifest identify the class.",
        )
        cases.append(
            _stop(
                number,
                family,
                evidence,
                actions,
                expected_actions,
                decision,
                observations,
            )
        )
    return tuple(cases)


def heldout_cases() -> tuple[LiftCase, ...]:
    """Return the sealed 60-case held-out pack in preregistered order."""
    cases = (
        *_hypothesis_cases(),
        *_discrimination_cases(),
        *_revision_cases(),
        *_derivation_cases(),
        *_stopping_cases(),
    )
    if len(cases) != 5 * HELDOUT_PER_OPERATOR:
        raise AssertionError("held-out pack must contain exactly 60 cases")
    return cases


def _case_vocabulary(case: LiftCase) -> set[str]:
    return _content_vocabulary(case)


def verify_heldout_isolation() -> dict[str, Any]:
    """Enforce the no-overlap invariant against the development bank."""
    development = _dev_vocabulary()
    collisions: set[str] = set()
    for case in heldout_cases():
        collisions |= _case_vocabulary(case) & development
    development_ids = {case.case_id for case in development_cases()}
    overlap_ids = development_ids & {case.case_id for case in heldout_cases()}
    return {
        "development_tokens": len(development),
        "collisions": sorted(collisions)[:40],
        "case_id_overlap": sorted(overlap_ids),
        "isolated": not collisions and not overlap_ids,
    }


def seal_heldout(output_path: Path | None = None) -> dict[str, Any]:
    """Seal the held-out pack and write its manifest next to the frozen runs."""
    cases = heldout_cases()
    blobs = [
        case.case_id.encode()
        + b"\x00"
        + "\n".join(content for _source_id, content in case.evidence).encode()
        for case in cases
    ]
    pack_sha256 = hashlib.sha256(b"\x00".join(blobs)).hexdigest()
    manifest = {
        "schema_version": 1,
        "purpose": "P6 held-out qualification pack",
        "operator_order": list(OPERATORS),
        "cases": len(cases),
        "first_numbers": {"per_operator_start": FIRST_HELDOUT_NUMBER},
        "pack_sha256": pack_sha256,
        "isolation": verify_heldout_isolation(),
    }
    if output_path is not None:
        output_path.write_text(
            __import__("json").dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return manifest

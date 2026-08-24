"""Public action-catalog planning for bounded adaptive evidence collection."""

from __future__ import annotations

import json
import re
from typing import Any

from cortheon.cognitive_core.models import CognitiveRuntimeError, Investigation
from cortheon.cognitive_core.tasks import _is_adaptive_stopping_goal

_STOP = frozenset(
    {
        "a",
        "all",
        "an",
        "and",
        "another",
        "as",
        "current",
        "every",
        "for",
        "from",
        "in",
        "is",
        "may",
        "of",
        "or",
        "read",
        "the",
        "to",
    }
)


def next_catalog_action(session: Investigation) -> dict[str, Any] | None:
    """Select the next cheap action covering the public sufficiency contract."""

    contract = _contract(session)
    if contract is None:
        return None
    actions, evidence = contract["actions"], contract["evidence"]
    evidence_terms = _terms(" ".join(str(item.get("content", "")) for item in evidence))
    required = [
        item
        for item in actions
        if _terms(f"{item['action_id']} {item['description']}") & evidence_terms
    ]
    candidates = required or sorted(actions, key=lambda item: (item["cost"], item["action_id"]))
    executed = {item[0] for item in _executed_actions(session)}
    return next((item for item in candidates if item["action_id"] not in executed), None)


def validate_adaptive_completion(session: Investigation, answer: str) -> None:
    """Bind claimed actions and cost to the host-executed action ledger."""

    if not _is_adaptive_stopping_goal(session.goal):
        return
    contract = _contract(session)
    if contract is None:
        return
    try:
        value = json.loads(answer)
    except json.JSONDecodeError as exc:
        raise CognitiveRuntimeError("adaptive completion must be a JSON object") from exc
    fields = {"actions", "decision", "total_cost", "stop_reason"}
    if not isinstance(value, dict) or set(value) != fields:
        raise CognitiveRuntimeError("adaptive completion has the wrong closed shape")
    actions = value.get("actions")
    if (
        not isinstance(actions, list)
        or not actions
        or len(set(actions)) != len(actions)
        or not all(isinstance(item, str) for item in actions)
        or value.get("decision") not in contract["decisions"]
        or type(value.get("total_cost")) is not int
        or value.get("stop_reason") != "sufficient"
    ):
        raise CognitiveRuntimeError("adaptive completion is outside the public contract")
    if actions != [item[0] for item in _executed_actions(session)]:
        raise CognitiveRuntimeError("adaptive completion claims an action that did not execute")
    expected_cost = sum(contract["costs"].get(action_id, -1) for action_id in actions)
    if expected_cost < 0 or value["total_cost"] != expected_cost:
        raise CognitiveRuntimeError("adaptive completion cost does not match executed actions")


def _contract(
    session: Investigation,
) -> dict[str, Any] | None:
    for observation in session.observations.values():
        if observation.status == "failed" or observation.quarantine_flags:
            continue
        try:
            payload = json.loads(observation.content)
        except json.JSONDecodeError:
            continue
        actions = payload.get("actions") if isinstance(payload, dict) else None
        evidence = payload.get("evidence") if isinstance(payload, dict) else None
        response = payload.get("response_schema") if isinstance(payload, dict) else None
        if not isinstance(actions, list) or not actions or not isinstance(evidence, list):
            continue
        if not isinstance(response, dict) or response.get("fields") != [
            "actions",
            "decision",
            "total_cost",
            "stop_reason",
        ]:
            continue
        decisions = response.get("decision_vocabulary")
        if not isinstance(decisions, list) or not all(isinstance(item, str) for item in decisions):
            raise CognitiveRuntimeError("adaptive decision vocabulary is invalid")
        normalized: list[dict[str, Any]] = []
        for item in actions:
            if not isinstance(item, dict) or set(item) != {"action_id", "description", "cost"}:
                return None
            action_id, description, cost = (
                item["action_id"],
                item["description"],
                item["cost"],
            )
            if not isinstance(action_id, str) or not isinstance(description, str):
                raise CognitiveRuntimeError("adaptive action catalogue is invalid")
            if type(cost) is not int or cost < 0:
                raise CognitiveRuntimeError("adaptive action catalogue is invalid")
            normalized.append(dict(item))
        return {
            "actions": normalized,
            "evidence": [item for item in evidence if isinstance(item, dict)],
            "decisions": decisions,
            "costs": {item["action_id"]: item["cost"] for item in normalized},
        }
    return None


def _executed_actions(session: Investigation) -> list[tuple[str, str]]:
    executed: list[tuple[str, str]] = []
    for observation in sorted(session.observations.values(), key=lambda item: item.sequence):
        receipt = observation.host_receipt
        arguments = receipt.get("args") if isinstance(receipt, dict) else None
        path = arguments.get("path") if isinstance(arguments, dict) else None
        match = re.fullmatch(r"actions/([A-Za-z0-9._-]+)\.txt", path or "")
        if match and observation.status != "failed" and not observation.quarantine_flags:
            executed.append((match.group(1), observation.evidence_id))
    return executed


def _terms(value: str) -> set[str]:
    terms = {token.rstrip("s") for token in re.findall(r"[a-z0-9]+", value.casefold())}
    return {term for term in terms if len(term) >= 3 and term not in _STOP}

"""Deterministic runtime-decision replay over the operator-lift case bank.

Not a claim gate: drives the real CognitiveRuntime on each sealed case with
a scripted responder, per operator condition, in milliseconds. No model, no
held-out grading.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cortheon.cognitive_core.models import CognitiveRuntimeError
from cortheon.cognitive_core.runtime import CognitiveRuntime
from cortheon.operator_lift.execution_schedule import case_goal
from cortheon.operator_lift.models import OPERATORS, LiftCase
from cortheon.operator_lift.oracles import grade_case

MAX_REPLAY_STEPS = 32


@dataclass(frozen=True)
class ReplayCell:
    case_id: str
    operator: str
    condition_id: str
    certified: bool
    correct: bool
    delivered: bool
    requests: tuple[str, ...]
    withheld_reasons: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def request_digest(self) -> str:
        return hashlib.sha256("\n".join(self.requests).encode()).hexdigest()[:16]


def answer_payload(case: LiftCase) -> dict[str, Any]:
    """Synthesize the oracle-correct structured answer for the case."""
    oracle = case.oracle
    if case.operator == "hypothesis_framing":
        leading = oracle["leading"]
        rival = oracle["rivals"][0]
        falsification = oracle["falsification"]
        return {
            "leading": dict(zip(("cause", "outcome", "scope"), leading, strict=True)),
            "rival": dict(zip(("cause", "outcome", "scope"), rival, strict=True)),
            "falsification": dict(
                zip(("intervention", "result", "refutes"), falsification, strict=True)
            ),
        }
    if case.operator == "discriminating_evidence":
        expected = oracle["expected"]
        return {
            "probe_id": expected[0],
            "positive_supports": expected[1],
            "negative_supports": expected[2],
        }
    if case.operator == "contradiction_revision":
        prior, prior_status, revised, decisive_source = oracle["expected"]
        return {
            "prior": prior,
            "prior_status": prior_status,
            "revised": revised,
            "decisive_source": decisive_source,
        }
    if case.operator == "cross_source_derivation":
        conclusion = oracle["conclusion"]
        return {
            "subject": conclusion[0],
            "relation": conclusion[1],
            "object": conclusion[2],
            "premises": [
                dict(zip(("source_id", "subject", "relation", "object"), premise, strict=True))
                for premise in oracle["premises"]
            ],
        }
    if case.operator == "adaptive_stopping":
        costs = {action[0]: action[2] for action in case.action_catalog}
        actions = list(oracle["expected_actions"])
        return {
            "actions": actions,
            "decision": oracle["decision"],
            "total_cost": sum(costs.get(action, 0) for action in actions),
            "stop_reason": "sufficient",
        }
    raise ValueError(f"unsupported case operator: {case.operator}")


def _materialize_workspace(root: Path, case: LiftCase) -> None:
    """Mirror the live runner's public workspace for replay reads."""
    from cortheon.operator_lift.sealing import public_case

    (root / "evidence").mkdir(parents=True, exist_ok=True)
    (root / "public-projection.json").write_text(
        json.dumps(public_case(case), sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    for source_id, content in case.evidence:
        (root / "evidence" / f"{source_id}.txt").write_text(content, encoding="utf-8")
    if case.operator == "adaptive_stopping":
        (root / "actions").mkdir(exist_ok=True)
        for action_id, observation in case.oracle["observations"]:
            (root / "actions" / f"{action_id}.txt").write_text(observation, encoding="utf-8")


def _profile(disabled_operator: str | None) -> dict[str, Any]:
    from cortheon.cognitive_protocol import EVALUATION_PROFILE_VERSION

    operators: dict[str, bool] = dict.fromkeys(
        ("retrieval", "verification", *OPERATORS),
        True,
    )
    if disabled_operator is not None:
        operators[disabled_operator] = False
    config: dict[str, Any] = {
        "schema_version": EVALUATION_PROFILE_VERSION,
        "operators": operators,
        "intercepts_final": True,
        "cleanup_before_answer": False,
        "hard_budgets_enforced": True,
        "sticky_terminal_safety": True,
        "transport_failure_fails_open": True,
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": EVALUATION_PROFILE_VERSION,
        "config": config,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "implementation_sha256": "a" * 64,
        "nonce": "c" * 32,
    }


def _read_receipt(path: str) -> str:
    return (
        '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
        '"args":{"filePath":"' + path + '"}}\n'
    )


class ReplayResponder:
    """Scripted host-and-model stand-in obeying the condition's allowed protocol."""

    def __init__(
        self,
        case: LiftCase,
        workspace_root: Path,
        operators: Mapping[str, bool],
    ) -> None:
        self.case = case
        self.root = Path(workspace_root).resolve()
        self.operators = operators
        self.accepted: list[str] = []
        self.requests: list[str] = []
        self.evidence_read = False
        self.answer = json.dumps(answer_payload(case), sort_keys=True)

    @property
    def framing(self) -> bool:
        return bool(self.operators.get("hypothesis_framing"))

    def _file_observation(self, path: str) -> dict[str, Any]:
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root) or not target.is_file():
            content = _read_receipt(path) + f"No such file: {path}"
        else:
            content = (
                _read_receipt(path) + target.read_text(encoding="utf-8", errors="replace")[:20_000]
            )
        return {
            "kind": "code" if path.endswith(".txt") else "documentation",
            "content": content,
            "source": f"replay:{path}",
        }

    def _observations_for(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        capability = str(request.get("capability") or "")
        parameters = request.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        if capability in {"read", "inspect", "search_or_read", "read_many"}:
            paths = parameters.get("paths")
            if not isinstance(paths, list) or not paths:
                path = parameters.get("path")
                paths = [path] if isinstance(path, str) and path else ["public-projection.json"]
            return [self._file_observation(str(path)) for path in paths]
        if capability == "search":
            pattern = str(parameters.get("pattern") or request.get("query") or ".")
            source = (self.root / "public-projection.json").read_text(
                encoding="utf-8", errors="replace"
            )
            present = pattern.casefold() in source.casefold()
            receipt = (
                '[CORTHEON_HOST_EVIDENCE] {"tool":"grep","outcome":"'
                + ("match" if present else "no_match")
                + '","args":{"pattern":"'
                + pattern
                + '","path":"public-projection.json"}}'
            )
            return [
                {
                    "kind": "code",
                    "content": receipt
                    + ("\nprojection mentions the pattern" if present else "\nNo matches found."),
                    "source": f"replay:grep:{pattern}",
                }
            ]
        raise ValueError(f"replay: unexpected host capability {capability!r}")

    def _claims(self) -> list[dict[str, Any]]:
        return [
            {
                "claim": re.sub(r"\b\d+(?:[.,]\d+)*\b", "N", content)[:400].strip(),
                "evidence_ids": list(self.accepted),
            }
            for _source_id, content in self.case.evidence
        ]

    def _hypotheses(self) -> list[dict[str, str]]:
        oracle = self.case.oracle
        if self.case.operator == "hypothesis_framing":
            leading = " ".join(oracle["leading"])
            rival = " ".join(oracle["rivals"][0])
            return [
                {
                    "statement": f"The leading cause is {leading}.",
                    "falsification_test": " ".join(oracle["falsification"]),
                },
                {
                    "statement": f"The rival cause is {rival}.",
                    "falsification_test": "The rival collapses when the leading cause is isolated.",
                },
            ]
        return [
            {
                "statement": f"{family} is the operative cause in {self.case.causal_family}.",
                "falsification_test": "Find evidence refuting the operative cause.",
            }
            for family in sorted({value for _, value in (self.case.evidence or ())})
        ] or [
            {
                "statement": "The case cause is operative.",
                "falsification_test": "Find contradicting evidence.",
            }
        ]

    def _completion_hypotheses(self) -> list[dict[str, Any]]:
        if not self.framing:
            return []
        return [
            {
                "statement": hypothesis["statement"],
                "falsification_test": hypothesis["falsification_test"],
                "status": "supported",
                "evidence_ids": list(self.accepted),
            }
            for hypothesis in self._hypotheses()
        ]

    # Loop ------------------------------------------------------------------

    def drive(
        self,
        runtime: CognitiveRuntime,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        stepped = False
        errors: list[str] = []
        self._verdict: str | None = None
        self._gaps: tuple[str, ...] = ()

        def guarded(call):
            try:
                return call()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                return None

        for _ in range(MAX_REPLAY_STEPS):
            action = payload.get("next_action")
            if not isinstance(action, dict):
                break
            action_type = action.get("type")
            if action_type == "harness_tool":
                request = action.get("request")
                if not isinstance(request, dict):
                    break
                self.requests.append(str(request.get("capability") or "request"))
                result = guarded(
                    lambda request=request: runtime.observe(
                        session_id,
                        self._observations_for(request),
                        request_id=str(request.get("request_id") or ""),
                    )
                )
                if result is None:
                    break
                payload = result
                accepted = payload.get("accepted_evidence_ids")
                if isinstance(accepted, list):
                    self.accepted.extend(
                        str(item) for item in accepted if item not in self.accepted
                    )
                continue
            if not self.evidence_read:
                for source_id, _content in self.case.evidence:
                    result = guarded(
                        lambda source_id=source_id: runtime.observe(
                            session_id,
                            [self._file_observation(f"evidence/{source_id}.txt")],
                        )
                    )
                    if result is None:
                        break
                    payload = result
                    accepted = payload.get("accepted_evidence_ids")
                    if isinstance(accepted, list):
                        self.accepted.extend(
                            str(item) for item in accepted if item not in self.accepted
                        )
                self.evidence_read = True
                if errors:
                    break
                continue
            if action_type == "reason" and not stepped:
                stepped = True
                hypotheses = self._hypotheses() if self.framing else []
                result = guarded(
                    lambda hypotheses=hypotheses: runtime.step(
                        session_id,
                        hypotheses=hypotheses,
                        open_questions=[],
                        draft=None,
                    )
                )
                if result is None:
                    break
                payload = result
                continue
            if action_type == "challenge":
                result = guarded(
                    lambda: runtime.challenge(
                        session_id,
                        draft=self.answer,
                        claims=self._claims(),
                    )
                )
                if result is None:
                    break
                payload = result
                continue
            self._verdict = None
            self._gaps: tuple[str, ...] = ()
            result = guarded(
                lambda: runtime.complete(
                    session_id,
                    answer=self.answer,
                    claims=self._claims(),
                    hypotheses=self._completion_hypotheses(),
                    completion_evidence_ids=list(self.accepted),
                )
            )
            if result is None:
                break
            payload = result
            verification = payload.get("verification") if isinstance(payload, dict) else None
            if isinstance(verification, dict):
                self._verdict = verification.get("verdict")
                gaps = verification.get("gaps")
                if isinstance(gaps, list):
                    self._gaps = tuple(str(item) for item in gaps)
                if self._verdict == "complete":
                    break
            elif payload.get("status") == "complete":
                # Terminal certification shape: status replaces the verification verdict.
                self._verdict = "complete"
                break
            # A withhhold is a terminal outcome for this runtime: stop here.
            break
        verdict = None
        if isinstance(payload, dict):
            verification = payload.get("verification")
            if isinstance(verification, dict):
                verdict = verification.get("verdict")
            if verdict is None and self._verdict is not None:
                verdict = self._verdict
        if payload.get("status") == "complete" and verdict != "complete":
            verdict = "complete"
        if verdict != "complete" and not errors and not self._gaps:
            errors.append("replay: runtime reached no terminal decision for the submission")
        self._errors = errors
        return payload


def replay_case(
    case: LiftCase,
    workspace_root: Path,
    *,
    disabled_operator: str | None = None,
    runtime: CognitiveRuntime | None = None,
    effort: str = "deep",
    strictness: str = "standard",
) -> ReplayCell:
    """Run one case through the runtime under one condition, deterministically."""
    errors: list[str] = []
    condition_id = (
        "full" if disabled_operator is None else f"ablation_{OPERATORS.index(disabled_operator)}"
    )
    rt = runtime or CognitiveRuntime(require_host_receipts=True)
    responder: ReplayResponder | None = None
    final: dict[str, Any] = {}
    certified = False
    correct = False
    withheld_reasons: tuple[str, ...] = ()
    try:
        profile = _profile(disabled_operator)
        operators = profile["config"]["operators"]
        payload = rt.start(
            case_goal(case),
            task_kind="general",
            effort=effort,
            strictness=strictness,
            evaluation_profile=profile,
        )
        responder = ReplayResponder(case, workspace_root, operators)
        session_id = str(payload["session"]["session_id"])
        final = responder.drive(rt, session_id, payload)
        verdict = None
        verification = final.get("verification") if isinstance(final, dict) else None
        if isinstance(verification, dict):
            verdict = verification.get("verdict")
        if verdict is None:
            verdict = getattr(responder, "_verdict", None)
        certified = verdict == "complete"
        if isinstance(verification, dict) and isinstance(verification.get("gaps"), list):
            withheld_reasons = tuple(str(item) for item in verification["gaps"])
        if not withheld_reasons:
            withheld_reasons = tuple(getattr(responder, "_gaps", ()))
        parsed = json.loads(responder.answer) if certified else None
        if parsed is not None:
            correct = grade_case(case, parsed).correct
    except (ValueError, KeyError, TypeError, CognitiveRuntimeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    drive_errors = tuple(getattr(responder, "_errors", ())) if responder is not None else ()
    return ReplayCell(
        case_id=case.case_id,
        operator=case.operator,
        condition_id=condition_id,
        certified=certified,
        correct=correct,
        delivered=bool(certified and correct),
        requests=tuple(responder.requests) if responder is not None else (),
        withheld_reasons=withheld_reasons,
        errors=tuple(errors) + drive_errors,
    )


def replay_bank(
    cases: tuple[LiftCase, ...],
    workspace_roots: dict[str, Path],
    *,
    conditions: tuple[str | None, ...] | None = None,
) -> list[ReplayCell]:
    """Run the bank over full and per-operator ablation conditions."""
    conditions = conditions if conditions is not None else (None, *OPERATORS)
    cells: list[ReplayCell] = []
    for case in cases:
        root = workspace_roots.get(case.case_id, workspace_roots.get(case.operator, Path.cwd()))
        cells.extend(
            replay_case(case, root, disabled_operator=condition) for condition in conditions
        )
    return cells


def replay_summary(cells: list[ReplayCell]) -> dict[str, Any]:
    """Compact per-condition tallies for fast development diffs."""

    by_condition: dict[str, list[ReplayCell]] = {}
    for cell in cells:
        by_condition.setdefault(cell.condition_id, []).append(cell)
    summary: dict[str, Any] = {}
    for condition_id, group in by_condition.items():
        summary[condition_id] = {
            "cells": len(group),
            "certified": sum(cell.certified for cell in group),
            "delivered": sum(cell.delivered for cell in group),
            "with_errors": sum(bool(cell.errors) for cell in group),
            "request_digests": sorted({cell.request_digest for cell in group})[:8],
        }
    summary["_as_of"] = datetime.now(UTC).isoformat()
    return summary

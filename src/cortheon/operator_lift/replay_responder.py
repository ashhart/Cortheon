"""ReplayResponder: scripted host-and-model stand-in for the operator-lift replay."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cortheon.cognitive_core.runtime import CognitiveRuntime
from cortheon.operator_lift.models import LiftCase

MAX_REPLAY_STEPS = 32


def _read_receipt(path: str) -> str:
    return (
        '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
        '"args":{"filePath":"' + path + '"}}\n'
    )


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


class ReplayResponder:
    """Scripted host-and-model stand-in obeying the condition's protocol."""

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
        base = (self.case.evidence[0][1] if self.case.evidence else self.case.causal_family)[:80]
        return [
            {
                "statement": f"{base} is the operative cause in {self.case.causal_family}.",
                "falsification_test": "Find evidence refuting the operative cause.",
            },
            {
                "statement": f"A distinct mechanism drives {self.case.causal_family}.",
                "falsification_test": "Find evidence refuting the alternative mechanism.",
            },
        ]

    def _completion_hypotheses(self) -> list[dict[str, Any]]:
        if not self.framing:
            return []
        return [
            {
                "statement": h["statement"],
                "falsification_test": h["falsification_test"],
                "status": "supported",
                "evidence_ids": list(self.accepted),
            }
            for h in self._hypotheses()
        ]

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

        served_actions: set[str] = set()

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
                observations = self._observations_for(request)
                served_actions.update(
                    str(o.get("source") or "").removeprefix("replay:")
                    for o in observations
                    if str(o.get("source") or "").startswith("replay:actions/")
                )
                if self.case.operator == "adaptive_stopping" and all(
                    f"actions/{action}.txt" in served_actions
                    for action in (self.case.oracle.get("expected_actions") or ())
                ):
                    pass
                else:
                    result = guarded(
                        lambda request=request, observations=observations: runtime.observe(
                            session_id,
                            observations,
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
                required = action.get("required_fields") or []
                draft = self.answer if "draft" in required else None
                hypotheses = [] if draft else (self._hypotheses() if self.framing else [])
                result = guarded(
                    lambda draft=draft, hypotheses=hypotheses: runtime.step(
                        session_id,
                        hypotheses=hypotheses,
                        open_questions=[],
                        draft=draft,
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
                self._verdict = "complete"
                break
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
            errors.append("replay: runtime reached no terminal decision")
        self._errors = errors
        return payload

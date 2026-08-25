"""BriefMixin for CognitiveRuntime."""

from __future__ import annotations

import copy
from itertools import pairwise
from typing import Any

from cortheon.cognitive_core.models import Investigation
from cortheon.cognitive_core.runtime_state import RuntimeState
from cortheon.cognitive_core.semantic_graph import _phrase_mentioned
from cortheon.cognitive_graph import rank_information_gain
from cortheon.cognitive_program import select_operator
from cortheon.cognitive_protocol import evaluation_operator


class BriefMixin(RuntimeState):
    """Brief responsibilities of CognitiveRuntime."""

    @staticmethod
    def _cognition_brief(
        session: Investigation,
        *,
        next_action: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Return one bounded public reasoning move for the host model."""

        derivations = [
            item for item in context.get("deterministic_derivations", ()) if isinstance(item, dict)
        ]
        action_type = str(next_action.get("type") or "")
        required_fields = {
            str(item) for item in next_action.get("required_fields", ()) if isinstance(item, str)
        }
        if any(item.get("operation") == "semantic_conflict" for item in derivations):
            stage = "challenge"
        elif any(
            item.get("operation")
            in {"diagnostic_chain", "ordered_plan", "semantic_chain", "semantic_rule"}
            for item in derivations
        ):
            stage = "connect"
        elif not session.observations:
            stage = "orient"
        elif "hypotheses" in required_fields:
            stage = "frame"
        elif "hypothesis_updates" in required_fields:
            stage = "update"
        elif action_type == "harness_tool":
            stage = "discover"
        elif action_type == "challenge":
            stage = "challenge"
        elif action_type in {"verify", "finish"}:
            stage = "verify"
        else:
            stage = "synthesize"

        active_operator = select_operator(
            session.program,
            next_action,
            has_derivation=any(
                item.get("operation")
                in {"diagnostic_chain", "ordered_plan", "semantic_chain", "semantic_rule"}
                for item in derivations
            ),
            has_conflict=any(item.get("operation") == "semantic_conflict" for item in derivations),
        )

        stage_moves = {
            "orient": [
                "Frame the exact deliverable and the strongest way an answer could be wrong.",
                "Acquire the requested live evidence before trusting model memory.",
                "Keep at least one alternative explanation for revision.",
            ],
            "frame": [
                "Generate genuinely different explanations, including a measurement or task-design artifact.",
                "Give each explanation an observable falsification test.",
                "Identify the cheapest evidence to separate the leading alternatives.",
            ],
            "discover": [
                "Use the pending request to reduce the highest-value uncertainty.",
                "Prefer evidence that separates alternatives over more confirmation.",
                "Revise the working hypothesis when the new observation conflicts.",
            ],
            "update": [
                "Classify the new evidence against the named hypothesis before gathering more.",
                "Use supported, refuted, or uncertain and cite only accepted evidence ids.",
                "If uncertainty remains, state what observation would change the classification.",
            ],
            "connect": [
                "Treat each derived bridge as a candidate inference, not a quoted fact.",
                "Check every edge against its separate source before accepting the bridge.",
                "State the new conclusion and the intermediate nodes that make it follow.",
            ],
            "challenge": [
                "Attack the strongest supported conclusion with the best live counterexample.",
                "Remove, qualify, or re-investigate any claim failing the attack.",
            ],
            "synthesize": [
                "Choose the hypothesis best supported after attempted falsification.",
                "Separate observed facts, derived conclusions, and unresolved uncertainty.",
                "Bind every factual claim to accepted evidence ids.",
            ],
            "verify": [
                "Check that the answer resolves the requested deliverable rather than only summarizing evidence.",
                "Require observable completion evidence for code changes and current claims.",
            ],
        }

        derived_insights: list[dict[str, Any]] = []
        for item in derivations:
            if item.get("operation") == "semantic_conflict":
                for conflict in item.get("conflicts", ()):
                    if not isinstance(conflict, dict):
                        continue
                    derived_insights.append(
                        {
                            "kind": "cross_source_conflict",
                            "statement": (
                                f"Sources disagree about {conflict.get('entity')} "
                                f"{conflict.get('relation')}: "
                                + " versus ".join(
                                    str(value) for value in conflict.get("targets", ())
                                )
                                + "."
                            ),
                            "sources": list(conflict.get("sources", ())),
                            "status": "requires_disambiguation",
                        }
                    )
                continue
            if item.get("operation") not in {
                "ordered_plan",
                "semantic_chain",
                "semantic_rule",
            }:
                continue
            nodes = [str(value) for value in item.get("nodes", ()) if str(value).strip()]
            sources = [str(value) for value in item.get("sources", ()) if str(value).strip()]
            relations = [str(value) for value in item.get("relations", ()) if str(value).strip()]
            if len(nodes) < 2:
                continue
            chain: list[dict[str, str]] = []
            for index, (source, target) in enumerate(pairwise(nodes)):
                edge: dict[str, str] = {
                    "from": source,
                    "relation": relations[index] if index < len(relations) else "linked_to",
                    "to": target,
                }
                if index < len(sources):
                    edge["source"] = sources[index]
                chain.append(edge)
            verbatim_in_one_observation = any(
                all(_phrase_mentioned(observation.content, node) for node in nodes)
                for observation in session.observations.values()
                if observation.status != "failed" and not observation.quarantine_flags
            )
            intermediate = ", then ".join(nodes[1:-1])
            statement = (
                f"{nodes[0]} is connected to {nodes[-1]}"
                + (f" through {intermediate}" if intermediate else "")
                + "."
            )
            derived_insights.append(
                {
                    "kind": "cross_source_inference",
                    "statement": statement,
                    "chain": chain,
                    "source_count": len(set(sources)),
                    "novel_cross_source_inference": (
                        len(set(sources)) >= 2 and not verbatim_in_one_observation
                    ),
                    "status": "candidate_until_challenged",
                    **(
                        {
                            "premises": copy.deepcopy(item.get("premises", [])),
                            "applied_rule": copy.deepcopy(item.get("rule", {})),
                        }
                        if item.get("operation") == "semantic_rule"
                        else {}
                    ),
                }
            )

        unresolved: list[str] = [str(item) for item in context.get("open_questions", ())][:4]
        unresolved.extend(
            hypothesis.statement
            for hypothesis in session.hypotheses.values()
            if hypothesis.status in {"open", "uncertain"}
        )
        request = next_action.get("request")
        evidence_target = None
        if isinstance(request, dict):
            unresolved_ids = {
                hypothesis.hypothesis_id: (
                    1.0 if hypothesis.status in {"open", "uncertain"} else 0.25
                )
                for hypothesis in session.hypotheses.values()
                if hypothesis.status != "refuted"
            }
            if len(unresolved_ids) < 2:
                unresolved_ids = {
                    str(item.get("requirement_id")): 1.0
                    for item in context.get("requirements", ())
                    if isinstance(item, dict)
                    and item.get("status") != "satisfied"
                    and item.get("requirement_id")
                }
            parameters = request.get("parameters", {})
            budget = parameters.get("tool_call_budget", 1) if isinstance(parameters, dict) else 1
            linked = str(request.get("hypothesis_id") or "")
            resolves = [linked] if linked in unresolved_ids else list(unresolved_ids)
            controller = parameters.get("controller") if isinstance(parameters, dict) else None
            if isinstance(controller, dict) and isinstance(
                controller.get("selected"),
                dict,
            ):
                selection = copy.deepcopy(controller["selected"])
                alternatives = copy.deepcopy(controller.get("alternatives", []))
                stop_when = controller.get("stop_when")
            else:
                ranked = rank_information_gain(
                    unresolved_ids,
                    [
                        {
                            "action_id": str(request.get("request_id") or "pending"),
                            "resolves": resolves,
                            "cost": max(1, int(budget)) if isinstance(budget, int) else 1,
                            "reliability": 0.9,
                        }
                    ],
                )
                selection = ranked[0] if ranked else None
                alternatives = []
                stop_when = None
            evidence_target = {
                "capability": request.get("capability"),
                "reason": request.get("reason"),
                "success_condition": request.get("success_condition"),
                "tool_call_budget": (
                    parameters.get("tool_call_budget") if isinstance(parameters, dict) else None
                ),
                "selection": selection,
                "alternatives": alternatives,
                "stop_when": stop_when,
            }
        stage_operator = {
            "frame": "hypothesis_framing",
            "discover": "discriminating_evidence",
            "update": "contradiction_revision",
            "connect": "cross_source_derivation",
            "challenge": "contradiction_revision",
            "verify": "verification",
        }.get(stage)
        reasoning_moves = (
            stage_moves[stage]
            if stage_operator is None
            or evaluation_operator(session.evaluation_profile, stage_operator)
            else []
        )
        return {
            "stage": stage,
            "task_frame": {
                "deliverable": session.deliverable,
                "constraints": list(context.get("constraints", ())),
                "requirements": list(context.get("requirements", ())),
            },
            "reasoning_moves": reasoning_moves,
            "program": {
                "program_id": session.program["program_id"],
                "active_operator": active_operator,
                "proof_obligations": copy.deepcopy(session.program["proof_obligations"]),
                "budgets": copy.deepcopy(session.program["budgets"]),
                "stop_conditions": list(session.program["stop_conditions"]),
            },
            "derived_insights": derived_insights,
            "unresolved": unresolved[:6],
            "evidence_target": evidence_target,
            "decision_rule": (
                "Revise when discriminating evidence changes the best explanation; "
                "finish only when the conclusion and its derivation survive challenge."
            ),
        }

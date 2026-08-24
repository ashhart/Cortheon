"""Small valid private cases for every closed North Star oracle class."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any

from cortheon.parity_benchmark_core.oracle_web import _truth_digest


def encoded(payload: dict[str, Any], explanation: str = "") -> str:
    value = dict(payload)
    if explanation:
        value["explanation"] = explanation
    return "```json\n" + json.dumps(value) + "\n```"


def cases() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    values = {
        "ambiguity_resolution": _ambiguity(),
        "constraint_bound_planning": _planning(),
        "cross_file_numeric_join": _numeric(),
        "current_web_research": _web(),
        "evidence_bound_debugging": _debugging(),
        "long_horizon_execution": _horizon(),
        "novel_abductive_synthesis": _abduction(),
        "semantic_cross_document_reasoning": _semantic(),
    }
    return {key: (case, deepcopy(answer)) for key, (case, answer) in values.items()}


def _base(task_class: str, grader_type: str, oracle: dict, documents: list[dict]) -> dict:
    prompt = "Solve the supplied task and return the requested structured result."
    if task_class == "current_web_research":
        prompt += (
            f" Use exact as_of {oracle['as_of']}. Return JSON fields as_of, sources with "
            "canonical_url, claims, and contradictions."
        )
    return {
        "id": task_class,
        "task_class": task_class,
        "category": task_class,
        "domain": task_class,
        "difficulty": "hard",
        "prompt": prompt,
        "documents": documents,
        "expected_verdict": "allow",
        "grader": {
            "type": grader_type,
            "oracle_version": 1,
            "oracle": oracle,
            "oracle_provenance": "frozen_external_pack",
        },
    }


def _document(uri: str, text: str) -> dict:
    return {"uri": uri, "title": uri.rsplit("/", 1)[-1], "text": text}


def _bindings(documents: list[dict]) -> list[dict]:
    return [
        {
            "id": document["uri"],
            "sha256": hashlib.sha256(document["text"].encode()).hexdigest(),
        }
        for document in documents
    ]


def _ambiguity():
    docs = [
        _document(
            "benchmark://ambiguity/brief",
            "Intent options are compare_net_cost_gbp and compare_net_cost_usd. "
            "Currency discriminator is GBP.",
        )
    ]
    answer = {
        "resolved_intent": "compare_net_cost_gbp",
        "decision": "answer",
        "discriminators": [{"id": "currency", "value": "GBP", "source_id": docs[0]["uri"]}],
    }
    oracle = {**answer, "accepted_clarification_ids": [], "source_bindings": _bindings(docs)}
    return _base("ambiguity_resolution", "ambiguity_oracle", oracle, docs), answer


def _planning():
    docs = [
        _document("benchmark://plan/order", "Add schema, then dual-write, then drop legacy."),
        _document("benchmark://plan/window", "Rollback window is 24h."),
    ]
    answer = {
        "steps": [
            {"id": "add", "action": "add_schema", "source_id": docs[0]["uri"]},
            {"id": "dual", "action": "dual_write", "source_id": docs[0]["uri"]},
            {"id": "drop", "action": "drop_legacy", "source_id": docs[0]["uri"]},
        ],
        "dependencies": [["add", "dual"], ["dual", "drop"]],
        "constraints": [
            {
                "id": "rollback",
                "step_id": "drop",
                "operator": "after",
                "value": "24h",
                "unit": "duration",
                "source_id": docs[1]["uri"],
            }
        ],
    }
    oracle = {
        **answer,
        "forbidden_dependencies": [["drop", "add"]],
        "source_bindings": _bindings(docs),
    }
    return _base("constraint_bound_planning", "constraint_graph", oracle, docs), answer


def _numeric():
    docs = [
        _document("benchmark://numeric/rate", "Daily output is 12 widgets/day."),
        _document("benchmark://numeric/days", "The run lasts 4 day."),
        _document("benchmark://numeric/factor", "Adjustment factor is 1.25 scalar."),
    ]
    answer = {
        "facts": [
            {"id": "rate", "value": 12, "unit": "widgets/day", "source_id": docs[0]["uri"]},
            {"id": "days", "value": 4, "unit": "day", "source_id": docs[1]["uri"]},
            {"id": "factor", "value": 1.25, "unit": "scalar", "source_id": docs[2]["uri"]},
        ],
        "derivation": [
            {"id": "subtotal", "op": "multiply", "args": ["rate", "days"], "unit": "widgets"},
            {"id": "total", "op": "multiply", "args": ["subtotal", "factor"], "unit": "widgets"},
        ],
        "result": {"ref": "total", "value": 60, "unit": "widgets"},
    }
    oracle = {
        **answer,
        "allowed_operations": ["multiply"],
        "necessary_fact_ids": ["rate", "days", "factor"],
        "source_bindings": _bindings(docs),
    }
    return _base("cross_file_numeric_join", "numeric_derivation", oracle, docs), answer


def _debugging():
    docs = [
        _document("benchmark://debug/log", "Symptom code WAIT_TIMEOUT appears under load."),
        _document("benchmark://debug/config", "Cause code POOL_UNDERSIZED: pool=8 workers=12."),
        _document("benchmark://debug/runbook", "Fix POOL_TO_12 then verify ZERO_WAITS."),
    ]
    answer = {
        "symptom": "WAIT_TIMEOUT",
        "cause": "POOL_UNDERSIZED",
        "fix": "POOL_TO_12",
        "verification": "ZERO_WAITS",
        "evidence": [
            {"stage": "symptom", "source_ids": [docs[0]["uri"]]},
            {"stage": "cause", "source_ids": [docs[1]["uri"]]},
            {"stage": "fix", "source_ids": [docs[2]["uri"]]},
            {"stage": "verification", "source_ids": [docs[2]["uri"]]},
        ],
    }
    oracle = {
        **answer,
        "evidence_facts": [
            {"stage": "symptom", "fact": "WAIT_TIMEOUT", "source_id": docs[0]["uri"]},
            {"stage": "cause", "fact": "POOL_UNDERSIZED", "source_id": docs[1]["uri"]},
            {"stage": "cause", "fact": "pool=8", "source_id": docs[1]["uri"]},
            {"stage": "fix", "fact": "POOL_TO_12", "source_id": docs[2]["uri"]},
            {"stage": "verification", "fact": "ZERO_WAITS", "source_id": docs[2]["uri"]},
        ],
        "source_bindings": _bindings(docs),
    }
    return _base("evidence_bound_debugging", "causal_debugging", oracle, docs), answer


def _horizon():
    docs = [
        _document("benchmark://horizon/a", "BACKFILL precedes MIGRATE."),
        _document("benchmark://horizon/b", "MIGRATE precedes VERIFY; gate is ZERO_GAPS."),
        _document("benchmark://horizon/c", "UNBLOCK. Final owner is OWNER_ELENA."),
    ]
    answer = {
        "steps": [
            {"id": "backfill", "action": "BACKFILL", "source_id": docs[0]["uri"]},
            {"id": "migrate", "action": "MIGRATE", "source_id": docs[1]["uri"]},
            {"id": "verify", "action": "VERIFY", "source_id": docs[1]["uri"]},
            {"id": "unblock", "action": "UNBLOCK", "source_id": docs[2]["uri"]},
        ],
        "dependencies": [["backfill", "migrate"], ["migrate", "verify"], ["verify", "unblock"]],
        "gates": [
            {
                "id": "ZERO_GAPS",
                "after_step": "verify",
                "condition": "ZERO_GAPS",
                "source_id": docs[1]["uri"],
            }
        ],
        "terminal_step_id": "unblock",
        "final_owner": "OWNER_ELENA",
    }
    oracle = {**answer, "owner_source_id": docs[2]["uri"], "source_bindings": _bindings(docs)}
    return _base("long_horizon_execution", "horizon_graph", oracle, docs), answer


def _abduction():
    docs = [
        _document("benchmark://abduction/a", "Premise ALPHA: failures align with region east."),
        _document("benchmark://abduction/b", "Premise BETA: only signed requests fail."),
        _document(
            "benchmark://abduction/c", "Observation GAMMA: rotating the east key clears failures."
        ),
    ]
    selected = {
        "subject": "east signing key",
        "relation": "causes",
        "object": "signed request failures",
    }
    rival = {
        "subject": "network path",
        "relation": "causes",
        "object": "regional failures",
    }
    second_rival = {
        "subject": "request payload",
        "relation": "causes",
        "object": "signed request failures",
    }
    answer = {
        "hypotheses": [
            {"proposition": selected, "status": "selected"},
            {"proposition": rival, "status": "ruled_out"},
        ],
        "selected_hypothesis": selected,
        "premises": [
            {"id": "p1", "proposition_id": "ALPHA", "fact": "ALPHA", "source_id": docs[0]["uri"]},
            {"id": "p2", "proposition_id": "BETA", "fact": "BETA", "source_id": docs[1]["uri"]},
            {"id": "p3", "proposition_id": "GAMMA", "fact": "GAMMA", "source_id": docs[2]["uri"]},
        ],
        "discriminator": {
            "observation": "GAMMA",
            "supports": selected,
            "rules_out": rival,
            "source_id": docs[2]["uri"],
        },
        "conclusion": "east_signing_key_mismatch",
    }
    oracle = {
        "selected_proposition": selected,
        "accepted_rivals": [rival, second_rival],
        "premises": answer["premises"],
        "discriminator": answer["discriminator"],
        "conclusion": answer["conclusion"],
        "necessary_source_ids": [document["uri"] for document in docs],
        "conclusion_dependencies": ["ALPHA", "BETA", "GAMMA"],
        "source_bindings": _bindings(docs),
    }
    case = _base("novel_abductive_synthesis", "abductive_oracle", oracle, docs)
    case["prompt"] += (
        " Hypothesis options: east signing key causes signed request failures; "
        "network path causes regional failures; request payload causes signed request failures."
    )
    return case, answer


def _semantic():
    docs = [
        _document("benchmark://semantic/a", "Fleet has 6 pods. Each pod holds 8 cells."),
        _document("benchmark://semantic/b", "All cells are active, not retired."),
    ]
    answer = {
        "hops": [
            {
                "id": "pod_count",
                "subject": "fleet",
                "relation": "has",
                "object": "pods",
                "polarity": "positive",
                "quantity": 6,
                "unit": "pods",
                "source_ids": [docs[0]["uri"]],
            },
            {
                "id": "cell_state",
                "subject": "cells",
                "relation": "state",
                "object": "active",
                "polarity": "positive",
                "quantity": 48,
                "unit": "cells",
                "source_ids": [docs[0]["uri"], docs[1]["uri"]],
            },
        ],
        "conclusion": "48_active_cells",
        "necessary_source_ids": [document["uri"] for document in docs],
    }
    oracle = {
        **answer,
        "source_premises": [
            {"source_id": docs[0]["uri"], "facts": ["Fleet", "6 pods", "8 cells"]},
            {"source_id": docs[1]["uri"], "facts": ["cells", "active", "not retired"]},
        ],
        "source_bindings": _bindings(docs),
    }
    return _base(
        "semantic_cross_document_reasoning", "semantic_document_graph", oracle, docs
    ), answer


def _web():
    now = datetime.now(UTC).replace(microsecond=0)
    acquired = (now - timedelta(minutes=10)).isoformat()
    revalidated = (now - timedelta(minutes=1)).isoformat()
    sources = [
        {
            "canonical_url": "https://primary.example/release",
            "origin_id": "primary",
            "syndication_group": "primary",
            "published_at": "2026-08-22T10:00:00+00:00",
            "retrieved_at": revalidated,
            "authority": "primary",
            "content_sha256": "a" * 64,
        },
        {
            "canonical_url": "https://analysis.example/report",
            "origin_id": "analysis",
            "syndication_group": "analysis",
            "published_at": "2026-08-22T12:00:00+00:00",
            "retrieved_at": revalidated,
            "authority": "secondary",
            "content_sha256": "b" * 64,
        },
        {
            "canonical_url": "https://mirror.example/story",
            "origin_id": "mirror",
            "syndication_group": "analysis",
            "published_at": "2026-08-21T12:00:00+00:00",
            "retrieved_at": revalidated,
            "authority": "secondary",
            "content_sha256": "c" * 64,
        },
    ]
    claims = [
        {
            "id": "release",
            "value": "2.0",
            "source_urls": [
                "https://primary.example/release",
                "https://analysis.example/report",
            ],
        }
    ]
    conflicts = [
        {
            "claim_id": "release",
            "source_url": "https://mirror.example/story",
            "rejected_value": "1.9",
            "resolved_by_url": "https://primary.example/release",
        }
    ]
    oracle = {
        "as_of": acquired,
        "revalidated_at": revalidated,
        "valid_until": (now + timedelta(days=1)).isoformat(),
        "truth_digest": "",
        "revalidated_truth_digest": "",
        "sources": sources,
        "origin_equivalence": [
            {"id": "primary", "hosts": ["primary.example"]},
            {"id": "analysis", "hosts": ["analysis.example"]},
            {"id": "mirror", "hosts": ["mirror.example"]},
        ],
        "claims": claims,
        "contradictions": conflicts,
        "acquisition_attestation": {
            "schema_version": 1,
            "evaluator_id": "lab",
            "policy_sha256": "d" * 64,
            "records": [
                {
                    "source_url": source["canonical_url"],
                    "requested_url": source["canonical_url"],
                    "final_url": source["canonical_url"],
                    "redirect_chain": [],
                    "initial_sha256": source["content_sha256"],
                    "revalidated_sha256": source["content_sha256"],
                    "acquired_at": acquired,
                    "revalidated_at": source["retrieved_at"],
                }
                for source in sources
            ],
        },
    }
    oracle["truth_digest"] = oracle["revalidated_truth_digest"] = _truth_digest(oracle)
    answer = {
        "as_of": oracle["as_of"],
        "sources": [{"canonical_url": item["canonical_url"]} for item in sources],
        "claims": claims,
        "contradictions": conflicts,
    }
    return _base("current_web_research", "current_web_claims", oracle, []), answer

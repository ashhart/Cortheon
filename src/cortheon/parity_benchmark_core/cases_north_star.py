"""Deterministic runnable cases for seven non-web, non-patch North Star classes."""

from __future__ import annotations

import hashlib
from typing import Any


def north_star_cases() -> list[dict[str, Any]]:
    return [
        _ambiguity_case(),
        _planning_case(),
        _numeric_case(),
        _debugging_case(),
        _horizon_case(),
        _abduction_case(),
        _semantic_case(),
    ]


def _case(task_class: str, grader_type: str, prompt: str, documents: list[dict], oracle: dict):
    public_identity = {
        "ambiguity_resolution": ("ns_dev_001", "reasoning", "reasoning"),
        "constraint_bound_planning": ("ns_dev_002", "planning", "reasoning"),
        "cross_file_numeric_join": ("ns_dev_003", "documents", "reasoning"),
        "evidence_bound_debugging": ("ns_dev_004", "debugging", "coding"),
        "long_horizon_execution": ("ns_dev_005", "execution", "planning"),
        "novel_abductive_synthesis": ("ns_dev_006", "reasoning", "research"),
        "semantic_cross_document_reasoning": ("ns_dev_007", "documents", "reasoning"),
    }[task_class]
    return {
        "id": public_identity[0],
        "task_class": task_class,
        "category": public_identity[1],
        "domain": public_identity[2],
        "difficulty": "hard",
        "prompt": prompt,
        "documents": documents,
        "expected_verdict": "allow",
        "grader": {"type": grader_type, "oracle_version": 1, "oracle": oracle},
    }


def _doc(uri: str, text: str) -> dict:
    return {"uri": uri, "title": uri.rsplit("/", 1)[-1], "text": text}


def _bindings(documents: list[dict]) -> list[dict]:
    return [
        {"id": item["uri"], "sha256": hashlib.sha256(item["text"].encode()).hexdigest()}
        for item in documents
    ]


def _ambiguity_case() -> dict:
    docs = [_doc("benchmark://north-star/ambiguity", "Currency is GBP, not USD.")]
    discriminator = [{"id": "currency", "value": "GBP", "source_id": docs[0]["uri"]}]
    oracle = {
        "resolved_intent": "net_cost_gbp",
        "decision": "answer",
        "discriminators": discriminator,
        "accepted_clarification_ids": [],
        "source_bindings": _bindings(docs),
    }
    prompt = (
        "Resolve intent IDs net_cost_gbp versus net_cost_usd. Return JSON with "
        "resolved_intent, decision, and discriminators[{id,value,source_id}]. "
        "Use the document URI as source_id; explanation is optional."
    )
    return _case("ambiguity_resolution", "ambiguity_oracle", prompt, docs, oracle)


def _planning_case() -> dict:
    docs = [
        _doc("benchmark://north-star/plan", "ADD_SCHEMA then DUAL_WRITE then DROP_LEGACY."),
        _doc("benchmark://north-star/window", "ROLLBACK_WINDOW is 24h."),
    ]
    steps = [
        {"id": action, "action": action, "source_id": docs[0]["uri"]}
        for action in ("ADD_SCHEMA", "DUAL_WRITE", "DROP_LEGACY")
    ]
    oracle = {
        "steps": steps,
        "dependencies": [["ADD_SCHEMA", "DUAL_WRITE"], ["DUAL_WRITE", "DROP_LEGACY"]],
        "constraints": [
            {
                "id": "ROLLBACK_WINDOW",
                "step_id": "DROP_LEGACY",
                "operator": "after",
                "value": "24h",
                "unit": "duration",
                "source_id": docs[1]["uri"],
            }
        ],
        "forbidden_dependencies": [["DROP_LEGACY", "ADD_SCHEMA"]],
        "source_bindings": _bindings(docs),
    }
    prompt = (
        "Return JSON steps[{id,action,source_id}], dependencies[[before,after]], and "
        "constraints[{id,step_id,operator,value,unit,source_id}]. Use visible uppercase "
        "codes as IDs and document URIs as source IDs."
    )
    return _case("constraint_bound_planning", "constraint_graph", prompt, docs, oracle)


def _numeric_case() -> dict:
    docs = [
        _doc("benchmark://north-star/rate", "RATE is 12 widgets/day."),
        _doc("benchmark://north-star/days", "DAYS is 4 day."),
        _doc("benchmark://north-star/factor", "FACTOR is 1.25 scalar."),
    ]
    facts = [
        {"id": item["uri"], "value": value, "unit": unit, "source_id": item["uri"]}
        for item, value, unit in zip(
            docs, (12, 4, 1.25), ("widgets/day", "day", "scalar"), strict=True
        )
    ]
    oracle = {
        "facts": facts,
        "allowed_operations": ["multiply"],
        "necessary_fact_ids": [item["id"] for item in facts],
        "derivation": [
            {
                "id": "subtotal",
                "op": "multiply",
                "args": [facts[0]["id"], facts[1]["id"]],
                "unit": "widgets",
            },
            {
                "id": "total",
                "op": "multiply",
                "args": ["subtotal", facts[2]["id"]],
                "unit": "widgets",
            },
        ],
        "result": {"ref": "total", "value": 60, "unit": "widgets"},
        "source_bindings": _bindings(docs),
    }
    prompt = (
        "Return JSON facts[{id,value,unit,source_id}], derivation[{id,op,args,unit}], "
        "and result{ref,value,unit}. Use each document URI as both fact id and source_id."
    )
    return _case("cross_file_numeric_join", "numeric_derivation", prompt, docs, oracle)


def _debugging_case() -> dict:
    docs = [
        _doc(
            "benchmark://north-star/debug-log",
            "Requests time out while waiting for an available connection slot.",
        ),
        _doc(
            "benchmark://north-star/debug-config",
            "There are 12 active workers, a pool size of 8, and each active worker "
            "requires one connection.",
        ),
        _doc(
            "benchmark://north-star/debug-runbook",
            "After changing capacity, replay the load and require zero waits for a slot.",
        ),
    ]
    evidence = [
        {"stage": stage, "source_ids": [docs[index]["uri"]]}
        for stage, index in (("symptom", 0), ("cause", 1), ("fix", 2), ("verification", 2))
    ]
    oracle = {
        "symptom": "WAIT_TIMEOUT",
        "cause": "POOL_UNDERSIZED",
        "fix": "POOL_TO_12",
        "verification": "ZERO_WAITS",
        "evidence": evidence,
        "evidence_facts": [
            {
                "stage": "symptom",
                "fact": "waiting for an available connection slot",
                "source_id": docs[0]["uri"],
            },
            {"stage": "cause", "fact": "12 active workers", "source_id": docs[1]["uri"]},
            {"stage": "cause", "fact": "pool size of 8", "source_id": docs[1]["uri"]},
            {"stage": "fix", "fact": "changing capacity", "source_id": docs[2]["uri"]},
            {"stage": "verification", "fact": "zero waits", "source_id": docs[2]["uri"]},
        ],
        "source_bindings": _bindings(docs),
    }
    prompt = (
        "Return JSON symptom, cause, fix, verification, and evidence[{stage,source_ids}]. "
        "Choose semantic IDs from: symptom WAIT_TIMEOUT; causes POOL_UNDERSIZED or "
        "NETWORK_CONGESTION; fixes POOL_TO_12 or ADD_RETRIES; verification ZERO_WAITS. "
        "Use document URIs; explanation is optional."
    )
    return _case("evidence_bound_debugging", "causal_debugging", prompt, docs, oracle)


def _horizon_case() -> dict:
    docs = [
        _doc("benchmark://north-star/horizon-a", "BACKFILL precedes MIGRATE."),
        _doc("benchmark://north-star/horizon-b", "MIGRATE precedes VERIFY; gate ZERO_GAPS."),
        _doc("benchmark://north-star/horizon-c", "VERIFY precedes UNBLOCK by OWNER_ELENA."),
    ]
    steps = [
        {"id": action, "action": action, "source_id": docs[min(index, 2)]["uri"]}
        for index, action in enumerate(("BACKFILL", "MIGRATE", "VERIFY", "UNBLOCK"))
    ]
    oracle = {
        "steps": steps,
        "dependencies": [["BACKFILL", "MIGRATE"], ["MIGRATE", "VERIFY"], ["VERIFY", "UNBLOCK"]],
        "gates": [
            {
                "id": "ZERO_GAPS",
                "after_step": "VERIFY",
                "condition": "ZERO_GAPS",
                "source_id": docs[1]["uri"],
            }
        ],
        "terminal_step_id": "UNBLOCK",
        "final_owner": "OWNER_ELENA",
        "owner_source_id": docs[2]["uri"],
        "source_bindings": _bindings(docs),
    }
    prompt = (
        "Return JSON steps, dependencies, gates, terminal_step_id, and final_owner. "
        "Use visible uppercase codes as IDs and document URIs as source IDs."
    )
    return _case("long_horizon_execution", "horizon_graph", prompt, docs, oracle)


def _abduction_case() -> dict:
    docs = [
        _doc("benchmark://north-star/abd-a", "ALPHA: failures align with east."),
        _doc("benchmark://north-star/abd-b", "BETA: only signed requests fail."),
        _doc("benchmark://north-star/abd-c", "GAMMA: rotating east key clears failures."),
    ]
    selected = {
        "subject": "east signing key",
        "relation": "causes",
        "object": "signed request failures",
    }
    rivals = [
        {"subject": "network path", "relation": "causes", "object": "regional failures"},
        {"subject": "request payload", "relation": "causes", "object": "signed request failures"},
    ]
    premises = [
        {"id": f"p{index + 1}", "proposition_id": code, "fact": code, "source_id": doc["uri"]}
        for index, (code, doc) in enumerate(zip(("ALPHA", "BETA", "GAMMA"), docs, strict=True))
    ]
    oracle = {
        "selected_proposition": selected,
        "accepted_rivals": rivals,
        "premises": premises,
        "discriminator": {
            "observation": "GAMMA",
            "supports": selected,
            "rules_out": rivals[0],
            "source_id": docs[2]["uri"],
        },
        "conclusion": "east_signing_key_mismatch",
        "necessary_source_ids": [doc["uri"] for doc in docs],
        "conclusion_dependencies": ["ALPHA", "BETA", "GAMMA"],
        "source_bindings": _bindings(docs),
    }
    prompt = (
        "Evaluate these typed hypothesis options: east signing key causes signed request "
        "failures; network path causes regional failures; request payload causes signed "
        "request failures. Return at least two as "
        "{proposition:{subject,relation,object},status}; "
        "return selected_hypothesis, source-bound premises, discriminator, and conclusion. "
        "Use ALPHA/BETA/GAMMA and document URIs where cited. Form conclusion IDs by "
        "lowercasing the selected subject with underscores and suffixing _mismatch."
    )
    return _case("novel_abductive_synthesis", "abductive_oracle", prompt, docs, oracle)


def _semantic_case() -> dict:
    docs = [
        _doc("benchmark://north-star/sem-a", "Fleet has 6 pods; each pod holds 8 cells."),
        _doc("benchmark://north-star/sem-b", "All cells are active, not retired."),
    ]
    hops = [
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
    ]
    oracle = {
        "hops": hops,
        "conclusion": "48_active_cells",
        "necessary_source_ids": [doc["uri"] for doc in docs],
        "source_premises": [
            {"source_id": docs[0]["uri"], "facts": ["fleet", "6 pods", "8 cells"]},
            {"source_id": docs[1]["uri"], "facts": ["cells", "active", "not retired"]},
        ],
        "source_bindings": _bindings(docs),
    }
    prompt = (
        "Return JSON hops[{id,subject,relation,object,polarity,quantity,unit,source_ids}], "
        "conclusion, and necessary_source_ids. Use hop IDs pod_count and cell_state, "
        "document URIs, and conclusion ID <quantity>_<state>_<object>."
    )
    return _case(
        "semantic_cross_document_reasoning", "semantic_document_graph", prompt, docs, oracle
    )

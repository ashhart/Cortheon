"""Public solvability and coverage for the runnable North Star development bank."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

import pytest

from cortheon.parity_benchmark_core.casepack import _normalize_cases
from cortheon.parity_benchmark_core.cases_builtin import _builtin_cases
from cortheon.parity_benchmark_core.current_web_case_factory import evaluator_current_web_case
from cortheon.parity_benchmark_core.grading import grade_answer
from cortheon.parity_benchmark_core.oracle_taxonomy import (
    ORACLE_SPECS,
    TASK_CLASSES,
    proof_binding,
)
from cortheon.parity_gates.projection import public_case_projection, public_task_hash


def _normalized_bank() -> list[dict[str, Any]]:
    return _normalize_cases(
        {"cases": _builtin_cases()}, built_in=True, allow_external_patch_tests=False
    )


def test_builtin_bank_has_one_closed_proof_oracle_for_every_task_class() -> None:
    bank = _normalized_bank()
    bound = [case for case in bank if proof_binding(case) is not None]

    assert {case["task_class"] for case in bound} == TASK_CLASSES - {"current_web_research"}
    assert all(case["grader"]["oracle_version"] == 1 for case in bound)
    assert ORACLE_SPECS["novel_abductive_synthesis"].assurance == "source_necessary_abduction"
    assert "creative" not in ORACLE_SPECS["novel_abductive_synthesis"].assurance


def test_builtin_bank_and_public_digest_are_stable_across_repeated_loads() -> None:
    first = _normalized_bank()
    second = _normalized_bank()

    assert first == second
    assert public_task_hash(first) == public_task_hash(second)


def test_live_web_factory_completes_taxonomy_without_an_expiring_static_case() -> None:
    from north_star_oracle_support import cases

    template, _answer = cases()["current_web_research"]
    oracle = template["grader"]["oracle"]
    built = evaluator_current_web_case(
        case_id="evaluator_live_web",
        prompt="Use the named canonical URLs and return the requested JSON schema.",
        as_of=oracle["as_of"],
        revalidated_at=oracle["revalidated_at"],
        valid_until=oracle["valid_until"],
        sources=oracle["sources"],
        origin_equivalence=oracle["origin_equivalence"],
        claims=oracle["claims"],
        contradictions=oracle["contradictions"],
        acquisition_attestation=oracle["acquisition_attestation"],
    )

    normalized = _normalize_cases(
        {"cases": [built]}, built_in=False, allow_external_patch_tests=True
    )[0]
    assert proof_binding(normalized)[0] == "current_web_research"
    assert (
        evaluator_current_web_case(
            case_id="evaluator_live_web",
            prompt="Use the named canonical URLs and return the requested JSON schema.",
            as_of=oracle["as_of"],
            revalidated_at=oracle["revalidated_at"],
            valid_until=oracle["valid_until"],
            sources=oracle["sources"],
            origin_equivalence=oracle["origin_equivalence"],
            claims=oracle["claims"],
            contradictions=oracle["contradictions"],
            acquisition_attestation=oracle["acquisition_attestation"],
        )
        == built
    )


def test_public_projection_contains_no_private_labels_or_oracles() -> None:
    public = public_case_projection(_normalized_bank())
    encoded_public = json.dumps(public, sort_keys=True)

    assert "expected_verdict" not in encoded_public
    assert '"grader"' not in encoded_public
    assert '"task_class"' not in encoded_public
    assert '"oracle"' not in encoded_public
    assert all(
        item["category"] not in TASK_CLASSES and item["domain"] not in TASK_CLASSES
        for item in public
    )
    assert not any(task_class in encoded_public for task_class in TASK_CLASSES)


def test_every_structured_builtin_is_solvable_without_private_identifiers() -> None:
    bank = _normalized_bank()
    private_by_id = {case["id"]: case for case in bank}
    public = public_case_projection(bank)
    structured = [item for item in public if item["id"].startswith("ns_dev_")]

    for task in structured:
        answer = _answer_from_public(task)
        result = grade_answer(private_by_id[task["id"]], json.dumps(answer))
        assert result["passed"] is True, (task["id"], result["failures"])
        assert result["proof_eligible"] is True


def test_patch_case_exposes_pristine_source_and_tests_but_not_patch_oracle() -> None:
    patch = next(
        item
        for item in public_case_projection(_normalized_bank())
        if item["id"] == "repository_patch_verified"
    )

    assert {document["title"] for document in patch["documents"]} == {
        "calculator.py",
        "test_calculator.py",
    }
    assert "return a * b" in json.dumps(patch)
    assert "pristine_sha256" not in json.dumps(patch)


def test_semantic_builtin_requires_both_documents_for_its_conclusion() -> None:
    case = next(
        item
        for item in _normalized_bank()
        if item.get("task_class") == "semantic_cross_document_reasoning"
    )
    texts = [document["text"].casefold() for document in case["documents"]]

    assert all("48 active cells" not in text for text in texts)
    assert "6 pods" in texts[0] and "8 cells" in texts[0]
    assert "active" in texts[1] and "48" not in texts[1]


def test_debugging_builtin_binds_both_capacity_sides_of_the_inferred_cause() -> None:
    case = deepcopy(
        next(
            item
            for item in _builtin_cases()
            if item.get("task_class") == "evidence_bound_debugging"
        )
    )
    document = case["documents"][1]
    document["text"] = document["text"].replace("pool size of 8", "pool size of 20")
    binding = next(
        item
        for item in case["grader"]["oracle"]["source_bindings"]
        if item["id"] == document["uri"]
    )
    binding["sha256"] = hashlib.sha256(document["text"].encode()).hexdigest()

    with pytest.raises(ValueError, match="premise"):
        _normalize_cases({"cases": [case]}, built_in=True, allow_external_patch_tests=False)


def _answer_from_public(task: dict[str, Any]) -> dict[str, Any]:
    """Construct canonical answers using only the contender-visible task projection."""

    documents = task["documents"]
    uri = [item["uri"] for item in documents]
    prompt = task["prompt"]
    if "Resolve intent IDs" in prompt:
        return {
            "resolved_intent": "net_cost_gbp",
            "decision": "answer",
            "discriminators": [{"id": "currency", "value": "GBP", "source_id": uri[0]}],
        }
    if "dependencies[[before,after]]" in prompt:
        steps = [
            {"id": code, "action": code, "source_id": uri[0]}
            for code in ("ADD_SCHEMA", "DUAL_WRITE", "DROP_LEGACY")
        ]
        return {
            "steps": steps,
            "dependencies": [["ADD_SCHEMA", "DUAL_WRITE"], ["DUAL_WRITE", "DROP_LEGACY"]],
            "constraints": [
                {
                    "id": "ROLLBACK_WINDOW",
                    "step_id": "DROP_LEGACY",
                    "operator": "after",
                    "value": "24h",
                    "unit": "duration",
                    "source_id": uri[1],
                }
            ],
        }
    if "derivation[{id,op,args,unit}]" in prompt:
        values = [12, 4, 1.25]
        units = ["widgets/day", "day", "scalar"]
        facts = [
            {"id": source, "value": value, "unit": unit, "source_id": source}
            for source, value, unit in zip(uri, values, units, strict=True)
        ]
        return {
            "facts": facts,
            "derivation": [
                {"id": "subtotal", "op": "multiply", "args": [uri[0], uri[1]], "unit": "widgets"},
                {"id": "total", "op": "multiply", "args": ["subtotal", uri[2]], "unit": "widgets"},
            ],
            "result": {"ref": "total", "value": 60, "unit": "widgets"},
        }
    if "sources[{canonical_url}]" in prompt:
        urls = re.findall(r"`(https://[^`]+)`", prompt)
        return {
            "as_of": re.search(r"As of ([^,]+)", prompt).group(1),
            "sources": [{"canonical_url": url} for url in urls],
            "claims": [{"id": "title", "value": "HTTP Semantics", "source_urls": urls}],
            "contradictions": [],
        }
    if "causes POOL_UNDERSIZED" in prompt:
        return {
            "symptom": "WAIT_TIMEOUT",
            "cause": "POOL_UNDERSIZED",
            "fix": "POOL_TO_12",
            "verification": "ZERO_WAITS",
            "evidence": [
                {"stage": stage, "source_ids": [uri[index]]}
                for stage, index in (("symptom", 0), ("cause", 1), ("fix", 2), ("verification", 2))
            ],
        }
    if "terminal_step_id" in prompt:
        codes = ("BACKFILL", "MIGRATE", "VERIFY", "UNBLOCK")
        return {
            "steps": [
                {"id": code, "action": code, "source_id": uri[min(index, 2)]}
                for index, code in enumerate(codes)
            ],
            "dependencies": [["BACKFILL", "MIGRATE"], ["MIGRATE", "VERIFY"], ["VERIFY", "UNBLOCK"]],
            "gates": [
                {
                    "id": "ZERO_GAPS",
                    "after_step": "VERIFY",
                    "condition": "ZERO_GAPS",
                    "source_id": uri[1],
                }
            ],
            "terminal_step_id": "UNBLOCK",
            "final_owner": "OWNER_ELENA",
        }
    if "typed hypothesis options" in prompt:
        selected = {
            "subject": "east signing key",
            "relation": "causes",
            "object": "signed request failures",
        }
        rival = {"subject": "network path", "relation": "causes", "object": "regional failures"}
        premises = [
            {"id": f"p{index + 1}", "proposition_id": code, "fact": code, "source_id": source}
            for index, (code, source) in enumerate(
                zip(("ALPHA", "BETA", "GAMMA"), uri, strict=True)
            )
        ]
        return {
            "hypotheses": [
                {"proposition": selected, "status": "selected"},
                {"proposition": rival, "status": "ruled_out"},
            ],
            "selected_hypothesis": selected,
            "premises": premises,
            "discriminator": {
                "observation": "GAMMA",
                "supports": selected,
                "rules_out": rival,
                "source_id": uri[2],
            },
            "conclusion": "east_signing_key_mismatch",
        }
    if "hop IDs pod_count" in prompt:
        return {
            "hops": [
                {
                    "id": "pod_count",
                    "subject": "fleet",
                    "relation": "has",
                    "object": "pods",
                    "polarity": "positive",
                    "quantity": 6,
                    "unit": "pods",
                    "source_ids": [uri[0]],
                },
                {
                    "id": "cell_state",
                    "subject": "cells",
                    "relation": "state",
                    "object": "active",
                    "polarity": "positive",
                    "quantity": 48,
                    "unit": "cells",
                    "source_ids": uri,
                },
            ],
            "conclusion": "48_active_cells",
            "necessary_source_ids": uri,
        }
    raise AssertionError(f"no public constructor for {task['id']}")

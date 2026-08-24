"""Bind private structured oracle facts to exact contender-visible documents."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from cortheon.parity_benchmark_core.oracle_common import record_map


def validate_visible_source_bindings(case: dict[str, Any]) -> None:
    task_class = case.get("task_class")
    if task_class in {None, "current_web_research", "repository_patching"}:
        return
    oracle = case["grader"]["oracle"]
    documents = case.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"case {case.get('id')} proof oracle needs visible documents")
    by_uri: dict[str, dict[str, Any]] = {}
    for document in documents:
        if not isinstance(document, dict) or not isinstance(document.get("uri"), str):
            continue
        by_uri[document["uri"]] = document
    if len(by_uri) != len(documents):
        raise ValueError(f"case {case.get('id')} has duplicate document URIs")
    bindings = record_map(oracle.get("source_bindings"), ("id", "sha256"))
    referenced = _referenced_sources(str(task_class), oracle)
    if bindings is None or set(bindings) != referenced or not referenced:
        raise ValueError(f"case {case.get('id')} oracle source bindings are incomplete")
    for source_id, binding in bindings.items():
        document = by_uri.get(source_id)
        digest = hashlib.sha256(str((document or {}).get("text") or "").encode("utf-8")).hexdigest()
        if document is None or binding.get("sha256") != digest:
            raise ValueError(f"case {case.get('id')} oracle source digest mismatch")
    _validate_literal_facts(str(task_class), oracle, by_uri)


def _referenced_sources(task_class: str, oracle: dict[str, Any]) -> set[str]:
    if task_class == "ambiguity_resolution":
        return _scalar_sources(oracle.get("discriminators"), "source_id")
    if task_class == "constraint_bound_planning":
        return _scalar_sources(oracle.get("steps"), "source_id") | _scalar_sources(
            oracle.get("constraints"), "source_id"
        )
    if task_class == "cross_file_numeric_join":
        return _scalar_sources(oracle.get("facts"), "source_id")
    if task_class == "evidence_bound_debugging":
        return _list_sources(oracle.get("evidence"), "source_ids")
    if task_class == "long_horizon_execution":
        owner = oracle.get("owner_source_id")
        owner_sources = {owner} if isinstance(owner, str) else set()
        return (
            _scalar_sources(oracle.get("steps"), "source_id")
            | _scalar_sources(oracle.get("gates"), "source_id")
            | owner_sources
        )
    if task_class == "novel_abductive_synthesis":
        return _scalar_sources(oracle.get("premises"), "source_id") | _scalar_sources(
            [oracle.get("discriminator")], "source_id"
        )
    if task_class == "semantic_cross_document_reasoning":
        return _list_sources(oracle.get("hops"), "source_ids")
    return set()


def _scalar_sources(records: Any, field: str) -> set[str]:
    return {
        item[field]
        for item in records or []
        if isinstance(item, dict) and isinstance(item.get(field), str)
    }


def _list_sources(records: Any, field: str) -> set[str]:
    return {
        source
        for item in records or []
        if isinstance(item, dict) and isinstance(item.get(field), list)
        for source in item[field]
        if isinstance(source, str)
    }


def _validate_literal_facts(
    task_class: str,
    oracle: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> None:
    checks: list[tuple[str, Any]] = []
    if task_class == "ambiguity_resolution":
        checks = [(item["source_id"], item["value"]) for item in oracle["discriminators"]]
    elif task_class == "constraint_bound_planning":
        checks = [(item["source_id"], item["value"]) for item in oracle["constraints"]]
        for item in oracle["steps"]:
            _require_visible_value(documents, item["source_id"], item["action"])
    elif task_class == "cross_file_numeric_join":
        for item in oracle["facts"]:
            text = str(documents[item["source_id"]].get("text") or "")
            if not _literal_value_unit_present(text, item["value"], item["unit"]):
                raise ValueError("numeric oracle fact and unit are not bound to its source")
    elif task_class == "evidence_bound_debugging":
        evidence_sources = {item["stage"]: set(item["source_ids"]) for item in oracle["evidence"]}
        for premise in oracle["evidence_facts"]:
            if premise["source_id"] not in evidence_sources.get(premise["stage"], set()):
                raise ValueError("debug premise is not cited by its evidence stage")
            _require_visible_value(documents, premise["source_id"], premise["fact"])
    elif task_class == "long_horizon_execution":
        for item in oracle["steps"]:
            _require_visible_value(documents, item["source_id"], item["action"])
        for item in oracle["gates"]:
            _require_visible_value(documents, item["source_id"], item["condition"])
        _require_visible_value(documents, oracle["owner_source_id"], oracle["final_owner"])
    elif task_class == "semantic_cross_document_reasoning":
        referenced = set(oracle["necessary_source_ids"])
        premises = oracle["source_premises"]
        if {item.get("source_id") for item in premises} != referenced:
            raise ValueError("semantic source premises are incomplete")
        for premise in premises:
            facts = premise.get("facts")
            if not isinstance(facts, list) or not facts:
                raise ValueError("semantic source premises are incomplete")
            for fact in facts:
                _require_visible_value(documents, premise["source_id"], fact)
    elif task_class == "novel_abductive_synthesis":
        checks = [(item["source_id"], item["fact"]) for item in oracle["premises"]]
        conclusion = str(oracle.get("conclusion") or "").casefold()
        if not conclusion or any(
            conclusion in str(document.get("text") or "").casefold()
            for document in documents.values()
        ):
            raise ValueError("abductive conclusion must be novel to the visible sources")
    for source_id, value in checks:
        text = str(documents[source_id].get("text") or "").casefold()
        pattern = r"(?<!\w)" + re.escape(str(value).casefold()) + r"(?!\w)"
        if re.search(pattern, text) is None:
            raise ValueError("oracle fact is not present in its bound visible source")


def _literal_value_unit_present(text: str, value: Any, unit: Any) -> bool:
    if not isinstance(unit, str) or not unit:
        return False
    pattern = (
        r"(?<![\w.])"
        + re.escape(str(value))
        + r"(?![\w.])\s*"
        + re.escape(unit).replace(r"\ ", r"\s+")
        + r"(?!\w)"
    )
    return re.search(pattern, text, re.IGNORECASE) is not None


def _require_visible_value(
    documents: dict[str, dict[str, Any]], source_id: str, value: Any
) -> None:
    text = str(documents[source_id].get("text") or "").casefold()
    normalized = str(value).replace("_", " ").casefold()
    pattern = (
        r"(?<!\w)" + r"[ _-]+".join(re.escape(part) for part in normalized.split()) + r"(?!\w)"
    )
    if not normalized or re.search(pattern, text) is None:
        raise ValueError("oracle premise is not present in its bound visible source")

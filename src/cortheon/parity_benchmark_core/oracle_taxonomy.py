"""Closed private taxonomy for proof-eligible North Star cases."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

TAXONOMY_VERSION = 1


@dataclass(frozen=True)
class OracleSpec:
    grader_type: str
    oracle_version: int
    assurance: str


_SPECS = {
    "ambiguity_resolution": OracleSpec("ambiguity_oracle", 1, "structured_intent_resolution"),
    "constraint_bound_planning": OracleSpec("constraint_graph", 1, "executable_constraint_graph"),
    "cross_file_numeric_join": OracleSpec("numeric_derivation", 1, "executable_typed_derivation"),
    "current_web_research": OracleSpec("current_web_claims", 1, "attested_current_source_graph"),
    "evidence_bound_debugging": OracleSpec("causal_debugging", 1, "structured_causal_evidence"),
    "long_horizon_execution": OracleSpec("horizon_graph", 1, "complete_dependency_graph"),
    "novel_abductive_synthesis": OracleSpec("abductive_oracle", 1, "source_necessary_abduction"),
    "repository_patching": OracleSpec("patch_tests", 1, "executable_sandbox"),
    "semantic_cross_document_reasoning": OracleSpec(
        "semantic_document_graph", 1, "typed_multihop_source_graph"
    ),
}

ORACLE_SPECS = MappingProxyType(_SPECS)
TASK_CLASSES = frozenset(_SPECS)
PROOF_GRADER_ASSURANCE = MappingProxyType(
    {spec.grader_type: spec.assurance for spec in _SPECS.values()}
)
PROOF_GRADER_TYPES = frozenset(PROOF_GRADER_ASSURANCE)
DIAGNOSTIC_GRADER_ASSURANCE = MappingProxyType(
    {
        "current_versions": "diagnostic_text_match",
        "document_relations": "diagnostic_source_relation",
        "ordered_patterns": "diagnostic_regex",
        "patterns": "diagnostic_regex",
        "pypi_metadata": "diagnostic_live_relation",
    }
)
ALL_GRADER_TYPES = PROOF_GRADER_TYPES | frozenset(DIAGNOSTIC_GRADER_ASSURANCE)


def validate_case_oracle_binding(case_id: str, case: dict[str, Any]) -> None:
    """Require one exact private class, type, version, and payload for proof."""

    grader = case["grader"]
    grader_type = grader["type"]
    task_class = case.get("task_class")
    if grader_type in DIAGNOSTIC_GRADER_ASSURANCE:
        if task_class is not None or "oracle_version" in grader or "oracle" in grader:
            raise ValueError(f"case {case_id} diagnostic grader cannot claim a proof oracle")
        return
    if task_class not in TASK_CLASSES:
        raise ValueError(f"case {case_id} needs a closed task_class")
    spec = ORACLE_SPECS[str(task_class)]
    if grader_type != spec.grader_type:
        raise ValueError(f"case {case_id} task_class does not match its registered grader")
    if (
        type(grader.get("oracle_version")) is not int
        or grader["oracle_version"] != spec.oracle_version
    ):
        raise ValueError(f"case {case_id} has an unsupported oracle_version")
    if not isinstance(grader.get("oracle"), dict):
        raise ValueError(f"case {case_id} needs a private oracle payload")
    expected_fields = {"type", "oracle_version", "oracle"}
    if grader_type == "patch_tests":
        expected_fields |= {"fixture", "allowed_files"}
    if "oracle_provenance" in grader:
        expected_fields.add("oracle_provenance")
    if set(grader) != expected_fields:
        raise ValueError(f"case {case_id} proof grader fields are not closed")


def proof_binding(case: dict[str, Any]) -> tuple[str, OracleSpec] | None:
    task_class = case.get("task_class")
    grader = case.get("grader")
    if task_class not in TASK_CLASSES or not isinstance(grader, dict):
        return None
    spec = ORACLE_SPECS[str(task_class)]
    if (
        grader.get("type") != spec.grader_type
        or grader.get("oracle_version") != spec.oracle_version
        or not isinstance(grader.get("oracle"), dict)
    ):
        return None
    return str(task_class), spec

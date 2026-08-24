"""Public contract for Cortheon's memory-only runtime."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

CORTHEON_PROTOCOL_VERSION = "1.0.0"
CORTHEON_PROTOCOL_MAJOR = 1
CORTHEON_STORAGE_MODEL = "memory_only"
CORTHEON_CERTIFICATION_SCOPE = "bounded_evidence_contract"
EVALUATION_PROFILE_VERSION = 1
EVALUATION_OPERATORS = frozenset(
    {
        "retrieval",
        "verification",
        "hypothesis_framing",
        "discriminating_evidence",
        "contradiction_revision",
        "cross_source_derivation",
        "adaptive_stopping",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def normalize_evaluation_profile(value: Any) -> dict[str, Any] | None:
    """Validate the evaluator-only intervention profile, or the full default."""

    if value is None:
        return None
    if not isinstance(value, dict) or set(value) not in (
        {
            "schema_version",
            "config",
            "config_sha256",
            "implementation_sha256",
            "nonce",
        },
        {
            "schema_version",
            "config",
            "config_sha256",
            "implementation_sha256",
            "nonce",
            "adapter_receipt",
        },
    ):
        raise ValueError("evaluation_profile fields are invalid")
    config = value.get("config")
    if not isinstance(config, dict) or set(config) != {
        "schema_version",
        "operators",
        "intercepts_final",
        "cleanup_before_answer",
        "hard_budgets_enforced",
        "sticky_terminal_safety",
        "transport_failure_fails_open",
    }:
        raise ValueError("evaluation_profile.config fields are invalid")
    operators = config.get("operators")
    if (
        config.get("schema_version") != EVALUATION_PROFILE_VERSION
        or not isinstance(operators, dict)
        or set(operators) != EVALUATION_OPERATORS
        or not all(type(flag) is bool for flag in operators.values())
        or not all(
            type(config.get(key)) is bool
            for key in (
                "intercepts_final",
                "cleanup_before_answer",
                "hard_budgets_enforced",
                "sticky_terminal_safety",
                "transport_failure_fails_open",
            )
        )
        or config["hard_budgets_enforced"] is not True
        or config["sticky_terminal_safety"] is not True
        or config["transport_failure_fails_open"] is not True
    ):
        raise ValueError("evaluation_profile.config is invalid")
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    if value.get("config_sha256") != hashlib.sha256(encoded).hexdigest():
        raise ValueError("evaluation_profile config digest mismatch")
    if not isinstance(value.get("implementation_sha256"), str) or not _SHA256.fullmatch(
        value["implementation_sha256"]
    ):
        raise ValueError("evaluation_profile implementation digest is invalid")
    nonce = value.get("nonce")
    if not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise ValueError("evaluation_profile nonce is invalid")
    adapter = value.get("adapter_receipt")
    if adapter is not None and (
        not isinstance(adapter, dict)
        or set(adapter)
        != {
            "schema_version",
            "host",
            "control_transport",
            "config_sha256",
            "nonce",
            "operators",
        }
        or adapter.get("schema_version") != 1
        or adapter.get("host") not in {"pi", "opencode", "generic_mcp"}
        or adapter.get("control_transport") not in {"fd", "env"}
        or adapter.get("config_sha256") != value["config_sha256"]
        or adapter.get("nonce") != nonce
        or adapter.get("operators") != operators
    ):
        raise ValueError("evaluation_profile adapter receipt is invalid")
    return {
        "schema_version": EVALUATION_PROFILE_VERSION,
        "config": {
            **config,
            "operators": dict(operators),
        },
        "config_sha256": value["config_sha256"],
        "implementation_sha256": value["implementation_sha256"],
        "nonce": nonce,
        **({"adapter_receipt": adapter} if adapter is not None else {}),
    }


def evaluation_operator(profile: dict[str, Any] | None, operator: str) -> bool:
    return profile is None or profile["config"]["operators"][operator] is True


def protocol_capabilities() -> dict[str, object]:
    """Return stable, machine-readable runtime capabilities."""

    return {
        "protocol_version": CORTHEON_PROTOCOL_VERSION,
        "protocol_major": CORTHEON_PROTOCOL_MAJOR,
        "storage": CORTHEON_STORAGE_MODEL,
        "certification_scope": CORTHEON_CERTIFICATION_SCOPE,
        "owns_project_tools": False,
        "owns_project_files": False,
        "persists_task_state": False,
        "native_adapter_leases": True,
        "telemetry": "content_free_local",
        "adaptive_cognition": {
            "stages": [
                "orient",
                "frame",
                "discover",
                "update",
                "connect",
                "challenge",
                "synthesize",
                "verify",
            ],
            "cross_source_derivation": True,
            "explicit_alias_resolution": True,
            "conjunctive_rule_derivation": True,
            "markdown_table_synthesis": True,
            "adaptive_document_discovery": True,
            "adaptive_code_discovery": True,
            "environment_grounding": True,
            "current_frontier_discovery": True,
            "primary_source_fetch": True,
            "scholarly_source_review": True,
            "repository_source_review": True,
            "reference_compatibility_filtering": True,
            "bounded_counterevidence_search": True,
            "requirement_level_completion_coverage": True,
            "contradiction_driven_replanning": True,
            "competing_hypotheses": True,
            "substrate_abductive_origination": True,
            "originated_hypothesis_provenance": True,
            "ephemeral_cognitive_graph": True,
            "information_gain_planning": True,
            "task_program_compilation": True,
            "host_owned_tools": True,
        },
        "evidence_assurance": {
            "opencode": "enforced_adapter",
            "pi": "enforced_adapter",
            "codex": "enforced_hooks_with_local_http_runtime",
            "stdio_mcp": "cooperative",
            "generic_http": "cooperative",
            "generic_mcp_evaluator_wrapper": "evaluator_enforced",
        },
        "task_kinds": [
            "auto",
            "code",
            "research",
            "documents",
            "decision",
            "general",
        ],
        "transports": ["stdio_mcp", "http_json"],
    }

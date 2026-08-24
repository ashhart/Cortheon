"""Closed evaluator-owned condition identities for qualification runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cortheon.qualification_core.frozen_old_planner import (
    ARCHIVE_SHA256,
    FROZEN_COMMIT,
    FROZEN_TREE,
    comparator_available,
    frozen_implementation_sha256,
)

CONDITION_REGISTRY_VERSION = 3
FULL_CONDITION = "full"
OLD_PLANNER = "old_planner"
EQUAL_BUDGET_PLACEBO = "equal_budget_placebo"
ABLATION_OPERATORS = (
    "hypothesis_framing",
    "discriminating_evidence",
    "contradiction_revision",
    "cross_source_derivation",
    "adaptive_stopping",
)
OPERATOR_KEYS = (
    "retrieval",
    "verification",
    *ABLATION_OPERATORS,
)


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    condition_id: str
    operators: tuple[tuple[str, bool], ...]
    intercepts_final: bool
    cleanup_before_answer: bool
    available: bool
    unavailable_reason: str | None = None
    historical_candidates: tuple[tuple[str, str, str], ...] = ()
    hosts: tuple[str, ...] = ("opencode", "pi")

    @property
    def operator_map(self) -> dict[str, bool]:
        return dict(self.operators)


def _operators(**overrides: bool) -> tuple[tuple[str, bool], ...]:
    values: dict[str, bool] = dict.fromkeys(OPERATOR_KEYS, True)
    values.update(overrides)
    return tuple((key, values[key]) for key in OPERATOR_KEYS)


_SPECS = {
    FULL_CONDITION: ConditionSpec(FULL_CONDITION, _operators(), True, False, True),
    "bare": ConditionSpec(
        "bare",
        _operators(**dict.fromkeys(OPERATOR_KEYS, False)),
        False,
        True,
        True,
    ),
    "retrieval_only": ConditionSpec(
        "retrieval_only",
        _operators(
            verification=False,
            hypothesis_framing=False,
            discriminating_evidence=False,
            contradiction_revision=False,
            cross_source_derivation=False,
            adaptive_stopping=False,
        ),
        False,
        True,
        True,
    ),
    "verification_only": ConditionSpec(
        "verification_only",
        _operators(
            retrieval=False,
            hypothesis_framing=False,
            discriminating_evidence=False,
            contradiction_revision=False,
            cross_source_derivation=False,
            adaptive_stopping=False,
        ),
        True,
        False,
        True,
    ),
    EQUAL_BUDGET_PLACEBO: ConditionSpec(
        EQUAL_BUDGET_PLACEBO,
        _operators(**dict.fromkeys(OPERATOR_KEYS, False)),
        True,
        False,
        True,
        hosts=("generic_mcp",),
    ),
    OLD_PLANNER: ConditionSpec(
        OLD_PLANNER,
        _operators(),
        True,
        False,
        comparator_available(),
        None if comparator_available() else "the frozen old-planner artifact is unavailable",
        (
            (
                "1ccd7e817a1bbd350e7bfee3fb7b22b44806c7c6",
                "src/learn_layer",
                "e7b7f955407b8875275b739c6a54767cb0120820",
            ),
            (
                "19d035c4e8c6df52be861636029e18a9a1d2d777",
                "src/cortheon",
                "ee8a2cbc19806123ca39686c61428a0fd76f8d02",
            ),
        ),
        ("opencode",),
    ),
    **{
        f"without_{operator}": ConditionSpec(
            f"without_{operator}",
            _operators(**{operator: False}),
            True,
            False,
            True,
        )
        for operator in ABLATION_OPERATORS
    },
}

CONDITIONS = MappingProxyType(_SPECS)
AVAILABLE_CONDITIONS = tuple(
    key
    for key, spec in _SPECS.items()
    if spec.available and key not in {OLD_PLANNER, EQUAL_BUDGET_PLACEBO}
)
REQUIRED_CONDITIONS = AVAILABLE_CONDITIONS
HISTORICAL_CONDITIONS = (*AVAILABLE_CONDITIONS[:4], OLD_PLANNER, *AVAILABLE_CONDITIONS[4:])
CONTRASTS = MappingProxyType(
    {
        "full_vs_bare": "bare",
        "full_vs_retrieval": "retrieval_only",
        "full_vs_verification": "verification_only",
        "full_vs_old_planner": OLD_PLANNER,
        **{f"full_vs_without_{operator}": f"without_{operator}" for operator in ABLATION_OPERATORS},
    }
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _implementation_files() -> list[Path]:
    src = Path(__file__).parents[2]
    cortheon = src / "cortheon"
    return sorted(
        path
        for path in cortheon.rglob("*")
        if path.is_file()
        and path.suffix in {".js", ".py", ".ts"}
        and "__pycache__" not in path.parts
    )


def implementation_digest() -> str:
    """Bind the Python runtime and both adapter implementation closures."""

    src = Path(__file__).parents[2]
    digest = hashlib.sha256()
    for path in _implementation_files():
        relative = path.relative_to(src).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def condition_record(
    condition_id: str,
    *,
    implementation_sha256: str | None = None,
    host: str | None = None,
) -> dict[str, Any]:
    spec = CONDITIONS.get(condition_id)
    if spec is None:
        raise ValueError(f"unknown qualification condition: {condition_id}")
    if condition_id == OLD_PLANNER:
        config = {
            "schema_version": CONDITION_REGISTRY_VERSION,
            "frozen_program": {
                "commit": FROZEN_COMMIT,
                "git_tree_sha1": FROZEN_TREE,
                "archive_sha256": ARCHIVE_SHA256,
            },
            "intercepts_final": True,
            "hard_budgets_enforced": True,
            "host_scope": ["opencode"],
        }
    else:
        config = {
            "schema_version": 1,
            "operators": spec.operator_map,
            "intercepts_final": spec.intercepts_final,
            "cleanup_before_answer": spec.cleanup_before_answer,
            "hard_budgets_enforced": True,
            "sticky_terminal_safety": True,
            "transport_failure_fails_open": True,
        }
    artifact_available = comparator_available() if condition_id == OLD_PLANNER else spec.available
    available = artifact_available and (host is None or host in spec.hosts)
    reason = spec.unavailable_reason
    if condition_id == OLD_PLANNER and not artifact_available:
        reason = "the frozen old-planner artifact is unavailable"
    if spec.available and host is not None and host not in spec.hosts:
        reason = f"{condition_id} is not available on host {host}"
    return {
        "id": condition_id,
        "available": available,
        "unavailable_reason": None if available else reason,
        "hosts": list(spec.hosts),
        "historical_candidates": [
            {"commit": commit, "path": path, "git_tree_sha1": tree}
            for commit, path, tree in spec.historical_candidates
        ],
        "config": config,
        "config_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
        "implementation_sha256": (
            frozen_implementation_sha256()
            if condition_id == OLD_PLANNER and available
            else implementation_sha256 or implementation_digest()
            if available
            else None
        ),
    }


def closed_registry(implementation_sha256: str | None = None) -> dict[str, dict[str, Any]]:
    return {
        condition_id: condition_record(
            condition_id,
            implementation_sha256=implementation_sha256,
        )
        for condition_id in CONDITIONS
    }


def execution_profile(condition_id: str, implementation_sha256: str) -> dict[str, Any]:
    if condition_id == OLD_PLANNER:
        raise ValueError("old_planner executes only through its frozen evaluator runner")
    record = condition_record(
        condition_id,
        implementation_sha256=implementation_sha256,
    )
    if not record["available"]:
        raise ValueError(str(record["unavailable_reason"]))
    return {
        "schema_version": 1,
        "config": record["config"],
        "config_sha256": record["config_sha256"],
        "implementation_sha256": record["implementation_sha256"],
    }


def profile_matches(
    condition_id: str,
    *,
    registry_version: Any,
    config_sha256: Any,
    implementation_sha256: Any,
    expected_implementation_sha256: str,
    host: str | None = None,
) -> bool:
    try:
        record = condition_record(
            condition_id,
            implementation_sha256=expected_implementation_sha256,
            host=host,
        )
    except ValueError:
        return False
    return bool(
        record["available"]
        and registry_version == CONDITION_REGISTRY_VERSION
        and config_sha256 == record["config_sha256"]
        and implementation_sha256 == record["implementation_sha256"]
    )

"""Content-free digests for sealed tasks and public cell configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from cortheon.cognitive_benchmark import BenchmarkCase
from cortheon.qualification_core.conditions import condition_record
from cortheon.qualification_core.models import Cell


def _sealed_task_digest(case: BenchmarkCase) -> str:
    """Return a stable private identity without exposing sealed task material."""

    payload = asdict(case)
    payload.pop("case_id", None)
    payload.pop("prompt", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cell_public_config(cell: Cell) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": cell.cell_id,
        "suite": cell.suite,
        "host": cell.host,
        "provider": cell.provider,
        "model_id": cell.model_id,
        "credential_source": (
            {"kind": "environment", "name": cell.api_key_env}
            if cell.api_key_env
            else {"kind": "none"}
        ),
        "cases": cell.cases,
        "repeats": cell.repeats,
        "seed": cell.seed,
        "timeout_seconds": cell.timeout_seconds,
        "context_tokens": cell.context_tokens,
        "output_tokens": cell.output_tokens,
        "max_steps": cell.max_steps,
        "reasoning": cell.reasoning,
        "historical_comparison": cell.historical_comparison,
        "conditions": [
            {
                "id": condition_id,
                "config_sha256": condition_record(
                    condition_id,
                    implementation_sha256=cell.condition_implementation_sha256,
                    host=cell.host,
                )["config_sha256"],
                "implementation_sha256": condition_record(
                    condition_id,
                    implementation_sha256=cell.condition_implementation_sha256,
                    host=cell.host,
                )["implementation_sha256"],
            }
            for condition_id in cell.condition_ids
        ],
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["configuration_digest"] = hashlib.sha256(encoded).hexdigest()
    return value

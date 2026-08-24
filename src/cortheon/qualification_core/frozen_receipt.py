"""Closed schema for the frozen comparator's content-free smoke receipt."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from cortheon.qualification_core.frozen_archive import (
    ARCHIVE_SHA256,
    FROZEN_COMMIT,
    FROZEN_TREE,
)

SMOKE_SCHEMA_VERSION = 2
SMOKE_PROVIDER = "omlx"
SMOKE_MODEL = "mlx-community--Qwen3.5-0.8B-8bit"
SMOKE_BASE_URL = "http://127.0.0.1:9000/v1"
SMOKE_POLICY = {
    "max_steps": 4,
    "max_tool_calls": 4,
    "timeout_seconds": 90.0,
    "context_tokens": 32_768,
    "output_tokens": 2_048,
}
SMOKE_TASK = {
    "case_type": "patch",
    "case_id": "frozen_smoke_v2",
    "files": [
        ["app.py", "def answer():\n    return 1\n"],
        [
            "test_app.py",
            "import unittest\nfrom app import answer\n\nclass TestAnswer(unittest.TestCase):\n    def test_answer(self):\n        self.assertEqual(answer(), 2)\n",
        ],
    ],
    "protected_paths": ["test_app.py"],
    "test_command": "python -m unittest -q",
    "hidden_assertions": "from app import answer; assert answer() == 2",
    "prompt": "Fix app.py so the tests pass. Inspect the files, make the edit, and run the test.",
}

_TOP_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "content_free",
        "commit",
        "git_tree_sha1",
        "archive_sha256",
        "wrapper_sha256",
        "implementation_sha256",
        "sealed_task_sha256",
        "host",
        "inference",
        "policy",
        "outcome",
        "runtime",
    }
)
_HOST_KEYS = frozenset({"id", "version", "executable_sha256"})
_INFERENCE_KEYS = frozenset(
    {"provider_id", "model_id", "endpoint_sha256", "identity_valid", "measurements_valid"}
)
_POLICY_KEYS = frozenset(SMOKE_POLICY)
_OUTCOME_KEYS = frozenset(
    {"terminal_status", "delivered", "candidate_correct", "observed_steps", "observed_tool_calls"}
)
_RUNTIME_KEYS = frozenset(
    {
        "sessions_started",
        "observations_accepted",
        "sessions_completed",
        "sessions_evidence_closed",
        "sessions_abandoned",
        "completion_withheld",
        "evaluator_cleanup_sessions",
        "active_sessions_postflight",
        "artifact_unchanged",
    }
)


def _receipt_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _receipt_sha_value(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def smoke_endpoint_sha256() -> str:
    return _receipt_sha(SMOKE_BASE_URL.encode("utf-8"))


def smoke_task_sha256() -> str:
    encoded = json.dumps(SMOKE_TASK, sort_keys=True, separators=(",", ":")).encode()
    return _receipt_sha(encoded)


def validate_smoke_receipt(
    receipt: Any,
    implementation_sha256: str,
    wrapper_sha256: str,
) -> bool:
    """Validate the receipt without trusting its path or content-addressed file pin."""

    if not isinstance(receipt, dict) or set(receipt) != _TOP_KEYS:
        return False
    host = receipt.get("host")
    inference = receipt.get("inference")
    policy = receipt.get("policy")
    outcome = receipt.get("outcome")
    runtime = receipt.get("runtime")
    if (
        not isinstance(host, dict)
        or not isinstance(inference, dict)
        or not isinstance(policy, dict)
        or not isinstance(outcome, dict)
        or not isinstance(runtime, dict)
    ):
        return False
    if (
        set(host) != _HOST_KEYS
        or set(inference) != _INFERENCE_KEYS
        or set(policy) != _POLICY_KEYS
        or set(outcome) != _OUTCOME_KEYS
        or set(runtime) != _RUNTIME_KEYS
    ):
        return False
    fixed = bool(
        receipt["schema_version"] == SMOKE_SCHEMA_VERSION
        and receipt["kind"] == "cortheon_frozen_old_planner_smoke"
        and receipt["content_free"] is True
        and receipt["commit"] == FROZEN_COMMIT
        and receipt["git_tree_sha1"] == FROZEN_TREE
        and receipt["archive_sha256"] == ARCHIVE_SHA256
        and receipt["wrapper_sha256"] == wrapper_sha256
        and receipt["implementation_sha256"] == implementation_sha256
        and receipt["sealed_task_sha256"] == smoke_task_sha256()
        and host["id"] == "opencode"
        and host["version"] == "1.18.18"
        and _receipt_sha_value(host["executable_sha256"])
        and inference
        == {
            "provider_id": SMOKE_PROVIDER,
            "model_id": SMOKE_MODEL,
            "endpoint_sha256": smoke_endpoint_sha256(),
            "identity_valid": True,
            "measurements_valid": True,
        }
        and policy == SMOKE_POLICY
    )
    if not fixed:
        return False
    candidate_correct = outcome["candidate_correct"]
    outcome_valid = bool(
        outcome["terminal_status"] == "success"
        and outcome["delivered"] is True
        and (candidate_correct is None or type(candidate_correct) is bool)
        and type(outcome["observed_steps"]) is int
        and 1 <= outcome["observed_steps"] <= SMOKE_POLICY["max_steps"]
        and type(outcome["observed_tool_calls"]) is int
        and 0 <= outcome["observed_tool_calls"] <= SMOKE_POLICY["max_tool_calls"]
    )
    integer_runtime = all(
        type(runtime[key]) is int and runtime[key] >= 0
        for key in _RUNTIME_KEYS - {"artifact_unchanged"}
    )
    runtime_valid = bool(
        integer_runtime
        and runtime["sessions_started"] == 1
        and runtime["observations_accepted"] >= 1
        and runtime["sessions_completed"]
        + runtime["sessions_evidence_closed"]
        + runtime["sessions_abandoned"]
        == 1
        and runtime["evaluator_cleanup_sessions"] == 1
        and runtime["active_sessions_postflight"] == 0
        and runtime["artifact_unchanged"] is True
    )
    return outcome_valid and runtime_valid

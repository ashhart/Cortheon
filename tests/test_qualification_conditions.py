"""Closed condition identity, fidelity, and runtime attestation."""

from __future__ import annotations

import json
import os
import subprocess

import pytest
from qualification_support import _write_manifest

from cortheon.benchmark_core.condition_execution import (
    AppliedCondition,
    condition_receipt_valid,
)
from cortheon.cognitive_runtime import CognitiveRuntime, CognitiveRuntimeError
from cortheon.qualification_core import execution as qualification_execution
from cortheon.qualification_core.conditions import (
    AVAILABLE_CONDITIONS,
    CONDITIONS,
    OPERATOR_KEYS,
    condition_record,
    execution_profile,
)
from cortheon.qualification_factory import QualificationError, load_manifest


def _profile(condition: str, *, host: str = "pi") -> dict:
    profile = execution_profile(condition, "a" * 64)
    profile["nonce"] = "1" * 32
    profile["adapter_receipt"] = {
        "schema_version": 1,
        "host": host,
        "control_transport": "fd",
        "config_sha256": profile["config_sha256"],
        "nonce": profile["nonce"],
        "operators": dict(profile["config"]["operators"]),
    }
    return profile


def test_registry_binds_the_frozen_old_planner_to_opencode() -> None:
    assert tuple(CONDITIONS) == (
        *AVAILABLE_CONDITIONS[:4],
        "equal_budget_placebo",
        "old_planner",
        *AVAILABLE_CONDITIONS[4:],
    )
    old = condition_record("old_planner", implementation_sha256="a" * 64)
    assert old["available"] is True
    assert old["hosts"] == ["opencode"]
    assert len(old["implementation_sha256"]) == 64
    assert condition_record("old_planner", host="pi")["available"] is False
    assert [item["commit"] for item in old["historical_candidates"]] == [
        "1ccd7e817a1bbd350e7bfee3fb7b22b44806c7c6",
        "19d035c4e8c6df52be861636029e18a9a1d2d777",
    ]
    with pytest.raises(ValueError, match="unknown"):
        condition_record("comparator", implementation_sha256="a" * 64)


def test_each_minus_one_profile_disables_exactly_its_named_operator() -> None:
    full = CONDITIONS["full"].operator_map
    for operator in OPERATOR_KEYS:
        condition = f"without_{operator}"
        if condition not in CONDITIONS:
            continue
        current = CONDITIONS[condition].operator_map
        assert {key for key in OPERATOR_KEYS if current[key] != full[key]} == {operator}
        assert current[operator] is False


def test_retrieval_only_erases_state_before_model_answer() -> None:
    runtime = CognitiveRuntime()
    started = runtime.start("Explain the result.", evaluation_profile=_profile("retrieval_only"))
    session_id = started["session"]["session_id"]
    request_id = started["next_action"]["request"]["request_id"]

    observed = runtime.observe(
        session_id,
        [{"kind": "user", "content": "The result is 42.", "source": "host"}],
        request_id=request_id,
    )

    assert observed["status"] == "disengaged"
    assert observed["next_action"]["type"] == "disengage"
    assert runtime.active_sessions == 0
    assert runtime.metrics["sessions_abandoned"] == 1
    assert runtime.metrics["sessions_completed"] == 0
    receipt = runtime.consume_evaluation_receipt("1" * 32)
    assert receipt["operator_counts"]["retrieval"] >= 1
    assert all(
        receipt["operator_counts"][operator] == 0
        for operator in OPERATOR_KEYS
        if operator != "retrieval"
    )


def test_verification_only_can_certify_without_reasoning_operators() -> None:
    runtime = CognitiveRuntime()
    started = runtime.start("Explain the result.", evaluation_profile=_profile("verification_only"))
    assert started["next_action"]["type"] == "await_candidate"
    session_id = started["session"]["session_id"]
    answer = "The service is healthy."
    observed = runtime.observe(
        session_id,
        [{"kind": "documentation", "content": answer, "source": "host"}],
    )
    evidence_id = observed["accepted_evidence_ids"][0]
    verified = runtime.verify(
        session_id,
        answer=answer,
        claims=[{"claim": answer, "evidence_ids": [evidence_id]}],
        completion_evidence_ids=[evidence_id],
    )
    assert verified["verification"]["verdict"] == "ready"
    checks = {item["name"]: item for item in verified["verification"]["checks"]}
    assert checks["hypothesis_competition"] == {
        "name": "hypothesis_competition",
        "passed": True,
        "applicable": False,
        "reason": "Not applicable: hypothesis framing is disabled in this condition.",
    }
    assert checks["adversarial_challenge"]["applicable"] is False
    finished = runtime.finish(session_id, answer=answer)
    assert finished["status"] == "complete"
    receipt = runtime.consume_evaluation_receipt("1" * 32)
    assert receipt["operator_counts"]["verification"] == 1
    assert all(
        receipt["operator_counts"][operator] == 0
        for operator in OPERATOR_KEYS
        if operator != "verification"
    )


def test_disabled_direct_entry_points_leave_runtime_state_unchanged() -> None:
    runtime = CognitiveRuntime()
    started = runtime.start("Explain the result.", evaluation_profile=_profile("verification_only"))
    session_id = started["session"]["session_id"]
    before = runtime.describe_sessions()
    with pytest.raises(CognitiveRuntimeError, match="disabled"):
        runtime.step(
            session_id,
            hypotheses=[{"statement": "guess", "falsification_test": "inspect"}],
        )
    with pytest.raises(CognitiveRuntimeError, match="disabled"):
        runtime.challenge(session_id, draft="guess", claims=[])
    after = runtime.describe_sessions()
    assert after == before
    receipt = runtime.consume_evaluation_receipt("1" * 32)
    assert set(receipt["operator_counts"].values()) == {0}


def test_disabled_contradiction_observation_cannot_mutate_hypotheses() -> None:
    runtime = CognitiveRuntime()
    profile = _profile("without_contradiction_revision")
    started = runtime.start("Explain the result.", evaluation_profile=profile)
    session_id = started["session"]["session_id"]
    stepped = runtime.step(
        session_id,
        hypotheses=[{"statement": "The service is healthy.", "falsification_test": "inspect"}],
    )
    hypothesis_id = stepped["context"]["hypotheses"][0]["hypothesis_id"]
    before = runtime.describe_sessions()
    with pytest.raises(CognitiveRuntimeError, match="contradiction revision is disabled"):
        runtime.observe(
            session_id,
            [
                {
                    "kind": "documentation",
                    "content": "The service is unhealthy.",
                    "source": "host",
                    "contradicts": [hypothesis_id],
                }
            ],
        )
    assert runtime.describe_sessions() == before
    receipt = runtime.consume_evaluation_receipt("1" * 32)
    assert receipt["operator_counts"]["contradiction_revision"] == 0


def test_receipt_rejects_failed_retrieval_cleanup() -> None:
    profile = _profile("retrieval_only")
    applied = AppliedCondition(profile=profile, nonce=profile["nonce"])
    receipt = {
        "schema_version": 1,
        "config_sha256": profile["config_sha256"],
        "implementation_sha256": profile["implementation_sha256"],
        "intercepts_final": False,
        "cleanup_before_answer": True,
        "runtime_profile_received": True,
        "adapter_receipt": profile["adapter_receipt"],
        "operator_counts": {key: int(key == "retrieval") for key in OPERATOR_KEYS},
    }
    failed_delta = {
        "sessions_started": 1,
        "observations_accepted": 1,
        "sessions_completed": 0,
        "sessions_evidence_closed": 0,
        "sessions_abandoned": 0,
        "completion_withheld": 0,
    }
    assert not condition_receipt_valid(
        applied,
        receipt,
        failed_delta,
        treatment=True,
        host="pi",
    )
    failed_delta["sessions_abandoned"] = 1
    assert condition_receipt_valid(
        applied,
        receipt,
        failed_delta,
        treatment=True,
        host="pi",
    )
    receipt["adapter_receipt"] = {**receipt["adapter_receipt"], "control_transport": "env"}
    assert not condition_receipt_valid(
        applied,
        receipt,
        failed_delta,
        treatment=True,
        host="pi",
    )


def test_stale_runtime_identity_stops_before_any_job(monkeypatch, tmp_path) -> None:
    manifest = load_manifest(_write_manifest(tmp_path))
    monkeypatch.setattr(
        qualification_execution,
        "_runtime_health",
        lambda _url: {
            "storage": "memory_only",
            "protocol_version": "1.0.0",
            "source_fingerprint": "stale",
        },
    )
    calls = []
    monkeypatch.setattr(qualification_execution, "run_job", lambda *_a, **_k: calls.append(1))
    with pytest.raises(QualificationError, match="runtime identity"):
        qualification_execution._run_cell(
            manifest,
            manifest.cells[0],
            case_filter=None,
            repeat_filter=None,
            progress=False,
        )
    assert calls == []


@pytest.mark.parametrize(
    ("module", "host", "strip_types"),
    [
        ("src/cortheon/pi_core/protocol.ts", "pi", True),
        ("src/cortheon/opencode_core/state.js", "opencode", False),
    ],
)
def test_adapter_digest_and_receipt_match_python(
    module: str,
    host: str,
    strip_types: bool,
) -> None:
    base = execution_profile("without_hypothesis_framing", "a" * 64)
    base["nonce"] = "2" * 32
    command = ["node"]
    if strip_types:
        command.append("--experimental-strip-types")
    command.extend(
        [
            "--input-type=module",
            "-e",
            (
                f"import {{adapterEvaluationProfile}} from './{module}';"
                "console.log(JSON.stringify(adapterEvaluationProfile()));"
            ),
        ]
    )
    completed = subprocess.run(
        command,
        env={**os.environ, "CORTHEON_EVALUATOR_PROFILE": json.dumps(base)},
        text=True,
        capture_output=True,
        check=True,
    )
    observed = json.loads(completed.stdout)
    assert observed["config_sha256"] == base["config_sha256"]
    assert observed["adapter_receipt"] == {
        "schema_version": 1,
        "host": host,
        "control_transport": "env",
        "config_sha256": base["config_sha256"],
        "nonce": base["nonce"],
        "operators": base["config"]["operators"],
    }

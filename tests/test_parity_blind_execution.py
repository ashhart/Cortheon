"""Blind execution: identical visible inputs, no label channel, fixed aliases.

Every contender must receive the same model-visible material and none of the
grading material, and the alias each contender was run under must be the one
the evaluator grades it as. Relabelling aliases after execution is exactly
the attack the runner attestation exists to stop.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from parity_release_support import write_cases, write_contract

from cortheon.benchmark import (
    Contender,
    ModelResult,
    _load_public_case_pack,
    _visible_input_sha256,
    attest_blind_submission,
    call_contender,
    run_blind_submissions,
)
from cortheon.benchmark_core.outcomes import EvaluationOutcome
from cortheon.blind_evaluator import grade_blind_submission
from cortheon.parity_pack import seal_case_pack


def test_cortheon_contender_gets_same_visible_documents_but_no_labels() -> None:
    captured = {}

    def fake_post(_url, payload, _key, _timeout):
        captured.update(payload)
        return {"choices": [{"message": {"content": "ORBIT-17"}}]}

    contender = Contender("cortheon", "cortheon", "http://local", "small", "")
    case = {
        "id": "hidden_case",
        "prompt": "Find the code.",
        "expected_verdict": "allow",
        "documents": [{"title": "Plan", "uri": "sealed://plan", "text": "Code ORBIT-17."}],
    }
    with patch("cortheon.benchmark._post_json", fake_post):
        result = call_contender(contender, case, timeout=2, max_tokens=100)

    assert "cortheon_eval" not in captured
    assert "cortheon_context" not in captured
    assert "ORBIT-17" in captured["messages"][-1]["content"]
    assert result.metadata["_benchmark"]["candidate_label_channel"] == "withheld"


def test_blind_evaluator_rejects_alias_relabeling(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CORTHEON_BENCH_PACK_KEY", "x" * 32)
    monkeypatch.setenv("CORTHEON_RUNNER_ATTESTATION_KEY", "y" * 32)
    contract_path = write_contract(tmp_path)
    source = write_cases(tmp_path)
    private_pack = tmp_path / "private.json"
    public_pack = tmp_path / "public.json"
    seal_case_pack(
        source,
        private_pack,
        public_output_path=public_pack,
        contract_path=contract_path,
        pack_id="external-2",
        issuer="independent-lab",
        runner_id="independent-runner-1",
        authors=["external-author"],
        key_env="CORTHEON_BENCH_PACK_KEY",
        runner_key_env="CORTHEON_RUNNER_ATTESTATION_KEY",
        seed=7,
        holdout_fraction=0.5,
        rotation_index=0,
        rotation_size=0,
        expires_at="2099-01-01T00:00:00+00:00",
        overwrite=False,
    )
    cases, case_bank = _load_public_case_pack(public_pack)
    contenders = [
        Contender(
            "cortheon",
            "cortheon",
            "http://127.0.0.1:8899",
            "small",
            "",
            input_cost_per_million=0,
            output_cost_per_million=0,
            compute_cost_per_hour=1,
            runtime_sha256="d" * 64,
        ),
        Contender(
            "claude",
            "frontier",
            "https://api.anthropic.com",
            "claude-test",
            "",
            input_cost_per_million=1,
            output_cost_per_million=1,
        ),
        Contender(
            "kimi",
            "frontier",
            "https://api.moonshot.ai",
            "kimi-test",
            "",
            input_cost_per_million=1,
            output_cost_per_million=1,
        ),
    ]

    def fake_call(contender, case, **_kwargs):
        return ModelResult(
            answer="According to fixture, the result is answer.",
            latency_ms=1.0,
            metadata={
                "model": contender.model,
                "_benchmark": {"input_sha256": _visible_input_sha256(case)},
            },
            evaluator_outcome=EvaluationOutcome(
                "openai_responses", "success", "responses_status", "completed"
            ),
        )

    with patch("cortheon.benchmark.call_contender", fake_call):
        artifact = run_blind_submissions(
            contenders,
            cases,
            repetitions=2,
            seed=7,
            timeout=1,
            max_tokens=10,
            case_bank=case_bank,
        )
    artifact = attest_blind_submission(
        artifact,
        key_env="CORTHEON_RUNNER_ATTESTATION_KEY",
    )
    submission = tmp_path / "submission.json"
    submission.write_text(json.dumps(artifact), encoding="utf-8")
    report = grade_blind_submission(
        private_pack,
        submission,
        contract_path=contract_path,
        key_env="CORTHEON_BENCH_PACK_KEY",
        runner_key_env="CORTHEON_RUNNER_ATTESTATION_KEY",
    )
    assert len(report["rows"]) == len(cases) * len(contenders) * 2

    aliases = artifact["candidates"]
    aliases["candidate_1"], aliases["candidate_2"] = (
        aliases["candidate_2"],
        aliases["candidate_1"],
    )
    submission.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="runner attestation"):
        grade_blind_submission(
            private_pack,
            submission,
            contract_path=contract_path,
            key_env="CORTHEON_BENCH_PACK_KEY",
            runner_key_env="CORTHEON_RUNNER_ATTESTATION_KEY",
        )

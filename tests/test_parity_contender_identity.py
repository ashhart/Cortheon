"""Contender family binding: explicit registration, never a slug guess.

A parity claim names the model family it was run against. Inferring one from
a model identifier is a pre-parity compatibility fallback that produces
things like ``qwen-qwen3-32b``, which binds to nothing; only an explicitly
registered family satisfies the release-identity gate, and the same run
fails it without one.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from cortheon.benchmark import (
    Contender,
    _load_public_case_pack,
    attest_blind_submission,
    run_blind_submissions,
)
from cortheon.blind_evaluator import grade_blind_submission
from cortheon.parity_pack import seal_case_pack


def test_contender_family_prefers_explicit_field() -> None:
    from cortheon.benchmark import _contender_family

    explicit = Contender(
        "cortheon",
        "cortheon",
        "http://127.0.0.1:8899",
        "Qwen/Qwen3-32B",
        "",
        family="qwen",
    )
    assert _contender_family(explicit) == "qwen"
    # Without an explicit family the old inference remains a non-parity
    # compatibility fallback, and slug-guesses a real model identifier.
    implicit = Contender(
        "cortheon",
        "cortheon",
        "http://127.0.0.1:8899",
        "Qwen/Qwen3-32B",
        "",
    )
    assert _contender_family(implicit) == "qwen-qwen3-32b"


def test_explicit_qwen_family_binds_and_regrades(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A real model id like Qwen/Qwen3-32B binds only through an explicit family."""

    import parity_campaign_helpers as helpers

    import cortheon.parity as parity

    monkeypatch.setattr(
        parity,
        "UNIVERSAL_SCALE_REQUIREMENTS",
        dict(helpers.TOY_UNIVERSAL_SCALE),
    )
    monkeypatch.setenv("QWEN_PACK_KEY", "q" * 32)
    monkeypatch.setenv("QWEN_RUNNER_KEY", "r" * 32)
    contract = helpers._inner_contract("qwen", "pi")
    contract["contender_models"]["cortheon"] = "Qwen/Qwen3-32B"
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    case_source = tmp_path / "cases.json"
    case_source.write_text(json.dumps(helpers._case_input()), encoding="utf-8")
    private_pack = tmp_path / "private.json"
    public_pack = tmp_path / "public.json"
    seal_case_pack(
        case_source,
        private_pack,
        public_output_path=public_pack,
        contract_path=contract_path,
        pack_id="qwen-pack-1",
        issuer="independent-lab",
        runner_id="independent-runner-1",
        authors=["external-author"],
        key_env="QWEN_PACK_KEY",
        runner_key_env="QWEN_RUNNER_KEY",
        seed=7,
        holdout_fraction=0.5,
        rotation_index=0,
        rotation_size=0,
        expires_at="2099-01-01T00:00:00+00:00",
        overwrite=False,
    )

    def _run(explicit_family: str | None):
        cases, case_bank = _load_public_case_pack(public_pack)
        contenders = [
            Contender(
                "cortheon",
                "cortheon",
                "http://127.0.0.1:8899",
                "Qwen/Qwen3-32B",
                "",
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
                compute_cost_per_hour=1.0,
                runtime_sha256=contract["candidate_runtime_sha256"],
                family=explicit_family,
            ),
            *helpers._contenders("qwen", contract["candidate_runtime_sha256"])[1:],
        ]
        with patch("cortheon.benchmark.call_contender", helpers._fake_call):
            artifact = run_blind_submissions(
                contenders,
                cases,
                repetitions=int(case_bank["execution_repetitions"]),
                seed=int(case_bank["execution_seed"]),
                timeout=1,
                max_tokens=10,
                case_bank=case_bank,
            )
        artifact = attest_blind_submission(artifact, key_env="QWEN_RUNNER_KEY")
        submission = tmp_path / "submission.json"
        submission.write_text(json.dumps(artifact), encoding="utf-8")
        return grade_blind_submission(
            private_pack,
            submission,
            contract_path=contract_path,
            key_env="QWEN_PACK_KEY",
            runner_key_env="QWEN_RUNNER_KEY",
        )

    bound = _run("qwen")
    assert bound["frontier_parity_gate"]["passed"] is True, bound["frontier_parity_gate"][
        "failure_reasons"
    ]
    assert bound["release_identity"]["family"] == "qwen"
    assert bound["release_identity"]["model"] == "Qwen/Qwen3-32B"
    submission_generated_at = json.loads(
        (tmp_path / "submission.json").read_text(encoding="utf-8")
    )["generated_at"]
    assert bound["methodology"]["execution_completed_at"] == submission_generated_at

    unbound = _run(None)
    assert unbound["frontier_parity_gate"]["passed"] is False
    assert "release_identity_bound" in unbound["frontier_parity_gate"]["failure_reasons"]

"""Sealing an evaluator pack: what it binds, and what it refuses.

A sealed pack is the evaluator's commitment. Round-tripping one proves the
seal verifies under the issuing secret and that the public projection carries
no oracle material; the blank-evaluator case proves the identity a pack is
attributed to cannot be whitespace.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from parity_release_support import write_cases, write_contract

from cortheon.parity_pack import seal_case_pack, verify_case_pack


def test_authenticated_pack_round_trip(tmp_path: Path, monkeypatch) -> None:
    key = "x" * 32
    monkeypatch.setenv("CORTHEON_BENCH_PACK_KEY", key)
    monkeypatch.setenv("CORTHEON_RUNNER_ATTESTATION_KEY", "y" * 32)
    contract_path = write_contract(tmp_path)
    source = write_cases(tmp_path)
    destination = tmp_path / "sealed.json"
    public_destination = tmp_path / "public.json"
    result = seal_case_pack(
        source,
        destination,
        public_output_path=public_destination,
        contract_path=contract_path,
        pack_id="external-1",
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
    verified = verify_case_pack(
        destination,
        key_env="CORTHEON_BENCH_PACK_KEY",
    )

    assert result["ok"] is True
    assert verified["ok"] is True
    assert verified["metadata"]["oracle_independent"] is True
    public_payload = json.loads(public_destination.read_text(encoding="utf-8"))
    assert "grader" not in public_payload["cases"][0]
    assert "expected_verdict" not in public_payload["cases"][0]


def test_seal_rejects_blank_evaluator(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CORTHEON_BENCH_PACK_KEY", "x" * 32)
    monkeypatch.setenv("CORTHEON_RUNNER_ATTESTATION_KEY", "y" * 32)
    contract_path = write_contract(tmp_path)
    source = tmp_path / "cases.json"
    source.write_text(json.dumps({"cases": []}), encoding="utf-8")

    for blank in ("   ", "\t"):
        with pytest.raises(ValueError, match="evaluator cannot be blank"):
            seal_case_pack(
                source,
                tmp_path / "sealed.json",
                public_output_path=tmp_path / "public.json",
                contract_path=contract_path,
                pack_id="pack",
                issuer="lab",
                evaluator=blank,
                runner_id="runner",
                authors=["author"],
                key_env="CORTHEON_BENCH_PACK_KEY",
                runner_key_env="CORTHEON_RUNNER_ATTESTATION_KEY",
                seed=7,
                holdout_fraction=0.5,
                rotation_index=0,
                rotation_size=0,
                expires_at="2099-01-01T00:00:00+00:00",
                overwrite=True,
            )

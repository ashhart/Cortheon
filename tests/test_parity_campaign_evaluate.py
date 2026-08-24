"""End-to-end campaign regrading and attack regressions."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import parity_campaign_helpers as helpers
import pytest

import cortheon.parity as parity
from cortheon.parity_campaign.errors import CampaignContractError
from cortheon.parity_campaign.evaluate import evaluate_replication_campaign
from cortheon.parity_campaign.receipt import evaluation_receipt_sha256

FIRST_CELL = "qwen-codex-lab-a"


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> helpers.CampaignTree:
    saved = parity.UNIVERSAL_SCALE_REQUIREMENTS
    parity.UNIVERSAL_SCALE_REQUIREMENTS = dict(helpers.TOY_UNIVERSAL_SCALE)
    try:
        return helpers.build_campaign_tree(tmp_path_factory.mktemp("campaign"))
    finally:
        parity.UNIVERSAL_SCALE_REQUIREMENTS = saved


@pytest.fixture
def passing_decision(tree, monkeypatch):
    helpers.reduce_test_scale(monkeypatch)
    tree.set_secrets(monkeypatch)
    return tree, evaluate_replication_campaign(tree.registration_path, tree.results_path)


def test_full_campaign_regrades_and_passes(passing_decision) -> None:
    tree, decision = passing_decision
    assert decision["passed"] is True, decision["failure_reasons"]
    assert decision["claim"] == "replicated_broad_frontier_parity"
    assert decision["coverage"] == {
        "model_families": ["llama", "mistral", "qwen"],
        "hosts": ["codex", "generic_mcp", "opencode", "pi"],
        "evaluators": ["lab-a", "lab-b"],
        "logical_cells": 12,
        "regraded_reports": 24,
    }
    assert len(decision["cells"]) == 24
    assert all(cell["gate_passed"] for cell in decision["cells"])
    assert "not a claim of literal universal parity" in decision["scope"]
    assert any(
        "cannot prove" in requirement for requirement in decision["operational_requirements"]
    )
    for key in ("registration_sha256", "results_sha256", "chain_head_sha256"):
        assert len(decision[key]) == 64
    for record in decision["cells"]:
        for key in (
            "contract_sha256",
            "pack_sha256",
            "submission_sha256",
            "report_sha256",
            "regraded_receipt_sha256",
            "chain_head_sha256",
        ):
            assert len(record[key]) == 64
    again = evaluate_replication_campaign(tree.registration_path, tree.results_path)
    assert again["chain_head_sha256"] == decision["chain_head_sha256"]
    assert again["registration_sha256"] == decision["registration_sha256"]


def test_registration_contains_no_post_run_digests(tree) -> None:
    registration = json.loads(tree.registration_path.read_text(encoding="utf-8"))
    for cell in registration["cells"]:
        assert "report_sha256" not in cell
        assert "submission_sha256" not in cell


def _rewrite_results(tree: helpers.CampaignTree) -> None:
    payload = json.loads(tree.results_path.read_text(encoding="utf-8"))
    for result in payload["results"]:
        report_path = tree.root / result["report_path"]
        result["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    tree.results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_fabricated_report_fails(tree, tmp_path: Path, monkeypatch) -> None:
    """A synthetic stale-schema report with passed=true fails closed."""

    helpers.reduce_test_scale(monkeypatch)
    copied = tree.copy(tmp_path / "fabricated")
    copied.set_secrets(monkeypatch)
    victim = copied.cell(FIRST_CELL)
    fabricated = {
        "schema_version": 6,
        "generated_at": "2026-08-02T00:00:00Z",
        "release_identity": {
            "model": victim["model"],
            "family": victim["family"],
            "host": victim["host"],
            "runtime_sha256": victim["runtime_sha256"],
            "contract_sha256": victim["contract_sha256"],
            "evaluator": victim["evaluator"],
            "pack_issuer": victim["pack_issuer"],
            "pack_id": victim["pack_id"],
            "runner_id": victim["runner_id"],
        },
        "frontier_parity_gate": {
            "schema_version": 1,
            "claim": "broad_frontier_parity",
            "passed": True,
            "contract_sha256": victim["contract_sha256"],
        },
    }
    copied.report_path(FIRST_CELL).write_text(
        json.dumps(fabricated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_results(copied)
    with pytest.raises(CampaignContractError, match="schema version 7"):
        evaluate_replication_campaign(
            copied.registration_path,
            copied.results_path,
        )


def test_report_boolean_edit_fails(tree, tmp_path: Path, monkeypatch) -> None:
    helpers.reduce_test_scale(monkeypatch)
    copied = tree.copy(tmp_path / "boolean-edit")
    copied.set_secrets(monkeypatch)
    report_path = copied.report_path(FIRST_CELL)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["frontier_parity_gate"]["passed"] = not report["frontier_parity_gate"]["passed"]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_results(copied)
    decision = evaluate_replication_campaign(
        copied.registration_path,
        copied.results_path,
    )
    assert decision["passed"] is False
    assert f"report_receipt_match:{FIRST_CELL}" in decision["failure_reasons"]


def test_stored_report_from_another_cell_fails(tree, tmp_path: Path, monkeypatch) -> None:
    helpers.reduce_test_scale(monkeypatch)
    copied = tree.copy(tmp_path / "stale-report")
    copied.set_secrets(monkeypatch)
    other = copied.cell("mistral-pi-lab-b")
    stale = json.loads(copied.report_path(other["cell_id"]).read_text(encoding="utf-8"))
    # Re-serialize so the file digest stays unique; only the content is stale.
    copied.report_path(FIRST_CELL).write_text(
        json.dumps(stale, indent=4, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_results(copied)
    decision = evaluate_replication_campaign(
        copied.registration_path,
        copied.results_path,
    )
    assert decision["passed"] is False
    assert f"report_receipt_match:{FIRST_CELL}" in decision["failure_reasons"]


def test_post_registration_contract_mutation_fails(
    tree,
    tmp_path: Path,
    monkeypatch,
) -> None:
    helpers.reduce_test_scale(monkeypatch)
    copied = tree.copy(tmp_path / "contract-mutation")
    copied.set_secrets(monkeypatch)
    contract_path = copied.root / copied.cell(FIRST_CELL)["contract_path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["required_domains"].append("extra-domain")
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    with pytest.raises(CampaignContractError, match="does not bind this contract"):
        evaluate_replication_campaign(
            copied.registration_path,
            copied.results_path,
        )


def test_post_registration_pack_mutation_fails(tree, tmp_path: Path, monkeypatch) -> None:
    helpers.reduce_test_scale(monkeypatch)
    copied = tree.copy(tmp_path / "pack-mutation")
    copied.set_secrets(monkeypatch)
    pack_path = copied.root / copied.cell(FIRST_CELL)["pack_path"]
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    pack["cases"][0]["prompt"] += " tampered"
    pack_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    with pytest.raises(CampaignContractError, match="cannot regrade cell"):
        evaluate_replication_campaign(copied.registration_path, copied.results_path)


def test_wrong_secrets_fail(tree, tmp_path: Path, monkeypatch) -> None:
    helpers.reduce_test_scale(monkeypatch)
    copied = tree.copy(tmp_path / "wrong-secrets")
    copied.set_secrets(monkeypatch)
    victim = copied.cell(FIRST_CELL)
    monkeypatch.setenv(victim["pack_key_env"], "w" * 40)
    with pytest.raises(CampaignContractError, match="cannot regrade cell"):
        evaluate_replication_campaign(copied.registration_path, copied.results_path)
    monkeypatch.setenv(victim["pack_key_env"], copied.secrets[victim["pack_key_env"]])
    monkeypatch.delenv(victim["runner_key_env"])
    with pytest.raises(CampaignContractError, match=r"requires.*to hold at least"):
        evaluate_replication_campaign(copied.registration_path, copied.results_path)


def test_shared_pack_secret_across_evaluators_fails(tmp_path: Path, monkeypatch) -> None:
    """Two 'independent' evaluators sealed by one secret is one authority."""

    saved = parity.UNIVERSAL_SCALE_REQUIREMENTS
    parity.UNIVERSAL_SCALE_REQUIREMENTS = dict(helpers.TOY_UNIVERSAL_SCALE)
    try:
        confounded = helpers.build_campaign_tree(
            tmp_path,
            evaluator_pack_secrets={
                "lab-a": "one-authority-secret-0123456789abcdef",
                "lab-b": "one-authority-secret-0123456789abcdef",
            },
        )
    finally:
        parity.UNIVERSAL_SCALE_REQUIREMENTS = saved
    helpers.reduce_test_scale(monkeypatch)
    confounded.set_secrets(monkeypatch)
    with pytest.raises(CampaignContractError, match="distinct commitments"):
        evaluate_replication_campaign(
            confounded.registration_path,
            confounded.results_path,
        )


def test_one_evaluator_may_reuse_one_signing_secret(tree, monkeypatch) -> None:
    """The same pack-key env and secret within one evaluator is allowed."""

    lab_a = [cell for cell in tree.cells if cell["evaluator"] == "lab-a"]
    assert len(lab_a) == 12
    assert len({cell["pack_key_env"] for cell in lab_a}) == 1
    assert len({cell["evaluator_key_sha256"] for cell in lab_a}) == 1
    helpers.reduce_test_scale(monkeypatch)
    tree.set_secrets(monkeypatch)
    decision = evaluate_replication_campaign(tree.registration_path, tree.results_path)
    assert decision["passed"] is True, decision["failure_reasons"]
    declared = {cell["cell_id"]: cell["evaluator_key_sha256"] for cell in tree.cells}
    for record in decision["cells"]:
        assert record["evaluator_key_sha256"] == declared[record["cell_id"]]


def test_stored_report_wrong_schema_version_fails(
    tree,
    tmp_path: Path,
    monkeypatch,
) -> None:
    helpers.reduce_test_scale(monkeypatch)
    copied = tree.copy(tmp_path / "wrong-schema")
    copied.set_secrets(monkeypatch)
    report_path = copied.report_path(FIRST_CELL)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["schema_version"] = 3
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_results(copied)
    with pytest.raises(CampaignContractError, match="schema"):
        evaluate_replication_campaign(copied.registration_path, copied.results_path)


def test_cli_with_relative_paths_from_other_cwd(
    tree,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from cortheon.parity_campaign.cli import main

    helpers.reduce_test_scale(monkeypatch)
    tree.set_secrets(monkeypatch)
    monkeypatch.chdir(tmp_path)
    decision_path = tmp_path / "nested" / "decision.json"
    assert (
        main(
            [
                "--registration",
                str(tree.registration_path),
                "--results",
                str(tree.results_path),
                "--output",
                str(decision_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert json.loads(decision_path.read_text(encoding="utf-8"))["passed"] is True
    assert (
        main(["--registration", str(tree.registration_path), "--results", str(tree.results_path)])
        == 0
    )
    assert (
        main(
            [
                "--registration",
                str(tree.registration_path),
                "--results",
                str(tree.results_path),
                "--output",
                str(decision_path),
            ]
        )
        == 2
    )
    assert (
        main(
            ["--registration", str(tmp_path / "missing.json"), "--results", str(tree.results_path)]
        )
        == 2
    )


def test_evaluation_receipt_excludes_only_generated_at() -> None:
    report = {
        "schema_version": 7,
        "generated_at": "2026-08-02T00:00:00Z",
        "rows": [{"passed": True}],
    }
    receipt = evaluation_receipt_sha256(report)
    report["generated_at"] = "2027-01-01T00:00:00Z"
    assert evaluation_receipt_sha256(report) == receipt
    report["rows"][0]["passed"] = False
    assert evaluation_receipt_sha256(report) != receipt


class _Clock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def set(self, current: datetime) -> None:
        self.current = current


def _fake_datetime_class(clock: _Clock) -> type:
    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.current if tz is not None else clock.current.replace(tzinfo=None)

    return FakeDateTime


@pytest.fixture(scope="module")
def frozen_tree(tmp_path_factory: pytest.TempPathFactory) -> helpers.CampaignTree:
    """A campaign executed under a faked clock whose packs have since expired.

    Packs were sealed on 2026-01-01, the registration was declared on
    2026-02-01, execution completed on 2026-03-01, and the packs expired on
    2026-06-01 — all before the real wall clock running these tests.
    """

    import cortheon.benchmark
    import cortheon.blind_evaluator
    import cortheon.parity_pack

    clock = _Clock(datetime(2026, 1, 1, tzinfo=UTC))
    fake = _fake_datetime_class(clock)
    modules = (
        cortheon.benchmark,
        cortheon.blind_evaluator,
        cortheon.parity_pack,
    )
    originals = [(module, module.datetime) for module in modules]
    saved_scale = parity.UNIVERSAL_SCALE_REQUIREMENTS
    parity.UNIVERSAL_SCALE_REQUIREMENTS = dict(helpers.TOY_UNIVERSAL_SCALE)
    for module in modules:
        module.datetime = fake
    try:
        return helpers.build_campaign_tree(
            tmp_path_factory.mktemp("frozen-campaign"),
            declared_at="2026-02-01T00:00:00+00:00",
            expires_at="2026-06-01T00:00:00+00:00",
            before_execution=lambda: clock.set(datetime(2026, 3, 1, tzinfo=UTC)),
        )
    finally:
        for module, original in originals:
            module.datetime = original
        parity.UNIVERSAL_SCALE_REQUIREMENTS = saved_scale


def _rewrite_execution_time(
    copied: helpers.CampaignTree,
    cell_id: str,
    generated_at: str,
    monkeypatch,
) -> None:
    """Move a cell's attested execution time and re-attest with its runner key."""

    from cortheon.benchmark import attest_blind_submission

    submission_path = copied.root / copied.result(cell_id)["submission_path"]
    artifact = json.loads(submission_path.read_text(encoding="utf-8"))
    artifact.pop("attestation", None)
    artifact["generated_at"] = generated_at
    monkeypatch.setenv(
        "RESEALED_RUNNER_KEY",
        copied.secrets[copied.cell(cell_id)["runner_key_env"]],
    )
    artifact = attest_blind_submission(artifact, key_env="RESEALED_RUNNER_KEY")
    submission_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    payload = json.loads(copied.results_path.read_text(encoding="utf-8"))
    for result in payload["results"]:
        if result["cell_id"] == cell_id:
            result["submission_sha256"] = hashlib.sha256(submission_path.read_bytes()).hexdigest()
    copied.results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_post_expiry_regrade_reproduces_gate_and_receipt(
    frozen_tree,
    monkeypatch,
) -> None:
    """Regrading archived artifacts after pack expiry is timeless."""

    helpers.reduce_test_scale(monkeypatch)
    frozen_tree.set_secrets(monkeypatch)
    decision = evaluate_replication_campaign(
        frozen_tree.registration_path,
        frozen_tree.results_path,
    )
    assert decision["passed"] is True, decision["failure_reasons"]
    stored = json.loads(frozen_tree.report_path(FIRST_CELL).read_text(encoding="utf-8"))
    assert stored["frontier_parity_gate"]["passed"] is True
    assert not stored["frontier_parity_gate"]["failure_reasons"]
    again = evaluate_replication_campaign(
        frozen_tree.registration_path,
        frozen_tree.results_path,
    )
    assert again["passed"] is True
    assert again["chain_head_sha256"] == decision["chain_head_sha256"]


def test_submission_before_registration_fails(
    frozen_tree,
    tmp_path: Path,
    monkeypatch,
) -> None:
    helpers.reduce_test_scale(monkeypatch)
    copied = frozen_tree.copy(tmp_path / "before-registration")
    copied.set_secrets(monkeypatch)
    _rewrite_execution_time(
        copied,
        FIRST_CELL,
        "2026-01-15T00:00:00+00:00",
        monkeypatch,
    )
    decision = evaluate_replication_campaign(
        copied.registration_path,
        copied.results_path,
    )
    assert decision["passed"] is False
    assert f"preregistration_sequence:{FIRST_CELL}" in decision["failure_reasons"]


def test_submission_before_pack_creation_fails(
    frozen_tree,
    tmp_path: Path,
    monkeypatch,
) -> None:
    helpers.reduce_test_scale(monkeypatch)
    copied = frozen_tree.copy(tmp_path / "before-creation")
    copied.set_secrets(monkeypatch)
    _rewrite_execution_time(
        copied,
        FIRST_CELL,
        "2025-12-01T00:00:00+00:00",
        monkeypatch,
    )
    with pytest.raises(CampaignContractError, match="cannot regrade cell"):
        evaluate_replication_campaign(copied.registration_path, copied.results_path)


def test_submission_after_pack_expiry_fails(
    frozen_tree,
    tmp_path: Path,
    monkeypatch,
) -> None:
    helpers.reduce_test_scale(monkeypatch)
    copied = frozen_tree.copy(tmp_path / "after-expiry")
    copied.set_secrets(monkeypatch)
    _rewrite_execution_time(
        copied,
        FIRST_CELL,
        "2026-07-01T00:00:00+00:00",
        monkeypatch,
    )
    with pytest.raises(CampaignContractError, match="cannot regrade cell"):
        evaluate_replication_campaign(copied.registration_path, copied.results_path)

"""Red contracts for content-free, independently replayable release records."""

from __future__ import annotations

import copy
import json
from argparse import Namespace

import pytest
from test_operator_lift_contrasts import _submissions

from cortheon.operator_lift import cli as operator_cli
from cortheon.operator_lift.case_bank import development_cases
from cortheon.operator_lift.cli import main
from cortheon.operator_lift.execution_models import ExecutionConfig, ExecutionOutcome
from cortheon.operator_lift.execution_release import (
    build_release,
    release_records,
    verify_release,
)
from cortheon.operator_lift.execution_report import content_free_report
from cortheon.operator_lift.execution_schedule import (
    execution_manifest,
    full_schedule,
    public_pack,
    run_descriptor,
)
from cortheon.operator_lift.execution_storage import validate_retained_artifacts


def _inputs(*, pilot: bool = False):
    cases = development_cases()
    manifest = execution_manifest(cases)
    schedule = full_schedule(manifest, cases)
    submissions = _submissions(cases, manifest)
    if pilot:
        selected = {schedule[0].case_id}
        schedule = tuple(cell for cell in schedule if cell.case_id in selected)
        submissions = [row for row in submissions if row["case_id"] in selected]
    pack = public_pack(cases)
    descriptor = run_descriptor(
        manifest,
        cases,
        schedule,
        ExecutionConfig("http://127.0.0.1:9000/v1", "local", "small", "private"),
        str(pack["pack_sha256"]),
    )
    run_sha256 = str(descriptor["run_sha256"])
    summaries = [
        {
            "condition_id": row["condition_id"],
            "identity_valid": True,
            "transcript_valid": True,
            "delivered": True,
            "safe": True,
            "correct": row["condition_id"] == "full",
            "model_steps": 2,
            "tokens": 20,
            "tool_calls": 1,
            "latency_seconds": 0.25,
            "timed_out": False,
        }
        for row in submissions
    ]
    projected = release_records(
        manifest,
        cases,
        schedule,
        submissions,
        summaries,
        run_sha256,
    )
    report = content_free_report(
        manifest,
        cases,
        submissions,
        summaries,
        run_sha256=run_sha256,
        event_chain_sha256=projected[-1]["record_sha256"],
        planned_cells=len(schedule),
    )
    return manifest, cases, schedule, submissions, summaries, descriptor, report


def _release(*, pilot: bool = False):
    values = _inputs(pilot=pilot)
    manifest, cases, schedule, submissions, summaries, descriptor, report = values
    return (
        build_release(
            manifest,
            cases,
            schedule,
            submissions,
            summaries,
            report,
            str(descriptor["run_sha256"]),
        ),
        values,
    )


def test_release_has_closed_content_free_schema_and_commitment_only_records() -> None:
    (
        release,
        (
            _manifest,
            _cases,
            schedule,
            _submissions,
            _summaries,
            _descriptor,
            _report,
        ),
    ) = _release()
    assert set(release) == {
        "schema_version",
        "content_free",
        "run_sha256",
        "manifest_sha256",
        "report_sha256",
        "records",
        "chain_root_sha256",
        "claim_eligible",
        "trust_anchor",
    }
    assert release["schema_version"] == 1
    assert release["content_free"] is True
    assert len(release["records"]) == len(schedule) == 540
    previous = "0" * 64
    for sequence, record in enumerate(release["records"], 1):
        assert set(record) == {
            "schema_version",
            "type",
            "sequence",
            "case_ordinal",
            "case_commitment",
            "condition_id",
            "repeat",
            "delivered",
            "safe",
            "correct",
            "output_present",
            "identity_valid",
            "transcript_valid",
            "measurements",
            "previous_record_sha256",
            "record_sha256",
        }
        assert record["type"] == "cell_evaluated"
        assert record["sequence"] == sequence
        assert record["previous_record_sha256"] == previous
        assert "case_id" not in record and "cell_id" not in record
        previous = record["record_sha256"]
    assert release["chain_root_sha256"] == previous


def test_release_serialization_contains_no_sensitive_content_or_content_commitments() -> None:
    release, _values = _release()
    serialized = json.dumps(release, sort_keys=True).casefold()
    forbidden = {
        "response",
        "prompt",
        "evidence",
        "answer",
        "path",
        "url",
        "credential",
        "api_key",
        "error",
        "arguments",
        "tool_result",
        "transcript_sha256",
        "source_a",
        "source_b",
    }
    assert all(token not in serialized for token in forbidden)


@pytest.mark.parametrize("mutation", ["delete", "reorder", "duplicate", "splice"])
def test_release_chain_rejects_structural_mutation(mutation: str) -> None:
    release, values = _release()
    changed = copy.deepcopy(release)
    records = changed["records"]
    if mutation == "delete":
        del records[3]
    elif mutation == "reorder":
        records[3], records[4] = records[4], records[3]
    elif mutation == "duplicate":
        records.insert(4, copy.deepcopy(records[3]))
    else:
        other, _ = _release(pilot=True)
        records[3] = copy.deepcopy(other["records"][0])
    with pytest.raises(ValueError):
        verify_release(changed, values[-1])


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("record", "correct", None),
        ("record", "tokens", 999),
        ("release", "run_sha256", "f" * 64),
        ("release", "manifest_sha256", "f" * 64),
        ("release", "report_sha256", "f" * 64),
    ],
)
def test_release_rejects_field_and_identity_mutation(
    target: str, field: str, value: object
) -> None:
    release, values = _release()
    changed = copy.deepcopy(release)
    if target == "release":
        changed[field] = value
    elif field == "tokens":
        changed["records"][0]["measurements"][field] = value
    elif field == "correct":
        changed["records"][0][field] = not changed["records"][0][field]
    else:
        changed["records"][0][field] = value
    with pytest.raises(ValueError):
        verify_release(changed, values[-1])


def test_release_rejects_unknown_fields_at_every_level() -> None:
    release, values = _release()
    for level in ("release", "record"):
        changed = copy.deepcopy(release)
        target = changed if level == "release" else changed["records"][0]
        target["unknown"] = True
        with pytest.raises(ValueError):
            verify_release(changed, values[-1])
    changed = copy.deepcopy(release)
    changed["records"][0]["measurements"]["unknown"] = 1
    with pytest.raises(ValueError):
        verify_release(changed, values[-1])


def test_replay_recomputes_counts_arms_measurements_and_root_deterministically() -> None:
    release, values = _release()
    replay = verify_release(release, values[-1])
    assert replay["record_count"] == 540
    assert replay["chain_root_sha256"] == release["chain_root_sha256"]
    assert set(replay["by_arm"]) == {"full", "ablation", "equal_budget_placebo"}
    for arm in replay["by_arm"].values():
        assert {"cells", "correct", "delivered", "safe"} <= set(arm)
    assert replay["measurements"] == {
        "model_steps": 1080,
        "tokens": 10800,
        "tool_calls": 540,
        "latency_seconds": 135.0,
    }
    rebuilt = build_release(
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[-1],
        str(values[-2]["run_sha256"]),
    )
    assert rebuilt == release


def test_replay_binds_descriptor_and_external_chain_root() -> None:
    release, values = _release(pilot=True)
    replay = verify_release(
        release,
        values[-1],
        values[-2],
        str(release["chain_root_sha256"]),
    )
    assert replay["descriptor_verified"] is True
    assert replay["trust_anchor_verified"] is True
    changed = copy.deepcopy(values[-2])
    changed["evaluator_identity"]["model_id"] = "other"
    with pytest.raises(ValueError, match="descriptor digest"):
        verify_release(release, values[-1], changed, str(release["chain_root_sha256"]))
    with pytest.raises(ValueError, match="externally pinned"):
        verify_release(release, values[-1], values[-2], "f" * 64)


def test_pilot_release_is_valid_but_never_claim_eligible() -> None:
    release, values = _release(pilot=True)
    replay = verify_release(release, values[-1])
    assert replay["record_count"] == len(values[2])
    assert release["claim_eligible"] is False
    assert replay["claim_eligible"] is False


def test_release_can_be_replayed_in_a_fresh_cli_process_contract(tmp_path, capsys) -> None:
    release, values = _release(pilot=True)
    release_path = tmp_path / "release.json"
    report_path = tmp_path / "report.json"
    run_path = tmp_path / "run.json"
    release_path.write_text(json.dumps(release), encoding="utf-8")
    report_path.write_text(json.dumps(values[-1]), encoding="utf-8")
    run_path.write_text(json.dumps(values[-2]), encoding="utf-8")
    assert (
        main(
            [
                "verify-release",
                "--release",
                str(release_path),
                "--report",
                str(report_path),
                "--run",
                str(run_path),
                "--expected-chain-root",
                str(release["chain_root_sha256"]),
            ]
        )
        == 0
    )
    replay = json.loads(capsys.readouterr().out)
    assert replay["record_count"] == 9
    assert replay["descriptor_verified"] is True
    assert replay["trust_anchor_verified"] is True


def test_successful_runner_retains_only_content_free_artifacts(tmp_path, monkeypatch) -> None:
    cases = development_cases()
    manifest = execution_manifest(cases)
    submissions = _submissions(cases, manifest)
    by_key = {(row["case_id"], row["condition_id"], row["repeat"]): row for row in submissions}

    def fake_run_cell(_config, _manifest, case, cell):
        submission = by_key[(case.case_id, cell.condition_id, cell.repeat)]
        return ExecutionOutcome(
            submission,
            {
                "condition_id": cell.condition_id,
                "identity_valid": True,
                "transcript_valid": True,
                "measurements_valid": True,
                "steps": 2,
                "inference_calls": 2,
                "tokens": 20,
                "tool_calls": 1,
                "latency_seconds": 0.25,
                "cost_usd": 0.0,
                "timed_out": False,
                "budget_reason": None,
                "terminal_status": "success",
            },
        )

    monkeypatch.setattr(operator_cli, "run_cell", fake_run_cell)
    monkeypatch.setenv("STAGE5_TEST_KEY", "private-api-key")
    operator_cli.run(
        Namespace(
            base_url="http://127.0.0.1:9000/v1",
            provider="local",
            model_id="small",
            api_key_env="STAGE5_TEST_KEY",
            output_dir=tmp_path,
            pilot_clusters=1,
            operator=None,
            timeout_seconds=120.0,
            context_tokens=16_384,
            output_tokens=2_048,
            max_steps=8,
            max_tool_calls=12,
        )
    )
    validate_retained_artifacts(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "release.json",
        "report.json",
        "run.json",
    }
    serialized = "".join(path.read_text() for path in tmp_path.iterdir())
    assert "private-api-key" not in serialized
    assert '"response"' not in serialized
    assert '"evidence"' not in serialized

from __future__ import annotations

import json

import pytest

from cortheon.operator_lift.execution_storage import (
    discard_private_state,
    freeze_submissions,
    initialize_run,
    load_checkpoint,
    save_checkpoint,
    save_release,
    save_report,
    validate_retained_artifacts,
)


def test_resume_reuses_exact_checkpoint_and_rejects_mutation(tmp_path) -> None:
    initialize_run(
        tmp_path,
        {"run_sha256": "a" * 64, "public_pack_sha256": "b" * 64},
        {"pack_sha256": "b" * 64},
    )
    assert not (tmp_path / "public-pack.json").exists()
    saved = save_checkpoint(
        tmp_path,
        "case--full--r0",
        "a" * 64,
        {"submission": {"case_id": "case"}, "summary": {"tokens": 1}},
    )
    assert load_checkpoint(tmp_path, "case--full--r0", "a" * 64) == saved
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(tmp_path, "case--full--r0", "b" * 64)
    path = tmp_path / "checkpoints/case--full--r0.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["summary"]["tokens"] = 2
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        load_checkpoint(tmp_path, "case--full--r0", "a" * 64)


def test_submission_freeze_is_create_once_and_content_addressed(tmp_path) -> None:
    records = [
        {
            "checkpoint_sha256": "c" * 64,
            "submission": {"case_id": "one", "response": {"answer": "private"}},
        }
    ]
    submissions, digest = freeze_submissions(tmp_path, "a" * 64, records)
    assert submissions == [records[0]["submission"]]
    assert len(digest) == 64
    assert "private" not in (tmp_path / "submissions.freeze.json").read_text()
    assert freeze_submissions(tmp_path, "a" * 64, records) == (submissions, digest)
    changed = [{**records[0], "checkpoint_sha256": "d" * 64}]
    with pytest.raises(ValueError, match="different bytes"):
        freeze_submissions(tmp_path, "a" * 64, changed)


def test_public_release_is_write_once(tmp_path) -> None:
    release = {"schema_version": 1, "content_free": True}
    save_release(tmp_path, release)
    save_release(tmp_path, release)
    assert json.loads((tmp_path / "release.json").read_text()) == release
    with pytest.raises(ValueError, match="different bytes"):
        save_release(tmp_path, {**release, "content_free": False})


def test_successful_finalization_discards_only_private_recovery_state(tmp_path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "cell.json").write_text('{"response":"private"}', encoding="utf-8")
    (tmp_path / "submissions.freeze.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run.json").write_text("{}", encoding="utf-8")
    save_report(tmp_path, {"raw_content_included": False})
    save_release(tmp_path, {"content_free": True})
    discard_private_state(tmp_path)
    validate_retained_artifacts(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "release.json",
        "report.json",
        "run.json",
    }


def test_retained_artifact_scanner_rejects_raw_content_fields(tmp_path) -> None:
    for name in ("run.json", "report.json", "release.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "report.json").write_text('{"response":"private"}', encoding="utf-8")
    with pytest.raises(ValueError, match="sensitive field"):
        validate_retained_artifacts(tmp_path)

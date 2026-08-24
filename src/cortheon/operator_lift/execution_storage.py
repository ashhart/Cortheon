"""Private recovery state and content-free finalized execution artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cortheon.operator_lift.execution_schedule import canonical_bytes

_SENSITIVE_FIELDS = frozenset(
    {
        "answer",
        "api_key",
        "arguments",
        "evidence",
        "path",
        "prompt",
        "response",
        "tool_result",
        "transcript_sha256",
        "url",
    }
)


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def initialize_run(root: Path, descriptor: dict[str, Any], pack: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if descriptor.get("public_pack_sha256") != pack.get("pack_sha256"):
        raise ValueError("public pack does not match the sealed run")
    path = root / "run.json"
    payload = canonical_bytes(descriptor)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("existing run.json does not match this sealed run")
    else:
        _write_once(path, payload)
    (root / "checkpoints").mkdir(exist_ok=True)


def checkpoint_path(root: Path, cell_id: str) -> Path:
    return root / "checkpoints" / f"{cell_id}.json"


def load_checkpoint(root: Path, cell_id: str, run_sha256: str) -> dict[str, Any] | None:
    path = checkpoint_path(root, cell_id)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("cell_id") != cell_id
        or value.get("run_sha256") != run_sha256
    ):
        raise ValueError("checkpoint identity is invalid")
    stored = value.get("checkpoint_sha256")
    unsigned = {key: item for key, item in value.items() if key != "checkpoint_sha256"}
    if stored != hashlib.sha256(canonical_bytes(unsigned)).hexdigest():
        raise ValueError("checkpoint digest is invalid")
    return value


def save_checkpoint(
    root: Path,
    cell_id: str,
    run_sha256: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": 1,
        "cell_id": cell_id,
        "run_sha256": run_sha256,
        **record,
    }
    value = {
        **unsigned,
        "checkpoint_sha256": hashlib.sha256(canonical_bytes(unsigned)).hexdigest(),
    }
    _write_once(checkpoint_path(root, cell_id), canonical_bytes(value))
    return value


def freeze_submissions(
    root: Path,
    run_sha256: str,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    freeze_path = root / "submissions.freeze.json"
    submissions = [record["submission"] for record in records]
    unsigned = {
        "schema_version": 2,
        "run_sha256": run_sha256,
        "checkpoint_sha256": [record["checkpoint_sha256"] for record in records],
        "submission_count": len(submissions),
    }
    digest = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    value = {**unsigned, "freeze_sha256": digest}
    payload = canonical_bytes(value)
    if freeze_path.exists():
        if freeze_path.read_bytes() != payload:
            raise ValueError("submission freeze already exists with different bytes")
    else:
        _write_once(freeze_path, payload)
    loaded = json.loads(freeze_path.read_text(encoding="utf-8"))
    check = {key: item for key, item in loaded.items() if key != "freeze_sha256"}
    if loaded.get("freeze_sha256") != hashlib.sha256(canonical_bytes(check)).hexdigest():
        raise ValueError("submission freeze digest is invalid")
    if loaded.get("submission_count") != len(submissions):
        raise ValueError("submission freeze count is invalid")
    return submissions, str(loaded["freeze_sha256"])


def save_release(root: Path, release: dict[str, Any]) -> None:
    """Persist the content-free public record chain without overwriting it."""

    path = root / "release.json"
    payload = canonical_bytes(release)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("release already exists with different bytes")
        return
    _write_once(path, payload)


def save_report(root: Path, report: dict[str, Any]) -> None:
    path = root / "report.json"
    payload = canonical_bytes(report)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("report already exists with different bytes")
        return
    _write_once(path, payload)


def discard_private_state(root: Path) -> None:
    """Remove exact evaluator recovery artifacts after public finalization."""

    checkpoints = root / "checkpoints"
    if checkpoints.exists():
        if not checkpoints.is_dir():
            raise ValueError("checkpoint location is invalid")
        for path in checkpoints.iterdir():
            if not path.is_file() or path.suffix != ".json":
                raise ValueError("checkpoint directory contains an unexpected entry")
            path.unlink()
        checkpoints.rmdir()
    for name in ("submissions.freeze.json", "public-pack.json"):
        (root / name).unlink(missing_ok=True)


def _reject_sensitive_fields(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in _SENSITIVE_FIELDS:
                raise ValueError(f"retained artifact contains sensitive field at {location}")
            _reject_sensitive_fields(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sensitive_fields(child, f"{location}[{index}]")


def validate_retained_artifacts(root: Path) -> None:
    """Fail if finalized artifacts retain raw task, model, or tool content."""

    for private_name in ("checkpoints", "submissions.freeze.json", "public-pack.json"):
        if (root / private_name).exists():
            raise ValueError("private evaluator state remains after finalization")
    for name in ("run.json", "report.json", "release.json"):
        path = root / name
        if not path.is_file():
            raise ValueError(f"finalized artifact {name} is missing")
        value = json.loads(path.read_text(encoding="utf-8"))
        _reject_sensitive_fields(value, name)

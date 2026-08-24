"""Frozen-program identity, isolation, lifecycle, and host-scope tests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import cast

import pytest
from qualification_support import (
    _historical_condition_entries,
    _result,
    _write_manifest,
)

from cortheon.benchmark_core.models import ImportCase
from cortheon.qualification_core import frozen_archive as archive
from cortheon.qualification_core import frozen_execution, frozen_receipt, frozen_smoke
from cortheon.qualification_core import frozen_old_planner as frozen
from cortheon.qualification_core import frozen_runtime as runtime_process
from cortheon.qualification_core.conditions import (
    CONDITION_REGISTRY_VERSION,
    HISTORICAL_CONDITIONS,
    OLD_PLANNER,
)
from cortheon.qualification_factory import QualificationError, load_manifest


def test_archive_and_smoke_receipt_are_content_addressed() -> None:
    assert frozen.archive_available()
    assert frozen.comparator_available()
    assert frozen.wrapper_sha256() == frozen.WRAPPER_SHA256
    assert hashlib.sha256(frozen.SMOKE.read_bytes()).hexdigest() == frozen.SMOKE_SHA256
    receipt = json.loads(frozen.SMOKE.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 2
    assert receipt["content_free"] is True
    assert receipt["host"]["id"] == "opencode"
    assert receipt["host"]["version"] == "1.18.18"
    assert len(receipt["host"]["executable_sha256"]) == 64
    assert receipt["inference"]["model_id"] == "mlx-community--Qwen3.5-0.8B-8bit"
    assert receipt["inference"]["identity_valid"] is True
    assert receipt["runtime"]["active_sessions_postflight"] == 0
    assert receipt["runtime"]["artifact_unchanged"] is True


def test_smoke_receipt_schema_rejects_mutation_and_private_material() -> None:
    receipt = json.loads(frozen.SMOKE.read_text(encoding="utf-8"))
    implementation = frozen.frozen_implementation_sha256()
    assert frozen_receipt.validate_smoke_receipt(
        receipt,
        implementation,
        frozen.WRAPPER_SHA256,
    )
    serialized = json.dumps(receipt, sort_keys=True).lower()
    for private in ("app.py", "test_app.py", "fix app.py", "api_key", "session_id"):
        assert private not in serialized

    mutations = []
    extra = copy.deepcopy(receipt)
    extra["unexpected"] = True
    mutations.append(extra)
    wrong_task = copy.deepcopy(receipt)
    wrong_task["sealed_task_sha256"] = "0" * 64
    mutations.append(wrong_task)
    wrong_host = copy.deepcopy(receipt)
    wrong_host["host"]["version"] = "1.18.19"
    mutations.append(wrong_host)
    wrong_endpoint = copy.deepcopy(receipt)
    wrong_endpoint["inference"]["endpoint_sha256"] = "0" * 64
    mutations.append(wrong_endpoint)
    bad_measurement = copy.deepcopy(receipt)
    bad_measurement["inference"]["measurements_valid"] = False
    mutations.append(bad_measurement)
    leaked_session = copy.deepcopy(receipt)
    leaked_session["runtime"]["active_sessions_postflight"] = 1
    mutations.append(leaked_session)
    for mutated in mutations:
        assert not frozen_receipt.validate_smoke_receipt(
            mutated,
            implementation,
            frozen.WRAPPER_SHA256,
        )


def test_candidate_generator_rejects_unregistered_identity_before_execution(monkeypatch) -> None:
    monkeypatch.setattr(
        frozen_smoke,
        "_run_frozen_evidence",
        lambda *_args, **_kwargs: pytest.fail("execution must not begin"),
    )
    with pytest.raises(ValueError, match="fixed identity"):
        frozen_smoke.generate_smoke_candidate(
            opencode="opencode",
            provider="spoofed",
            model_id=frozen_receipt.SMOKE_MODEL,
            base_url=frozen_receipt.SMOKE_BASE_URL,
            api_key="secret",
        )


def test_verified_runtime_imports_only_the_extracted_program() -> None:
    with frozen.frozen_old_planner() as runtime:
        assert runtime.health()["active_sessions"] == 0
        assert runtime.unchanged()
        assert runtime.adapter.name == "adapter.js"
        assert not str(runtime.root).lower().endswith("old_planner")
        assert isinstance(runtime.process.args, list)
        argv = " ".join(str(item) for item in runtime.process.args)
        assert "19d035c" not in argv
        assert "old_planner" not in argv
        assert "historical" not in argv
        assert runtime.token not in argv
        assert runtime.url not in argv
        wrapper = runtime.adapter.read_text(encoding="utf-8")
        assert 'await import("./program.js")' in wrapper
        assert "src/cortheon" not in wrapper
        program = runtime.adapter.with_name("program.js").read_bytes()
        assert (
            hashlib.sha256(program).hexdigest()
            == archive.MEMBER_SHA256["src/cortheon/opencode_plugin.js"]
        )


def test_archive_duplicate_member_fails_before_extraction(monkeypatch, tmp_path) -> None:
    duplicate = tmp_path / "duplicate.tar.gz"
    with tarfile.open(archive.ARCHIVE, "r:gz") as source:
        members = source.getmembers()
        payloads = {}
        for member in members:
            if not member.isfile():
                continue
            stream = source.extractfile(member)
            assert stream is not None
            payloads[member.name] = stream.read()
    with tarfile.open(duplicate, "w:gz") as target:
        for member in members:
            data = payloads.get(member.name)
            target.addfile(member, io.BytesIO(data) if data is not None else None)
        repeated = next(member for member in members if member.isfile())
        target.addfile(repeated, io.BytesIO(payloads[repeated.name]))
    monkeypatch.setattr(archive, "ARCHIVE", duplicate)
    monkeypatch.setattr(archive, "ARCHIVE_BYTES", duplicate.stat().st_size)
    monkeypatch.setattr(
        archive,
        "ARCHIVE_SHA256",
        hashlib.sha256(duplicate.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="duplicate"):
        archive.extract_verified(tmp_path / "out")


def test_primary_error_survives_historical_runtime_death() -> None:
    with (
        pytest.raises(RuntimeError, match="primary failure"),
        frozen.frozen_old_planner() as runtime,
    ):
        runtime.process.kill()
        runtime.process.wait(timeout=3)
        raise RuntimeError("primary failure")


def test_readiness_timeout_reaps_the_child_and_closes_stderr(monkeypatch, tmp_path) -> None:
    archive.extract_verified(tmp_path)
    processes = []
    real_popen = runtime_process.subprocess.Popen

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    class EmptySelector:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def register(self, *_args):
            return None

        def select(self, _timeout):
            return []

    monkeypatch.setattr(runtime_process.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(runtime_process.selectors, "DefaultSelector", EmptySelector)
    with pytest.raises(ValueError, match="readiness"):
        runtime_process.start_runtime(tmp_path, tmp_path / "adapter.js", "t" * 64)
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stderr is not None and processes[0].stderr.closed


def test_malformed_readiness_reaps_the_child(monkeypatch, tmp_path) -> None:
    archive.extract_verified(tmp_path)
    processes = []
    real_popen = runtime_process.subprocess.Popen

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runtime_process.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(
        runtime_process,
        "BOOTSTRAP",
        """
import json,os,time
control=int(os.environ.pop('CORTHEON_CONTROL_FD'))
status=int(os.environ.pop('CORTHEON_STATUS_FD'))
with os.fdopen(control,'rb',closefd=True) as stream: json.load(stream)
with os.fdopen(status,'w',closefd=True) as stream: json.dump({},stream); stream.flush()
time.sleep(10)
""",
    )
    with pytest.raises(ValueError, match="readiness record"):
        runtime_process.start_runtime(tmp_path, tmp_path / "adapter.js", "t" * 64)
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stderr is not None and processes[0].stderr.closed


def test_historical_manifest_is_explicit_and_opencode_only(tmp_path) -> None:
    designated = _write_manifest(
        tmp_path,
        cells=[
            {
                "id": "flagship",
                "suite": "semantic",
                "host": "opencode",
                "provider": "Local",
                "model_id": "small-model",
                "cases": 2,
                "repeats": 1,
                "historical_comparison": True,
                "conditions": _historical_condition_entries(),
            }
        ],
    )
    cell = load_manifest(designated).cells[0]
    assert cell.historical_comparison is True
    assert cell.condition_ids == HISTORICAL_CONDITIONS

    value = json.loads(designated.read_text(encoding="utf-8"))
    value["cells"][0]["host"] = "pi"
    designated.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(QualificationError, match="only for OpenCode"):
        load_manifest(designated)


def test_historical_manifest_rejects_spoofed_or_missing_condition(tmp_path) -> None:
    path = _write_manifest(
        tmp_path,
        cells=[
            {
                "id": "flagship",
                "suite": "semantic",
                "host": "opencode",
                "provider": "Local",
                "model_id": "small-model",
                "cases": 2,
                "repeats": 1,
                "historical_comparison": True,
                "conditions": _historical_condition_entries(),
            }
        ],
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    old = next(item for item in value["cells"][0]["conditions"] if item["id"] == OLD_PLANNER)
    old["implementation_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(QualificationError, match="digest mismatch"):
        load_manifest(path)

    value["cells"][0]["conditions"] = [
        item for item in value["cells"][0]["conditions"] if item["id"] != OLD_PLANNER
    ]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(QualificationError, match="condition matrix"):
        load_manifest(path)


class _FakeRuntime:
    url = "http://127.0.0.1:12345"
    token = "t" * 64
    adapter = Path("/tmp/neutral-adapter.js")

    def __init__(self) -> None:
        self.cleaned = False

    def health(self):
        return {"active_sessions": 0}

    def metrics(self):
        if not self.cleaned:
            return dict.fromkeys(
                (
                    "sessions_started",
                    "observations_accepted",
                    "sessions_completed",
                    "completion_withheld",
                    "sessions_evidence_closed",
                    "sessions_abandoned",
                    "controller_decisions",
                    "controller_alternatives_considered",
                ),
                0,
            )
        return {
            "sessions_started": 1,
            "observations_accepted": 1,
            "sessions_completed": 0,
            "completion_withheld": 0,
            "sessions_evidence_closed": 0,
            "sessions_abandoned": 1,
            "controller_decisions": 0,
            "controller_alternatives_considered": 0,
        }

    def abandon_active(self):
        self.cleaned = True
        return 1

    def unchanged(self):
        return True

    def control_payload(self):
        return b"private-control"


def test_frozen_job_binds_identity_lifecycle_and_neutral_runner(monkeypatch) -> None:
    result = _result("case", 0, OLD_PLANNER, True, telemetry=True)
    result.execution_identity_valid = True
    result.execution_measurements_valid = True
    calls = []

    def fake_run(args, case, **kwargs):
        calls.append((args, case, kwargs))
        return result

    monkeypatch.setattr(frozen_execution, "run_job", fake_run)
    runtime = _FakeRuntime()
    observed = frozen_execution.run_frozen_job(
        argparse.Namespace(host="opencode"),
        ImportCase("case", "file.py", "module", True, "prompt"),
        repeat=0,
        runtime=cast(frozen.FrozenRuntime, runtime),
    )
    args, _case, kwargs = calls[0]
    assert args.runtime_url == runtime.url
    assert args.evaluation_plugin_path == runtime.adapter
    assert kwargs["evaluator_control_payload"] == b"private-control"
    assert kwargs["control_token"] == runtime.token
    assert observed.condition_registry_version == CONDITION_REGISTRY_VERSION
    assert observed.condition_profile_receipt_valid is True
    assert observed.runtime_sessions_abandoned == 1
    assert observed.condition_operator_counts is None

"""Installed-byte and adapter-closure identity tests for scaling reports."""

from __future__ import annotations

import zipfile
from types import SimpleNamespace

import pytest

from cortheon.benchmark_core.scaling_identity import (
    _scaling_adapter_digest,
    _scaling_experiment_identity,
    _scaling_identity_valid,
    _scaling_tree_digest,
)


def test_source_and_archive_traversables_hash_the_same_installed_bytes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "runtime.py").write_text("VERSION = 1\n", encoding="utf-8")
    (source / "adapter.js").write_text("export const x = 1\n", encoding="utf-8")
    archive_path = tmp_path / "package.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(source / "adapter.js", "package/adapter.js")
        archive.write(source / "runtime.py", "package/runtime.py")
    with zipfile.ZipFile(archive_path) as archive:
        archived = zipfile.Path(archive, "package/")
        assert _scaling_tree_digest(source) == _scaling_tree_digest(archived)

    original = _scaling_tree_digest(source)
    (source / "runtime.py").write_text("VERSION = 2\n", encoding="utf-8")
    assert _scaling_tree_digest(source) != original


def test_producer_binds_the_model_id_observed_by_inference_health(monkeypatch, tmp_path):
    import cortheon.benchmark_core.scaling_identity as identity_module

    monkeypatch.setattr(
        identity_module,
        "_scaling_command_identity",
        lambda _command: {
            "configured_command": "pi",
            "executable_sha256": "2" * 64,
            "version": "pi 1.0",
        },
    )
    monkeypatch.setattr(identity_module, "_scaling_tree_digest", lambda _root: "4" * 64)
    monkeypatch.setattr(identity_module, "_scaling_adapter_digest", lambda *_args: "5" * 64)
    args = SimpleNamespace(
        repository=tmp_path,
        suite="semantic",
        seed=7,
        repeats=1,
        host="pi",
        pi="pi",
        opencode="opencode",
        runtime_url="http://127.0.0.1:8743",
        provider="local",
        model_id="registered-model",
        base_url="http://127.0.0.1:18081/v1",
        inference_artifact_sha256="8" * 64,
        reasoning=False,
        timeout_seconds=60.0,
        context_tokens=8192,
        output_tokens=512,
        max_tool_calls=16,
        frontier_cli="",
        frontier_model_id="",
        frontier_inference_artifact_sha256="",
        max_steps=4,
    )
    case = SimpleNamespace(case_id="case_0")
    identity = _scaling_experiment_identity(
        args,
        health={
            "service": "cortheon-cognitive",
            "version": "1.0",
            "protocol_version": "1.0.0",
            "source_fingerprint": "6" * 64,
        },
        inference={"model_id": "served-model"},
        frontier_inference=None,
        repository_snapshot="1" * 64,
        blinded_cases=[{"case_id": "case_0"}],
        jobs=[(case, 0, "baseline"), (case, 0, "cortheon")],
    )

    assert identity["inference"]["observed_model_id"] == "served-model"
    assert _scaling_identity_valid(identity) is False


def test_generic_identity_binds_stable_wrapper_runtime_condition_and_web(
    monkeypatch,
    tmp_path,
):
    import cortheon.benchmark_core.scaling_identity as identity_module

    monkeypatch.setattr(identity_module, "_scaling_tree_digest", lambda _root: "4" * 64)
    pre = {
        "wrapper_sha256": "6" * 64,
        "runtime_sha256": "a" * 64,
        "condition_sha256": "a" * 64,
        "web_provider_sha256": "b" * 64,
        "host_identity_sha256": "5" * 64,
    }
    args = SimpleNamespace(
        repository=tmp_path,
        suite="research",
        seed=7,
        repeats=1,
        host="generic_mcp",
        pi="pi",
        opencode="opencode",
        runtime_url="embedded",
        provider="local",
        model_id="small",
        base_url="http://127.0.0.1:9000/v1",
        inference_artifact_sha256="8" * 64,
        reasoning=False,
        timeout_seconds=60.0,
        context_tokens=8192,
        output_tokens=512,
        max_tool_calls=16,
        frontier_cli="",
        frontier_model_id="",
        frontier_inference_artifact_sha256="",
        max_steps=4,
        generic_implementation_pre=pre,
        generic_implementation_post=dict(pre),
    )
    case = SimpleNamespace(case_id="case_0")
    identity = _scaling_experiment_identity(
        args,
        health={
            "service": "cortheon-generic-mcp-evaluator",
            "version": "1",
            "protocol_version": "1.0.0",
            "source_fingerprint": "6" * 64,
        },
        inference={"model_id": "small"},
        frontier_inference=None,
        repository_snapshot="1" * 64,
        blinded_cases=[{"case_id": "case_0"}],
        jobs=[(case, 0, "baseline"), (case, 0, "cortheon")],
    )

    assert identity["schema_version"] == 2
    assert identity["cortheon_runtime"]["adapter_sha256"] == "5" * 64
    assert _scaling_identity_valid(identity)
    identity["generic_implementation"]["web_provider_post_sha256"] = "c" * 64
    assert not _scaling_identity_valid(identity)


@pytest.mark.parametrize(
    ("host", "facade", "core", "suffix"),
    [
        ("pi", "pi_extension.ts", "pi_core", ".ts"),
        ("opencode", "opencode_plugin.js", "opencode_core", ".js"),
    ],
)
def test_adapter_digest_binds_sibling_content_and_membership(
    tmp_path,
    host,
    facade,
    core,
    suffix,
):
    (tmp_path / facade).write_text("export facade\n", encoding="utf-8")
    core_path = tmp_path / core
    core_path.mkdir()
    sibling = core_path / f"sibling{suffix}"
    sibling.write_text("export const value = 1\n", encoding="utf-8")
    original = _scaling_adapter_digest(tmp_path, host)

    (tmp_path / facade).write_text("export changed facade\n", encoding="utf-8")
    assert _scaling_adapter_digest(tmp_path, host) != original
    (tmp_path / facade).write_text("export facade\n", encoding="utf-8")
    sibling.write_text("export const value = 2\n", encoding="utf-8")
    assert _scaling_adapter_digest(tmp_path, host) != original
    sibling.write_text("export const value = 1\n", encoding="utf-8")
    sibling.unlink()
    assert _scaling_adapter_digest(tmp_path, host) != original
    sibling.write_text("export const value = 1\n", encoding="utf-8")
    added = core_path / f"added{suffix}"
    added.write_text("export const added = 1\n", encoding="utf-8")
    assert _scaling_adapter_digest(tmp_path, host) != original
    added.unlink()
    assert _scaling_adapter_digest(tmp_path, host) == original

    ignored = core_path / "notes.txt"
    ignored.write_text("not an invoked module\n", encoding="utf-8")
    assert _scaling_adapter_digest(tmp_path, host) == original


@pytest.mark.parametrize(
    ("host", "facade", "core", "sibling"),
    [
        ("pi", "pi_extension.ts", "pi_core", "session.ts"),
        ("opencode", "opencode_plugin.js", "opencode_core", "session.js"),
    ],
)
def test_adapter_digest_is_identical_for_source_and_archive_traversables(
    tmp_path,
    host,
    facade,
    core,
    sibling,
):
    source = tmp_path / "source"
    (source / core).mkdir(parents=True)
    (source / facade).write_text("export facade\n", encoding="utf-8")
    (source / core / sibling).write_text("export sibling\n", encoding="utf-8")
    archive_path = tmp_path / "package.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(source / facade, f"package/{facade}")
        archive.write(source / core / sibling, f"package/{core}/{sibling}")
    with zipfile.ZipFile(archive_path) as archive:
        archived = zipfile.Path(archive, "package/")
        assert _scaling_adapter_digest(source, host) == _scaling_adapter_digest(
            archived,
            host,
        )

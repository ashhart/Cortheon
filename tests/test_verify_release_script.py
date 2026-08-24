from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "verify-release"


def _module():
    loader = importlib.machinery.SourceFileLoader("verify_release", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_release_script_is_executable_and_within_the_line_cap() -> None:
    assert SCRIPT.stat().st_mode & 0o111
    assert len(SCRIPT.read_bytes().splitlines()) <= 500


def test_release_inventory_covers_authored_languages_and_has_no_god_files() -> None:
    module = _module()
    inventory = module._line_inventory()
    assert inventory["files"] > 100
    assert inventory["maximum_lines"] <= 500
    files = {path.suffix for path in module._authored_files()}
    assert {".py", ".js", ".ts"} <= files


def test_release_artifact_record_enforces_the_fixed_cap(tmp_path: Path) -> None:
    module = _module()
    artifact = tmp_path / "artifact.whl"
    artifact.write_bytes(b"1234")
    assert module._artifact_record(artifact, 4)["bytes"] == 4
    with pytest.raises(module.GateFailure, match="limit"):
        module._artifact_record(artifact, 3)


def test_release_gate_rejects_untracked_benchmark_material(monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(
        module,
        "_git_paths",
        lambda *_arguments: ["benchmarks/private-pack.json"],
    )
    with pytest.raises(module.GateFailure, match="untracked benchmark"):
        module._reject_untracked_benchmark_material()


def test_release_privacy_inventory_rejects_personal_paths_and_private_keys(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _module()
    clean = tmp_path / "clean.py"
    clean.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "_authored_files", lambda: [clean])
    assert module._privacy_inventory() == {"files_scanned": 1, "findings": 0}

    clean.write_text("path = '/" + "Users/person/project'\n", encoding="utf-8")
    with pytest.raises(module.GateFailure, match="personal or secret"):
        module._privacy_inventory()

    clean.write_text("-----BEGIN " + "OPENSSH PRIVATE KEY-----\n", encoding="utf-8")
    with pytest.raises(module.GateFailure, match="personal or secret"):
        module._privacy_inventory()

"""Manifest loading: strict schema, no literal credentials, gates only tighten."""

import json

import pytest
from qualification_support import _condition_entries, _write_manifest

from cortheon.qualification_factory import QualificationError, load_manifest


def test_manifest_is_strict_and_rejects_literal_credentials(tmp_path):
    path = _write_manifest(tmp_path)
    value = json.loads(path.read_text())
    value["cells"][0]["api_key"] = "definitely-secret"
    path.write_text(json.dumps(value))

    with pytest.raises(QualificationError, match="api_key_env"):
        load_manifest(path)


def test_manifest_schema_two_fails_closed(tmp_path):
    path = _write_manifest(tmp_path)
    value = json.loads(path.read_text())
    value["schema_version"] = 2
    path.write_text(json.dumps(value))

    with pytest.raises(QualificationError, match="must be 3"):
        load_manifest(path)


def test_manifest_rejects_credentials_hidden_in_endpoint_queries(tmp_path):
    path = _write_manifest(tmp_path)
    value = json.loads(path.read_text())
    value["cells"][0]["base_url"] = "http://127.0.0.1:18081/v1?api_key=definitely-secret"
    path.write_text(json.dumps(value))

    with pytest.raises(QualificationError, match="query"):
        load_manifest(path)


def test_tier_policy_can_only_be_tightened(tmp_path):
    path = _write_manifest(
        tmp_path,
        tier="weekly",
        gates={"max_false_block_rate": 0.5},
    )

    with pytest.raises(QualificationError, match="only tighten"):
        load_manifest(path)


def test_toml_manifest_and_environment_reference_validate(tmp_path):
    path = tmp_path / "qualification.toml"
    condition_toml = "\n\n".join(
        "[[cells.conditions]]\n"
        f'id = "{entry["id"]}"\n'
        f'config_sha256 = "{entry["config_sha256"]}"\n'
        f'implementation_sha256 = "{entry["implementation_sha256"]}"'
        for entry in _condition_entries()
    )
    path.write_text(
        """
schema_version = 3
tier = "pr"
repository = "."
seed = 9

[[cells]]
id = "local-semantic"
suite = "semantic"
host = "opencode"
provider = "Local"
model_id = "small-model"
api_key_env = "LOCAL_MODEL_KEY"
cases = 2
repeats = 1
""".strip()
        + "\n\n"
        + condition_toml
    )

    manifest = load_manifest(path)

    assert manifest.cells[0].api_key_env == "LOCAL_MODEL_KEY"
    assert manifest.gates["min_independent_cases"] == 2

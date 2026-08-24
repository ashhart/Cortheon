from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from cortheon.threat_model.catalog import THREAT_MANIFEST
from cortheon.threat_model.validation import validate_threat_model

ROOT = Path(__file__).parents[1]


def _expected_nodes() -> tuple[str, ...]:
    return tuple(node for risk in THREAT_MANIFEST.risks for node in risk.test_node_ids)


def test_every_bound_hostile_test_exists_and_collects_exactly() -> None:
    result = validate_threat_model(ROOT, THREAT_MANIFEST)
    assert result.errors == ()
    assert result.valid is True
    assert result.catalog_valid is True
    assert result.collection_valid is True
    assert result.collected_node_ids == _expected_nodes()
    assert result.hostile_tests_executed is False
    assert result.hostile_tests_passed is False


def test_reported_source_digests_bind_exact_current_test_bytes() -> None:
    result = validate_threat_model(ROOT, THREAT_MANIFEST)
    for relative, digest in result.test_source_sha256:
        assert digest == hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_collection_rename_or_missing_node_fails_closed() -> None:
    def incomplete(command: tuple[str, ...], root: Path) -> subprocess.CompletedProcess[str]:
        del root
        output = "\n".join(command[6:-1]) + "\n"
        return subprocess.CompletedProcess(command, 0, output, "")

    result = validate_threat_model(ROOT, THREAT_MANIFEST, runner=incomplete)
    assert result.valid is False
    assert result.collection_valid is False
    assert "exact_test_collection_mismatch" in result.errors


def test_execution_failure_is_distinct_from_collection_failure() -> None:
    nodes = _expected_nodes()
    calls = 0

    def runner(command: tuple[str, ...], root: Path) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        del root
        calls += 1
        if "--collect-only" in command:
            return subprocess.CompletedProcess(command, 0, "\n".join(nodes) + "\n", "")
        return subprocess.CompletedProcess(command, 1, "failed", "")

    result = validate_threat_model(
        ROOT,
        THREAT_MANIFEST,
        execute_hostile_tests=True,
        runner=runner,
    )
    assert calls == 2
    assert result.collection_valid is True
    assert result.hostile_tests_executed is True
    assert result.hostile_tests_passed is False
    assert result.valid is False
    assert result.errors == ("hostile_test_execution_failed",)

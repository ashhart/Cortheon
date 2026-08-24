"""Fail-closed catalog, source-binding, collection, and execution validation."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from cortheon.threat_model.catalog import REQUIRED_RISK_IDS
from cortheon.threat_model.models import (
    MODEL_VERSION,
    SCHEMA_VERSION,
    ThreatManifest,
    ValidationResult,
    manifest_sha256,
)

CommandRunner = Callable[[tuple[str, ...], Path], subprocess.CompletedProcess[str]]


def _default_runner(command: tuple[str, ...], root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)


def _catalog_errors(manifest: ThreatManifest, root: Path) -> tuple[str, ...]:
    errors: list[str] = []
    if manifest.schema_version != SCHEMA_VERSION or manifest.model_version != MODEL_VERSION:
        errors.append("threat_model_version_mismatch")
    ids = [risk.risk_id for risk in manifest.risks]
    if tuple(ids) != REQUIRED_RISK_IDS:
        errors.append("risk_inventory_missing_renamed_reordered_or_duplicated")
    nodes: list[str] = []
    for index, risk in enumerate(manifest.risks):
        try:
            risk.validate()
        except ValueError as exc:
            errors.append(f"risk_{index}_invalid:{exc}")
        nodes.extend(risk.test_node_ids)
    if len(nodes) != len(set(nodes)):
        errors.append("test_node_owned_by_multiple_risks")
    residual_ids = [residual.residual_id for residual in manifest.residuals]
    if len(residual_ids) != len(set(residual_ids)) or not residual_ids:
        errors.append("residual_inventory_invalid")
    for index, residual in enumerate(manifest.residuals):
        try:
            residual.validate()
        except ValueError as exc:
            errors.append(f"residual_{index}_invalid:{exc}")
    for node in nodes:
        relative = node.split("::", 1)[0]
        path = root / relative
        if not path.is_file() or path.suffix != ".py":
            errors.append(f"test_source_missing:{relative}")
    return tuple(sorted(set(errors)))


def _parse_collected(stdout: str) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in stdout.splitlines()
        if line.strip().startswith("tests/") and "::test_" in line
    )


def validate_threat_model(
    root: Path,
    manifest: ThreatManifest,
    *,
    execute_hostile_tests: bool = False,
    runner: CommandRunner | None = None,
) -> ValidationResult:
    """Validate the closed inventory and optionally execute its exact test nodes."""

    root = root.resolve()
    runner = runner or _default_runner
    errors = list(_catalog_errors(manifest, root))
    expected_nodes = tuple(node for risk in manifest.risks for node in risk.test_node_ids)
    collect_command = (
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        *expected_nodes,
    )
    collected: tuple[str, ...] = ()
    collection_valid = False
    if not errors:
        completed = runner(collect_command, root)
        collected = _parse_collected(completed.stdout)
        collection_valid = completed.returncode == 0 and collected == expected_nodes
        if not collection_valid:
            errors.append("exact_test_collection_mismatch")
    hostile_tests_passed = False
    pytest_command: tuple[str, ...] = ()
    if execute_hostile_tests and collection_valid:
        pytest_command = (sys.executable, "-m", "pytest", "-q", *expected_nodes)
        completed = runner(pytest_command, root)
        hostile_tests_passed = completed.returncode == 0
        if not hostile_tests_passed:
            errors.append("hostile_test_execution_failed")
    source_paths = sorted({node.split("::", 1)[0] for node in expected_nodes})
    source_digests = tuple(
        (relative, hashlib.sha256((root / relative).read_bytes()).hexdigest())
        for relative in source_paths
        if (root / relative).is_file()
    )
    catalog_valid = not any(
        error != "exact_test_collection_mismatch" and error != "hostile_test_execution_failed"
        for error in errors
    )
    valid = (
        catalog_valid
        and collection_valid
        and (hostile_tests_passed if execute_hostile_tests else True)
    )
    return ValidationResult(
        valid=valid,
        catalog_valid=catalog_valid,
        collection_valid=collection_valid,
        hostile_tests_executed=execute_hostile_tests and collection_valid,
        hostile_tests_passed=hostile_tests_passed,
        manifest_sha256=manifest_sha256(manifest),
        errors=tuple(sorted(set(errors))),
        collected_node_ids=collected,
        test_source_sha256=source_digests,
        pytest_command=pytest_command,
    )

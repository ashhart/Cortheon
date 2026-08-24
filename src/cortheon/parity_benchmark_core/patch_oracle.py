"""Closed fixture and executable-test bindings for repository patch cases."""

from __future__ import annotations

import hashlib
import json
from typing import Any

TEST_COMMAND = ["python", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"]


def content_digest(files: dict[str, str]) -> str:
    canonical = json.dumps(
        files,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_patch_oracle(case_id: str, grader: dict[str, Any]) -> None:
    oracle = grader.get("oracle")
    fixture = grader.get("fixture")
    if not isinstance(oracle, dict) or set(oracle) != {
        "pristine_sha256",
        "test_files",
        "tests_sha256",
        "test_command",
    }:
        raise ValueError(f"case {case_id} has an invalid patch oracle")
    if not isinstance(fixture, dict):
        raise ValueError(f"case {case_id} has no patch fixture")
    test_files = oracle.get("test_files")
    allowed_files = grader.get("allowed_files")
    if (
        not isinstance(test_files, list)
        or not test_files
        or len(set(test_files)) != len(test_files)
        or any(not isinstance(path, str) or path not in fixture for path in test_files)
        or not isinstance(allowed_files, list)
        or bool(set(test_files) & set(allowed_files))
    ):
        raise ValueError(f"case {case_id} patch oracle has invalid test_files")
    tests = {path: fixture[path] for path in test_files}
    if (
        oracle.get("pristine_sha256") != content_digest(fixture)
        or oracle.get("tests_sha256") != content_digest(tests)
        or oracle.get("test_command") != TEST_COMMAND
    ):
        raise ValueError(f"case {case_id} patch oracle digest or command mismatch")


def patch_evidence_digest(grader: dict[str, Any], patch: str) -> str:
    oracle = grader["oracle"]
    canonical = json.dumps(
        {
            "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
            "pristine_sha256": oracle["pristine_sha256"],
            "tests_sha256": oracle["tests_sha256"],
            "test_command": oracle["test_command"],
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()

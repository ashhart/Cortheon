"""Exact repository source identity for generic MCP evaluator execution."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

EXPECTED_DIGEST_ENV = "CORTHEON_GENERIC_SOURCE_SHA256"
VERIFIED_DIGEST_ENV = "CORTHEON_VERIFIED_GENERIC_SOURCE_SHA256"
SOURCE_FILES = (
    "generic_mcp_answer_schemas.py",
    "generic_mcp_brief.py",
    "generic_mcp_auto.py",
    "generic_mcp_claim_validation.py",
    "generic_mcp_diagnostics.py",
    "generic_mcp_executor.py",
    "generic_mcp_host.py",
    "generic_mcp_launcher.py",
    "generic_mcp_lifecycle.py",
    "generic_mcp_model.py",
    "generic_mcp_message_validation.py",
    "generic_mcp_process.py",
    "generic_mcp_projection.py",
    "generic_mcp_protocol.py",
    "generic_mcp_runner.py",
    "generic_mcp_runtime.py",
    "generic_mcp_runtime_projection.py",
    "generic_mcp_search_projection.py",
    "generic_mcp_schema_validation.py",
    "generic_mcp_source.py",
    "generic_mcp_terminal.py",
    "generic_mcp_terminal_validation.py",
    "generic_mcp_tools.py",
    "generic_mcp_turns.py",
    "generic_mcp_validation.py",
)


def generic_source_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        data = (root / name).read_bytes()
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def resource_records(root: Path, paths: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    root = root.resolve(strict=True)
    records: list[dict[str, Any]] = []
    for relative in paths:
        candidate = (root / relative).resolve(strict=True)
        if not candidate.is_file() or candidate.is_symlink() or not candidate.is_relative_to(root):
            raise ValueError("generic MCP resource is outside the evaluator workspace")
        payload = candidate.read_bytes()
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    return tuple(records)

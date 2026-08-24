"""Check a sealed pack against the evaluator secret that issued it."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def verify_case_pack(path: Path, *, key_env: str) -> dict[str, Any]:
    from cortheon.benchmark import _load_case_pack

    loaded = _load_case_pack(path, key_env=key_env)
    seal = loaded.metadata.get("seal")
    return {
        "ok": bool(isinstance(seal, dict) and seal.get("verified") is True),
        "cases": len(loaded.cases),
        "metadata": loaded.metadata,
    }

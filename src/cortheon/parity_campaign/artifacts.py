"""Path resolution and digesting for campaign artifacts.

Relative artifact paths are resolved against the file that references them
(the registration for contracts and packs, the results file for submissions
and reports), never against the process working directory. Paths are
normalized by resolution so aliases of the same file are detectable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cortheon.parity_campaign.errors import CampaignContractError


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.expanduser().read_bytes()
    except OSError as exc:
        raise CampaignContractError(f"cannot read {label} {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CampaignContractError(f"invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CampaignContractError(f"{label} {path} must be a JSON object")
    return payload


def resolve_artifact(base_dir: Path, raw: str) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def file_digest(path: Path, *, label: str, cell_id: str) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CampaignContractError(
            f"cannot read {label} for cell {cell_id}: {path}: {exc}"
        ) from exc
    return hashlib.sha256(raw).hexdigest()

"""Writing an evaluator artifact: canonical JSON, owner-only permissions.

Packs and contracts are both evaluator-owned files that a digest is taken
over, so both are written the same way -- stable key order, two-space indent,
a trailing newline, mode 0600.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PRIVATE_MODE = 0o600


def write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(PRIVATE_MODE)

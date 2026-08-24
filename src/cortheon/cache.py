from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

# Bump when cached-value semantics change so stale entries are not read.
SYMBOLS_SCHEMA_VERSION = "1"


class FactCache:
    """Cache for version-pinned, immutable facts.

    The freshness contract has a deliberate loophole: facts about a pinned
    artifact never change — httpx 0.28.1's symbol table is the same forever.
    Those are safe to cache indefinitely. Anything about "latest" must NOT go
    through this cache.
    """

    def __init__(self, base_dir: Path | str = ".cortheon/cache") -> None:
        self.base_dir = Path(base_dir)
        self.enabled = os.environ.get("CORTHEON_NO_CACHE", "") != "1"

    def get(self, *key_parts: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(key_parts)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(payload, dict) or list(payload.get("key") or []) != list(key_parts):
            return None
        return payload.get("value")

    def put(self, value: Any, *key_parts: str) -> None:
        if not self.enabled:
            return
        path = self._path(key_parts)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"key": list(key_parts), "value": value}, sort_keys=True),
                encoding="utf-8",
            )
            tmp.replace(path)
        except OSError:
            # A cache write failure must never break the underlying operation.
            return

    def _path(self, key_parts: tuple[str, ...]) -> Path:
        digest = hashlib.sha256("\x1f".join(key_parts).encode("utf-8")).hexdigest()
        return self.base_dir / digest[:2] / f"{digest}.json"

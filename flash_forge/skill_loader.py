"""Phase-scoped Local-Model Skill Pack loader.

Local MoEs ignore 3,000-word frontier prose skills. FlashForge loads ONLY the
tight <250-token protocol card for the active phase + 1-2 dynamic verified
exemplars matching the task domain.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SKILL_DIR = Path(__file__).resolve().parent / "_skill"
CARDS_DIR = SKILL_DIR / "cards"
EXEMPLARS_PATH = SKILL_DIR / "exemplars.json"


class SkillPack:
    def __init__(self, skill_dir: Path | None = None) -> None:
        self.skill_dir = skill_dir or SKILL_DIR
        self.cards_dir = self.skill_dir / "cards"
        self.exemplars_path = self.skill_dir / "exemplars.json"
        self._cards_cache: dict[str, str] = {}
        self._exemplars: list[dict[str, Any]] = []
        self._load_exemplars()

    def _load_exemplars(self) -> None:
        if self.exemplars_path.exists():
            try:
                data = json.loads(self.exemplars_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._exemplars = data
            except Exception:
                self._exemplars = []

    def card(self, name: str) -> str:
        if name not in self._cards_cache:
            path = self.cards_dir / f"{name}.md"
            if not path.exists():
                raise FileNotFoundError(f"Missing skill micro-card: {path}")
            self._cards_cache[name] = path.read_text(encoding="utf-8").strip()
        return self._cards_cache[name]

    def match_exemplars(self, phase: str, query: str, limit: int = 1) -> str:
        """Select compact verified exemplars by keyword overlap for the current phase."""
        q_tokens = set(re.findall(r"[a-z0-9_]{3,}", query.lower()))
        scored: list[tuple[int, dict[str, Any]]] = []
        for ex in self._exemplars:
            if ex.get("phase") != phase:
                continue
            tags = set(t.lower() for t in ex.get("tags", []))
            score = len(q_tokens & tags)
            scored.append((score, ex))
        if not scored:
            return ""
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [ex for _, ex in scored[:limit]]
        blocks = []
        for ex in chosen:
            blocks.append(
                f"EXAMPLE ({ex.get('title', phase)}):\n"
                f"Input: {ex['input']}\n"
                f"Verified Approach:\n{ex['output']}"
            )
        return "\n\n".join(blocks)

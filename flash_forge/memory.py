"""Zero-weight Episodic Reflexion Memory backed by SQLite + FTS5.

Whenever FlashForge recovers from a Wave-1 failure or Red-Team kill, it distills
a compact causal lesson (symptom -> dead_end -> verified_rule) into SQLite.
Future runs retrieve the top-K relevant lessons by BM25 and inject them into the
shared cached prefix — improving reliability across runs without touching weights.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_DB_PATH = Path.home() / ".flashforge" / "lessons.db"


@dataclass
class Lesson:
    id: int
    domain: str
    symptom: str
    dead_end: str
    verified_rule: str
    created_at: float
    hits: int = 0

    def format_compact(self) -> str:
        return f"- [{self.domain}] Watch for: {self.symptom} | Avoid: {self.dead_end} | Rule: {self.verified_rule}"


class EpisodicMemory:
    def __init__(self, db_path: str | Path | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                symptom TEXT NOT NULL,
                dead_end TEXT NOT NULL,
                verified_rule TEXT NOT NULL,
                created_at REAL NOT NULL,
                hits INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS lessons_fts USING fts5(
                domain, symptom, dead_end, verified_rule,
                content='lessons', content_rowid='id'
            )
            """
        )
        self._conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS lessons_ai AFTER INSERT ON lessons BEGIN
                INSERT INTO lessons_fts(rowid, domain, symptom, dead_end, verified_rule)
                VALUES (new.id, new.domain, new.symptom, new.dead_end, new.verified_rule);
            END
            """
        )
        self._conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS lessons_ad AFTER DELETE ON lessons BEGIN
                INSERT INTO lessons_fts(lessons_fts, rowid, domain, symptom, dead_end, verified_rule)
                VALUES ('delete', old.id, old.domain, old.symptom, old.dead_end, old.verified_rule);
            END
            """
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def add(self, domain: str, symptom: str, dead_end: str, verified_rule: str) -> int:
        if not self.enabled or self._conn is None:
            return -1
        # Deduplicate near-identical verified rules
        cur = self._conn.execute(
            "SELECT id FROM lessons WHERE lower(verified_rule) = lower(?) LIMIT 1",
            (verified_rule.strip(),),
        )
        row = cur.fetchone()
        if row:
            return int(row[0])
        cur = self._conn.execute(
            "INSERT INTO lessons(domain, symptom, dead_end, verified_rule, created_at) VALUES (?, ?, ?, ?, ?)",
            (domain.strip()[:48], symptom.strip()[:220], dead_end.strip()[:220], verified_rule.strip()[:260], time.time()),
        )
        self._conn.commit()
        return int(cur.lastrowid or -1)

    def retrieve(self, query: str, domain: Optional[str] = None, limit: int = 3) -> list[Lesson]:
        if not self.enabled or self._conn is None:
            return []
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", query.lower())
        stop = {
            "the", "and", "for", "with", "that", "this", "from", "return", "write",
            "function", "python", "given", "find", "what", "how", "should", "must",
            "each", "when", "where", "which", "have", "into", "your", "there", "their",
        }
        keywords = []
        seen = set()
        for t in tokens:
            if t not in stop and t not in seen:
                seen.add(t)
                keywords.append(t)
            if len(keywords) >= 12:
                break
        if not keywords:
            return []
        fts_query = " OR ".join(f'"{k}"' for k in keywords)
        try:
            if domain:
                sql = (
                    "SELECT l.id, l.domain, l.symptom, l.dead_end, l.verified_rule, l.created_at, l.hits "
                    "FROM lessons_fts f JOIN lessons l ON f.rowid = l.id "
                    "WHERE lessons_fts MATCH ? AND l.domain = ? "
                    "ORDER BY bm25(lessons_fts) LIMIT ?"
                )
                rows = self._conn.execute(sql, (fts_query, domain, limit)).fetchall()
            else:
                sql = (
                    "SELECT l.id, l.domain, l.symptom, l.dead_end, l.verified_rule, l.created_at, l.hits "
                    "FROM lessons_fts f JOIN lessons l ON f.rowid = l.id "
                    "WHERE lessons_fts MATCH ? "
                    "ORDER BY bm25(lessons_fts) LIMIT ?"
                )
                rows = self._conn.execute(sql, (fts_query, limit)).fetchall()
        except sqlite3.OperationalError:
            return []

        lessons = [Lesson(*r) for r in rows]
        if lessons:
            self._conn.executemany("UPDATE lessons SET hits = hits + 1 WHERE id = ?", [(l.id,) for l in lessons])
            self._conn.commit()
        return lessons

    def count(self) -> int:
        if not self.enabled or self._conn is None:
            return 0
        row = self._conn.execute("SELECT COUNT(*) FROM lessons").fetchone()
        return int(row[0]) if row else 0

    def list_recent(self, limit: int = 20) -> list[Lesson]:
        if not self.enabled or self._conn is None:
            return []
        rows = self._conn.execute(
            "SELECT id, domain, symptom, dead_end, verified_rule, created_at, hits "
            "FROM lessons ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [Lesson(*r) for r in rows]

    def format_prompt_block(self, query: str, domain: Optional[str] = None, limit: int = 3) -> str:
        lessons = self.retrieve(query, domain=domain, limit=limit)
        if not lessons:
            return ""
        lines = ["VERIFIED LESSONS FROM PAST RUNS (apply only if directly relevant):"]
        lines.extend(l.format_compact() for l in lessons)
        return "\n".join(lines)

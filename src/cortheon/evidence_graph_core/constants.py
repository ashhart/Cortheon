"""Tokenization and redaction constants for the evidence graph."""

from __future__ import annotations

import re

TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.:/+-]{2,}")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?i)\b(api[_ -]?key|access[_ -]?token|password|passwd|secret|credential)"
        r"(\s*[:=]\s*[\"']?)[^\s,\"']{8,}"
    ),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~-]{12,}"),
)
STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "among",
        "and",
        "are",
        "because",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "can",
        "could",
        "does",
        "each",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "into",
        "its",
        "may",
        "more",
        "most",
        "not",
        "only",
        "other",
        "our",
        "should",
        "than",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "under",
        "using",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

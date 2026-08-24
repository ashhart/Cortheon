"""Safe document normalization, chunking, and term extraction."""

from __future__ import annotations

import re
from typing import Any


def clean_document(
    text: str,
    *,
    secret_patterns: tuple[re.Pattern[str], ...],
    scanner: Any,
) -> tuple[str, list[str]]:
    redacted = text[:2_000_000].replace("\r\n", "\n").replace("\r", "\n")
    for pattern in secret_patterns:
        redacted = pattern.sub("[REDACTED SECRET]", redacted)
    scan = scanner(redacted, preserve_layout=True)
    cleaned = "\n".join(line.rstrip() for line in scan.clean_text.splitlines()).strip()
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
    return cleaned, list(dict.fromkeys(scan.flags))


def chunk_document(
    text: str,
    *,
    target_chars: int = 900,
    overlap_chars: int = 140,
    max_chunks: int = 250,
    sentence_boundary: re.Pattern[str],
) -> list[tuple[str, int, int]]:
    sentences = sentence_boundary.split(text)
    chunks: list[tuple[str, int, int]] = []
    buffer = ""
    buffer_start = 0
    cursor = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        start = text.find(sentence, cursor)
        if start < 0:
            start = cursor
        cursor = start + len(sentence)
        if not buffer:
            buffer = sentence
            buffer_start = start
        elif len(buffer) + 1 + len(sentence) <= target_chars:
            buffer += " " + sentence
        else:
            chunks.append((buffer, buffer_start, buffer_start + len(buffer)))
            if len(chunks) >= max_chunks:
                break
            overlap = buffer[-overlap_chars:].lstrip()
            buffer = f"{overlap} {sentence}".strip()
            buffer_start = max(0, start - len(overlap) - 1)
    if buffer and len(chunks) < max_chunks:
        chunks.append((buffer, buffer_start, min(len(text), buffer_start + len(buffer))))
    return chunks


def content_terms(
    text: str,
    *,
    token_pattern: re.Pattern[str],
    stopwords: frozenset[str],
) -> set[str]:
    terms: set[str] = set()
    for token in token_pattern.findall(text):
        normalized = token.casefold().strip("._:/+-")
        candidates = [normalized, *re.split(r"[^a-z0-9]+", normalized)]
        for candidate in candidates:
            if len(candidate) >= 3 and candidate not in stopwords:
                terms.add(candidate)
    return terms

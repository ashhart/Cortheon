"""Scholarly XML namespaces and ranking constants."""

from __future__ import annotations

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
MAX_SCHOLARLY_XML_BYTES = 5_000_000
RECENCY_STEPS: tuple[tuple[int, float], ...] = (
    (180, 0.98),
    (365, 0.9),
    (730, 0.75),
    (1095, 0.6),
    (1825, 0.4),
)
RECENCY_FLOOR = 0.25
UNDATED_RECENCY = 0.5

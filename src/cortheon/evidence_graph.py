"""Persistent local evidence graph for cross-document retrieval and joins.

The public API stays here. Focused internal modules own ingestion, storage,
retrieval, ranking, and text processing.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cortheon.embeddings import EmbeddingProvider, cosine_similarity
from cortheon.evidence_graph_core import ingestion, ranking, retrieval, storage
from cortheon.evidence_graph_core import text as text_core
from cortheon.evidence_graph_core.constants import (
    SECRET_PATTERNS as _SECRET_PATTERNS,
)
from cortheon.evidence_graph_core.constants import (
    SENTENCE_BOUNDARY as _SENTENCE_BOUNDARY,
)
from cortheon.evidence_graph_core.constants import STOPWORDS as _STOPWORDS
from cortheon.evidence_graph_core.constants import TOKEN as _TOKEN
from cortheon.sanitize import scan_text


class EvidenceGraph:
    """SQLite-backed document/chunk store with deterministic cross-source joins."""

    def __init__(
        self,
        path: str | Path,
        *,
        namespace: str = "default",
        max_documents: int = 1_000,
        embedder: EmbeddingProvider | None = None,
        vector_weight: float = 0.35,
    ) -> None:
        if not 0.0 <= vector_weight <= 1.0:
            raise ValueError("vector_weight must be between 0 and 1")
        self.path = Path(path).expanduser().resolve()
        self.namespace = re.sub(r"[^a-zA-Z0-9_.-]+", "-", namespace)[:80] or "default"
        self.max_documents = max(10, max_documents)
        self.embedder = embedder
        self.vector_weight = vector_weight
        self._ensure()

    def ingest(
        self,
        *,
        uri: str,
        text: str,
        title: str | None = None,
        source_type: str = "document",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return ingestion.ingest(
            self,
            uri=uri,
            text=text,
            title=title,
            source_type=source_type,
            metadata=metadata,
            clean_document=_clean_document,
            chunk_document=_chunk_document,
            content_terms=_content_terms,
            sha256=hashlib.sha256,
            dumps=json.dumps,
            now=lambda: datetime.now(UTC).replace(microsecond=0).isoformat(),
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return retrieval.search(
            self,
            query,
            limit=limit,
            document_ids=document_ids,
            content_terms=_content_terms,
            rank_chunks=_rank_chunks,
            public_chunk=_public_chunk,
        )

    def join(
        self,
        question: str,
        *,
        max_paths: int = 4,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return retrieval.join(
            self,
            question,
            max_paths=max_paths,
            document_ids=document_ids,
            content_terms=_content_terms,
            rank_chunks=_rank_chunks,
            public_chunk=_public_chunk,
            join_reason=_join_reason,
        )

    def stats(self) -> dict[str, Any]:
        return storage.stats(self)

    def list_documents(self, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        return storage.list_documents(self, limit=limit, offset=offset)

    def delete_document(self, document_id: str) -> bool:
        return storage.delete_document(self, document_id, fullmatch=re.fullmatch)

    def _load_chunks(self, document_ids: list[str] | None) -> list[dict[str, Any]]:
        return storage.load_chunks(self, document_ids, loads=json.loads)

    def _query_embedding(self, query: str) -> tuple[list[float] | None, str | None]:
        return retrieval.query_embedding(self, query)

    def _embedding_status(self, error: str | None) -> dict[str, Any]:
        return retrieval.embedding_status(self, error)

    def _retrieval_status(
        self,
        query_embedding: list[float] | None,
        error: str | None,
    ) -> dict[str, Any]:
        return retrieval.retrieval_status(self, query_embedding, error)

    def _ensure(self) -> None:
        storage.ensure(self)

    def _connect(self) -> sqlite3.Connection:
        return storage.connect(self.path, sqlite_module=sqlite3)

    def _prune(self, connection: sqlite3.Connection) -> None:
        storage.prune(self, connection)


def _clean_document(text: str) -> tuple[str, list[str]]:
    return text_core.clean_document(
        text,
        secret_patterns=_SECRET_PATTERNS,
        scanner=scan_text,
    )


def _chunk_document(
    text: str,
    *,
    target_chars: int = 900,
    overlap_chars: int = 140,
    max_chunks: int = 250,
) -> list[tuple[str, int, int]]:
    return text_core.chunk_document(
        text,
        target_chars=target_chars,
        overlap_chars=overlap_chars,
        max_chunks=max_chunks,
        sentence_boundary=_SENTENCE_BOUNDARY,
    )


def _content_terms(text: str) -> set[str]:
    return text_core.content_terms(
        text,
        token_pattern=_TOKEN,
        stopwords=_STOPWORDS,
    )


def _rank_chunks(
    query: str,
    query_terms: set[str],
    rows: list[dict[str, Any]],
    *,
    query_embedding: list[float] | None = None,
    vector_weight: float = 0.35,
) -> list[dict[str, Any]]:
    return ranking.rank_chunks(
        query,
        query_terms,
        rows,
        query_embedding=query_embedding,
        vector_weight=vector_weight,
        retrieval_terms=_retrieval_terms,
        cosine=cosine_similarity,
        logarithm=math.log,
    )


def _retrieval_terms(row: dict[str, Any]) -> list[str]:
    """Include source identity in retrieval while keeping content-only bridges."""
    return ranking.retrieval_terms(row, content_terms=_content_terms)


def _public_chunk(item: dict[str, Any]) -> dict[str, Any]:
    return ranking.public_chunk(item)


def _join_reason(bridges: list[str], complementary: int) -> str:
    return ranking.join_reason(bridges, complementary)

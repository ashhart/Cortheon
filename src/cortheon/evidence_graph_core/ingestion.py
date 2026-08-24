"""Document ingestion and optional embedding persistence."""

from __future__ import annotations

from typing import Any


def ingest(
    graph: Any,
    *,
    uri: str,
    text: str,
    title: str | None = None,
    source_type: str = "document",
    metadata: dict[str, Any] | None = None,
    clean_document: Any,
    chunk_document: Any,
    content_terms: Any,
    sha256: Any,
    dumps: Any,
    now: Any,
) -> dict[str, Any]:
    clean_text, quarantine_flags = clean_document(text)
    if not clean_text:
        raise ValueError("document has no safe text to ingest")
    uri = uri.strip()[:2_000]
    if not uri:
        raise ValueError("document URI must not be empty")
    content_hash = sha256(clean_text.encode("utf-8")).hexdigest()
    document_id = "doc_" + sha256(f"{graph.namespace}\0{uri}".encode()).hexdigest()[:20]
    chunks = chunk_document(clean_text)
    with graph._connect() as connection:
        existing = connection.execute(
            "SELECT content_hash FROM documents WHERE id = ? AND namespace = ?",
            (document_id, graph.namespace),
        ).fetchone()
        embedded_count = (
            connection.execute(
                """
                SELECT COUNT(*) FROM chunks
                WHERE document_id = ? AND embedding_model = ?
                """,
                (document_id, graph.embedder.model),
            ).fetchone()[0]
            if existing and graph.embedder is not None
            else 0
        )
        if (
            existing
            and existing[0] == content_hash
            and (graph.embedder is None or embedded_count == len(chunks))
        ):
            count = connection.execute(
                "SELECT COUNT(*) FROM chunks WHERE document_id = ?",
                (document_id,),
            ).fetchone()[0]
            return {
                "document_id": document_id,
                "uri": uri,
                "chunks": count,
                "changed": False,
                "quarantine_flags": quarantine_flags,
                "embedding": graph._embedding_status(None),
            }
    embeddings: list[list[float] | None] = [None] * len(chunks)
    embedding_error: str | None = None
    if graph.embedder is not None and chunks:
        try:
            embedded = graph.embedder.embed([item[0] for item in chunks])
            if len(embedded) != len(chunks):
                raise ValueError("embedding provider returned the wrong vector count")
            embeddings = [vector for vector in embedded]  # noqa: C416 - widens item type
        except Exception as exc:
            embedding_error = f"{type(exc).__name__}: {exc}"[:500]
    timestamp = now()

    with graph._connect() as connection:
        connection.execute(
            """
            INSERT INTO documents(
                id, namespace, uri, title, source_type, content_hash,
                metadata_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                uri = excluded.uri,
                title = excluded.title,
                source_type = excluded.source_type,
                content_hash = excluded.content_hash,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                document_id,
                graph.namespace,
                uri,
                (title or "")[:500] or None,
                source_type[:100],
                content_hash,
                dumps(metadata or {}, sort_keys=True, default=str),
                timestamp,
            ),
        )
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        for ordinal, (chunk_text, start, end) in enumerate(chunks):
            chunk_id = f"{document_id}_c{ordinal:04d}"
            connection.execute(
                """
                INSERT INTO chunks(
                    id, document_id, ordinal, char_start, char_end,
                    text, terms_json, embedding_json, embedding_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    document_id,
                    ordinal,
                    start,
                    end,
                    chunk_text,
                    dumps(sorted(content_terms(chunk_text))),
                    (
                        dumps(embeddings[ordinal], separators=(",", ":"))
                        if embeddings[ordinal] is not None
                        else None
                    ),
                    (
                        graph.embedder.model
                        if embeddings[ordinal] is not None and graph.embedder is not None
                        else None
                    ),
                ),
            )
        graph._prune(connection)

    return {
        "document_id": document_id,
        "uri": uri,
        "chunks": len(chunks),
        "changed": True,
        "quarantine_flags": quarantine_flags,
        "embedding": graph._embedding_status(embedding_error),
    }

"""SQLite schema, queries, namespace isolation, and retention."""

from __future__ import annotations

from typing import Any


def load_chunks(
    graph: Any,
    document_ids: list[str] | None,
    *,
    loads: Any,
) -> list[dict[str, Any]]:
    parameters: list[Any] = [graph.namespace]
    where = "documents.namespace = ?"
    selected = [item for item in (document_ids or []) if item][:50]
    if selected:
        placeholders = ",".join("?" for _ in selected)
        where += f" AND documents.id IN ({placeholders})"
        parameters.extend(selected)
    with graph._connect() as connection:
        records = connection.execute(
            f"""
            SELECT
                chunks.id, chunks.document_id, chunks.ordinal,
                chunks.char_start, chunks.char_end, chunks.text,
                chunks.terms_json, documents.uri, documents.title,
                documents.source_type, documents.updated_at,
                chunks.embedding_json, chunks.embedding_model
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            WHERE {where}
            ORDER BY documents.updated_at DESC, chunks.ordinal ASC
            LIMIT 10000
            """,
            parameters,
        ).fetchall()
    return [
        {
            "chunk_id": item[0],
            "document_id": item[1],
            "ordinal": item[2],
            "char_start": item[3],
            "char_end": item[4],
            "text": item[5],
            "_terms": loads(item[6]),
            "uri": item[7],
            "title": item[8],
            "source_type": item[9],
            "updated_at": item[10],
            "_embedding": loads(item[11]) if item[11] is not None else None,
            "_embedding_model": item[12],
        }
        for item in records
    ]


def stats(graph: Any) -> dict[str, Any]:
    with graph._connect() as connection:
        documents = connection.execute(
            "SELECT COUNT(*) FROM documents WHERE namespace = ?",
            (graph.namespace,),
        ).fetchone()[0]
        chunks = connection.execute(
            """
            SELECT COUNT(*) FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            WHERE documents.namespace = ?
            """,
            (graph.namespace,),
        ).fetchone()[0]
        embedded_chunks = connection.execute(
            """
            SELECT COUNT(*) FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            WHERE documents.namespace = ?
              AND chunks.embedding_json IS NOT NULL
            """,
            (graph.namespace,),
        ).fetchone()[0]
    return {
        "namespace": graph.namespace,
        "documents": documents,
        "chunks": chunks,
        "embedded_chunks": embedded_chunks,
        "embedding_model": graph.embedder.model if graph.embedder else None,
        "retrieval_mode": "hybrid" if graph.embedder else "lexical",
    }


def list_documents(graph: Any, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    bounded_limit = max(1, min(limit, 500))
    bounded_offset = max(0, offset)
    with graph._connect() as connection:
        records = connection.execute(
            """
            SELECT documents.id, documents.uri, documents.title,
                   documents.source_type, documents.updated_at,
                   COUNT(chunks.id)
            FROM documents
            LEFT JOIN chunks ON chunks.document_id = documents.id
            WHERE documents.namespace = ?
            GROUP BY documents.id
            ORDER BY documents.updated_at DESC, documents.id
            LIMIT ? OFFSET ?
            """,
            (graph.namespace, bounded_limit, bounded_offset),
        ).fetchall()
    return {
        "documents": [
            {
                "document_id": item[0],
                "uri": item[1],
                "title": item[2],
                "source_type": item[3],
                "updated_at": item[4],
                "chunks": item[5],
            }
            for item in records
        ],
        "limit": bounded_limit,
        "offset": bounded_offset,
    }


def delete_document(graph: Any, document_id: str, *, fullmatch: Any) -> bool:
    if not fullmatch(r"doc_[a-f0-9]{20}", document_id):
        raise ValueError("invalid document id")
    with graph._connect() as connection:
        cursor = connection.execute(
            "DELETE FROM documents WHERE id = ? AND namespace = ?",
            (document_id, graph.namespace),
        )
    return cursor.rowcount > 0


def ensure(graph: Any) -> None:
    graph.path.parent.mkdir(parents=True, exist_ok=True)
    with graph._connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents(
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                uri TEXT NOT NULL,
                title TEXT,
                source_type TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS documents_namespace_updated
                ON documents(namespace, updated_at DESC);
            CREATE TABLE IF NOT EXISTS chunks(
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                char_start INTEGER NOT NULL,
                char_end INTEGER NOT NULL,
                text TEXT NOT NULL,
                terms_json TEXT NOT NULL,
                embedding_json TEXT,
                embedding_model TEXT
            );
            CREATE INDEX IF NOT EXISTS chunks_document
                ON chunks(document_id, ordinal);
            """
        )
        chunk_columns = {
            str(item[1]) for item in connection.execute("PRAGMA table_info(chunks)").fetchall()
        }
        if "embedding_json" not in chunk_columns:
            connection.execute("ALTER TABLE chunks ADD COLUMN embedding_json TEXT")
        if "embedding_model" not in chunk_columns:
            connection.execute("ALTER TABLE chunks ADD COLUMN embedding_model TEXT")


def connect(path: Any, *, sqlite_module: Any) -> Any:
    connection = sqlite_module.connect(path, timeout=10)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


def prune(graph: Any, connection: Any) -> None:
    stale = connection.execute(
        """
        SELECT id FROM documents
        WHERE namespace = ?
        ORDER BY updated_at DESC
        LIMIT -1 OFFSET ?
        """,
        (graph.namespace, graph.max_documents),
    ).fetchall()
    if stale:
        connection.executemany(
            "DELETE FROM documents WHERE id = ?",
            [(item[0],) for item in stale],
        )

"""Search, cross-document joins, and embedding fallback status."""

from __future__ import annotations

from typing import Any


def search(
    graph: Any,
    query: str,
    *,
    limit: int = 8,
    document_ids: list[str] | None = None,
    content_terms: Any,
    rank_chunks: Any,
    public_chunk: Any,
) -> dict[str, Any]:
    query_terms = content_terms(query)
    if not query_terms:
        raise ValueError("document search query has no content terms")
    rows = graph._load_chunks(document_ids)
    query_embedding, embedding_error = graph._query_embedding(query)
    ranked = rank_chunks(
        query,
        query_terms,
        rows,
        query_embedding=query_embedding,
        vector_weight=graph.vector_weight,
    )
    matches = [public_chunk(item) for item in ranked[: max(1, min(limit, 20))]]
    return {
        "query": query,
        "matches": matches,
        "documents_considered": len({item["document_id"] for item in rows}),
        "chunks_considered": len(rows),
        "retrieval": graph._retrieval_status(query_embedding, embedding_error),
    }


def join(
    graph: Any,
    question: str,
    *,
    max_paths: int = 4,
    document_ids: list[str] | None = None,
    content_terms: Any,
    rank_chunks: Any,
    public_chunk: Any,
    join_reason: Any,
) -> dict[str, Any]:
    query_terms = content_terms(question)
    if not query_terms:
        raise ValueError("document join question has no content terms")
    rows = graph._load_chunks(document_ids)
    query_embedding, embedding_error = graph._query_embedding(question)
    ranked = rank_chunks(
        question,
        query_terms,
        rows,
        query_embedding=query_embedding,
        vector_weight=graph.vector_weight,
    )[:24]
    paths: list[dict[str, Any]] = []
    for left_index, left in enumerate(ranked):
        for right in ranked[left_index + 1 :]:
            if left["document_id"] == right["document_id"]:
                continue
            left_terms = set(left["_terms"])
            right_terms = set(right["_terms"])
            bridges = sorted(
                (left_terms & right_terms) - query_terms,
                key=lambda term: (-len(term), term),
            )[:8]
            left_query = left_terms & query_terms
            right_query = right_terms & query_terms
            complementary = len(left_query | right_query) - max(len(left_query), len(right_query))
            path_score = (
                float(left["score"])
                + float(right["score"])
                + min(3.0, len(bridges) * 0.55)
                + complementary * 0.45
            )
            paths.append(
                {
                    "path_id": "",
                    "score": round(path_score, 4),
                    "bridge_terms": bridges,
                    "query_coverage": sorted(left_query | right_query),
                    "left": public_chunk(left),
                    "right": public_chunk(right),
                    "reason": join_reason(bridges, complementary),
                }
            )
    paths.sort(key=lambda item: item["score"], reverse=True)
    deduped: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for item in paths:
        pair = tuple(sorted((item["left"]["document_id"], item["right"]["document_id"])))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        item["path_id"] = f"join_{len(deduped)}"
        deduped.append(item)
        if len(deduped) >= max(1, min(max_paths, 8)):
            break
    return {
        "question": question,
        "paths": deduped,
        "documents_considered": len({item["document_id"] for item in rows}),
        "chunks_considered": len(rows),
        "status": "linked" if deduped else "insufficient_cross_document_evidence",
        "gap": None if deduped else "No relevant chunks from two different documents were found.",
        "retrieval": graph._retrieval_status(query_embedding, embedding_error),
    }


def query_embedding(graph: Any, query: str) -> tuple[list[float] | None, str | None]:
    if graph.embedder is None:
        return None, None
    try:
        vectors = graph.embedder.embed([query])
        if len(vectors) != 1:
            raise ValueError("embedding provider returned the wrong vector count")
        return vectors[0], None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:500]


def embedding_status(graph: Any, error: str | None) -> dict[str, Any]:
    return {
        "model": graph.embedder.model if graph.embedder else None,
        "mode": "hybrid" if graph.embedder else "lexical",
        "error": error,
    }


def retrieval_status(
    graph: Any,
    query_embedding: list[float] | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "mode": "hybrid" if query_embedding is not None else "lexical",
        "embedding_model": graph.embedder.model if graph.embedder else None,
        "vector_weight": graph.vector_weight if query_embedding is not None else 0.0,
        "embedding_error": error,
    }

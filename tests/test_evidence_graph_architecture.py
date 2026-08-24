"""Architecture contract for the repository-only evidence graph."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import get_type_hints

import pytest

import cortheon.evidence_graph as facade

ROOT = Path(__file__).parents[1]
FACADE = ROOT / "src" / "cortheon" / "evidence_graph.py"
CORE = ROOT / "src" / "cortheon" / "evidence_graph_core"

CLASS_SIGNATURE = "(path: 'str | Path', *, namespace: 'str' = 'default', max_documents: 'int' = 1000, embedder: 'EmbeddingProvider | None' = None, vector_weight: 'float' = 0.35) -> 'None'"
METHOD_SIGNATURES = {
    "ingest": "(self, *, uri: 'str', text: 'str', title: 'str | None' = None, source_type: 'str' = 'document', metadata: 'dict[str, Any] | None' = None) -> 'dict[str, Any]'",
    "search": "(self, query: 'str', *, limit: 'int' = 8, document_ids: 'list[str] | None' = None) -> 'dict[str, Any]'",
    "join": "(self, question: 'str', *, max_paths: 'int' = 4, document_ids: 'list[str] | None' = None) -> 'dict[str, Any]'",
    "stats": "(self) -> 'dict[str, Any]'",
    "list_documents": "(self, *, limit: 'int' = 100, offset: 'int' = 0) -> 'dict[str, Any]'",
    "delete_document": "(self, document_id: 'str') -> 'bool'",
    "_load_chunks": "(self, document_ids: 'list[str] | None') -> 'list[dict[str, Any]]'",
    "_query_embedding": "(self, query: 'str') -> 'tuple[list[float] | None, str | None]'",
    "_embedding_status": "(self, error: 'str | None') -> 'dict[str, Any]'",
    "_retrieval_status": "(self, query_embedding: 'list[float] | None', error: 'str | None') -> 'dict[str, Any]'",
    "_ensure": "(self) -> 'None'",
    "_connect": "(self) -> 'sqlite3.Connection'",
    "_prune": "(self, connection: 'sqlite3.Connection') -> 'None'",
}
FUNCTION_SIGNATURES = {
    "_clean_document": "(text: 'str') -> 'tuple[str, list[str]]'",
    "_chunk_document": "(text: 'str', *, target_chars: 'int' = 900, overlap_chars: 'int' = 140, max_chunks: 'int' = 250) -> 'list[tuple[str, int, int]]'",
    "_content_terms": "(text: 'str') -> 'set[str]'",
    "_rank_chunks": "(query: 'str', query_terms: 'set[str]', rows: 'list[dict[str, Any]]', *, query_embedding: 'list[float] | None' = None, vector_weight: 'float' = 0.35) -> 'list[dict[str, Any]]'",
    "_retrieval_terms": "(row: 'dict[str, Any]') -> 'list[str]'",
    "_public_chunk": "(item: 'dict[str, Any]') -> 'dict[str, Any]'",
    "_join_reason": "(bridges: 'list[str]', complementary: 'int') -> 'str'",
}
EXPECTED_SOURCE_CONSUMERS: set[str] = set()


def _imports(path: Path, module: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            return True
        if isinstance(node, ast.Import) and any(alias.name == module for alias in node.names):
            return True
    return False


def test_facade_and_core_stay_below_file_limit() -> None:
    paths = [FACADE, *sorted(CORE.glob("*.py"))]
    assert paths
    for path in paths:
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 500, path


def test_core_imports_are_acyclic_and_do_not_reenter_facade() -> None:
    for path in CORE.glob("*.py"):
        assert not _imports(path, "cortheon.evidence_graph"), path
    assert _imports(FACADE, "cortheon.evidence_graph_core")


def test_source_consumers_are_explicit() -> None:
    consumers = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*.py")
        if path != FACADE and _imports(path, "cortheon.evidence_graph")
    }
    assert consumers == EXPECTED_SOURCE_CONSUMERS


def test_evidence_graph_is_not_in_the_lean_distribution_allowlist() -> None:
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    manifest_text = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert '"evidence_graph"' not in setup_text
    assert "src/cortheon/evidence_graph.py" not in manifest_text


def test_public_identity_signatures_and_type_hints_are_stable() -> None:
    assert facade.EvidenceGraph.__module__ == "cortheon.evidence_graph"
    assert str(inspect.signature(facade.EvidenceGraph)) == CLASS_SIGNATURE
    for name, expected in METHOD_SIGNATURES.items():
        method = getattr(facade.EvidenceGraph, name)
        assert method.__module__ == "cortheon.evidence_graph"
        assert str(inspect.signature(method)) == expected
        assert get_type_hints(method)
    for name, expected in FUNCTION_SIGNATURES.items():
        function = getattr(facade, name)
        assert function.__module__ == "cortheon.evidence_graph"
        assert str(inspect.signature(function)) == expected
        assert get_type_hints(function)


def test_search_keeps_facade_helper_monkeypatch_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = facade.EvidenceGraph(tmp_path / "graph.sqlite3")
    marker = RuntimeError("patched content terms")

    def patched_terms(_text: str) -> set[str]:
        raise marker

    monkeypatch.setattr(facade, "_content_terms", patched_terms)
    with pytest.raises(RuntimeError) as raised:
        graph.search("query")
    assert raised.value is marker

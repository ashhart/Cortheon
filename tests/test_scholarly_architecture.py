"""Architecture contract for the repository-only scholarly research stack."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import get_type_hints

import pytest

import cortheon.scholarly as facade

ROOT = Path(__file__).parents[1]
FACADE = ROOT / "src" / "cortheon" / "scholarly.py"
CORE = ROOT / "src" / "cortheon" / "scholarly_core"
EXPECTED_CONSUMERS = {
    "src/cortheon/benchmark_core/fixtures_research.py",
}
CLASS_SIGNATURES = {
    "ScholarlyDiscoveryResult": "(works: 'list[ScholarlyWork]', evidence: 'list[Evidence]', errors: 'list[str]') -> None",
    "ScholarlyConnector": "()",
    "CompositeScholarlyDiscovery": "(connectors: 'list[ScholarlyConnector] | None' = None) -> 'None'",
    "ArxivConnector": "(client: 'JsonHttpClient | None' = None) -> 'None'",
    "OpenAlexConnector": "(client: 'JsonHttpClient | None' = None) -> 'None'",
    "PubMedConnector": "(client: 'JsonHttpClient | None' = None) -> 'None'",
}
METHOD_SIGNATURES = {
    (
        "ScholarlyConnector",
        "search",
    ): "(self, query: 'str', limit: 'int') -> 'ScholarlyDiscoveryResult'",
    ("ScholarlyConnector", "source_profile"): "(self) -> 'dict[str, object]'",
    ("CompositeScholarlyDiscovery", "source_profiles"): "(self) -> 'list[dict[str, object]]'",
    (
        "CompositeScholarlyDiscovery",
        "search",
    ): "(self, query: 'str', limit: 'int', connector_names: 'list[str] | None' = None) -> 'ScholarlyDiscoveryResult'",
    (
        "CompositeScholarlyDiscovery",
        "_selected_connectors",
    ): "(self, connector_names: 'list[str] | None') -> 'list[ScholarlyConnector]'",
    (
        "ArxivConnector",
        "search",
    ): "(self, query: 'str', limit: 'int') -> 'ScholarlyDiscoveryResult'",
    (
        "OpenAlexConnector",
        "search",
    ): "(self, query: 'str', limit: 'int') -> 'ScholarlyDiscoveryResult'",
    (
        "PubMedConnector",
        "search",
    ): "(self, query: 'str', limit: 'int') -> 'ScholarlyDiscoveryResult'",
    (
        "PubMedConnector",
        "_search_ids",
    ): "(self, query: 'str', limit: 'int') -> 'tuple[list[str], str]'",
}
FUNCTION_SIGNATURES = {
    "pubmed_esearch_url": "(query: 'str', limit: 'int') -> 'str'",
    "pubmed_efetch_url": "(ids: 'list[str]') -> 'str'",
    "pubmed_base_params": "() -> 'dict[str, str]'",
    "parse_pubmed_articles": "(body: 'bytes', limit: 'int') -> 'list[ScholarlyWork]'",
    "bounded_xml_root": "(body: 'bytes', *, max_bytes: 'int' = 5000000) -> 'ET.Element'",
    "pubmed_abstract": "(article: 'ET.Element') -> 'str | None'",
    "pubmed_authors": "(article: 'ET.Element') -> 'list[str]'",
    "pubmed_article_id": "(article: 'ET.Element', id_type: 'str') -> 'str | None'",
    "pubmed_published_at": "(article: 'ET.Element') -> 'datetime | None'",
    "date_from_parts": "(year_value: 'str | None', month_value: 'str | None', day_value: 'str | None') -> 'datetime | None'",
    "parse_month": "(value: 'str | None') -> 'int'",
    "element_text": "(element: 'ET.Element | None') -> 'str'",
    "abstract_from_inverted_index": "(value: 'Any') -> 'str | None'",
    "scholarly_query_variants": "(query: 'str') -> 'list[str]'",
    "arxiv_query": "(query: 'str') -> 'str'",
    "key_phrases": "(query: 'str') -> 'list[str]'",
    "dedupe_works": "(works: 'list[ScholarlyWork]') -> 'list[ScholarlyWork]'",
    "score_work_relevance": "(work: 'ScholarlyWork', query: 'str') -> 'ScholarlyWork'",
    "score_work_recency": "(work: 'ScholarlyWork', now: 'datetime | None' = None) -> 'ScholarlyWork'",
    "work_recency_score": "(published_at: 'datetime | None', now: 'datetime | None' = None) -> 'float'",
    "scholarly_rank_key": "(work: 'ScholarlyWork') -> 'float'",
    "query_terms": "(query: 'str') -> 'list[str]'",
    "minimum_relevance": "(query: 'str') -> 'float'",
    "clean_text": "(value: 'str') -> 'str'",
    "normalize_title": "(value: 'str') -> 'str'",
    "normalize_for_match": "(value: 'str') -> 'str'",
    "arxiv_id_from_url": "(url: 'str') -> 'str'",
}


def _imports(path: Path, module: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        (isinstance(node, ast.ImportFrom) and node.module == module)
        or (isinstance(node, ast.Import) and any(alias.name == module for alias in node.names))
        for node in ast.walk(tree)
    )


def test_facade_and_core_stay_below_file_limit() -> None:
    for path in [FACADE, *sorted(CORE.glob("*.py"))]:
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 500, path


def test_core_does_not_reenter_the_facade() -> None:
    assert _imports(FACADE, "cortheon.scholarly_core")
    for path in CORE.glob("*.py"):
        assert not _imports(path, "cortheon.scholarly"), path


def test_direct_consumers_are_explicit() -> None:
    consumers = {
        path.relative_to(ROOT).as_posix()
        for directory in (ROOT / "src", ROOT / "benchmarks")
        for path in directory.rglob("*.py")
        if path != FACADE and _imports(path, "cortheon.scholarly")
    }
    assert consumers == EXPECTED_CONSUMERS


def test_scholarly_stack_is_not_in_the_lean_distribution() -> None:
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    manifest_text = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert '"scholarly"' not in setup_text
    assert "src/cortheon/scholarly.py" not in manifest_text


def test_public_identity_signatures_and_type_hints_are_stable() -> None:
    for name, expected in CLASS_SIGNATURES.items():
        public = getattr(facade, name)
        assert public.__module__ == "cortheon.scholarly"
        assert str(inspect.signature(public)) == expected
    assert get_type_hints(facade.ScholarlyDiscoveryResult)
    for (class_name, method_name), expected in METHOD_SIGNATURES.items():
        method = getattr(getattr(facade, class_name), method_name)
        assert method.__module__ == "cortheon.scholarly"
        assert str(inspect.signature(method)) == expected
        assert get_type_hints(method)
    for name, expected in FUNCTION_SIGNATURES.items():
        function = getattr(facade, name)
        assert function.__module__ == "cortheon.scholarly"
        assert str(inspect.signature(function)) == expected
        assert get_type_hints(function)


def test_composite_keeps_query_variant_monkeypatch_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = RuntimeError("patched query variants")

    def patched_variants(_query: str) -> list[str]:
        raise marker

    monkeypatch.setattr(facade, "scholarly_query_variants", patched_variants)
    discovery = facade.CompositeScholarlyDiscovery(connectors=[facade.ScholarlyConnector()])
    with pytest.raises(RuntimeError) as raised:
        discovery.search("query", 1)
    assert raised.value is marker

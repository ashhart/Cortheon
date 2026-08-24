"""Graph-based semantic join analysis."""

from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import product
from typing import Any

from cortheon.cognitive_core.models import Observation, SemanticEdge, SemanticRule
from cortheon.cognitive_core.plan_joins import _ordered_plan_analysis
from cortheon.cognitive_core.semantic_graph import (
    _semantic_edges,
    _semantic_key,
    _semantic_rules,
    _semantic_table_edges,
)


def _semantic_join_analysis(
    goal: str,
    paths: list[str],
    observations: Iterable[Observation],
    *,
    require_all_documents: bool = True,
) -> dict[str, Any] | None:
    observations = list(observations)
    ordered = _ordered_plan_analysis(
        goal,
        paths,
        observations,
        require_all_documents=require_all_documents,
    )
    if ordered is not None:
        return ordered
    requested = {path.casefold(): path for path in paths}
    edges: list[SemanticEdge] = []
    rules: list[SemanticRule] = []
    seen_edges: set[tuple[str, str, str, str, int]] = set()
    for observation in observations:
        receipt = observation.host_receipt
        if receipt is None or receipt.get("tool") not in {"read", "grep"}:
            continue
        arguments = receipt.get("args", {})
        if not isinstance(arguments, dict):
            continue
        document = str(arguments.get("filePath") or arguments.get("path") or "")
        if not document:
            continue
        document = requested.get(document.casefold(), document)
        extracted = [
            *_semantic_table_edges(observation.content, document),
            *(
                edge
                for line in observation.content.splitlines()
                for edge in _semantic_edges(line, document)
            ),
        ]
        for line in observation.content.splitlines():
            rules.extend(_semantic_rules(line, document))
        for edge in extracted:
            identity = (
                edge.source_key,
                edge.target_key,
                edge.document.casefold(),
                edge.relation,
                edge.priority,
            )
            if identity in seen_edges:
                continue
            seen_edges.add(identity)
            edges.append(edge)
    if not edges:
        return None

    def functional_relation(edge: SemanticEdge) -> str | None:
        if edge.relation in {"mapping", "ownership"}:
            return "assignment"
        if edge.relation == "classification":
            return edge.relation
        return None

    grouped: dict[tuple[str, str], list[SemanticEdge]] = {}
    for edge in edges:
        relation = functional_relation(edge)
        if relation is not None:
            grouped.setdefault((edge.source_key, relation), []).append(edge)

    suppressed: set[SemanticEdge] = set()
    unresolved_groups: dict[tuple[str, str], list[SemanticEdge]] = {}
    for identity, related in grouped.items():
        targets = {edge.target_key for edge in related}
        if len(targets) <= 1:
            continue
        highest = max(edge.priority for edge in related)
        preferred_targets = {edge.target_key for edge in related if edge.priority == highest}
        if len(preferred_targets) == 1 and any(edge.priority < highest for edge in related):
            preferred = next(iter(preferred_targets))
            suppressed.update(
                edge for edge in related if edge.priority < highest or edge.target_key != preferred
            )
            continue
        unresolved_groups[identity] = related

    usable_edges = [edge for edge in edges if edge not in suppressed]
    adjacency: dict[str, list[SemanticEdge]] = {}
    displays: dict[str, str] = {}
    for edge in usable_edges:
        adjacency.setdefault(edge.source_key, []).append(edge)
        displays.setdefault(edge.source_key, edge.source)
        displays.setdefault(edge.target_key, edge.target)
        if edge.relation == "alias":
            adjacency.setdefault(edge.target_key, []).append(
                SemanticEdge(
                    source_key=edge.target_key,
                    source=edge.target,
                    target_key=edge.source_key,
                    target=edge.source,
                    document=edge.document,
                    relation=edge.relation,
                    priority=edge.priority,
                )
            )

    goal_key = _semantic_key(goal)
    starts = sorted(
        {
            edge.source_key
            for edge in edges
            if re.search(
                rf"(?:^|\s){re.escape(edge.source_key)}(?:\s|$)",
                goal_key,
            )
        },
        key=len,
        reverse=True,
    )
    required_documents = {path.casefold() for path in paths}
    superseded_documents = {edge.document.casefold() for edge in suppressed}

    def has_sufficient_sources(sources: Iterable[str]) -> bool:
        covered = {
            *(source.casefold() for source in sources),
            *superseded_documents,
        }
        if require_all_documents:
            return required_documents.issubset(covered)
        return len(covered - superseded_documents) >= 2

    rule_candidates: list[dict[str, Any]] = []
    for rule in rules:
        for subject_key in starts:
            options = [
                [
                    edge
                    for edge in usable_edges
                    if edge.source_key == subject_key
                    and edge.relation == relation
                    and edge.target_key == target_key
                ]
                for relation, target_key, _display in rule.conditions
            ]
            if any(not matches for matches in options):
                continue
            for premises in product(*options):
                queue: list[tuple[str, list[SemanticEdge], set[str]]] = [
                    (rule.target_key, [], {subject_key, rule.target_key})
                ]
                while queue:
                    node, continuation, visited = queue.pop(0)
                    if continuation:
                        sources = [
                            *(edge.document for edge in premises),
                            rule.document,
                            *(edge.document for edge in continuation),
                        ]
                        if has_sufficient_sources(sources):
                            condition_nodes = [
                                display for _relation, _key, display in rule.conditions
                            ]
                            nodes = [
                                premises[0].source,
                                *condition_nodes,
                                rule.target,
                                *(edge.target for edge in continuation),
                            ]
                            rule_candidates.append(
                                {
                                    "status": "rule",
                                    "nodes": nodes,
                                    "relations": [
                                        *(edge.relation for edge in premises),
                                        rule.relation,
                                        *(edge.relation for edge in continuation),
                                    ],
                                    "sources": sources,
                                    "premises": [
                                        {
                                            "subject": edge.source,
                                            "relation": edge.relation,
                                            "object": edge.target,
                                            "source": edge.document,
                                        }
                                        for edge in premises
                                    ],
                                    "rule": {
                                        "conditions": [
                                            {
                                                "relation": relation,
                                                "object": display,
                                            }
                                            for relation, _key, display in rule.conditions
                                        ],
                                        "conclusion": rule.target,
                                        "source": rule.document,
                                    },
                                    "excluded_nodes": [],
                                }
                            )
                    if len(continuation) >= 3:
                        continue
                    for edge in adjacency.get(node, []):
                        group = (
                            edge.source_key,
                            functional_relation(edge) or edge.relation,
                        )
                        if group in unresolved_groups or edge.target_key in visited:
                            continue
                        queue.append(
                            (
                                edge.target_key,
                                [*continuation, edge],
                                {*visited, edge.target_key},
                            )
                        )
    if rule_candidates:
        selected_rule = max(
            rule_candidates,
            key=lambda item: (
                len(set(item["sources"])),
                len(item["nodes"]),
            ),
        )
        selected_keys = {_semantic_key(str(node)) for node in selected_rule["nodes"]}
        selected_rule["excluded_nodes"] = [
            display for key, display in displays.items() if key not in selected_keys
        ][:24]
        return selected_rule

    candidates: list[list[SemanticEdge]] = []
    for start in starts:
        queue: list[tuple[str, list[SemanticEdge], set[str]]] = [(start, [], {start})]
        while queue:
            node, chain, visited = queue.pop(0)
            if len(chain) >= 2:
                candidates.append(chain)
            if len(chain) >= 6:
                continue
            for edge in adjacency.get(node, []):
                if edge.target_key in visited:
                    continue
                queue.append(
                    (
                        edge.target_key,
                        [*chain, edge],
                        {*visited, edge.target_key},
                    )
                )
    clean_candidates = [
        chain
        for chain in candidates
        if not any(
            (
                edge.source_key,
                functional_relation(edge) or edge.relation,
            )
            in unresolved_groups
            for edge in chain
        )
        and has_sufficient_sources(edge.document for edge in chain)
    ]
    if not clean_candidates:
        candidate_groups: set[tuple[str, str]] = set()
        for chain in candidates:
            chain_groups = {
                (
                    edge.source_key,
                    functional_relation(edge) or edge.relation,
                )
                for edge in chain
                if (
                    edge.source_key,
                    functional_relation(edge) or edge.relation,
                )
                in unresolved_groups
            }
            covered = {edge.document.casefold() for edge in chain}
            covered.update(
                edge.document.casefold()
                for identity in chain_groups
                for edge in unresolved_groups[identity]
            )
            if (
                required_documents.issubset(covered)
                if require_all_documents
                else len(
                    {
                        edge.document.casefold()
                        for edge in chain
                        if edge.document.casefold() not in superseded_documents
                    }
                )
                >= 2
            ):
                candidate_groups.update(chain_groups)
        conflicts = []
        for identity, related in unresolved_groups.items():
            if identity not in candidate_groups:
                continue
            conflicts.append(
                {
                    "entity": related[0].source,
                    "relation": related[0].relation,
                    "targets": sorted({edge.target for edge in related}),
                    "sources": sorted({edge.document for edge in related}),
                    "resolution": (
                        "Find an explicit current or effective mapping from a live "
                        "authoritative source before selecting either branch."
                    ),
                }
            )
        if conflicts:
            return {
                "status": "conflicted",
                "conflicts": conflicts,
            }
        return None

    chain = max(
        clean_candidates,
        key=lambda item: (
            sum(edge.priority for edge in item),
            item[-1].priority,
            item[-1].relation == "mapping",
            len({edge.document.casefold() for edge in item}),
            len(item),
        ),
    )
    chain_keys = {chain[0].source_key, *(edge.target_key for edge in chain)}
    excluded = [display for key, display in displays.items() if key not in chain_keys][:24]
    return {
        "status": "chain",
        "nodes": [chain[0].source, *(edge.target for edge in chain)],
        "relations": [edge.relation for edge in chain],
        "sources": [edge.document for edge in chain],
        "excluded_nodes": excluded,
    }

"""Small, memory-only primitives for inspectable test-time reasoning."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import re
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any


def content_id(prefix: str, value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:16]}"


def _entropy(weights: Iterable[float]) -> float:
    values = [value for value in weights if value > 0]
    total = sum(values)
    return -sum((value / total) * math.log2(value / total) for value in values) if total else 0.0


def _candidate_expressions(expression: str, parameters: list[str]) -> set[str]:
    """Enumerate bounded expression mutations for repair planning."""

    candidates: set[str] = set()
    for operator_text in (" + ", " - ", " * ", " / ", " % "):
        if operator_text not in expression:
            continue
        for replacement in (" + ", " - ", " * ", " / ", " % "):
            if replacement != operator_text:
                candidates.add(expression.replace(operator_text, replacement, 1))
    if re.search(r"\b(?:min|max)\s*\(", expression):
        candidates.add(
            re.sub(
                r"\bmax\s*\(",
                "min(",
                re.sub(r"\bmin\s*\(", "__CORTHEON_MAX__(", expression),
            ).replace("__CORTHEON_MAX__(", "max(")
        )
    for left, right in (
        ("== 0", "== 1"),
        ("== 1", "== 0"),
        ("!= 0", "!= 1"),
        ("!= 1", "!= 0"),
    ):
        if left in expression:
            candidates.add(expression.replace(left, right, 1))
    comparisons = (" < ", " <= ", " > ", " >= ")
    for comparison in comparisons:
        if comparison not in expression:
            continue
        for replacement in comparisons:
            if replacement != comparison:
                candidates.add(expression.replace(comparison, replacement, 1))
    for match in re.finditer(r"\b\d+\b", expression):
        value = int(match.group(0))
        for adjusted in (value - 1, value + 1):
            if adjusted >= 0:
                candidates.add(
                    expression[: match.start()] + str(adjusted) + expression[match.end() :]
                )
    operand_swap = re.fullmatch(
        r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*([-/%])\s*([A-Za-z_][A-Za-z0-9_]*)\s*",
        expression,
    )
    if operand_swap is not None:
        left_name, operator_symbol, right_name = operand_swap.groups()
        candidates.add(f"{right_name} {operator_symbol} {left_name}")
    absolute = re.fullmatch(r"abs\((.+)\)", expression)
    if absolute is not None:
        candidates.add(absolute.group(1))
    elif len(expression) <= 60:
        candidates.add(f"abs({expression})")
    if expression.startswith("not "):
        candidates.add(expression[4:])
    elif len(expression) <= 60:
        candidates.add(f"not {expression}")
    if len(parameters) == 2:
        left, right = parameters
        candidates.update(
            {
                f"{left} + {right}",
                f"{left} - {right}",
                f"{left} * {right}",
                f"{left} + {left} * {right}",
                f"{left} - {left} * {right}",
                f"{left} * (1 + {right})",
                f"{left} * (1 - {right})",
            }
        )
    candidates.discard(expression)
    candidates.discard("")
    return candidates


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}
_COMPARE_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_CALLS = {"min": min, "max": max, "abs": abs}


def _evaluate_expression(
    expression: str,
    parameters: list[str],
    values: tuple[int | float | bool, ...],
) -> Any:
    """Evaluate one small arithmetic expression without executing code."""

    if len(expression) > 300:
        return None
    try:
        tree = ast.parse(expression, mode="eval")
        return _evaluate_node(tree.body, dict(zip(parameters, values, strict=True)))
    except (ArithmeticError, SyntaxError, TypeError, ValueError):
        return None


def _evaluate_node(node: ast.AST, environment: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant) and isinstance(
        node.value,
        (int, float, bool),
    ):
        return node.value
    if isinstance(node, ast.Name) and node.id in environment:
        return environment[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        return _BINARY_OPERATORS[type(node.op)](
            _evaluate_node(node.left, environment),
            _evaluate_node(node.right, environment),
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand, environment))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _CALLS
        and not node.keywords
    ):
        return _CALLS[node.func.id](*(_evaluate_node(item, environment) for item in node.args))
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and len(node.comparators) == 1
        and type(node.ops[0]) in _COMPARE_OPERATORS
    ):
        return _COMPARE_OPERATORS[type(node.ops[0])](
            _evaluate_node(node.left, environment),
            _evaluate_node(node.comparators[0], environment),
        )
    raise ValueError("unsupported expression")


def _matches_expected(observed: Any, expected: int | float | bool) -> bool:
    if isinstance(expected, bool):
        return observed is expected
    return (
        isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and abs(float(observed) - float(expected)) <= 1e-9
    )


def rank_information_gain(
    hypotheses: Mapping[str, float],
    actions: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank evidence actions by expected entropy reduction per unit cost."""

    weights = {key: max(0.0, float(value)) for key, value in hypotheses.items()}
    if not any(weights.values()):
        weights = dict.fromkeys(weights, 1.0)
    prior = _entropy(weights.values())
    ranked: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        partitions = action.get("partitions")
        reported_resolves: set[str] = set()
        if not isinstance(partitions, (list, tuple)):
            resolved = {str(item) for item in action.get("resolves", ()) if str(item) in weights}
            reported_resolves = resolved
            partitions = [[item] for item in sorted(resolved)]
            unresolved = sorted(set(weights) - resolved)
            if unresolved:
                partitions.append(unresolved)
        seen: set[str] = set()
        expected = 0.0
        total = sum(weights.values())
        for group in partitions:
            members = [
                str(item) for item in group if str(item) in weights and str(item) not in seen
            ]
            seen.update(members)
            mass = sum(weights[item] for item in members)
            if total and mass:
                expected += (mass / total) * _entropy(weights[item] for item in members)
        omitted = sorted(set(weights) - seen)
        omitted_mass = sum(weights[item] for item in omitted)
        if total and omitted_mass:
            expected += (omitted_mass / total) * _entropy(weights[item] for item in omitted)
        gain = max(0.0, prior - expected)
        cost = max(0.001, float(action.get("cost", 1.0)))
        reliability = min(1.0, max(0.0, float(action.get("reliability", 1.0))))
        ranked.append(
            {
                "action_id": str(action.get("action_id") or f"action_{index + 1}"),
                "information_gain_bits": round(gain, 6),
                "expected_utility": round(gain * reliability / cost, 6),
                "cost": cost,
                "reliability": reliability,
                "resolves": sorted(reported_resolves),
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            -item["expected_utility"],
            -item["information_gain_bits"],
            item["action_id"],
        ),
    )


class CognitiveGraph:
    """A bounded proposition graph with provenance and deterministic identity."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: set[tuple[str, str, str, str, bool]] = set()

    def add_node(self, node_id: str, kind: str, label: str, **state: Any) -> str:
        self._nodes[node_id] = {
            "node_id": node_id,
            "kind": kind,
            "label": label,
            **{key: value for key, value in state.items() if value is not None},
        }
        return node_id

    def proposition(self, subject: str, relation: str, target: str) -> tuple[str, str]:
        source_id = content_id("p", {"label": subject})
        target_id = content_id("p", {"label": target})
        self.add_node(source_id, "proposition", subject)
        self.add_node(target_id, "proposition", target)
        self.add_edge(source_id, relation, target_id, functional=True)
        return source_id, target_id

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        evidence_id: str = "",
        functional: bool = False,
    ) -> None:
        if source not in self._nodes or target not in self._nodes:
            raise ValueError("graph edges require existing nodes")
        self._edges.add((source, relation, target, evidence_id, functional))

    def proof_path(self, source: str, target: str) -> list[dict[str, Any]]:
        adjacent: dict[str, list[tuple[str, str, str]]] = {}
        for start, relation, end, evidence_id, _functional in self._edges:
            adjacent.setdefault(start, []).append((end, relation, evidence_id))
        queue = deque([(source, [])])
        visited = {source}
        while queue:
            node, path = queue.popleft()
            if node == target:
                return path
            for end, relation, evidence_id in sorted(adjacent.get(node, ())):
                if end in visited:
                    continue
                visited.add(end)
                queue.append(
                    (
                        end,
                        [
                            *path,
                            {
                                "from": node,
                                "relation": relation,
                                "to": end,
                                **({"evidence_id": evidence_id} if evidence_id else {}),
                            },
                        ],
                    )
                )
        return []

    def snapshot(self) -> dict[str, Any]:
        nodes = [self._nodes[key] for key in sorted(self._nodes)]
        edges = [
            {
                "from": source,
                "relation": relation,
                "to": target,
                **({"evidence_id": evidence_id} if evidence_id else {}),
                **({"functional": True} if functional else {}),
            }
            for source, relation, target, evidence_id, functional in sorted(self._edges)
        ]
        targets: dict[tuple[str, str], set[str]] = {}
        evidence: dict[tuple[str, str], set[str]] = {}
        for source, relation, target, evidence_id, functional in self._edges:
            if not functional:
                continue
            targets.setdefault((source, relation), set()).add(target)
            if evidence_id:
                evidence.setdefault((source, relation), set()).add(evidence_id)
        contradictions = [
            {
                "source": source,
                "relation": relation,
                "targets": sorted(values),
                "evidence_ids": sorted(evidence.get((source, relation), ())),
            }
            for (source, relation), values in sorted(targets.items())
            if len(values) > 1
        ]
        graph = {"nodes": nodes, "edges": edges, "contradictions": contradictions}
        return {
            **graph,
            "digest": content_id("cg", graph),
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

"""Architecture and compatibility guards for bounded repair planning."""

from __future__ import annotations

from pathlib import Path

import cortheon.cognitive_graph as cognitive_graph
import cortheon.cognitive_repair as repair

ROOT = Path(__file__).parents[1]
REPAIR = ROOT / "src/cortheon/cognitive_repair.py"
GRAPH = ROOT / "src/cortheon/cognitive_graph.py"
PUBLIC_API = {
    "RepairPlan",
    "TestInvocation",
    "changed_paths_from_diff",
    "derive_repair_candidates",
    "derive_simple_repair",
    "is_test_path",
    "protected_test_paths",
    "protects_tests",
    "requested_check_invocation",
    "requested_test_invocation",
}


def test_repair_modules_have_real_headroom() -> None:
    assert len(REPAIR.read_text(encoding="utf-8").splitlines()) <= 450
    assert len(GRAPH.read_text(encoding="utf-8").splitlines()) <= 450


def test_public_repair_api_stays_on_the_original_module() -> None:
    for name in PUBLIC_API:
        assert hasattr(repair, name), name
    assert repair.RepairPlan.__module__ == "cortheon.cognitive_repair"
    assert repair.TestInvocation.__module__ == "cortheon.cognitive_repair"


def test_expression_engine_has_one_implementation_owner() -> None:
    assert repair._candidate_expressions is cognitive_graph._candidate_expressions
    assert repair._evaluate_expression is cognitive_graph._evaluate_expression
    assert repair._matches_expected is cognitive_graph._matches_expected


def test_derive_keeps_the_original_module_monkeypatch_seams(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    def candidates(expression: str, parameters: list[str]) -> set[str]:
        calls.append(("candidates", (expression, parameters)))
        return {"left + right"}

    def evaluate(expression: str, parameters: list[str], values: tuple) -> int:
        calls.append(("evaluate", (expression, parameters, values)))
        return 5

    def matches(observed: object, expected: object) -> bool:
        calls.append(("matches", (observed, expected)))
        return True

    monkeypatch.setattr(repair, "_candidate_expressions", candidates)
    monkeypatch.setattr(repair, "_evaluate_expression", evaluate)
    monkeypatch.setattr(repair, "_matches_expected", matches)

    plans = repair.derive_repair_candidates(
        [
            ("calculator.py", "def add(left, right):\n    return left - right\n"),
            ("test_calculator.py", "assert add(2, 3) == 5\n"),
        ]
    )

    assert plans[0].new_text == "    return left + right"
    assert [name for name, _payload in calls] == ["candidates", "evaluate", "matches"]


def test_both_implementation_modules_remain_in_the_runtime_allowlist() -> None:
    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
    runtime_block = setup_source.split("RUNTIME_MODULES", 1)[1].split(")", 1)[0]
    assert '"cognitive_repair"' in runtime_block
    assert '"cognitive_graph"' in runtime_block

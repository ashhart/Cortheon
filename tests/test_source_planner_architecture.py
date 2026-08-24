import ast
from pathlib import Path

import cortheon.source_planner as planner

ROOT = Path(__file__).parents[1]
CORE = ROOT / "src" / "cortheon" / "source_planner_core"


def test_source_planner_has_no_model_transport() -> None:
    assert {path.name for path in CORE.glob("*.py")} == {
        "__init__.py",
        "_compat.py",
        "heuristic.py",
        "profiles.py",
        "types.py",
    }
    tree = ast.parse((ROOT / "src" / "cortheon" / "source_planner.py").read_text())
    imports = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(module.endswith((".model", ".json_io")) for module in imports)


def test_public_planner_is_heuristic() -> None:
    assert planner.SourcePlanner.__module__ == "cortheon.source_planner"
    assert planner.default_source_planner().__class__ is planner.SourcePlanner

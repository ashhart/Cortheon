import ast
from pathlib import Path

import cortheon.decision as decision

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src" / "cortheon"


def test_decision_layer_has_no_model_client_path() -> None:
    paths = [SOURCE / "decision.py", *(SOURCE / "decision_core").glob("*.py")]
    imports = {
        node.module
        for path in paths
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any("cortheon_llm" in module or "deliberation" in module for module in imports)
    assert not (SOURCE / "decision_core" / "cortheon_bridge.py").exists()


def test_facade_patches_drive_the_layer(monkeypatch) -> None:
    monkeypatch.setattr(decision, "build_checks", lambda _text, _evidence: [])
    monkeypatch.setattr(decision, "verdict_for", lambda _checks: "allow")
    monkeypatch.setattr(decision, "confidence_for", lambda _checks, _verdict: 0.123)
    report = decision.DecisionLayer().evaluate("patched")
    assert report.verdict == "allow"
    assert report.confidence == 0.123

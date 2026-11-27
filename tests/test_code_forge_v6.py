from __future__ import annotations

from flash_forge.code_forge import (
    _run_modular_suite_with_diff,
    evaluate_oracle_mutation_score,
    generate_ast_mutants,
)


def test_rich_diff_probe_captures_created_files_and_scrubs_object_addresses():
    sol = """
class DummyPlot:
    pass

def make_artifacts():
    with open("report.csv", "w") as f:
        f.write("a,b\\n1,2\\n")
    return DummyPlot()
"""
    tests = """
def test_basic():
    obj = make_artifacts()
    assert obj is not None

def diff_probe():
    obj = make_artifacts()
    return {"obj": obj}
"""
    rows, diff_out, _, fatal = _run_modular_suite_with_diff(sol, tests)
    assert fatal is None
    assert rows and rows[0]["ok"] is True
    assert "<ADDR>" in diff_out["obj"]
    assert "0x" not in diff_out["obj"]
    assert "__created_files__" in diff_out
    assert "report.csv:True" in diff_out["__created_files__"]


def test_ast_mutation_score_distinguishes_strong_vs_tautological_oracle():
    sol = """
def clamp_positive(x, limit):
    if x < 0:
        return 0
    if x > limit:
        return limit
    return x + 0
"""
    mutants = generate_ast_mutants(sol, max_mutants=3)
    assert len(mutants) == 3

    strong_tests = """
def test_boundaries():
    assert clamp_positive(-5, 10) == 0
    assert clamp_positive(0, 10) == 0
    assert clamp_positive(5, 10) == 5
    assert clamp_positive(10, 10) == 10
    assert clamp_positive(15, 10) == 10
"""
    weak_tests = """
def test_rubber_stamp():
    assert isinstance(clamp_positive(5, 10), int)
"""
    strong_killed, total = evaluate_oracle_mutation_score(sol, strong_tests, max_mutants=3)
    weak_killed, _ = evaluate_oracle_mutation_score(sol, weak_tests, max_mutants=3)
    assert total == 3
    assert strong_killed >= 2
    assert strong_killed > weak_killed

"""Proof-schema and distribution guards for repository-only qualification."""

from pathlib import Path

import cortheon.qualification_factory as facade

ROOT = Path(__file__).parents[1]
CORE_DIR = ROOT / "src/cortheon/qualification_core"


def _core_modules() -> list[Path]:
    return sorted(path for path in CORE_DIR.glob("*.py") if path.name != "__init__.py")


def test_report_schema_version_seven_has_one_owner() -> None:
    assert facade.SCHEMA_VERSION == 3
    assert facade.REPORT_SCHEMA_VERSION == 7
    emitters = {
        path.stem
        for path in _core_modules()
        if "REPORT_SCHEMA_VERSION" in path.read_text(encoding="utf-8")
    }
    assert emitters == {"constants", "report", "cli"}


def test_block_taxonomy_has_one_owner() -> None:
    taxonomy_source = (CORE_DIR / "taxonomy.py").read_text(encoding="utf-8")
    for token in (
        "classify_block",
        "FALSE_BLOCK",
        "SAFE_BLOCK",
        "UNCLASSIFIED_BLOCK",
        "DELIVERY_FAILURE",
    ):
        assert token in taxonomy_source, token
    classifiers = {
        path.stem
        for path in _core_modules()
        if "classify_block" in path.read_text(encoding="utf-8")
    }
    assert classifiers == {"taxonomy"}, f"block classification leaked into {sorted(classifiers)}"
    gate_owners = {
        path.stem
        for path in _core_modules()
        if '"all_blocks_classified"' in path.read_text(encoding="utf-8")
    }
    assert gate_owners == {"cell_gates", "report"}


def test_qualification_stays_repository_only() -> None:
    for relative in ("setup.py", "MANIFEST.in", "pyproject.toml"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "qualification" not in source, f"{relative} would ship qualification"
    distribution_test = (ROOT / "tests/test_lightweight_distribution.py").read_text(
        encoding="utf-8"
    )
    assert "qualification" not in distribution_test, "the shipped-module list must not grow"

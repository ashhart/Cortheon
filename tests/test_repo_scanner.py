import tempfile
import unittest
from pathlib import Path

from cortheon.engine import CortheonEngine
from cortheon.ledger import EvidenceLedger
from cortheon.models import PackageMetadata
from cortheon.repo_scanner import (
    parse_requirement_line,
    python_compatibility,
    scan_repo,
)

PYPROJECT = """
[project]
name = "mypkg"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "httpx>=0.27",
  "rich",
]

[project.optional-dependencies]
dev = ["pytest"]

[tool.pytest.ini_options]
testpaths = ["tests"]
"""

REQUIREMENTS = """
# pinned web framework
flask==3.0.0
-r other-requirements.txt
"""

CORE_PY = """
import json
import httpx
import yaml


def fetch(url):
    return httpx.get(url)
"""


def build_fixture_repo(root: Path) -> None:
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (root / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    package_dir = root / "src" / "mypkg"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("from mypkg.core import fetch\n", encoding="utf-8")
    (package_dir / "core.py").write_text(CORE_PY, encoding="utf-8")
    tests_dir = root / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_core.py").write_text("import mypkg\n", encoding="utf-8")


class RepoScannerTests(unittest.TestCase):
    def test_scans_fixture_repo_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_fixture_repo(root)

            report = scan_repo(root)

        self.assertEqual(report.dependency_managers, ["pyproject", "requirements"])
        self.assertEqual(report.python_requirement, ">=3.11")
        names = {(item.name, item.constraint) for item in report.declared_dependencies}
        self.assertIn(("httpx", ">=0.27"), names)
        self.assertIn(("rich", None), names)
        self.assertIn(("flask", "==3.0.0"), names)
        self.assertIn(("pytest", None), names)
        self.assertTrue(report.src_layout)
        self.assertIn("PYTHONPATH=src python3 -m pytest", report.test_commands)
        self.assertEqual(report.python_file_count, 3)
        self.assertIn("httpx", report.imported_third_party)
        self.assertIn("yaml", report.imported_third_party)
        # yaml is imported but never declared; rich is declared but never imported.
        self.assertEqual(report.undeclared_imports, ["yaml"])
        self.assertEqual(report.unused_declared, ["rich"])
        self.assertIn("httpx", report.framework_signals)
        self.assertIn("flask", report.framework_signals)
        self.assertEqual(report.errors, [])
        self.assertEqual(report.evidence[0].support.value, "observed")

    def test_missing_repo_fails_honestly(self) -> None:
        report = scan_repo("/definitely/not/a/repo/path")

        self.assertTrue(report.errors)
        self.assertEqual(report.evidence[0].support.value, "failed")

    def test_python_compatibility_bounds(self) -> None:
        self.assertEqual(python_compatibility(">=3.11", ">=3.8")[0], True)
        self.assertEqual(python_compatibility(">=3.11", ">=3.12")[0], False)
        self.assertIsNone(python_compatibility(None, ">=3.8")[0])
        self.assertIsNone(python_compatibility(">=3.11", None)[0])

    def test_requirement_line_parsing(self) -> None:
        parsed = parse_requirement_line(
            "httpx[http2]>=0.27 ; python_version >= '3.9'  # http", source="requirements.txt"
        )
        self.assertEqual(parsed.name, "httpx")
        self.assertEqual(parsed.constraint, ">=0.27")
        self.assertIsNone(parse_requirement_line("# comment only", source="r"))
        self.assertIsNone(parse_requirement_line("-r base.txt", source="r"))
        self.assertIsNone(parse_requirement_line("https://example.org/pkg.whl", source="r"))


class FakePyPI:
    def __init__(self, requires_python: str) -> None:
        self.requires_python = requires_python

    def fetch(self, package, version=None):
        metadata = PackageMetadata(
            name=package,
            version="9.9.9",
            summary=None,
            requires_python=self.requires_python,
            license="MIT",
            project_urls={},
            classifiers=[],
            requires_dist=[],
            release_upload_time=None,
            release_count=1,
            artifacts=[],
            source_url=f"https://pypi.org/pypi/{package}/json",
        )
        return metadata, []


class RepoFitTests(unittest.TestCase):
    def test_declared_and_compatible_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_fixture_repo(root)
            engine = CortheonEngine(
                pypi=FakePyPI(">=3.8"), ledger=EvidenceLedger(root / ".cortheon")
            )

            fit = engine.check_repo_fit("httpx", str(root), write_report=False)

        self.assertTrue(fit.already_declared)
        self.assertEqual(fit.declared_constraint, ">=0.27")
        self.assertTrue(fit.already_imported)
        self.assertIs(fit.python_compatible, True)
        self.assertEqual(fit.risks, [])

    def test_python_floor_raise_is_a_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_fixture_repo(root)
            engine = CortheonEngine(
                pypi=FakePyPI(">=3.13"), ledger=EvidenceLedger(root / ".cortheon")
            )

            fit = engine.check_repo_fit("newpkg", str(root), write_report=False)

        self.assertFalse(fit.already_declared)
        self.assertIs(fit.python_compatible, False)
        self.assertTrue(any("raise the repository's minimum Python" in risk for risk in fit.risks))
        self.assertTrue(any("new dependency" in note for note in fit.notes))


if __name__ == "__main__":
    unittest.main()

import difflib
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

from cortheon.patch_runner import _host_test_invocation, run_patch_verification

PYPROJECT = """
[project]
name = "mypkg"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
"""

CORE_OLD = """def greet(name):
    return f"hello {name}"
"""

CORE_NEW = """def greet(name):
    return f"hello, {name}"
"""

TEST_OLD = """from mypkg.core import greet


def test_greet():
    assert greet("x") == "hello x"
"""

TEST_NEW = """from mypkg.core import greet


def test_greet():
    assert greet("x") == "hello, x"
"""


def build_repo(root: Path) -> None:
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    package = root / "src" / "mypkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "core.py").write_text(CORE_OLD, encoding="utf-8")
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_core.py").write_text(TEST_OLD, encoding="utf-8")


def unified(path: str, old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


class PatchRunnerTests(unittest.TestCase):
    def test_host_test_command_is_parsed_without_shell_semantics(self) -> None:
        argv, environment = _host_test_invocation(
            'PYTHONPATH=src python3 -m pytest -q "tests/test core.py"'
        )

        self.assertEqual(
            argv,
            ["python3", "-m", "pytest", "-q", "tests/test core.py"],
        )
        self.assertEqual(environment, {"PYTHONPATH": "src"})
        with self.assertRaisesRegex(ValueError, "shell operators"):
            _host_test_invocation("python3 -m pytest && touch escaped")

    def test_good_patch_earns_allow_and_tests_passed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_repo(root)
            patch = unified("src/mypkg/core.py", CORE_OLD, CORE_NEW) + unified(
                "tests/test_core.py", TEST_OLD, TEST_NEW
            )

            report = run_patch_verification(root, patch)

        self.assertTrue(report.applied)
        self.assertEqual(sorted(report.files_changed), ["src/mypkg/core.py", "tests/test_core.py"])
        self.assertTrue(report.baseline.passed)
        self.assertTrue(report.after.passed)
        self.assertEqual(report.verdict, "allow")
        self.assertIn("tests_passed", report.earned_evidence_tags)
        self.assertEqual(report.evidence[0].support.value, "verified")
        self.assertTrue(any("git apply -R" in step for step in report.rollback_plan))

    def test_regressing_patch_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_repo(root)
            # Change behavior without updating the test: a regression.
            patch = unified("src/mypkg/core.py", CORE_OLD, CORE_NEW)

            report = run_patch_verification(root, patch)

        self.assertTrue(report.applied)
        self.assertTrue(report.baseline.passed)
        self.assertFalse(report.after.passed)
        self.assertEqual(report.verdict, "block")
        self.assertEqual(report.earned_evidence_tags, [])
        self.assertTrue(any("REGRESSION" in note for note in report.notes))
        self.assertEqual(report.evidence[0].support.value, "failed")

    def test_malformed_patch_blocks_without_running_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_repo(root)

            report = run_patch_verification(root, "--- not a real diff ---\n")

        self.assertFalse(report.applied)
        self.assertEqual(report.verdict, "block")
        self.assertIsNone(report.after)
        self.assertTrue(any("does not apply" in error for error in report.errors))

    def test_no_test_command_yields_needs_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
            package = root / "src" / "mypkg"
            package.mkdir(parents=True)
            (package / "core.py").write_text(CORE_OLD, encoding="utf-8")
            patch = unified("src/mypkg/core.py", CORE_OLD, CORE_NEW)

            report = run_patch_verification(root, patch)

        self.assertTrue(report.applied)
        self.assertEqual(report.verdict, "needs_evidence")
        self.assertTrue(any("no test command" in note.lower() for note in report.notes))

    def test_original_repo_is_never_touched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_repo(root)
            patch = unified("src/mypkg/core.py", CORE_OLD, CORE_NEW)

            run_patch_verification(root, patch)

            self.assertEqual(
                (root / "src" / "mypkg" / "core.py").read_text(encoding="utf-8"), CORE_OLD
            )
            self.assertFalse((root / ".git").exists())

    def test_after_run_does_not_reuse_same_size_baseline_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = "def divide(a, b):\n    return a * b\n"
            new = "def divide(a, b):\n    return a / b\n"
            (root / "calculator.py").write_text(old, encoding="utf-8")
            (root / "test_calculator.py").write_text(
                "import unittest\n"
                "from calculator import divide\n\n"
                "class TestCalculator(unittest.TestCase):\n"
                "    def test_divide(self):\n"
                "        self.assertEqual(divide(12, 4), 3)\n",
                encoding="utf-8",
            )
            report = run_patch_verification(
                root,
                unified("calculator.py", old, new),
                test_command=(
                    f'{shlex.quote(sys.executable)} -m unittest discover -s . -p "test_*.py"'
                ),
            )

        self.assertFalse(report.baseline.passed)
        self.assertTrue(report.after.passed)
        self.assertEqual(report.verdict, "allow")


if __name__ == "__main__":
    unittest.main()

import ast
import unittest

from cortheon.api_diff import build_api_diff_report, diff_symbols
from cortheon.api_indexer import extract_symbols_from_ast

OLD_SOURCE = """
class Client:
    def __init__(self, base_url, proxies=None):
        pass

    def stream(self, method, url):
        pass

    def old_helper(self):
        pass


def parse(data):
    pass
"""

NEW_SOURCE = '''
import warnings


class Client:
    def __init__(self, base_url):
        pass

    def stream(self, method, url):
        pass


def parse(data):
    """Deprecated: use parse_bytes instead."""
    pass


def parse_bytes(data):
    pass


def legacy_run():
    warnings.warn("legacy_run is going away", DeprecationWarning)
'''


def symbols_from(source: str):
    return extract_symbols_from_ast(ast.parse(source), "pkg", "pkg/__init__.py")


class ApiDiffTests(unittest.TestCase):
    def test_diff_detects_added_removed_changed_and_deprecated(self) -> None:
        old_symbols = symbols_from(OLD_SOURCE)
        new_symbols = symbols_from(NEW_SOURCE)

        added, removed, changed, deprecated = diff_symbols(old_symbols, new_symbols)

        added_names = [symbol.qualname for symbol in added]
        removed_names = [symbol.qualname for symbol in removed]
        changed_names = [change.qualname for change in changed]
        deprecated_names = [symbol.qualname for symbol in deprecated]

        self.assertIn("pkg.parse_bytes", added_names)
        self.assertIn("pkg.legacy_run", added_names)
        self.assertIn("pkg.Client.old_helper", removed_names)
        # Constructor kwargs are visible: the removed proxies parameter shows
        # up as a Client.__init__ signature change.
        self.assertIn("pkg.Client.__init__", changed_names)
        change = next(item for item in changed if item.qualname == "pkg.Client.__init__")
        self.assertIn("proxies", change.old_signature or "")
        self.assertNotIn("proxies", change.new_signature or "")
        # Docstring deprecation and warnings.warn(DeprecationWarning) both count.
        self.assertIn("pkg.parse", deprecated_names)
        self.assertIn("pkg.legacy_run", deprecated_names)

    def test_deprecated_method_does_not_mark_class(self) -> None:
        source = '''
class Widget:
    def fine(self):
        pass

    def going_away(self):
        """Deprecated since 2.0."""
        pass
'''
        symbols = symbols_from(source)
        by_name = {symbol.qualname: symbol for symbol in symbols}

        self.assertFalse(by_name["pkg.Widget"].deprecated)
        self.assertTrue(by_name["pkg.Widget.going_away"].deprecated)
        self.assertFalse(by_name["pkg.Widget.fine"].deprecated)

    def test_report_carries_counts_and_verified_evidence(self) -> None:
        old_symbols = symbols_from(OLD_SOURCE)
        new_symbols = symbols_from(NEW_SOURCE)

        report = build_api_diff_report("pkg", "1.0.0", "2.0.0", old_symbols, new_symbols)

        self.assertEqual(report.added_count, len(report.added))
        self.assertEqual(report.removed_count, len(report.removed))
        self.assertGreaterEqual(report.changed_count, 1)
        self.assertGreaterEqual(report.deprecated_count, 2)
        self.assertEqual(report.evidence[0].support.value, "verified")
        self.assertIn("2.0.0", report.evidence[0].claim)

    def test_unparsed_versions_produce_failed_evidence(self) -> None:
        report = build_api_diff_report("pkg", "1.0.0", "2.0.0", [], [], errors=["1.0.0: boom"])

        self.assertEqual(report.evidence[0].support.value, "failed")
        self.assertEqual(report.errors, ["1.0.0: boom"])


class MigrationGateTests(unittest.TestCase):
    def test_migration_task_requires_api_diff_evidence(self) -> None:
        from cortheon.decision import DecisionLayer

        report = DecisionLayer().evaluate(
            "Upgrade our service from pydantic v1 to v2",
            proposed_action="Rewrite the validators for pydantic v2.",
        )

        self.assertEqual(report.verdict, "needs_evidence")
        self.assertIn("api_diff_evidence", report.required_evidence)
        self.assertIn("cortheon_api_diff", report.recommended_tools)

    def test_migration_with_diff_evidence_passes_that_check(self) -> None:
        from cortheon.decision import DecisionLayer

        report = DecisionLayer().evaluate(
            "Upgrade our service from pydantic v1 to v2",
            proposed_action="Rewrite the validators for pydantic v2.",
            evidence=["api_diff_evidence"],
        )

        self.assertNotIn("api_diff_evidence", report.required_evidence)


if __name__ == "__main__":
    unittest.main()

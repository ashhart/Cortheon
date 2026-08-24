"""Protocol tests for the unknowable benchmark's credibility improvements.

Verifies (offline, no model): the neutral signature exists and uses intuitive
names (isolating access from prior-punishment), the adversarial one does not,
and each minted package is unique (so n>1 runs are genuinely independent). These
are the methodological invariants the demo's credibility rests on.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_unknowable():
    spec = importlib.util.spec_from_file_location(
        "unknowable", REPO / "benchmarks" / "unknowable.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["unknowable"] = module
    spec.loader.exec_module(module)
    return module


class UnknowableProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_unknowable()

    def test_neutral_signature_uses_intuitive_names(self) -> None:
        # The neutral control: a model's prior (data/encryption/shards) would
        # actually match here. So a stock pass under neutral is the prior
        # working, NOT access — and the demo must say so.
        src = self.mod.module_source(neutral=True)
        self.assertIn("def transmit(data", src)
        self.assertIn("encryption", src)
        self.assertIn("shards", src)
        self.assertNotIn("cargo", src)

    def test_adversarial_signature_contradicts_priors(self) -> None:
        src = self.mod.module_source(neutral=False)
        self.assertIn("cargo", src)
        self.assertIn("cipher_suite", src)
        self.assertIn("shard_factor", src)
        self.assertNotIn("data,", src)

    def test_neutral_signature_is_valid_python_and_parseable(self) -> None:
        # The substrate must be able to AST-read both signatures (the runtime
        # grader depends on it).
        import ast

        from cortheon.api_indexer import extract_symbols_from_ast

        for neutral in (True, False):
            src = self.mod.module_source(neutral=neutral)
            tree = ast.parse(src)  # must not raise
            symbols = extract_symbols_from_ast(tree, "anyname", "anyname.py")
            names = [s.name for s in symbols]
            self.assertIn("transmit", names, f"neutral={neutral}")

    def test_each_minted_package_is_unique(self) -> None:
        # n>1 is only meaningful if each run mints a genuinely fresh package.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = {self.mod.mint_package(root, neutral=False)[0] for _ in range(5)}
        self.assertEqual(len(names), 5, "mint_package must produce unique names per call")

    def test_minted_neutral_package_source_matches_module_source(self) -> None:
        # The written file must carry the neutral signature (no accidental
        # revert to adversarial).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name, pkg_dir = self.mod.mint_package(root, neutral=True)
            written = (pkg_dir / f"{name}.py").read_text()
        self.assertIn("def transmit(data", written)
        self.assertNotIn("cargo", written)


if __name__ == "__main__":
    unittest.main()

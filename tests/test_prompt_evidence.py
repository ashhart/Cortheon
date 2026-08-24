import unittest
from types import SimpleNamespace

from cortheon import prompt_evidence
from cortheon.models import ApiSymbol
from cortheon.prompt_evidence import (
    EVIDENCE_HEADER,
    build_evidence,
    detect_packages,
    predict_failures,
    wrap_for_prompt,
)

KNOWN = {
    "httpx",
    "pyyaml",
    "numpy",
    "polars",
    "beautifulsoup4",
    "requests",
    "fastapi",
    "pydantic",
}
# Real PyPI names that are dead metadata-less stubs squatting English words.
SPARSE_STUBS = {"automatic", "service"}


class ProbedMeta:
    def __init__(self, requires_python: str) -> None:
        self.requires_python = requires_python


def probe(name: str) -> ProbedMeta | None:
    if name.lower() in KNOWN:
        return ProbedMeta(">=3.9")
    if name.lower() in SPARSE_STUBS:
        return ProbedMeta("")
    return None


def symbol(
    qualname: str, kind: str = "function", signature: str | None = None, deprecated: bool = False
) -> ApiSymbol:
    return ApiSymbol(
        name=qualname.split(".")[-1],
        kind=kind,
        module=qualname.rsplit(".", 1)[0],
        qualname=qualname,
        signature=signature,
        file_path="src.py",
        line=1,
        docstring=None,
        deprecated=deprecated,
    )


HTTPX_SYMBOLS = [
    symbol("httpx.Client", kind="class"),
    symbol("httpx.Client.__init__", signature="__init__(self, *, timeout=None, transport=None)"),
    symbol("httpx.HTTPTransport", kind="class"),
    symbol("httpx.HTTPTransport.__init__", signature="__init__(self, *, retries=0, verify=True)"),
    symbol("httpx.stream", signature="stream(method, url, **kwargs)"),
    symbol("httpx.stream_file", signature="stream_file(url, path)", deprecated=True),
]


class FakeEngine:
    """Duck-types engine.pypi.fetch and engine.api_extractor.load_symbols."""

    class _Meta:
        def __init__(self, name: str, version: str) -> None:
            self.name = name
            self.version = version

    def __init__(self, symbols: list[ApiSymbol]) -> None:
        self._symbols = symbols
        self.pypi = self
        self.api_extractor = self

    def fetch(self, package: str, version: str | None = None):
        if package.lower() != "httpx":
            raise ValueError(f"unknown package {package}")
        return self._Meta("httpx", "0.99.0"), []

    def load_symbols(self, metadata):
        return None, self._symbols, []

    def diff_api(self, package, old_version, new_version, *, write_report=False):
        return SimpleNamespace(
            added=[
                symbol("httpx.NewClient", kind="class"),
                symbol("httpx.experimental._Private"),
            ]
        )

    def fetch_docs(self, package, *, max_pages=12, write_report=False):
        return SimpleNamespace(
            pages=[
                SimpleNamespace(
                    final_url="https://docs.example/streaming",
                    code_blocks=['with httpx.stream("GET", url) as response:\n    ...'],
                    text=(
                        "Streaming responses use .stream(). Within a stream block, "
                        "request data is available with .iter_bytes()."
                    ),
                )
            ]
        )


class DetectPackagesTests(unittest.TestCase):
    def test_imports_win_with_overrides_and_stdlib_filtered(self) -> None:
        text = "import httpx\nimport yaml\nimport json, os\nfrom pathlib import Path"
        self.assertEqual(detect_packages(text, probe), ["httpx", "pyyaml"])

    def test_import_priority_beats_prose_order(self) -> None:
        text = "polars is nice for this.\nimport httpx"
        self.assertEqual(detect_packages(text, probe), ["httpx", "polars"])

    def test_pip_install_with_extras_and_pins(self) -> None:
        text = "run pip install 'polars[lazy]>=1.0' beautifulsoup4 first"
        self.assertEqual(detect_packages(text, probe), ["polars", "beautifulsoup4", "first"])

    def test_dotted_usage_maps_aliases(self) -> None:
        self.assertEqual(detect_packages("then np.array(x) does it", probe), ["numpy"])

    def test_backticked_names_map_import_overrides(self) -> None:
        self.assertEqual(detect_packages("see the `bs4` docs", probe), ["beautifulsoup4"])

    def test_prose_stopwords_and_nonpackages_skipped(self) -> None:
        self.assertEqual(detect_packages("my client sends retries to the server", probe), [])
        self.assertEqual(detect_packages("the response.json() call", probe), [])

    def test_bare_prose_mention_of_real_package_detected(self) -> None:
        self.assertEqual(
            detect_packages("should I use requests or httpx here?", probe),
            ["requests", "httpx"],
        )

    def test_english_words_squatted_as_dead_stubs_skipped_in_prose(self) -> None:
        text = "Add automatic retries to my httpx client for a production service."
        self.assertEqual(detect_packages(text, probe), ["httpx"])

    def test_sparse_package_still_detected_from_explicit_import(self) -> None:
        self.assertEqual(detect_packages("import service", probe), ["service"])

    def test_urls_do_not_leak_domain_roots(self) -> None:
        text = "see https://pypi.org/project/httpx and pypi.org for details"
        self.assertEqual(detect_packages(text, probe), ["httpx"])

    def test_package_cap(self) -> None:
        text = "import httpx\nimport yaml\nimport numpy\nimport polars"
        self.assertEqual(len(detect_packages(text, probe)), 3)

    def test_probe_budget_bounds_lookups(self) -> None:
        fillers = " ".join(f"zzza{c}" for c in "abcdefgh")  # 8 probe-eligible unknowns
        self.assertEqual(detect_packages(f"{fillers} httpx", probe), [])
        self.assertEqual(detect_packages("zzzaa zzzab httpx", probe), ["httpx"])

    def test_explicit_package_list_beats_earlier_prose_candidates(self) -> None:
        text = (
            "State the exact current stable version of each of these packages "
            "on PyPI right now: fastapi, pydantic, httpx."
        )
        self.assertEqual(
            detect_packages(text, probe),
            ["fastapi", "pydantic", "httpx"],
        )


class BuildEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FakeEngine(HTTPX_SYMBOLS)

    def test_version_line_and_audit_details(self) -> None:
        facts, details = build_evidence(self.engine, "use httpx", ["httpx"])
        self.assertIn("current version of httpx is 0.99.0.", facts)
        self.assertEqual(details["packages"], {"httpx": "0.99.0"})
        self.assertEqual(details["facts"], len(facts.splitlines()))

    def test_mentioned_class_gets_exact_constructor_parameters(self) -> None:
        facts, _ = build_evidence(self.engine, "make a Client with httpx", ["httpx"])
        self.assertIn(
            "VERIFIED: httpx.Client(*, timeout=None, transport=None) "
            "— these are the exact constructor parameters. Use them. Do not invent others.",
            facts,
        )

    def test_mentioned_function_gets_exact_signature(self) -> None:
        facts, _ = build_evidence(self.engine, "how do I stream with httpx?", ["httpx"])
        self.assertIn(
            "VERIFIED: httpx.stream — exact signature: stream(method, url, **kwargs)",
            facts,
        )

    def test_deprecated_mention_gets_fate_and_replacement(self) -> None:
        facts, _ = build_evidence(self.engine, "use stream_file from httpx", ["httpx"])
        self.assertIn("'stream_file' is DEPRECATED in the current version", facts)
        self.assertIn("use instead: stream", facts)

    def test_bound_ghost_name_flagged_bare_camelcase_not(self) -> None:
        facts, _ = build_evidence(
            self.engine, "from httpx import Ghost\nmy DataFetcher class uses httpx", ["httpx"]
        )
        self.assertIn("'Ghost' does NOT exist in the current source", facts)
        self.assertNotIn("DataFetcher", facts)

    def test_nested_bound_ghost_is_detected(self) -> None:
        facts, _ = build_evidence(
            self.engine,
            "call httpx.Client.stream_to_file()",
            ["httpx"],
        )
        self.assertIn("'stream_to_file' does NOT exist", facts)

    def test_bound_ghost_gets_verified_replacement_signature(self) -> None:
        facts, _ = build_evidence(
            self.engine,
            "call httpx.stream_to_file()",
            ["httpx"],
        )
        self.assertIn("'stream_to_file' does NOT exist", facts)
        self.assertIn(
            "Closest verified current alternative is: "
            "httpx.stream with signature stream(method, url, **kwargs)",
            facts,
        )
        self.assertIn("VERIFIED OFFICIAL DOCS USAGE SHAPE", facts)
        self.assertIn(".iter_bytes()", facts)

    def test_bound_existing_name_not_flagged(self) -> None:
        facts, _ = build_evidence(self.engine, "from httpx import Client", ["httpx"])
        self.assertNotIn("defines no public symbol", facts)

    def test_keyword_ownership_fact_for_mentioned_kwarg(self) -> None:
        facts, _ = build_evidence(self.engine, "add retries to my httpx setup", ["httpx"])
        self.assertIn(
            "VERIFIED: 'retries' is accepted by HTTPTransport; no other verified callable takes it.",
            facts,
        )

    def test_unknown_package_contributes_nothing(self) -> None:
        facts, details = build_evidence(self.engine, "use ghostlib", ["ghostlib"])
        self.assertEqual(facts, "")
        self.assertEqual(details, {"packages": {}, "facts": 0})

    def test_evidence_is_char_capped_at_line_boundary(self) -> None:
        original = prompt_evidence.MAX_EVIDENCE_CHARS
        prompt_evidence.MAX_EVIDENCE_CHARS = 60
        self.addCleanup(setattr, prompt_evidence, "MAX_EVIDENCE_CHARS", original)
        facts, details = build_evidence(
            self.engine, "make a Client that can stream with httpx", ["httpx"]
        )
        self.assertLessEqual(len(facts), 60 + 1)
        self.assertEqual(details["facts"], 0)  # version line is longer than cap
        self.assertTrue(facts == "")

    def test_since_version_request_gets_live_public_api_diff(self) -> None:
        facts, _ = build_evidence(
            self.engine,
            "Name additions to httpx since version 0.20.0.",
            ["httpx"],
        )
        self.assertIn("public API additions in httpx since 0.20.0: NewClient", facts)
        self.assertNotIn("_Private", facts)

    def test_failure_prediction_never_treats_prose_as_missing_symbols(self) -> None:
        prediction = predict_failures(
            self.engine,
            "httpx has grown new public additions to its interface",
            ["httpx"],
        )
        self.assertEqual(prediction, "")


class WrapForPromptTests(unittest.TestCase):
    def test_block_carries_header_and_facts(self) -> None:
        block = wrap_for_prompt("fact one\nfact two")
        self.assertIn(EVIDENCE_HEADER, block)
        self.assertIn("fact one\nfact two", block)
        self.assertTrue(block.startswith("\n\n["))
        self.assertTrue(block.endswith("]"))


if __name__ == "__main__":
    unittest.main()

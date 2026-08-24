import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cortheon.api_indexer import ApiEvidenceExtractor
from cortheon.cache import FactCache
from cortheon.models import DistributionArtifact, PackageMetadata

SOURCE = b"""
class Client:
    def stream(self, method, url):
        pass
"""


def sdist_bytes() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("examplepkg-1.0.0/examplepkg/__init__.py")
        info.size = len(SOURCE)
        archive.addfile(info, io.BytesIO(SOURCE))
    return buffer.getvalue()


class CountingClient:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.calls = 0

    def get(self, url, headers=None):
        self.calls += 1

        class Response:
            body = self.body

        return Response()


def metadata() -> PackageMetadata:
    return PackageMetadata(
        name="examplepkg",
        version="1.0.0",
        summary=None,
        requires_python=">=3.11",
        license="MIT",
        project_urls={},
        classifiers=[],
        requires_dist=[],
        release_upload_time=None,
        release_count=1,
        artifacts=[
            DistributionArtifact(
                filename="examplepkg-1.0.0.tar.gz",
                package_type="sdist",
                url="https://files.example/examplepkg-1.0.0.tar.gz",
                size=None,
                upload_time=None,
                digests={"sha256": "abc123"},
            )
        ],
        source_url="https://pypi.org/pypi/examplepkg/json",
    )


class FactCacheTests(unittest.TestCase):
    def test_roundtrip_and_key_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = FactCache(tmp)
            cache.put({"a": 1}, "symbols", "1", "pkg")

            self.assertEqual(cache.get("symbols", "1", "pkg"), {"a": 1})
            self.assertIsNone(cache.get("symbols", "2", "pkg"))

    def test_corrupted_entry_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = FactCache(tmp)
            cache.put([1, 2], "k")
            path = cache._path(("k",))
            path.write_text("{not json", encoding="utf-8")

            self.assertIsNone(cache.get("k"))

    def test_kill_switch_disables_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict("os.environ", {"CORTHEON_NO_CACHE": "1"}):
                cache = FactCache(tmp)
                cache.put({"a": 1}, "k")
                self.assertIsNone(cache.get("k"))
            self.assertEqual(list(Path(tmp).rglob("*.json")), [])


class SymbolCacheIntegrationTests(unittest.TestCase):
    def test_second_load_hits_cache_and_skips_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = CountingClient(sdist_bytes())
            extractor = ApiEvidenceExtractor(client=client, cache=FactCache(tmp))

            _, first, _ = extractor.load_symbols(metadata())
            _, second, _ = extractor.load_symbols(metadata())

        self.assertEqual(client.calls, 1)
        self.assertEqual(
            [symbol.qualname for symbol in first],
            [symbol.qualname for symbol in second],
        )
        self.assertEqual(second[0].deprecated, False)
        self.assertEqual(second[0].qualname, "examplepkg.Client")

    def test_digest_change_invalidates_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = CountingClient(sdist_bytes())
            extractor = ApiEvidenceExtractor(client=client, cache=FactCache(tmp))
            extractor.load_symbols(metadata())

            changed = metadata()
            changed.artifacts[0].digests = {"sha256": "different"}
            extractor.load_symbols(changed)

        self.assertEqual(client.calls, 2)

    def test_corrupt_cached_schema_falls_back_to_fresh_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = CountingClient(sdist_bytes())
            cache = FactCache(tmp)
            extractor = ApiEvidenceExtractor(client=client, cache=cache)
            extractor.load_symbols(metadata())
            # Poison the cached value with entries missing required fields.
            from cortheon.api_indexer import symbols_cache_key

            key = symbols_cache_key(metadata(), metadata().artifacts[0])
            cache.put([{"bogus": True}], *key)

            _, symbols, _ = extractor.load_symbols(metadata())

        self.assertEqual(client.calls, 2)
        self.assertEqual(symbols[0].qualname, "examplepkg.Client")


if __name__ == "__main__":
    unittest.main()

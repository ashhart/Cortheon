import io
import unittest
import zipfile

from cortheon.api_indexer import extract_symbols_from_archive, match_symbols


class ApiIndexerTests(unittest.TestCase):
    def test_extracts_public_class_and_method_from_wheel(self) -> None:
        body = io.BytesIO()
        with zipfile.ZipFile(body, "w") as archive:
            archive.writestr(
                "example/__init__.py",
                '''
class Client:
    """Example client."""

    def stream(self, data: bytes, *, model: str = "default") -> str:
        """Stream data."""
        return "ok"

def helper(value: int) -> int:
    return value
''',
            )

        symbols = extract_symbols_from_archive(
            "example-1.0.0-py3-none-any.whl", body.getvalue(), 20
        )
        matches = match_symbols(symbols, "Client.stream")

        self.assertEqual(matches[0].qualname, "example.Client.stream")
        self.assertEqual(
            matches[0].signature,
            "stream(self, data: bytes, *, model: str = 'default') -> str",
        )


if __name__ == "__main__":
    unittest.main()

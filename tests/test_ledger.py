import json
import tempfile
import unittest
from pathlib import Path

from cortheon.ledger import EvidenceLedger
from cortheon.models import Evidence


class LedgerTests(unittest.TestCase):
    def test_append_evidence_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(Path(tmp))
            ledger.append_evidence(
                [
                    Evidence(
                        claim="PyPI reports latest package version.",
                        source_type="pypi_metadata",
                        source_url="https://pypi.org/pypi/example/json",
                        package="example",
                        version="1.0.0",
                    )
                ]
            )

            lines = (Path(tmp) / "evidence.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["claim"], "PyPI reports latest package version.")
            self.assertEqual(payload["status"], "current")


if __name__ == "__main__":
    unittest.main()

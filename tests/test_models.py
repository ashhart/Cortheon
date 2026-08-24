import unittest
from datetime import timedelta

from cortheon.models import Evidence, EvidenceStatus, utc_now


class EvidenceFreshnessTests(unittest.TestCase):
    def test_expired_current_evidence_becomes_stale(self) -> None:
        now = utc_now()
        evidence = Evidence(
            claim="Old package claim",
            source_type="unit_test",
            source_url=None,
            retrieved_at=now - timedelta(days=10),
            expires_at=now - timedelta(days=1),
        )

        self.assertEqual(evidence.refresh_status(now), EvidenceStatus.STALE)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from datetime import UTC, datetime

from cognitive_runtime_cases_common import RuntimeTestCase


class ResearchMixin(RuntimeTestCase):
    def test_research_requires_fresh_independent_cited_and_challenged_sources(self) -> None:
        started = self.runtime.start(
            "Research the current status of Project Atlas from live sources.",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        initial = started["next_action"]["request"]
        self.assertEqual(initial["capability"], "search")
        self.assertEqual(
            initial["parameters"]["purpose"],
            "contradiction_check",
        )
        retrieved_at = datetime.now(UTC).isoformat()

        discovered = self.runtime.observe(
            session_id,
            [
                {
                    "kind": "web",
                    "content": "Project Atlas launched on July 20, 2026.",
                    "source": "https://agency.gov/atlas",
                    "url": "https://agency.gov/atlas",
                    "retrieved_at": retrieved_at,
                    "published_at": "2026-07-20",
                    "purpose": "contradiction_check",
                },
                {
                    "kind": "web",
                    "content": (
                        "No credible correction contradicts the reported Project Atlas launch date."
                    ),
                    "source": "https://independent.example/report",
                    "url": "https://independent.example/report",
                    "retrieved_at": retrieved_at,
                    "published_at": "2026-07-21",
                    "purpose": "contradiction_check",
                },
            ],
            request_id="req1",
        )
        self.assertEqual(
            discovered["next_action"]["request"]["capability"],
            "fetch",
        )

        fetched = self.runtime.observe(
            session_id,
            [
                {
                    "kind": "web",
                    "content": "Primary announcement: Project Atlas launched July 20.",
                    "source": "https://agency.gov/atlas",
                    "url": "https://agency.gov/atlas",
                    "retrieved_at": retrieved_at,
                    "published_at": "2026-07-20",
                    "purpose": "primary_fetch",
                }
            ],
            request_id="req2",
        )
        self.assertEqual(fetched["next_action"]["type"], "reason")
        all_evidence_ids = ["ev1", "ev2", "ev3"]
        common = {
            "claims": [
                {
                    "claim": "Project Atlas launched on July 20, 2026.",
                    "evidence_ids": all_evidence_ids,
                }
            ],
            "hypotheses": [
                {
                    "statement": "Project Atlas launched on July 20, 2026.",
                    "falsification_test": "Check primary and contradictory sources.",
                    "status": "supported",
                    "evidence_ids": all_evidence_ids,
                }
            ],
            "completion_evidence_ids": all_evidence_ids,
        }

        rejected = self.runtime.complete(
            session_id,
            answer="Project Atlas launched on July 20, 2026.",
            **common,
        )
        alignment = next(
            item
            for item in rejected["verification"]["checks"]
            if item["name"] == "evidence_alignment"
        )
        self.assertFalse(alignment["passed"])
        self.assertIn("clickable citations", alignment["reason"])

        conflict_rejected = self.runtime.complete(
            session_id,
            answer=(
                "Project Atlas launched on July 20, 2026. Sources: "
                "https://agency.gov/atlas and https://independent.example/report."
            ),
            **common,
        )
        conflict_alignment = next(
            item
            for item in conflict_rejected["verification"]["checks"]
            if item["name"] == "evidence_alignment"
        )
        self.assertFalse(conflict_alignment["passed"])
        self.assertIn("polarity conflict", conflict_alignment["reason"])

        accepted = self.runtime.complete(
            session_id,
            answer=(
                "The source tension matters: the agency reports that Project Atlas "
                "launched on July 20, 2026, while the independent source reports no "
                "credible correction; both support the same date. Sources: "
                "https://agency.gov/atlas and https://independent.example/report."
            ),
            **common,
        )
        self.assertEqual(accepted["status"], "complete")
        self.assertEqual(self.runtime.active_sessions, 0)

    def test_research_requires_every_declared_evidence_domain(self) -> None:
        started = self.runtime.start(
            "Research Project Atlas from current web sources and ground the answer "
            "in the local SmartTensor workspace.",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        retrieved_at = datetime.now(UTC).isoformat()
        discovered = self.runtime.observe(
            session_id,
            [
                {
                    "kind": "web",
                    "content": "Project Atlas uses a staged execution engine.",
                    "source": "https://agency.gov/atlas",
                    "url": "https://agency.gov/atlas",
                    "retrieved_at": retrieved_at,
                    "published_at": "2026-07-20",
                    "purpose": "contradiction_check",
                },
                {
                    "kind": "web",
                    "content": "Independent review confirms the staged execution engine.",
                    "source": "https://independent.example/report",
                    "url": "https://independent.example/report",
                    "retrieved_at": retrieved_at,
                    "published_at": "2026-07-21",
                    "purpose": "contradiction_check",
                },
            ],
            request_id="req1",
        )
        self.assertEqual(
            discovered["next_action"]["request"]["capability"],
            "fetch",
        )

        fetched = self.runtime.observe(
            session_id,
            [
                {
                    "kind": "web",
                    "content": "Primary design: Project Atlas uses staged execution.",
                    "source": "https://agency.gov/atlas",
                    "url": "https://agency.gov/atlas",
                    "retrieved_at": retrieved_at,
                    "published_at": "2026-07-20",
                    "purpose": "primary_fetch",
                }
            ],
            request_id="req2",
        )
        self.assertEqual(fetched["next_action"]["type"], "harness_tool")
        self.assertEqual(
            fetched["next_action"]["request"]["capability"],
            "inspect",
        )
        self.assertIn(
            "local workspace",
            fetched["next_action"]["request"]["reason"].casefold(),
        )

        unrelated = self.runtime.observe(
            session_id,
            [
                {
                    "kind": "command",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"bash","outcome":"result",'
                        '"args":{"command":"date -Iseconds"}}\n'
                        "2026-07-26T16:00:00+01:00"
                    ),
                    "source": "host clock",
                }
            ],
            request_id="req3",
        )
        self.assertEqual(unrelated["next_action"]["type"], "harness_tool")
        self.assertEqual(
            unrelated["next_action"]["request"]["capability"],
            "inspect",
        )
        local_request_id = unrelated["next_action"]["request"]["request_id"]

        inspected = self.runtime.observe(
            session_id,
            [
                {
                    "kind": "code",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
                        '"args":{"filePath":"src/local_engine.py"}}\n'
                        "class LocalEngine: strategy = 'staged'"
                    ),
                    "source": "src/local_engine.py",
                }
            ],
            request_id=local_request_id,
        )
        self.assertEqual(inspected["next_action"]["type"], "reason")

        evidence_ids = ["ev1", "ev2", "ev3", "ev5"]
        accepted = self.runtime.complete(
            session_id,
            answer=(
                "Project Atlas uses staged execution, and the local workspace implements "
                "the same strategy in LocalEngine. Sources: https://agency.gov/atlas "
                "and https://independent.example/report."
            ),
            claims=[
                {
                    "claim": "Current sources describe staged execution.",
                    "evidence_ids": ["ev1", "ev2", "ev3"],
                },
                {
                    "claim": "The local workspace implements staged execution.",
                    "evidence_ids": ["ev5"],
                },
            ],
            hypotheses=[
                {
                    "statement": "The live sources and local workspace use one strategy.",
                    "falsification_test": "Compare the sources with the local engine.",
                    "status": "supported",
                    "evidence_ids": evidence_ids,
                }
            ],
            completion_evidence_ids=evidence_ids,
        )
        self.assertEqual(accepted["status"], "complete")

    def test_research_rejects_invalid_provenance_metadata(self) -> None:
        started = self.runtime.start("Research current API availability.")
        session_id = started["session"]["session_id"]

        with self.assertRaisesRegex(ValueError, "absolute http"):
            self.runtime.observe(
                session_id,
                [
                    {
                        "kind": "web",
                        "content": "result",
                        "url": "file:///tmp/result",
                    }
                ],
                request_id="req1",
            )
        with self.assertRaisesRegex(ValueError, "timezone"):
            self.runtime.observe(
                session_id,
                [
                    {
                        "kind": "web",
                        "content": "result",
                        "url": "https://example.com/result",
                        "retrieved_at": "2026-07-26T12:00:00",
                    }
                ],
                request_id="req1",
            )

    def test_research_with_a_requested_report_is_not_document_synthesis(self) -> None:
        started = self.runtime.start(
            "Research the latest stable Python release and report contradictions.",
            effort="quick",
        )

        self.assertEqual(started["session"]["task_kind"], "research")
        self.assertEqual(started["session"]["deliverable"], "research_answer")
        self.assertEqual(
            started["next_action"]["request"]["parameters"]["purpose"],
            "contradiction_check",
        )

    def test_latest_release_requires_cross_origin_version_consensus(self) -> None:
        started = self.runtime.start(
            "Research the latest released uv version from current live sources.",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        retrieved_at = datetime.now(UTC).isoformat()
        searched = self.runtime.observe(
            session_id,
            [
                {
                    "kind": "web",
                    "content": "Release 0.11.32 · astral-sh/uv · GitHub",
                    "source": "https://github.com/astral-sh/uv/releases/latest",
                    "url": "https://github.com/astral-sh/uv/releases/latest",
                    "retrieved_at": retrieved_at,
                    "published_at": "2026-07-23",
                    "purpose": "contradiction_check",
                },
                {
                    "kind": "web",
                    "content": "package-header__name uv 0.11.32",
                    "source": "https://pypi.org/project/uv/",
                    "url": "https://pypi.org/project/uv/",
                    "retrieved_at": retrieved_at,
                    "published_at": "2026-07-23",
                    "purpose": "contradiction_check",
                },
            ],
            request_id="req1",
        )
        self.assertEqual(
            searched["next_action"]["request"]["parameters"]["purpose"],
            "primary_fetch",
        )
        fetched = self.runtime.observe(
            session_id,
            [
                {
                    "kind": "web",
                    "content": "Official release version 0.11.32.",
                    "source": "https://github.com/astral-sh/uv/releases/latest",
                    "url": "https://github.com/astral-sh/uv/releases/latest",
                    "retrieved_at": retrieved_at,
                    "published_at": "2026-07-23",
                    "purpose": "primary_fetch",
                }
            ],
            request_id="req2",
        )
        release = next(
            item
            for item in fetched["context"]["deterministic_derivations"]
            if item["operation"] == "release_version"
        )
        self.assertEqual(release["value"], "0.11.32")
        evidence_ids = ["ev1", "ev2", "ev3"]
        common = {
            "claims": [
                {
                    "claim": "The sources establish the current uv release.",
                    "evidence_ids": evidence_ids,
                }
            ],
            "hypotheses": [
                {
                    "statement": "The sources agree on the uv release.",
                    "falsification_test": "Compare the release labels.",
                    "status": "supported",
                    "evidence_ids": evidence_ids,
                }
            ],
            "completion_evidence_ids": evidence_ids,
        }
        rejected = self.runtime.complete(
            session_id,
            answer=(
                "The latest release is v52bc061. Sources: "
                "https://github.com/astral-sh/uv/releases/latest and "
                "https://pypi.org/project/uv/."
            ),
            **common,
        )
        alignment = next(
            item
            for item in rejected["verification"]["checks"]
            if item["name"] == "evidence_alignment"
        )
        self.assertFalse(alignment["passed"])
        self.assertIn("0.11.32", alignment["reason"])

        accepted = self.runtime.complete(
            session_id,
            answer=(
                "The latest release is 0.11.32. The origins agree and no material "
                "contradiction was found. Sources: "
                "https://github.com/astral-sh/uv/releases/latest and "
                "https://pypi.org/project/uv/."
            ),
            **common,
        )
        self.assertEqual(accepted["status"], "complete")

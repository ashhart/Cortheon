from __future__ import annotations

from cognitive_runtime_cases_common import RuntimeTestCase


class ReasoningMixin(RuntimeTestCase):
    def test_runtime_originates_competing_abductive_hypotheses(self) -> None:
        started = self.runtime.start(
            "Explain why activation fell and investigate competing hypotheses.",
            task_kind="general",
            effort="standard",
        )
        observed = self.runtime.observe(
            started["session"]["session_id"],
            [
                {
                    "kind": "analysis",
                    "content": (
                        "Activation fell only for accounts created during the "
                        "weekend migration window."
                    ),
                    "source": "host:cohort-analysis",
                    "status": "verified",
                }
            ],
            request_id="req1",
        )

        hypotheses = observed["context"]["hypotheses"]
        self.assertEqual(len(hypotheses), 2)
        self.assertTrue(all(item["origin"] == "substrate_abduction" for item in hypotheses))
        self.assertTrue(all(item["origin_evidence_ids"] == ["ev1"] for item in hypotheses))
        self.assertTrue(
            all(not item["supporting_evidence"] for item in hypotheses),
            "Candidate origination must not promote its clues to proof.",
        )
        self.assertEqual(observed["next_action"]["type"], "harness_tool")
        self.assertIn(
            observed["next_action"]["request"]["hypothesis_id"],
            {"h1", "h2"},
        )
        graph = observed["context"]["cognitive_graph"]
        self.assertEqual(
            sum(edge["relation"] == "inspired_candidate" for edge in graph["edges"]),
            2,
        )
        self.assertEqual(self.runtime.metrics["hypotheses_originated"], 2)

    def test_runtime_does_not_originate_hypotheses_for_simple_lookup(self) -> None:
        started = self.runtime.start(
            "Report the configured activation threshold.",
            task_kind="general",
            effort="standard",
        )
        observed = self.runtime.observe(
            started["session"]["session_id"],
            [
                {
                    "kind": "documentation",
                    "content": "The activation threshold is 0.72.",
                    "source": "host:settings",
                    "status": "verified",
                }
            ],
            request_id="req1",
        )

        self.assertEqual(observed["context"]["hypotheses"], [])
        self.assertEqual(
            observed["next_action"]["required_fields"],
            ["hypotheses"],
        )

    def test_read_only_abductive_clues_are_not_misclassified_as_code_by_test_word(
        self,
    ) -> None:
        started = self.runtime.start(
            "Connect the separate clues to explain the sensor dropout. Generate "
            "competing explanations and state a test that could disprove the strongest "
            "one. Do not modify files.",
            effort="quick",
        )

        self.assertEqual(started["session"]["task_kind"], "documents")
        self.assertEqual(started["session"]["deliverable"], "document_synthesis")
        self.assertEqual(
            started["next_action"]["request"]["parameters"]["operation"],
            "document_discovery",
        )

    def test_read_only_causal_language_without_hypothesis_word_is_document_synthesis(
        self,
    ) -> None:
        for goal in (
            "Explain the mismatch, evaluate an alternative, and give a falsification "
            "test. Do not modify files.",
            "Derive the access failure across the records and state a test that would "
            "disprove it. Do not modify files.",
        ):
            with self.subTest(goal=goal):
                started = self.runtime.start(goal, effort="quick")
                self.assertEqual(started["session"]["task_kind"], "documents")
                self.assertEqual(
                    started["session"]["deliverable"],
                    "document_synthesis",
                )

    def test_evidence_only_closure_is_terminal_without_certifying_an_answer(
        self,
    ) -> None:
        started = self.runtime.start(
            "Synthesize the competing records and give a falsification test. Do not modify files.",
            effort="quick",
        )
        observed = self.runtime.observe(
            started["session"]["session_id"],
            [
                {
                    "kind": "documentation",
                    "content": "Record A establishes the live cohort boundary.",
                    "source": "host:record-a",
                    "status": "verified",
                }
            ],
            request_id=started["next_action"]["request"]["request_id"],
        )

        closed = self.runtime.close_evidence(observed["session"]["session_id"])

        self.assertEqual(closed["status"], "evidence_closed")
        self.assertFalse(closed["answer_certified"])
        self.assertTrue(closed["discarded"])
        self.assertFalse(closed["retained_project_data"])
        self.assertEqual(self.runtime.active_sessions, 0)
        self.assertEqual(self.runtime.metrics["sessions_evidence_closed"], 1)
        self.assertEqual(self.runtime.metrics["sessions_completed"], 0)

    def test_abductive_completion_accepts_bounded_inference_without_claiming_deduction(
        self,
    ) -> None:
        paths_and_facts = [
            (
                "cohort_notes.md",
                "Activation fell only for weekend migration accounts.",
            ),
            (
                "routing_map.md",
                "Weekend migrations use the legacy token broker.",
            ),
            (
                "capacity_limits.md",
                "The legacy broker rejects above 500; migration bursts reach 900.",
            ),
        ]
        started = self.runtime.start(
            "Read cohort_notes.md, routing_map.md, and capacity_limits.md. "
            "Infer why activation fell, compare competing hypotheses, and state one "
            "observation that would falsify the best explanation.",
            effort="quick",
        )
        observed = self.runtime.observe(
            started["session"]["session_id"],
            [
                {
                    "kind": "documentation",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
                        f'"args":{{"filePath":"{path}"}}}}\n{fact}'
                    ),
                    "source": f"opencode:read:{path}",
                    "status": "verified",
                }
                for path, fact in paths_and_facts
            ],
            request_id="req1",
        )
        self.assertEqual(observed["next_action"]["type"], "reason")

        evidence_ids = observed["accepted_evidence_ids"]
        answer = (
            "The strongest explanation is legacy-broker overload because weekend "
            "migrations route through that broker and bursts of 900 exceed its 500 "
            "limit. A competing alternative is a weekend-only measurement artifact. "
            "Falsify the leading explanation by finding a 900-request weekend burst "
            "through the legacy broker with normal activation."
        )
        completed = self.runtime.complete(
            started["session"]["session_id"],
            answer=answer,
            claims=[{"claim": answer, "evidence_ids": evidence_ids}],
            hypotheses=[
                {
                    "statement": "Legacy-broker overload explains the fall.",
                    "falsification_test": (
                        "Find a 900-request weekend burst with normal activation."
                    ),
                    "status": "supported",
                    "evidence_ids": evidence_ids,
                },
                {
                    "statement": "A weekend-only measurement artifact explains the fall.",
                    "falsification_test": "Compare an independent activation measure.",
                    "status": "uncertain",
                    "evidence_ids": evidence_ids,
                },
            ],
            completion_evidence_ids=evidence_ids,
        )

        self.assertEqual(completed["status"], "complete")
        self.assertEqual(completed["answer"], answer)

    def test_ambiguity_completion_certifies_clarification_instead_of_guessing(
        self,
    ) -> None:
        paths_and_facts = [
            ("product.md", "Atlas Portal is the customer analytics application."),
            ("platform.md", "Atlas Pipeline is the ingestion service."),
            (
                "request.md",
                "The request says deploy Atlas without a component or environment.",
            ),
        ]
        started = self.runtime.start(
            "Read product.md, platform.md, and request.md. Preserve ambiguity rather "
            "than guessing, enumerate the viable interpretations, and ask the smallest "
            "clarification needed for 'deploy Atlas'.",
            effort="quick",
        )
        observed = self.runtime.observe(
            started["session"]["session_id"],
            [
                {
                    "kind": "documentation",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
                        f'"args":{{"filePath":"{path}"}}}}\n{fact}'
                    ),
                    "source": f"pi:read:{path}",
                    "status": "verified",
                }
                for path, fact in paths_and_facts
            ],
            request_id="req1",
        )
        self.assertEqual(observed["next_action"]["type"], "reason")

        evidence_ids = observed["accepted_evidence_ids"]
        answer = (
            "The request is ambiguous: Atlas could mean Atlas Portal or Atlas Pipeline, "
            "so neither deployment is justified. Clarify which component and environment "
            "the requester intends."
        )
        completed = self.runtime.complete(
            started["session"]["session_id"],
            answer=answer,
            claims=[{"claim": answer, "evidence_ids": evidence_ids}],
            hypotheses=[
                {
                    "statement": "The request refers to Atlas Portal.",
                    "falsification_test": "Ask which component is intended.",
                    "status": "uncertain",
                    "evidence_ids": evidence_ids,
                },
                {
                    "statement": "The request refers to Atlas Pipeline.",
                    "falsification_test": "Ask which component is intended.",
                    "status": "supported",
                    "evidence_ids": evidence_ids,
                },
            ],
            completion_evidence_ids=evidence_ids,
        )

        self.assertEqual(completed["status"], "complete")
        self.assertEqual(completed["answer"], answer)

    def test_unnamed_ambiguity_with_no_mutation_uses_document_discovery(
        self,
    ) -> None:
        started = self.runtime.start(
            "Assess the request to 'optimize latency'. Enumerate the incompatible "
            "metrics and ask the minimum clarification needed before changing "
            "anything. Do not modify files.",
            effort="quick",
        )

        self.assertEqual(started["session"]["task_kind"], "documents")
        self.assertEqual(started["session"]["deliverable"], "document_synthesis")
        request = started["next_action"]["request"]
        self.assertEqual(request["capability"], "search")
        self.assertEqual(
            request["parameters"]["operation"],
            "document_discovery",
        )

    def test_joining_language_uses_native_document_discovery(self) -> None:
        started = self.runtime.start(
            "Explain the anomaly by joining the cohort, credit, conversion, and "
            "control records. Reject an alternative and give a falsification test. "
            "Do not modify files.",
            effort="quick",
        )

        request = started["next_action"]["request"]
        self.assertEqual(started["session"]["deliverable"], "document_synthesis")
        self.assertEqual(request["capability"], "search")
        self.assertEqual(request["parameters"]["operation"], "document_discovery")

    def test_cross_document_language_uses_native_document_discovery(self) -> None:
        started = self.runtime.start(
            "Resolve the aliases, show the cross-document mechanism, compare an "
            "alternative, and give a falsification test. Do not modify files.",
            effort="quick",
        )

        request = started["next_action"]["request"]
        self.assertEqual(request["capability"], "search")
        self.assertEqual(request["parameters"]["operation"], "document_discovery")

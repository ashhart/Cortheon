from __future__ import annotations

from cognitive_runtime_cases_common import RuntimeTestCase


class SemanticRulesMixin(RuntimeTestCase):
    def test_semantic_conflict_stops_after_bounded_scoped_nulls(self) -> None:
        facts = [
            ("service_catalog.md", "Checkout is change class Coral."),
            (
                "change_policy.md",
                "Coral changes require approval from the Duty Security Officer.",
            ),
            ("old_directory.md", "Duty Security Officer: Amara Okafor"),
            ("new_directory.md", "Duty Security Officer: Lin Wei"),
        ]
        started = self.runtime.start(
            "Read service_catalog.md, change_policy.md, old_directory.md, and "
            "new_directory.md as separate documents. Who currently approves Checkout?",
            effort="quick",
        )
        result = self.runtime.observe(
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
                for path, fact in facts
            ],
            request_id="req1",
        )

        for round_number in (1, 2):
            request = result["next_action"]["request"]
            result = self.runtime.observe(
                started["session"]["session_id"],
                [
                    {
                        "kind": "documentation",
                        "content": (
                            '[CORTHEON_HOST_EVIDENCE] {"tool":"read",'
                            '"outcome":"result",'
                            f'"args":{{"filePath":"scoped_search_{round_number}.txt"}}}}\n'
                            f"Scoped search {round_number} found no current mapping."
                        ),
                        "source": f"codex:read:scoped_search_{round_number}.txt",
                        "status": "verified",
                    }
                ],
                request_id=request["request_id"],
            )

        self.assertEqual(result["next_action"]["type"], "finish")
        self.assertIn(
            "disambiguation budget is exhausted",
            result["next_action"]["instruction"],
        )
        self.assertIn("doom loop", result["guidance"])

    def test_semantic_join_applies_a_rule_only_when_all_documented_conditions_hold(
        self,
    ) -> None:
        facts = [
            ("data_profile.md", "Kepler processes biometric templates."),
            ("market_scope.md", "Kepler serves EEA residents."),
            (
                "privacy_policy.md",
                "Systems that process biometric templates and serve EEA residents "
                "require approval from the Data Protection Officer.",
            ),
            ("governance_roster.md", "Data Protection Officer: Noor Patel"),
        ]
        started = self.runtime.start(
            "Read data_profile.md, market_scope.md, privacy_policy.md, and "
            "governance_roster.md separately. Who must approve Kepler, and why?",
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
                for path, fact in facts
            ],
            request_id="req1",
        )

        derivation = observed["context"]["deterministic_derivations"][0]
        self.assertEqual(derivation["operation"], "semantic_rule")
        self.assertEqual(
            derivation["nodes"],
            [
                "Kepler",
                "biometric templates",
                "EEA residents",
                "Data Protection Officer",
                "Noor Patel",
            ],
        )
        self.assertEqual(len(derivation["premises"]), 2)
        insight = observed["cognition"]["derived_insights"][0]
        self.assertEqual(insight["applied_rule"]["conclusion"], "Data Protection Officer")
        self.assertTrue(insight["novel_cross_source_inference"])
        answer = (
            "Kepler processes biometric templates and serves EEA residents. "
            "The privacy policy therefore requires the Data Protection Officer, "
            "Noor Patel, to approve Kepler."
        )
        completed = self.runtime.complete(
            started["session"]["session_id"],
            answer=answer,
            claims=[
                {
                    "claim": answer,
                    "evidence_ids": observed["accepted_evidence_ids"],
                }
            ],
            hypotheses=[
                {
                    "statement": "All policy conditions hold, so Noor Patel must approve.",
                    "falsification_test": (
                        "Check whether either condition or the role mapping is absent."
                    ),
                    "status": "supported",
                    "evidence_ids": observed["accepted_evidence_ids"],
                }
            ],
            completion_evidence_ids=observed["accepted_evidence_ids"],
        )
        self.assertEqual(completed["status"], "complete")

    def test_semantic_join_does_not_fire_a_conjunctive_rule_on_a_near_miss(self) -> None:
        facts = [
            ("data_profile.md", "Kepler processes biometric templates."),
            ("market_scope.md", "Kepler serves US residents."),
            (
                "privacy_policy.md",
                "Systems that process biometric templates and serve EEA residents "
                "require approval from the Data Protection Officer.",
            ),
            ("governance_roster.md", "Data Protection Officer: Noor Patel"),
        ]
        started = self.runtime.start(
            "Read data_profile.md, market_scope.md, privacy_policy.md, and "
            "governance_roster.md separately. Who must approve Kepler, and why?",
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
                for path, fact in facts
            ],
            request_id="req1",
        )

        self.assertEqual(
            observed["context"]["deterministic_derivations"],
            [],
        )
        self.assertEqual(observed["cognition"]["derived_insights"], [])
        unsupported = self.runtime.complete(
            started["session"]["session_id"],
            answer=(
                "Kepler requires the Data Protection Officer, Noor Patel, because "
                "it processes biometric templates and serves EEA residents."
            ),
            claims=[
                {
                    "claim": "Noor Patel must approve Kepler.",
                    "evidence_ids": observed["accepted_evidence_ids"],
                }
            ],
            hypotheses=[
                {
                    "statement": "Both policy conditions hold.",
                    "falsification_test": "Check the market-scope document.",
                    "status": "supported",
                    "evidence_ids": observed["accepted_evidence_ids"],
                }
            ],
            completion_evidence_ids=observed["accepted_evidence_ids"],
        )
        self.assertEqual(unsupported["verification"]["verdict"], "needs_evidence")
        self.assertTrue(
            any(
                "do not yield a deterministic evidence-constrained derivation" in gap
                for gap in unsupported["verification"]["gaps"]
            )
        )

    def test_semantic_join_requires_the_answer_to_draw_from_every_document(self) -> None:
        started = self.runtime.start(
            "Read service_catalog.md, change_policy.md, and org_directory.md as "
            "separate documents. Who approves the Checkout rollback?",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        facts = [
            (
                "service_catalog.md",
                "Search is classified as change class Indigo.\n"
                "Checkout is classified as change class Coral.\n"
                "Messaging is classified as change class Silver.",
            ),
            (
                "change_policy.md",
                "Indigo changes require approval from the Product Director.\n"
                "Coral changes require approval from the Duty Security Officer.\n"
                "Silver changes require approval from the Communications Lead.",
            ),
            (
                "org_directory.md",
                "Product Director: Lin.\n"
                "Duty Security Officer: Amara.\n"
                "Communications Lead: Javier.",
            ),
        ]
        observations = [
            {
                "kind": "documentation",
                "content": (
                    '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
                    f'"args":{{"filePath":"{path}"}}}}\n{fact}'
                ),
                "source": f"opencode:read:{path}",
                "status": "verified",
            }
            for path, fact in facts
        ]
        observed = self.runtime.observe(
            session_id,
            observations,
            request_id="req1",
        )
        evidence_ids = observed["accepted_evidence_ids"]
        derivation = observed["context"]["deterministic_derivations"][0]
        self.assertEqual(
            derivation["nodes"],
            [
                "Checkout",
                "Coral",
                "Duty Security Officer",
                "Amara",
            ],
        )
        self.assertEqual(
            derivation["sources"],
            [
                "service_catalog.md",
                "change_policy.md",
                "org_directory.md",
            ],
        )
        completion = {
            "claims": [
                {
                    "claim": "The three documents establish the approval chain.",
                    "evidence_ids": evidence_ids,
                }
            ],
            "hypotheses": [
                {
                    "statement": "Amara is the required approver.",
                    "falsification_test": "Trace Checkout through its class and role owner.",
                    "status": "supported",
                    "evidence_ids": evidence_ids,
                }
            ],
            "completion_evidence_ids": evidence_ids,
        }

        rejected = self.runtime.complete(
            session_id,
            answer="The relevant change class is Coral.",
            **completion,
        )
        alignment = next(
            item
            for item in rejected["verification"]["checks"]
            if item["name"] == "evidence_alignment"
        )
        self.assertFalse(alignment["passed"])
        self.assertIn("deterministic cross-document chain", alignment["reason"])
        self.assertIn("Amara", alignment["reason"])

        overbroad = self.runtime.complete(
            session_id,
            answer=(
                "Checkout is Coral; Coral requires the Duty Security Officer, Amara. "
                "The Product Director Lin and Communications Lead Javier also approve."
            ),
            **completion,
        )
        alignment = next(
            item
            for item in overbroad["verification"]["checks"]
            if item["name"] == "evidence_alignment"
        )
        self.assertFalse(alignment["passed"])
        self.assertIn("unrelated branches", alignment["reason"])

        accepted = self.runtime.complete(
            session_id,
            answer=(
                "Checkout is Coral; Coral requires the Duty Security Officer, "
                "and the directory names Amara, so Amara approves."
            ),
            **completion,
        )
        self.assertEqual(accepted["status"], "complete")

    def test_semantic_join_fails_closed_for_unrecognized_relations(self) -> None:
        started = self.runtime.start(
            "Read first.md, second.md, and third.md as separate documents. Who owns the result?",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        facts = [
            ("first.md", "Alpha dances beside Beta."),
            ("second.md", "Beta dreams about Gamma."),
            ("third.md", "Gamma occasionally greets Dana."),
        ]
        observations = [
            {
                "kind": "documentation",
                "content": (
                    '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
                    f'"args":{{"filePath":"{path}"}}}}\n{fact}'
                ),
                "source": f"opencode:read:{path}",
                "status": "verified",
            }
            for path, fact in facts
        ]
        observed = self.runtime.observe(
            session_id,
            observations,
            request_id="req1",
        )
        evidence_ids = observed["accepted_evidence_ids"]

        result = self.runtime.complete(
            session_id,
            answer="Alpha leads to Beta, Gamma, and Dana.",
            claims=[
                {
                    "claim": "The documents establish the chain.",
                    "evidence_ids": evidence_ids,
                }
            ],
            hypotheses=[
                {
                    "statement": "Dana is the owner.",
                    "falsification_test": "Trace every relation from Alpha to Dana.",
                    "status": "supported",
                    "evidence_ids": evidence_ids,
                }
            ],
            completion_evidence_ids=evidence_ids,
        )
        alignment = next(
            item
            for item in result["verification"]["checks"]
            if item["name"] == "evidence_alignment"
        )
        self.assertFalse(alignment["passed"])
        self.assertIn("will not substitute lexical overlap", alignment["reason"])

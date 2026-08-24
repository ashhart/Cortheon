from __future__ import annotations

from cognitive_runtime_cases_common import RuntimeTestCase

from cortheon.cognitive_runtime import (
    CognitiveRuntime,
)


class SemanticRelationsMixin(RuntimeTestCase):
    def test_semantic_join_supports_dependency_and_risk_chains(self) -> None:
        cases = [
            (
                (
                    "Read dependency_map.md, asset_register.md, and oncall_roster.md "
                    "as separate documents. Which current responder owns the blocker "
                    "preventing Atlas ingestion from resuming?"
                ),
                [
                    (
                        "dependency_map.md",
                        "Atlas ingestion cannot resume until certificate Nimbus is renewed.",
                    ),
                    (
                        "asset_register.md",
                        "Certificate Nimbus belongs to Identity Platform.",
                    ),
                    ("oncall_roster.md", "Identity Platform: Elena Voss"),
                ],
                [
                    "Atlas ingestion",
                    "certificate Nimbus",
                    "Identity Platform",
                    "Elena Voss",
                ],
            ),
            (
                (
                    "Read release_brief.md, risk_policy.md, and regional_roles.md as "
                    "separate documents. Who authorizes Project Aster's EMEA launch?"
                ),
                [
                    (
                        "release_brief.md",
                        "Project Aster requires launch authorization for risk band Saffron.",
                    ),
                    (
                        "risk_policy.md",
                        "Saffron releases are approved by the Regional Risk Delegate.",
                    ),
                    (
                        "regional_roles.md",
                        "Regional Risk Delegate: Kwame Mensah",
                    ),
                ],
                [
                    "Project Aster",
                    "Saffron",
                    "Regional Risk Delegate",
                    "Kwame Mensah",
                ],
            ),
        ]
        for goal, facts, expected_nodes in cases:
            with self.subTest(expected=expected_nodes[-1]):
                runtime = CognitiveRuntime()
                started = runtime.start(goal, effort="quick")
                observations = [
                    {
                        "kind": "documentation",
                        "content": (
                            '[CORTHEON_HOST_EVIDENCE] {"tool":"read",'
                            '"outcome":"result",'
                            f'"args":{{"filePath":"{path}"}}}}\n{fact}'
                        ),
                        "source": f"pi:read:{path}",
                        "status": "verified",
                    }
                    for path, fact in facts
                ]
                observed = runtime.observe(
                    started["session"]["session_id"],
                    observations,
                    request_id="req1",
                )
                derivation = observed["context"]["deterministic_derivations"][0]
                self.assertEqual(derivation["nodes"], expected_nodes)
                cognition = observed["cognition"]
                self.assertEqual(cognition["stage"], "connect")
                insight = cognition["derived_insights"][0]
                self.assertEqual(insight["kind"], "cross_source_inference")
                self.assertEqual(insight["source_count"], 3)
                self.assertTrue(insight["novel_cross_source_inference"])
                self.assertEqual(insight["status"], "candidate_until_challenged")
                self.assertEqual(
                    [edge["from"] for edge in insight["chain"]] + [insight["chain"][-1]["to"]],
                    expected_nodes,
                )
                self.assertFalse(
                    any(insight["statement"] in fact for _path, fact in facts),
                    "The derived conclusion must not merely copy a source.",
                )

    def test_semantic_join_resolves_explicit_aliases_and_entity_type_variants(self) -> None:
        facts = [
            (
                "service_index.md",
                "Payments API is internally called Lantern.",
            ),
            (
                "dependency_notes.md",
                "Service Lantern depends on Queue Zephyr.",
            ),
            (
                "duty_roster.md",
                "Current owner for Queue Zephyr is Rina Sol.",
            ),
        ]
        started = self.runtime.start(
            "Read service_index.md, dependency_notes.md, and duty_roster.md as "
            "separate documents. Who currently owns the Payments API dependency?",
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
                    "source": f"codex:read:{path}",
                    "status": "verified",
                }
                for path, fact in facts
            ],
            request_id="req1",
        )

        derivation = observed["context"]["deterministic_derivations"][0]
        self.assertEqual(
            derivation["nodes"],
            ["Payments API", "Lantern", "Queue Zephyr", "Rina Sol"],
        )
        self.assertEqual(
            derivation["relations"],
            ["alias", "dependency", "ownership"],
        )
        self.assertTrue(
            observed["cognition"]["derived_insights"][0]["novel_cross_source_inference"]
        )

    def test_semantic_join_connects_markdown_tables_across_documents(self) -> None:
        facts = [
            (
                "product_registry.md",
                "| Product | Runtime service |\n"
                "| --- | --- |\n"
                "| Order Console | Helios |\n"
                "| Search Console | Boreal |",
            ),
            (
                "integration_matrix.md",
                "| Service | Critical dataset |\n"
                "| --- | --- |\n"
                "| Helios | Ledger Aurora |\n"
                "| Boreal | Index Cobalt |",
            ),
            (
                "data_stewards.md",
                "| Dataset | Current steward |\n"
                "| --- | --- |\n"
                "| Ledger Aurora | Imani Brooks |\n"
                "| Index Cobalt | Pavel Novak |",
            ),
        ]
        started = self.runtime.start(
            "Read product_registry.md, integration_matrix.md, and data_stewards.md "
            "as separate documents. Who is the current steward of the critical "
            "dataset used by Order Console?",
            effort="quick",
        )
        observed = self.runtime.observe(
            started["session"]["session_id"],
            [
                {
                    "kind": "documentation",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
                        f'"args":{{"filePath":"{path}"}}}}\n{content}'
                    ),
                    "source": f"codex:read:{path}",
                    "status": "verified",
                }
                for path, content in facts
            ],
            request_id="req1",
        )

        derivation = observed["context"]["deterministic_derivations"][0]
        self.assertEqual(
            derivation["nodes"],
            ["Order Console", "Helios", "Ledger Aurora", "Imani Brooks"],
        )
        self.assertEqual(
            derivation["relations"],
            ["mapping", "dependency", "ownership"],
        )
        self.assertEqual(derivation["sources"], [item[0] for item in facts])
        self.assertIn(
            "Pavel Novak",
            derivation["exclude_unless_explicitly_negated"],
        )
        insight = observed["cognition"]["derived_insights"][0]
        self.assertTrue(insight["novel_cross_source_inference"])
        self.assertEqual(insight["source_count"], 3)

    def test_semantic_table_join_prefers_explicit_current_assignment(self) -> None:
        facts = [
            (
                "product_registry.md",
                "| Product | Dataset |\n| --- | --- |\n| Order Console | Ledger Aurora |",
            ),
            (
                "archived_stewards.md",
                "| Dataset | Steward |\n| --- | --- |\n| Ledger Aurora | Pavel Novak |",
            ),
            (
                "current_stewards.md",
                "| Dataset | Current steward |\n| --- | --- |\n| Ledger Aurora | Imani Brooks |",
            ),
        ]
        started = self.runtime.start(
            "Read product_registry.md, archived_stewards.md, and "
            "current_stewards.md. Who currently stewards the Order Console dataset?",
            effort="quick",
        )
        observed = self.runtime.observe(
            started["session"]["session_id"],
            [
                {
                    "kind": "documentation",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
                        f'"args":{{"filePath":"{path}"}}}}\n{content}'
                    ),
                    "source": f"codex:read:{path}",
                    "status": "verified",
                }
                for path, content in facts
            ],
            request_id="req1",
        )

        derivation = observed["context"]["deterministic_derivations"][0]
        self.assertEqual(
            derivation["nodes"],
            ["Order Console", "Ledger Aurora", "Imani Brooks"],
        )
        self.assertNotIn("Pavel Novak", derivation["nodes"])

    def test_semantic_conflict_replans_then_accepts_explicit_current_authority(self) -> None:
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
        conflicted = self.runtime.observe(
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

        self.assertEqual(conflicted["cognition"]["stage"], "challenge")
        conflict = conflicted["cognition"]["derived_insights"][0]
        self.assertEqual(conflict["kind"], "cross_source_conflict")
        self.assertEqual(conflict["status"], "requires_disambiguation")
        request = conflicted["next_action"]["request"]
        self.assertEqual(request["parameters"]["operation"], "semantic_disambiguation")
        self.assertEqual(request["parameters"]["tool_call_budget"], 1)

        resolved = self.runtime.observe(
            started["session"]["session_id"],
            [
                {
                    "kind": "documentation",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
                        '"args":{"filePath":"current_authority.md"}}\n'
                        "Current owner for Duty Security Officer is Amara Okafor."
                    ),
                    "source": "codex:read:current_authority.md",
                    "status": "verified",
                }
            ],
            request_id=request["request_id"],
        )

        derivation = resolved["context"]["deterministic_derivations"][0]
        self.assertEqual(derivation["operation"], "semantic_chain")
        self.assertEqual(derivation["nodes"][-1], "Amara Okafor")
        self.assertIn("current_authority.md", derivation["sources"])
        self.assertEqual(resolved["cognition"]["stage"], "connect")

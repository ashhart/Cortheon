from __future__ import annotations

from cognitive_runtime_cases_common import RuntimeTestCase

from cortheon.cognitive_runtime import (
    CognitiveRuntime,
)


class DiscoveryMixin(RuntimeTestCase):
    def test_document_discovery_narrows_search_then_joins_only_relevant_sources(
        self,
    ) -> None:
        started = self.runtime.start(
            "Search across project documents and connect the evidence to determine "
            "the steward of the critical dataset used by Order Console.",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        discovery = started["next_action"]["request"]
        self.assertEqual(started["session"]["task_kind"], "documents")
        self.assertEqual(discovery["capability"], "search")
        self.assertEqual(
            discovery["parameters"]["operation"],
            "document_discovery",
        )

        narrowed = self.runtime.observe(
            session_id,
            [
                {
                    "kind": "documentation",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"grep","outcome":"match",'
                        '"args":{"command":"bounded project document search"}}\n'
                        "docs/product_registry.md:3: Order Console uses Helios.\n"
                        "docs/integration_matrix.md:3: Helios uses the critical "
                        "dataset Ledger Aurora.\n"
                        "docs/data_stewards.md:3: Ledger Aurora steward Imani Brooks.\n"
                        "README.md:1: General project documents and installation."
                    ),
                    "source": "opencode:grep:documents",
                    "status": "verified",
                }
            ],
            request_id=discovery["request_id"],
        )
        read_request = narrowed["next_action"]["request"]
        self.assertEqual(read_request["capability"], "read_many")
        self.assertEqual(read_request["parameters"]["operation"], "semantic_join")
        self.assertTrue(read_request["parameters"]["discovered"])
        requested_paths = read_request["parameters"]["paths"]
        self.assertTrue(
            {
                "docs/product_registry.md",
                "docs/integration_matrix.md",
                "docs/data_stewards.md",
            }.issubset(requested_paths)
        )
        self.assertIn("README.md", requested_paths)

        contents = {
            "docs/product_registry.md": (
                "| Product | Runtime service |\n| --- | --- |\n| Order Console | Helios |"
            ),
            "docs/integration_matrix.md": (
                "| Service | Critical dataset |\n| --- | --- |\n| Helios | Ledger Aurora |"
            ),
            "docs/data_stewards.md": (
                "| Dataset | Steward |\n| --- | --- |\n| Ledger Aurora | Imani Brooks |"
            ),
            "README.md": "# Installation\nRun the package installer.",
        }
        connected = self.runtime.observe(
            session_id,
            [
                {
                    "kind": "documentation",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
                        f'"args":{{"filePath":"{path}"}}}}\n{contents[path]}'
                    ),
                    "source": f"opencode:read:{path}",
                    "status": "verified",
                }
                for path in requested_paths
            ],
            request_id=read_request["request_id"],
        )
        derivation = connected["context"]["deterministic_derivations"][0]
        self.assertEqual(
            derivation["nodes"],
            ["Order Console", "Helios", "Ledger Aurora", "Imani Brooks"],
        )
        self.assertNotIn("README.md", derivation["sources"])

        relevant_ids = [
            evidence_id
            for path, evidence_id in zip(
                requested_paths,
                connected["accepted_evidence_ids"],
                strict=True,
            )
            if path != "README.md"
        ]
        completed = self.runtime.complete(
            session_id,
            answer=(
                "Order Console maps to Helios, which uses Ledger Aurora; "
                "Imani Brooks stewards Ledger Aurora."
            ),
            claims=[
                {
                    "claim": (
                        "Order Console maps to Helios, which uses Ledger Aurora, "
                        "stewarded by Imani Brooks."
                    ),
                    "evidence_ids": relevant_ids,
                }
            ],
            hypotheses=[
                {
                    "statement": "Imani Brooks stewards the required dataset.",
                    "falsification_test": "Check the discovered stewardship table.",
                    "status": "supported",
                    "evidence_ids": relevant_ids,
                }
            ],
            completion_evidence_ids=relevant_ids,
        )
        self.assertEqual(completed["status"], "complete")

    def test_document_discovery_stops_after_two_scoped_nulls(self) -> None:
        started = self.runtime.start(
            "Search across project documents and connect the records needed to "
            "identify the launch approver.",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        request = started["next_action"]["request"]
        for round_number in (1, 2):
            result = self.runtime.observe(
                session_id,
                [
                    {
                        "kind": "documentation",
                        "content": (
                            '[CORTHEON_HOST_EVIDENCE] {"tool":"grep",'
                            '"outcome":"no_match",'
                            f'"args":{{"round":{round_number}}}}}\n'
                            "No matching project documents were found."
                        ),
                        "source": f"opencode:grep:round-{round_number}",
                        "status": "verified",
                    }
                ],
                request_id=request["request_id"],
            )
            if round_number == 1:
                request = result["next_action"]["request"]
                self.assertEqual(
                    request["parameters"]["discovery_round"],
                    2,
                )

        self.assertEqual(result["session"]["phase"], "inconclusive")
        self.assertEqual(result["next_action"]["type"], "finish")
        self.assertIn("scoped null", result["next_action"]["instruction"])

    def test_document_discovery_rejects_absolute_and_traversal_candidates(self) -> None:
        started = self.runtime.start(
            "Search across project documents and connect the records needed to "
            "identify the launch approver.",
            effort="quick",
        )
        result = self.runtime.observe(
            started["session"]["session_id"],
            [
                {
                    "kind": "documentation",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"find","outcome":"match",'
                        '"args":{"pattern":"*.md"}}\n'
                        "/etc/private_policy.md\n"
                        "../../secrets.md\n"
                        "https://example.com/remote_policy.md\n"
                        "docs/release_policy.md: Saffron launch approval"
                    ),
                    "source": "opencode:find:documents",
                    "status": "verified",
                }
            ],
            request_id="req1",
        )

        retry = result["next_action"]["request"]
        self.assertEqual(retry["parameters"]["discovery_round"], 2)
        self.assertNotEqual(retry["capability"], "read_many")

    def test_code_discovery_prioritizes_implementation_and_test(self) -> None:
        started = self.runtime.start(
            "Fix the parser's empty-input behavior and run the focused test.",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        discovery = started["next_action"]["request"]
        self.assertEqual(started["session"]["task_kind"], "code")
        self.assertEqual(discovery["capability"], "search")
        self.assertEqual(discovery["parameters"]["operation"], "code_discovery")
        self.assertTrue(discovery["parameters"]["prefer_tests"])

        narrowed = self.runtime.observe(
            session_id,
            [
                {
                    "kind": "code",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"grep","outcome":"match",'
                        '"args":{"command":"bounded code search"}}\n'
                        "src/parser.py:18: def parse(value):\n"
                        "tests/test_parser.py:8: def test_empty_input():\n"
                        "src/unrelated.py:1: def helper():"
                    ),
                    "source": "opencode:grep:code",
                    "status": "verified",
                }
            ],
            request_id=discovery["request_id"],
        )

        request = narrowed["next_action"]["request"]
        self.assertEqual(request["capability"], "read_many")
        self.assertEqual(request["parameters"]["operation"], "code_context")
        self.assertTrue(request["parameters"]["discovered"])
        self.assertEqual(
            request["parameters"]["paths"][:2],
            ["src/parser.py", "tests/test_parser.py"],
        )

    def test_cooperative_search_receipt_advances_code_discovery(self) -> None:
        runtime = CognitiveRuntime(require_host_receipts=True)
        started = runtime.start(
            "Extend the paired benchmark and add a regression test.",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        discovery = started["next_action"]["request"]

        narrowed = runtime.observe(
            session_id,
            [
                {
                    "kind": "code",
                    "content": (
                        '[CORTHEON_HOST_EVIDENCE] {"tool":"search",'
                        '"outcome":"match","args":{"command":'
                        "\"rg -n 'host|benchmark' src tests\"}}\n"
                        "src/cortheon/cognitive_benchmark.py:2458: host dispatch\n"
                        "tests/test_cognitive_benchmark.py:610: benchmark CLI"
                    ),
                    "source": "codex:search:benchmark-surface",
                    "status": "verified",
                }
            ],
            request_id=discovery["request_id"],
        )

        request = narrowed["next_action"]["request"]
        self.assertEqual(request["capability"], "read_many")
        self.assertEqual(request["parameters"]["operation"], "code_context")
        self.assertEqual(
            request["parameters"]["paths"][:2],
            [
                "src/cortheon/cognitive_benchmark.py",
                "tests/test_cognitive_benchmark.py",
            ],
        )

    def test_unnamed_correction_is_classified_as_a_code_change(self) -> None:
        started = self.runtime.start(
            "An arithmetic component has a multiplication defect. Locate the "
            "implementation and focused test, then make the smallest correction.",
            effort="quick",
        )

        self.assertEqual(started["session"]["task_kind"], "code")
        self.assertEqual(started["session"]["deliverable"], "code_change")
        discovery = started["next_action"]["request"]
        self.assertEqual(discovery["parameters"]["operation"], "code_discovery")
        self.assertTrue(discovery["parameters"]["prefer_tests"])

    def test_code_discovery_stops_without_an_implementation_test_pair(self) -> None:
        started = self.runtime.start(
            "Fix the parser's empty-input behavior and run the focused test.",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        request = started["next_action"]["request"]
        for round_number in (1, 2):
            result = self.runtime.observe(
                session_id,
                [
                    {
                        "kind": "code",
                        "content": (
                            '[CORTHEON_HOST_EVIDENCE] {"tool":"grep",'
                            '"outcome":"match",'
                            f'"args":{{"round":{round_number}}}}}\n'
                            "src/parser.py:18: def parse(value):"
                        ),
                        "source": f"opencode:grep:code-{round_number}",
                        "status": "verified",
                    }
                ],
                request_id=request["request_id"],
            )
            if round_number == 1:
                request = result["next_action"]["request"]
                self.assertEqual(request["parameters"]["discovery_round"], 2)

        self.assertEqual(result["session"]["phase"], "inconclusive")
        self.assertEqual(result["next_action"]["type"], "finish")
        self.assertIn("do not roam", result["next_action"]["instruction"])

    def test_explicit_task_kind_is_honored(self) -> None:
        result = self.runtime.start(
            "Analyze this dependency risk without reading project files.",
            effort="quick",
            task_kind="decision",
        )

        self.assertEqual(result["session"]["task_kind"], "decision")
        self.assertEqual(result["next_action"]["request"]["capability"], "inspect")

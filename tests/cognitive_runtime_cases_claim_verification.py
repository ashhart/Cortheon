from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime

from cortheon.cognitive_runtime import (
    CognitiveRuntime,
    _claim_type,
)


class ClaimVerificationEngineTests(unittest.TestCase):
    def test_unrelated_evidence_cannot_establish_a_claim(self) -> None:
        runtime = CognitiveRuntime()
        started = runtime.start(
            "Determine whether the deployment is safe.",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        runtime.observe(
            session_id,
            [
                {
                    "kind": "other",
                    "content": "The weather is sunny and mild.",
                    "source": "weather note",
                }
            ],
            request_id=started["next_action"]["request"]["request_id"],
        )

        completed = runtime.complete(
            session_id,
            answer="The deployment is safe.",
            claims=[
                {
                    "claim": "The deployment is safe.",
                    "evidence_ids": ["ev1"],
                }
            ],
            hypotheses=[
                {
                    "statement": "The deployment is safe.",
                    "falsification_test": "Inspect deployment evidence.",
                    "status": "supported",
                    "evidence_ids": ["ev1"],
                }
            ],
            completion_evidence_ids=["ev1"],
        )

        profile = completed["verification"]["claim_verification"][0]
        self.assertEqual(completed["verification"]["verdict"], "needs_evidence")
        self.assertEqual(profile["dimensions"]["entailment"], "not_established")
        self.assertEqual(profile["established_level"], "source_attributed")
        self.assertEqual(completed["next_action"]["type"], "harness_tool")
        self.assertEqual(completed["next_action"]["request"]["capability"], "inspect")

    def test_code_behavior_requires_a_verified_test(self) -> None:
        runtime = CognitiveRuntime()
        started = runtime.start("Fix parser.py so empty input works.", effort="quick")
        session_id = started["session"]["session_id"]
        runtime.observe(
            session_id,
            [
                {
                    "kind": "code",
                    "content": "def parse(value): raise ParseError",
                    "source": "parser.py",
                }
            ],
            request_id=started["next_action"]["request"]["request_id"],
        )
        runtime.step(
            session_id,
            hypotheses=[
                {
                    "statement": "The parser rejects empty input.",
                    "falsification_test": "Run the empty-input test.",
                }
            ],
        )
        runtime.observe(
            session_id,
            [
                {
                    "kind": "diff",
                    "content": "- raise ParseError\n+ return empty_result",
                    "source": "git diff",
                    "supports": ["h1"],
                }
            ],
            request_id="req2",
        )

        completed = runtime.complete(
            session_id,
            answer="The parser now handles empty input.",
            claims=[
                {
                    "claim": "parser.py now handles empty input correctly.",
                    "evidence_ids": ["ev2"],
                }
            ],
            hypotheses=[
                {
                    "statement": "The parser rejects empty input.",
                    "falsification_test": "Run the empty-input test.",
                    "status": "supported",
                    "evidence_ids": ["ev2"],
                }
            ],
            completion_evidence_ids=["ev2"],
        )

        profile = completed["verification"]["claim_verification"][0]
        self.assertEqual(profile["claim_type"], "code_behavior")
        self.assertFalse(profile["passed"])
        self.assertTrue(any("passing test" in gap for gap in profile["gaps"]))
        self.assertIn("focused host test", profile["next_truth_operation"])
        self.assertEqual(completed["next_action"]["type"], "harness_tool")
        self.assertEqual(completed["next_action"]["request"]["capability"], "test")

    def test_claim_truth_operation_uses_predicate_not_incidental_tense_words(self) -> None:
        runtime = CognitiveRuntime()
        started = runtime.start("Implement the focused runtime changes.", effort="quick")
        session = runtime._sessions[started["session"]["session_id"]]

        self.assertEqual(
            _claim_type(
                session,
                "ARCHITECTURE.md now defines the runtime boundary and lifecycle.",
            ),
            "code_static",
        )
        self.assertEqual(
            _claim_type(
                session,
                "The implementation adds a cognitive graph and audit manifest.",
            ),
            "code_change",
        )
        self.assertEqual(
            _claim_type(
                session,
                "After the final diff, 131 focused tests passed.",
            ),
            "code_behavior",
        )

    def test_near_identical_web_origins_are_one_effective_lineage(self) -> None:
        runtime = CognitiveRuntime()
        started = runtime.start(
            "Research the current Widget Atlas deployment safety position.",
            effort="quick",
            strictness="strict",
        )
        session_id = started["session"]["session_id"]
        retrieved_at = datetime.now(UTC).isoformat()
        copied = (
            "Widget Atlas deployment safety controls reduce incidents across "
            "regulated production environments according to the official report."
        )
        searched = runtime.observe(
            session_id,
            [
                {
                    "kind": "web",
                    "content": copied,
                    "source": "https://agency.gov/widget",
                    "url": "https://agency.gov/widget",
                    "retrieved_at": retrieved_at,
                    "purpose": "contradiction_check",
                },
                {
                    "kind": "web",
                    "content": copied,
                    "source": "https://mirror.example/widget",
                    "url": "https://mirror.example/widget",
                    "retrieved_at": retrieved_at,
                    "purpose": "contradiction_check",
                },
            ],
            request_id=started["next_action"]["request"]["request_id"],
        )
        self.assertEqual(
            searched["next_action"]["request"]["parameters"]["purpose"],
            "corroboration",
        )

        evidence_ids = ["ev1", "ev2"]
        completed = runtime.complete(
            session_id,
            answer=(
                "The current report says Widget Atlas safety controls reduce incidents. "
                "Sources: https://agency.gov/widget and https://mirror.example/widget."
            ),
            claims=[
                {
                    "claim": "Widget Atlas safety controls reduce incidents.",
                    "evidence_ids": evidence_ids,
                }
            ],
            hypotheses=[
                {
                    "statement": "The controls reduce incidents.",
                    "falsification_test": "Search for contrary evidence.",
                    "status": "supported",
                    "evidence_ids": evidence_ids,
                }
            ],
            completion_evidence_ids=evidence_ids,
        )

        profile = completed["verification"]["claim_verification"][0]
        self.assertEqual(profile["raw_url_origins"], 2)
        self.assertEqual(profile["effective_source_lineages"], 1)
        self.assertTrue(profile["likely_syndicated_sources"])
        self.assertFalse(profile["passed"])

    def test_private_record_needs_attribution_but_not_public_corroboration(self) -> None:
        runtime = CognitiveRuntime()
        started = runtime.start(
            "Use the private handbook.md to identify the widget approval threshold.",
            effort="quick",
        )
        session_id = started["session"]["session_id"]
        runtime.observe(
            session_id,
            [
                {
                    "kind": "documentation",
                    "content": "Widget purchases above £5,000 require director approval.",
                    "source": "handbook.md",
                }
            ],
            request_id=started["next_action"]["request"]["request_id"],
        )

        completed = runtime.complete(
            session_id,
            answer="The private handbook requires director approval above £5,000.",
            claims=[
                {
                    "claim": "Widget purchases above £5,000 require director approval.",
                    "evidence_ids": ["ev1"],
                }
            ],
            hypotheses=[
                {
                    "statement": "Director approval is required above £5,000.",
                    "falsification_test": "Read the handbook threshold.",
                    "status": "supported",
                    "evidence_ids": ["ev1"],
                }
            ],
            completion_evidence_ids=["ev1"],
        )

        self.assertEqual(completed["status"], "complete")
        profile = completed["claim_verification"][0]
        self.assertEqual(profile["claim_type"], "private_record")
        self.assertEqual(profile["dimensions"]["independence"], "not_applicable")
        self.assertEqual(profile["established_level"], "source_attributed")
        self.assertIn("identified source", profile["allowed_wording"])

    def test_ordered_plan_is_derived_and_wrong_order_is_rejected(self) -> None:
        runtime = CognitiveRuntime(require_host_receipts=True)
        started = runtime.start(
            "Read owners.md, dependencies.md, and policy.md. Produce the safe "
            "ordered rollout plan with every owner."
        )
        session_id = started["session"]["session_id"]
        request_id = started["next_action"]["request"]["request_id"]
        documents = {
            "owners.md": (
                "| Step | Owner |\n| --- | --- |\n"
                "| Freeze schema | Priya |\n| Migrate data | Tomas |\n"
                "| Deploy API | Mei |\n"
            ),
            "dependencies.md": (
                "Migrate data depends on Freeze schema.\nDeploy API depends on Migrate data.\n"
            ),
            "policy.md": "Notify customers must follow Deploy API.\n",
        }
        observations = []
        for path, content in documents.items():
            receipt = "[CORTHEON_HOST_EVIDENCE] " + json.dumps(
                {
                    "tool": "read",
                    "outcome": "result",
                    "args": {"filePath": path},
                },
                separators=(",", ":"),
            )
            observations.append(
                {
                    "kind": "documentation",
                    "content": receipt + "\n" + content,
                    "source": "opencode:read:" + path,
                    "status": "verified",
                }
            )
        observed = runtime.observe(
            session_id,
            observations,
            request_id=request_id,
        )
        derivation = observed["context"]["deterministic_derivations"][0]
        self.assertEqual(derivation["operation"], "ordered_plan")
        self.assertEqual(
            derivation["nodes"],
            ["Freeze schema", "Migrate data", "Deploy API", "Notify customers"],
        )

        evidence_ids = observed["accepted_evidence_ids"]
        wrong = runtime.complete(
            session_id,
            answer=(
                "Mei deploys Deploy API, Priya does Freeze schema, Tomas does "
                "Migrate data, then Notify customers."
            ),
            claims=[
                {
                    "claim": "The documents establish a rollout dependency order.",
                    "evidence_ids": evidence_ids,
                }
            ],
            hypotheses=[
                {
                    "statement": "The documents establish a rollout dependency order.",
                    "falsification_test": "Compare the order with every dependency.",
                    "status": "supported",
                    "evidence_ids": evidence_ids,
                }
            ],
            completion_evidence_ids=evidence_ids,
        )

        self.assertEqual(wrong["verification"]["verdict"], "needs_evidence")
        alignment = next(
            item for item in wrong["verification"]["checks"] if item["name"] == "evidence_alignment"
        )
        self.assertFalse(alignment["passed"])
        self.assertIn("order contradicts", alignment["reason"])

    def test_requirement_language_does_not_change_the_deliverable(self) -> None:
        runtime = CognitiveRuntime()
        read_only = runtime.start(
            "Inspect src/left.py and src/right.py. What is their sum? Do not modify files."
        )
        self.assertEqual(read_only["session"]["deliverable"], "code_understanding")
        self.assertNotIn(
            "mutation",
            {item["proof"] for item in read_only["cognition"]["task_frame"]["requirements"]},
        )

        documents = runtime.start(
            "Read catalog.md and policy.md. Connect who approves the Checkout "
            "change and explain why. Do not modify files."
        )
        document_proofs = {
            item["proof"] for item in documents["cognition"]["task_frame"]["requirements"]
        }
        self.assertIn("synthesis", document_proofs)
        self.assertNotIn("mutation", document_proofs)

        abductive = runtime.start(
            "Read cohorts.md and design.md. Compare two causal hypotheses and "
            "give a discriminating falsification test. Do not modify files."
        )
        abductive_proofs = {
            item["proof"] for item in abductive["cognition"]["task_frame"]["requirements"]
        }
        self.assertIn("synthesis", abductive_proofs)
        self.assertNotIn("verification", abductive_proofs)
        self.assertNotIn("mutation", abductive_proofs)

        disproof = runtime.start(
            "Read incident_timeline.md and sampling_contract.md. Generate competing "
            "explanations, identify the strongest one, and state a test that could "
            "disprove it. Do not modify files."
        )
        self.assertNotIn(
            "verification",
            {item["proof"] for item in disproof["cognition"]["task_frame"]["requirements"]},
        )
        code_named_disproof = runtime.start(
            "Read README.md, ARCHITECTURE.md, and pyproject.toml. Generate competing "
            "explanations and state a test that could disprove the strongest one. "
            "Do not modify files."
        )
        self.assertNotIn(
            "verification",
            {
                item["proof"]
                for item in code_named_disproof["cognition"]["task_frame"]["requirements"]
            },
        )

        research = runtime.start(
            "Research the latest release from the current web. Check freshness "
            "and contradictions and include both source URLs."
        )
        self.assertEqual(
            {item["proof"] for item in research["cognition"]["task_frame"]["requirements"]},
            {"research"},
        )

        decision = runtime.start(
            "Design a graph-memory plugin and produce an implementable build plan.",
            task_kind="decision",
        )
        decision_requirements = decision["cognition"]["task_frame"]["requirements"]
        self.assertTrue(any("build plan" in item["statement"] for item in decision_requirements))
        self.assertNotIn(
            "mutation",
            {item["proof"] for item in decision_requirements},
        )

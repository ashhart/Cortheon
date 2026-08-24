from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor

from cognitive_adversarial_cases_common import hypothesis

from cortheon.cognitive_runtime import CognitiveRuntime


class CognitiveConcurrencyTests(unittest.TestCase):
    @staticmethod
    def _workflow(runtime: CognitiveRuntime, index: int) -> str:
        started = runtime.start(
            f"Inspect behavior {index}",
            effort="quick",
            task_kind="general",
        )
        session_id = started["session"]["session_id"]
        runtime.observe(
            session_id,
            [{"kind": "analysis", "content": f"initial evidence {index}", "source": f"s{index}"}],
            request_id="req1",
        )
        runtime.step(session_id, hypotheses=[hypothesis(f"explanation {index}")])
        runtime.observe(
            session_id,
            [
                {
                    "kind": "analysis",
                    "content": f"supporting evidence {index}",
                    "source": f"s{index}",
                    "supports": ["h1"],
                }
            ],
            request_id="req2",
        )
        answer = f"Supporting evidence {index} was observed."
        claims = [{"claim": answer, "evidence_ids": ["ev2"]}]
        runtime.step(session_id, draft=answer)
        runtime.challenge(session_id, draft=answer, claims=claims)
        verified = runtime.verify(
            session_id,
            answer=answer,
            claims=claims,
            completion_evidence_ids=["ev2"],
        )
        if verified["verification"]["verdict"] != "ready":
            raise AssertionError(verified)
        runtime.finish(session_id, answer=answer)
        return session_id

    def test_250_parallel_investigations_are_isolated_and_erased(self) -> None:
        runtime = CognitiveRuntime(max_sessions=512)

        with ThreadPoolExecutor(max_workers=32) as executor:
            session_ids = list(executor.map(lambda i: self._workflow(runtime, i), range(250)))

        self.assertEqual(len(set(session_ids)), 250)
        self.assertEqual(runtime.active_sessions, 0)

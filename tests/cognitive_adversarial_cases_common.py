from __future__ import annotations

import unittest

from cortheon.cognitive_runtime import CognitiveRuntime


def hypothesis(statement: str) -> dict[str, str]:
    return {
        "statement": statement,
        "falsification_test": f"Try to disprove: {statement}",
    }


class AdversarialTestCase(unittest.TestCase):
    pass


class CompletionCase(AdversarialTestCase):
    def _code_session(self) -> tuple[CognitiveRuntime, str]:
        runtime = CognitiveRuntime()
        started = runtime.start(
            "Fix the parser bug in parser.py",
            effort="quick",
            task_kind="code",
        )
        session_id = started["session"]["session_id"]
        runtime.observe(
            session_id,
            [{"kind": "code", "content": "def parse(value): return value", "source": "parser.py"}],
            request_id="req1",
        )
        runtime.step(session_id, hypotheses=[hypothesis("the parser is defective")])
        runtime.observe(
            session_id,
            [
                {
                    "kind": "code",
                    "content": "empty values are returned unchanged",
                    "source": "parser.py:1",
                    "supports": ["h1"],
                }
            ],
            request_id="req2",
        )
        return runtime, session_id

from __future__ import annotations

import unittest

from cortheon.cognitive_runtime import CognitiveRuntime

ANSWER = "The parser now handles empty input and all parser tests pass."
CLAIMS = [
    {
        "claim": "The empty-input behavior is fixed and tested.",
        "evidence_ids": ["ev5", "ev6"],
    }
]


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class RuntimeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = CognitiveRuntime()

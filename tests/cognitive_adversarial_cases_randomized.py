from __future__ import annotations

import json
import random
import string
import unittest

from cortheon.cognitive_runtime import CognitiveRuntime


class CognitiveRandomizedStateTests(unittest.TestCase):
    def test_seeded_random_workloads_preserve_bounds_and_jsonability(self) -> None:
        randomizer = random.Random(8675309)
        alphabet = string.ascii_letters + string.digits + " \n_-"
        runtime = CognitiveRuntime(max_sessions=128)

        for _ in range(1_000):
            goal = "".join(randomizer.choice(alphabet) for _ in range(randomizer.randint(3, 200)))
            result = runtime.start(
                goal,
                effort=randomizer.choice(["quick", "standard", "deep"]),
                task_kind=randomizer.choice(
                    ["code", "research", "documents", "decision", "general"]
                ),
            )
            json.dumps(result)
            context = result["context"]
            self.assertLessEqual(
                context["context_chars_used"],
                context["context_char_limit"],
            )
            runtime.finish(result["session"]["session_id"], mode="abandon")

        self.assertEqual(runtime.active_sessions, 0)

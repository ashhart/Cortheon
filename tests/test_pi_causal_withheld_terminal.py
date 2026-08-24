"""Direct state regressions for the terminal causal withhold.

The forced-answer tool guard sets answer-only before the model's single
forced answer. When ``causalAnswerResult`` then withholds — deliberation
empty here, since the stub context carries no model — the withhold must be
terminal: the answer is marked delivered and a sticky terminal disposition
is held before the session is abandoned, so ``agent_end`` can never
schedule another answer-only continuation after ``abandonActive`` and no
later raw text in the abandoned window escapes interception.

Node's type-stripping loader executes the real reviewed TypeScript sources
(merge, state, causal_answer); the mutation copy removes exactly the
terminal marking and proves the escape reopens.
"""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
CORE = ROOT / "src" / "cortheon" / "pi_core"
SOURCE_DIR = ROOT / "src" / "cortheon"

RUN_SCRIPT = """
const fs = await import("node:fs");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const merge = await import(input.core + "/merge.ts");
const state = await import(input.core + "/state.ts");
const causal = await import(input.core + "/causal_answer.ts");
const terminal = await import(input.core + "/terminal.ts");
merge.mergePayload(input.payload);
state.setEnabled(true);
state.markAnswerOnly();
const message = {
  content: [{ type: "text", text: "draft answer" }],
  usage: {
    input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
    cost: { input: 0, output: 0, cacheWrite: 0, cacheRead: 0, total: 0 },
  },
};
const result = await causal.causalAnswerResult(
  {},
  {
    model: { id: "mock-small" },
    modelRegistry: {
      getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "k", headers: {}, env: {} }),
    },
    signal: undefined,
  },
  message,
  "draft answer",
);
const pi = { appendEntry() {}, sendMessage() {} };
const later = await terminal.terminalDispositionResult(pi, {
  content: [{ type: "text", text: "raw later text" }],
});
const report = {
  withheld: Boolean(result?.message?.content?.[0]?.text?.startsWith("[Cortheon withheld:")),
  sessionAbandoned: state.getActive() === undefined,
  answerDelivered: state.answerAlreadyDelivered(),
  noSessionMeansNoFollowUp: state.getActive() === undefined,
  rawLaterReplaced: Boolean(
    later?.message?.content?.[0]?.text?.startsWith("[Cortheon withheld:"),
  ),
  disposition: state.peekTerminalDisposition() ?? null,
};
console.log(JSON.stringify(report));
process.exit(0);
"""

SUFFICIENT_PAYLOAD = {
    "session_id": "s1",
    "status": "observing",
    "session": {"deliverable": "document_synthesis"},
    "context": {
        "goal": (
            "Diagnose the causal explanation for the clash between the two "
            "ledgers, disprove the rival hypothesis, and give a "
            "discriminating test."
        ),
        "evidence": [
            {
                "evidence_id": "ev-1",
                "source": "pi:read:facts/a.txt",
                "content": "clean fact one",
            },
            {
                "evidence_id": "ev-2",
                "source": "pi:read:facts/b.txt",
                "content": "clean fact two",
            },
        ],
    },
    "next_action": {"type": "reason"},
}


def _stage(core_target: Path) -> None:
    """Copy pi_core next to a stub node_modules providing the only external
    runtime dependency of the chain (repair.ts's pi-ai); the stubs are
    never called in these probes (the context carries no model)."""
    core_target.mkdir(parents=True)
    for path in sorted(CORE.glob("*.ts")):
        (core_target / path.name).write_text(path.read_text(encoding="utf-8"))
    vendor = core_target.parent / "node_modules" / "@earendil-works" / "pi-ai"
    vendor.mkdir(parents=True)
    (vendor / "package.json").write_text(
        json.dumps(
            {
                "name": "@earendil-works/pi-ai",
                "exports": {".": "./index.js", "./compat": "./compat.js"},
            }
        ),
        encoding="utf-8",
    )
    (vendor / "index.js").write_text("export const uuidv7 = () => 'stub';\n", encoding="utf-8")
    (vendor / "compat.js").write_text(
        "export const complete = async () => ({\n"
        '  stopReason: "stop",\n'
        '  content: [{ type: "text", text: "no structured sections" }],\n'
        "  usage: {\n"
        "    input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,\n"
        "    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },\n"
        "  },\n"
        "});\n",
        encoding="utf-8",
    )


def _probe(core_dir: Path) -> dict[str, object]:
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", RUN_SCRIPT],
        input=json.dumps({"core": str(core_dir), "payload": SUFFICIENT_PAYLOAD}),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise AssertionError(f"node failed: {completed.stderr}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


class CausalWithheldTerminalTests(unittest.TestCase):
    def test_final_withhold_marks_delivery_and_holds_disposition(self) -> None:
        root = Path(
            subprocess.run(
                ["mktemp", "-d"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        _stage(root / "pi_core")
        report = _probe(root / "pi_core")
        self.assertTrue(report["withheld"])
        self.assertTrue(report["sessionAbandoned"])
        self.assertTrue(report["answerDelivered"])
        # The unified budget needs a retained session: with the session
        # abandoned no follow-up can ever be scheduled.
        self.assertTrue(report["noSessionMeansNoFollowUp"])
        # Every later raw text in the abandoned window is replaced by the
        # held terminal disposition.
        self.assertTrue(report["rawLaterReplaced"])
        disposition = report["disposition"]
        self.assertIsInstance(disposition, dict)
        self.assertTrue(disposition["causal"])

    def test_mutation_removing_terminal_marking_reopens_escape(self) -> None:
        """Removing the terminal marking reopens the review's escape: the
        withheld answer is not delivered and no disposition is held, so raw
        later text in the abandoned window has nothing to intercept it."""
        root = Path(
            subprocess.run(
                ["mktemp", "-d"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        _stage(root / "pi_core")
        mutation = (
            "\t\tmarkFinalWithhold(\n"
            '\t\t\t"causal deliberation produced no validated candidate before " +\n'
            '\t\t\t\t"certification",\n'
            "\t\t);\n"
            "\t\tawait abandonActive();",
            "\t\tawait abandonActive();",
        )
        target = root / "pi_core" / "causal_answer.ts"
        text = target.read_text(encoding="utf-8")
        old, new = mutation
        assert text.count(old) == 1, old
        target.write_text(text.replace(old, new), encoding="utf-8")
        report = _probe(root / "pi_core")
        self.assertTrue(report["withheld"])
        self.assertTrue(report["sessionAbandoned"])
        self.assertFalse(report["answerDelivered"])
        # The escape: no disposition, so raw later text passes through.
        self.assertFalse(report["rawLaterReplaced"])
        self.assertIsNone(report["disposition"])


if __name__ == "__main__":
    unittest.main()

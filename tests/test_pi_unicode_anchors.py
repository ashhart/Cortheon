"""Unicode research grounding for the Pi synthesis validator.

Deterministic, dependency-free Unicode word segmentation must let CJK,
Cyrillic, and Greek evidence ground a synthesis (or correctly fail to), with
script-aware anchor thresholds: Han/kana bigrams anchor at 2 units,
alphabetic words at 5. One shared token never clears grounding. Node
executes the real TypeScript directly.
"""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
TEXT = ROOT / "src" / "cortheon" / "pi_core" / "text.ts"
GROUNDING = ROOT / "src" / "cortheon" / "pi_core" / "grounding.ts"

RUN_SCRIPT = """
const fs = await import("node:fs");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const out = {};
if (input.tokens) {
  const text = await import(input.tokens.module);
  out.tokens = [...text.anchorTokens(input.tokens.value)];
}
if (input.validate) {
  const grounding = await import(input.validate.module);
  out.validate = grounding.groundingFailures(input.validate.sections, input.validate.records);
}
console.log(JSON.stringify(out));
process.exit(0);
"""


def _run(tokens_value: str | None = None, validate=None) -> dict[str, object]:
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", RUN_SCRIPT],
        input=json.dumps(
            {
                "tokens": (
                    {"module": str(TEXT), "value": tokens_value}
                    if tokens_value is not None
                    else None
                ),
                "validate": ({"module": str(GROUNDING), **validate} if validate else None),
            }
        ),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise AssertionError(f"node failed: {completed.stderr}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


ZH_SECTIONS = {
    "evidence": "host ledger",
    "cause": "冲突发生的原因是两条路径复用了同一个配置键「琥珀」。",
    "rival": "Instead, 缓存压缩是竞争性的 competing alternative。",
    "test": "Cause predicts 冲突消失 whereas Rival predicts 冲突仍然存在。",
}
ZH_RECORD = {"source": "src/南京.txt", "fact": "配置键为「琥珀」，两条路径复用它。"}  # noqa: RUF001
EN_SECTIONS = {
    "evidence": "host ledger",
    "cause": "The collision occurs because both paths reuse the Northstar key amber.",
    "rival": "Instead, cache compaction is the competing alternative.",
    "test": "Cause predicts the collision disappears whereas Rival predicts it remains.",
}
EN_CONTROL_RECORDS = [
    {
        "source": "pi:read:a.txt",
        "fact": "Northstar path A uses collision key amber.",
    },
    {
        "source": "pi:read:b.txt",
        "fact": "Path B reuses key amber; collision keys collide.",
    },
]


class AnchorTokenTests(unittest.TestCase):
    def test_cjk_bigrams_and_alphabetic_thresholds(self) -> None:
        tokens = _run(tokens_value="配置键为「琥珀」 Amber. path")["tokens"]
        self.assertIn("配置", tokens)
        self.assertIn("琥珀", tokens)
        # Short alphabetic words and one-character CJK function words never
        # become anchors.
        self.assertNotIn("path", tokens)
        self.assertNotIn("uses", tokens)
        self.assertNotIn("为", tokens)

    def test_punctuation_only_is_never_an_anchor(self) -> None:
        tokens = _run(tokens_value="「」，。；：！？—")["tokens"]  # noqa: RUF001
        self.assertEqual(tokens, [])

    def test_cyrillic_and_greek_words_are_anchors(self) -> None:
        cyrillic = _run(tokens_value="Коллизия сохраняется при отключённом сжатии.")["tokens"]
        self.assertIn("коллизия", cyrillic)
        self.assertIn("сжатии", cyrillic)
        greek = _run(tokens_value="Η σύγκρουση παραμένει.")["tokens"]  # noqa: RUF001
        self.assertIn("σύγκρουση", greek)


class UnicodeGroundingTests(unittest.TestCase):
    def _validate(self, sections, records) -> list[str]:
        result = _run(validate={"sections": sections, "records": records})
        value = result["validate"]
        assert isinstance(value, list)
        return value

    def test_verbatim_cjk_synthesis_grounds(self) -> None:
        failures = self._validate(ZH_SECTIONS, [ZH_RECORD])
        self.assertEqual(
            [f for f in failures if "grounded" in f],
            [],
            failures,
        )

    def test_english_synthesis_with_unrelated_cjk_source_fails_grounding(self) -> None:
        failures = self._validate(
            EN_SECTIONS,
            [
                EN_CONTROL_RECORDS[0],
                {"source": "src/南京.txt", "fact": "配置键为「琥珀」。"},
            ],
        )
        self.assertTrue(any("grounded" in f for f in failures), failures)

    def test_cyrillic_source_unreflected_in_english_synthesis_fails(self) -> None:
        failures = self._validate(
            EN_SECTIONS,
            [{"source": "b", "fact": "Коллизия сохраняется при отключённом сжатии."}],
        )
        self.assertTrue(any("grounded" in f for f in failures), failures)

    def test_english_control_still_grounds(self) -> None:
        failures = self._validate(EN_SECTIONS, EN_CONTROL_RECORDS)
        self.assertEqual(failures, [], failures)

    def test_lone_shared_project_name_never_clears_grounding(self) -> None:
        # Both records share only the project name with the synthesis body:
        # a single incidental anchor is not grounding.
        sections = {
            "evidence": "host ledger",
            "cause": (
                "The Northstar shard rotation leads to the collision because "
                "two writers hash to the identical slot."
            ),
            "rival": "Instead, cache compaction is the competing alternative that evicts entries.",
            "test": (
                "Disable compaction while holding the shard map constant: Cause "
                "predicts the collision persists whereas Rival predicts the "
                "collision disappears, which would falsify the wrong mechanism."
            ),
        }
        records = [
            {
                "source": "pi:read:facts/a.txt",
                "fact": "Northstar was renamed from Polaris during the platform migration.",
            },
            {
                "source": "pi:read:facts/b.txt",
                "fact": "The Northstar steering group meets weekly in the annex.",
            },
        ]
        failures = self._validate(sections, records)
        self.assertTrue(any("grounded" in f for f in failures), failures)


if __name__ == "__main__":
    unittest.main()

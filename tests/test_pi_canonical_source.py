"""Unit tests for the Pi adapter's canonical evidence-source identity.

``canonicalEvidenceSource`` is a deterministic TypeScript function; Node's
type-stripping loader executes the real reviewed source directly, so these
tests exercise the shipped logic rather than a Python reimplementation.
Contract under test: non-Pi sources are byte-preserved exactly (no
trimming, no case folding), rooted POSIX paths stay distinct from relative
ones, redundant separators and dot/dot-dot segments collapse lexically
without touching the filesystem or resolving symlinks, and same-file
read/grep aliases collapse to one origin.
"""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
HOST_EVIDENCE = ROOT / "src" / "cortheon" / "pi_core" / "host_evidence.ts"

RUN_SCRIPT = """
const fs = await import("node:fs");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const module = await import(input.module);
const result = input.sources.map((source) =>
  module.canonicalEvidenceSource(source));
console.log(JSON.stringify({ result }));
process.exit(0);
"""


def _canonical(sources: list[str]) -> list[str]:
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", RUN_SCRIPT],
        input=json.dumps({"module": str(HOST_EVIDENCE), "sources": sources}),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise AssertionError(f"node failed: {completed.stderr}")
    return json.loads(completed.stdout.strip().splitlines()[-1])["result"]


def _canonical_one(source: str) -> str:
    return _canonical([source])[0]


class NonPiBytePreservationTests(unittest.TestCase):
    def test_non_pi_sources_keep_every_byte(self) -> None:
        # Leading/trailing whitespace, internal spacing, case, and
        # separators are identity, not noise: nothing may be trimmed,
        # folded, or rewritten on a non-Pi source.
        for source in (
            "  https://example.com/docs/a%20b",
            "\tdocuments/Report Final.pdf\n",
            "HTTP://Example.COM/X",
            "repo//a/.././a.txt",
            " ",
        ):
            self.assertEqual(_canonical_one(source), source)

    def test_pi_prefixed_text_with_leading_padding_is_non_pi(self) -> None:
        # Only an exact-anchored pi:<tool>[:origin] form is a host alias;
        # anything with leading or embedded padding is a non-Pi source
        # preserved byte-for-byte. A trailing space stays a Pi alias whose
        # origin normalizes ("a.txt " -> "a.txt").
        for source in (" pi:read:a.txt", "x pi:read:a.txt", "pi :read:a.txt"):
            self.assertEqual(_canonical_one(source), source)
        self.assertEqual(_canonical_one("pi:read:a.txt "), "a.txt")


class RootedVersusRelativeIdentityTests(unittest.TestCase):
    def test_leading_slash_is_preserved(self) -> None:
        self.assertEqual(_canonical_one("pi:read:/repo/a.txt"), "/repo/a.txt")
        self.assertEqual(_canonical_one("pi:read:repo/a.txt"), "repo/a.txt")
        # Absolute and relative never collide.
        self.assertNotEqual(
            _canonical_one("pi:read:/repo/a.txt"),
            _canonical_one("pi:read:repo/a.txt"),
        )

    def test_rooted_normalization_stays_rooted(self) -> None:
        self.assertEqual(_canonical_one("pi:read:/repo/./sub/../a.txt"), "/repo/a.txt")
        self.assertEqual(_canonical_one("pi:read://repo//a.txt"), "/repo/a.txt")

    def test_relative_normalization_stays_relative(self) -> None:
        self.assertEqual(_canonical_one("pi:read:./a.txt"), "a.txt")
        self.assertEqual(_canonical_one("pi:read:repo/./a/../a.txt"), "repo/a.txt")

    def test_leading_dot_dot_segments_are_kept(self) -> None:
        # Lexical only: ".." above the root of the expression cannot be
        # resolved and must not collapse into a sibling identity.
        self.assertEqual(_canonical_one("pi:read:../a.txt"), "../a.txt")
        self.assertEqual(_canonical_one("pi:read:../../a.txt"), "../../a.txt")
        self.assertEqual(_canonical_one("pi:read:../a/../b"), "../b")

    def test_path_case_is_preserved(self) -> None:
        self.assertEqual(_canonical_one("pi:read:Docs/A.TXT"), "Docs/A.TXT")
        self.assertNotEqual(
            _canonical_one("pi:read:Docs/A.TXT"), _canonical_one("pi:read:docs/a.txt")
        )


class SameFileAliasTests(unittest.TestCase):
    def test_read_and_grep_aliases_collapse(self) -> None:
        aliases = [
            "pi:read:facts/a.txt",
            "pi:grep:facts/a.txt",
            "PI:READ:./facts/a.txt",
            "Pi:Grep:facts//a.txt",
        ]
        canonical = set(_canonical(aliases))
        self.assertEqual(canonical, {"facts/a.txt"})

    def test_bare_tool_label_proves_nothing(self) -> None:
        self.assertEqual(_canonical(["pi:read", "PI:GREP"]), ["", ""])

    def test_distinct_origins_stay_distinct(self) -> None:
        canonical = set(_canonical(["pi:read:a.txt", "pi:read:b.txt", "pi:grep:/a.txt"]))
        self.assertEqual(canonical, {"a.txt", "b.txt", "/a.txt"})


if __name__ == "__main__":
    unittest.main()

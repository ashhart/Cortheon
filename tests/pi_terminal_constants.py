"""Constants for the bounded-completion terminal tests.

Prompts, answers, accepted-evidence fixtures, and the custom-entry markers
the terminal scripts and their tests share.
"""

from __future__ import annotations

from pathlib import Path

EXTENSION = Path(__file__).parents[1] / "src" / "cortheon" / "pi_extension.ts"
SOURCE_DIR = Path(__file__).parents[1] / "src" / "cortheon"
WITHHELD_MARKER = "[Cortheon withheld:"
CANDIDATE_ENTRY_TYPE = "cortheon-benchmark-candidate-v1"

CAUSAL_PROMPT = (
    "Diagnose the causal explanation for the clash between the two ledgers, "
    "disprove the rival hypothesis, and give a discriminating test."
)
CAUSAL_CERTIFIED = "CORTHEON CERTIFIED: the clash is the shared shard key."

AMBIGUITY_PROMPT = (
    "The phrase 'improve margin by 5' in the spec is ambiguous; clarify "
    "which meaning is justified rather than guessing."
)
AMBIGUITY_ANSWER = (
    "The justified meaning is the profit-margin reading; the rival reading is not supported."
)

EVIDENCE = [
    {
        "evidence_id": "ev-1",
        "source": "pi:read:facts/a.txt",
        "content": "Ledger alpha writes shard key copper during nightly rotation.",
    },
    {
        "evidence_id": "ev-2",
        "source": "pi:read:facts/b.txt",
        "content": (
            "Ledger beta reuses shard key copper; the clash persists when archiving is disabled."
        ),
    },
]

GOOD_SYNTHESIS = (
    "Cause: The clash occurs because both ledgers reuse shard key copper.\n"
    "Rival: Instead, nightly archiving is the competing alternative because "
    "the clash persists when archiving is disabled.\n"
    "Test: Reassign distinct shard keys while holding archiving constant — "
    "this distinguishing test would falsify the wrong mechanism: Cause "
    "predicts the clash disappears whereas Rival predicts the clash remains."
)

TERMINAL_STATUS_TYPE = "cortheon-terminal-status-v1"
TERMINAL_STATUS_VERSION = 1
TERMINAL_STATUS_MARKER = "ended without a certified answer because"

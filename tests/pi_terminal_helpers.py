"""Aggregating facade for the bounded-completion terminal helpers.

The helpers live in focused modules — constants, causal scripts,
withholding scripts, event parsing, and source mutation — and are
re-exported here so the terminal test modules keep one import site.
"""

from __future__ import annotations

from pi_causal_scripts import (
    always_pending_request_script,
    evidence_insufficient_wandering_script,
    evidence_ready_wandering_script,
    repeated_evidence_wandering_script,
    single_batch_sufficient_script,
)
from pi_terminal_constants import (
    AMBIGUITY_ANSWER,
    AMBIGUITY_PROMPT,
    CANDIDATE_ENTRY_TYPE,
    CAUSAL_CERTIFIED,
    CAUSAL_PROMPT,
    EVIDENCE,
    EXTENSION,
    GOOD_SYNTHESIS,
    SOURCE_DIR,
    TERMINAL_STATUS_MARKER,
    TERMINAL_STATUS_TYPE,
    TERMINAL_STATUS_VERSION,
    WITHHELD_MARKER,
)
from pi_terminal_events import custom_entry_data, terminal_status_messages
from pi_terminal_sources import mutated_source
from pi_withholding_scripts import (
    varying_action_withholding_script,
    withhold_then_finish_script,
    withholding_ambiguity_script,
)

__all__ = [
    "AMBIGUITY_ANSWER",
    "AMBIGUITY_PROMPT",
    "CANDIDATE_ENTRY_TYPE",
    "CAUSAL_CERTIFIED",
    "CAUSAL_PROMPT",
    "EVIDENCE",
    "EXTENSION",
    "GOOD_SYNTHESIS",
    "SOURCE_DIR",
    "TERMINAL_STATUS_MARKER",
    "TERMINAL_STATUS_TYPE",
    "TERMINAL_STATUS_VERSION",
    "WITHHELD_MARKER",
    "always_pending_request_script",
    "custom_entry_data",
    "evidence_insufficient_wandering_script",
    "evidence_ready_wandering_script",
    "mutated_source",
    "repeated_evidence_wandering_script",
    "single_batch_sufficient_script",
    "terminal_status_messages",
    "varying_action_withholding_script",
    "withhold_then_finish_script",
    "withholding_ambiguity_script",
]

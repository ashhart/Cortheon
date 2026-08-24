"""The parity contract failure type and the registered contender universes."""

from __future__ import annotations


class ParityContractError(ValueError):
    """A parity contract is incomplete, ambiguous, or not pre-registrable."""


_TRUSTED_FRONTIER_HOSTS = {
    "anthropic": {"api.anthropic.com"},
    "google": {
        "aiplatform.googleapis.com",
        "generativelanguage.googleapis.com",
    },
    "moonshot": {"api.moonshot.ai"},
    "openai": {"api.openai.com"},
}

SUPPORTED_CANDIDATE_HOSTS = frozenset({"codex", "generic_mcp", "opencode", "pi"})

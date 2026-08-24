"""Manifest schema constants and per-tier promotion policy defaults."""

from __future__ import annotations

import re

SCHEMA_VERSION = 3
# Version 7 binds designated OpenCode cells to the frozen historical program.
REPORT_SCHEMA_VERSION = 7
MAX_MANIFEST_BYTES = 1_000_000
MAX_CELLS = 128
MAX_JOBS = 20_000
SUITES = frozenset({"imports", "joins", "patches", "semantic", "research", "mixed"})
HOSTS = frozenset({"opencode", "pi"})
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
FORBIDDEN_CREDENTIAL_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
        "access_token",
        "headers",
    }
)
ROOT_KEYS = frozenset(
    {
        "schema_version",
        "tier",
        "repository",
        "seed",
        "cells",
        "gates",
    }
)
CELL_KEYS = frozenset(
    {
        "id",
        "suite",
        "host",
        "provider",
        "base_url",
        "api_key_env",
        "model_id",
        "runtime_url",
        "cases",
        "repeats",
        "seed",
        "timeout_seconds",
        "context_tokens",
        "output_tokens",
        "max_steps",
        "reasoning",
        "opencode",
        "pi",
        "conditions",
        "historical_comparison",
    }
)
GATE_KEYS = frozenset(
    {
        "min_independent_cases",
        "max_false_allows",
        "max_false_block_rate",
        "min_full_accuracy",
        "min_full_vs_bare_accuracy_delta",
        "min_full_vs_bare_accuracy_delta_ci_lower",
        "min_full_vs_reduced_accuracy_delta_ci_lower",
        "max_invalid_pairs",
    }
)
TIER_DEFAULTS: dict[str, dict[str, int | float]] = {
    "pr": {
        "default_repeats": 1,
        "min_independent_cases": 2,
        "max_false_allows": 0,
        "max_false_block_rate": 0.25,
        "min_full_accuracy": 0.50,
        "min_full_vs_bare_accuracy_delta": 0.0,
        "min_full_vs_bare_accuracy_delta_ci_lower": -1.0,
        "min_full_vs_reduced_accuracy_delta_ci_lower": -1.0,
        "max_invalid_pairs": 0,
    },
    "nightly": {
        "default_repeats": 2,
        "min_independent_cases": 8,
        "max_false_allows": 0,
        "max_false_block_rate": 0.10,
        "min_full_accuracy": 0.80,
        "min_full_vs_bare_accuracy_delta": 0.0,
        "min_full_vs_bare_accuracy_delta_ci_lower": -0.25,
        "min_full_vs_reduced_accuracy_delta_ci_lower": -0.25,
        "max_invalid_pairs": 0,
    },
    "weekly": {
        "default_repeats": 3,
        "min_independent_cases": 24,
        "max_false_allows": 0,
        "max_false_block_rate": 0.02,
        "min_full_accuracy": 0.90,
        "min_full_vs_bare_accuracy_delta": 0.0,
        "min_full_vs_bare_accuracy_delta_ci_lower": 0.05,
        "min_full_vs_reduced_accuracy_delta_ci_lower": 0.03,
        "max_invalid_pairs": 0,
    },
}

"""Synthetic held-out diagnostic fixtures with forbidden answers."""

from __future__ import annotations

import hashlib
import random

from cortheon.benchmark_core.models import DiagnosticCase


def discover_diagnostic_cases(*, count: int, seed: int) -> list[DiagnosticCase]:
    """Return held-out diagnosis tasks with evidence split across code and logs."""

    definitions = [
        (
            "retry_off_by_one",
            (
                ("diagnostics/retry_config.py", "MAX_RETRIES = 3\n"),
                (
                    "diagnostics/retry_client.py",
                    "def send(request, max_retries):\n"
                    "    for attempt in range(max_retries + 1):\n"
                    "        if request(attempt):\n"
                    "            return True\n"
                    "    return False\n",
                ),
                (
                    "diagnostics/retry_trace.log",
                    "attempt=0 status=503\nattempt=1 status=503\n"
                    "attempt=2 status=503\nattempt=3 status=503\n",
                ),
            ),
            ("range(max_retries + 1)", "attempts", "off-by-one"),
            ("dns", "timeout"),
            (
                "Diagnose why the client makes four attempts although "
                "diagnostics/retry_config.py sets three retries. Read "
                "diagnostics/retry_client.py and diagnostics/retry_trace.log, identify "
                "the exact root cause, and distinguish it from plausible network "
                "failures. Do not modify files."
            ),
        ),
        (
            "pagination_origin",
            (
                (
                    "diagnostics/pager.py",
                    "def collect(fetch):\n"
                    "    page = 0\n"
                    "    rows = []\n"
                    "    while True:\n"
                    "        batch = fetch(page)\n"
                    "        if not batch:\n"
                    "            return rows\n"
                    "        rows.extend(batch)\n"
                    "        page += 1\n",
                ),
                (
                    "diagnostics/api_contract.md",
                    "The list endpoint uses one-based page numbers. Page 1 is the "
                    "first valid page; page 0 returns an empty result.\n",
                ),
                (
                    "diagnostics/pager_trace.log",
                    "request page=0 status=200 rows=0\ncollector returned rows=0\n",
                ),
            ),
            ("page = 0", "one-based", "empty result"),
            ("database", "authentication"),
            (
                "Diagnose why collect in diagnostics/pager.py always returns no rows. "
                "Use diagnostics/api_contract.md and diagnostics/pager_trace.log to "
                "establish the root cause and the failing boundary assumption. "
                "Do not modify files."
            ),
        ),
        (
            "audience_mismatch",
            (
                (
                    "diagnostics/auth_settings.py",
                    'EXPECTED_AUDIENCE = "orders-api"\nEXPECTED_ISSUER = "identity"\n',
                ),
                (
                    "diagnostics/token_factory.py",
                    'def claims():\n    return {"iss": "identity", "aud": "order-api"}\n',
                ),
                (
                    "diagnostics/auth_trace.log",
                    "issuer check: passed\n"
                    "audience check: expected=orders-api actual=order-api failed\n",
                ),
            ),
            ("orders-api", "order-api", "audience"),
            ("issuer", "signature"),
            (
                "Diagnose the authentication rejection using "
                "diagnostics/auth_settings.py, diagnostics/token_factory.py, and "
                "diagnostics/auth_trace.log. Identify the exact mismatched claim and "
                "rule out the tempting issuer explanation. Do not modify files."
            ),
        ),
        (
            "ttl_unit_mismatch",
            (
                ("diagnostics/session_settings.py", "SESSION_TTL_SECONDS = 300\n"),
                (
                    "diagnostics/session_cache.py",
                    "def expires_at(issued_at, ttl_seconds):\n"
                    "    return issued_at + ttl_seconds * 1000\n",
                ),
                (
                    "diagnostics/session_trace.log",
                    "issued_at=5000 ttl_seconds=300 expires_at=305000\n"
                    "session remained valid far beyond five minutes\n",
                ),
            ),
            ("ttl_seconds * 1000", "seconds", "unit"),
            ("clock skew", "database"),
            (
                "Diagnose why sessions outlive the configured five-minute TTL. Read "
                "diagnostics/session_settings.py, diagnostics/session_cache.py, and "
                "diagnostics/session_trace.log. State the exact unit error and reject "
                "unsupported alternatives. Do not modify files."
            ),
        ),
    ]
    if count > len(definitions):
        raise ValueError(
            f"diagnostic suite has {len(definitions)} held-out cases; requested {count}"
        )
    random.Random(seed ^ 0xD1A6).shuffle(definitions)
    return [
        DiagnosticCase(
            case_id="diagnostic_"
            + hashlib.sha256(f"{seed}\0{name}\0{expected}".encode()).hexdigest()[:12],
            files=case_files,
            expected=expected,
            forbidden_answers=forbidden,
            prompt=prompt,
        )
        for name, case_files, expected, forbidden, prompt in definitions[:count]
    ]

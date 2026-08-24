"""Source-bound isolated launcher for the repository-only generic evaluator."""

from __future__ import annotations

import os
import re
import signal
import sys
from pathlib import Path

_SHA256 = re.compile(r"[0-9a-f]{64}")


def launch() -> int:
    source_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(source_root))
    from cortheon.benchmark_core.generic_mcp_source import (
        EXPECTED_DIGEST_ENV,
        VERIFIED_DIGEST_ENV,
        generic_source_sha256,
    )

    expected = os.environ.pop(EXPECTED_DIGEST_ENV, None)
    observed = generic_source_sha256()
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise RuntimeError("evaluator-owned generic source digest is required")
    if expected != observed:
        raise RuntimeError("generic evaluator source digest changed before launch")
    os.environ[VERIFIED_DIGEST_ENV] = observed
    from cortheon.benchmark_core.generic_mcp_process import (
        _interrupted,
        serve_controlled_process,
    )

    signal.signal(signal.SIGTERM, _interrupted)
    signal.signal(signal.SIGINT, _interrupted)
    return serve_controlled_process()


if __name__ == "__main__":
    raise SystemExit(launch())

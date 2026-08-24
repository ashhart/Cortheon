"""Evaluator-only execution of the immutable 19d035c OpenCode substrate."""

from __future__ import annotations

import inspect
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from cortheon.qualification_core.frozen_archive import (
    _ROOT,
    ARCHIVE_SHA256,
    FROZEN_COMMIT,
    FROZEN_TREE,
    _sha,
    archive_available,
    extract_verified,
)
from cortheon.qualification_core.frozen_runtime import (
    FrozenRuntime,
    start_runtime,
    stop_runtime,
)

if TYPE_CHECKING:
    import cortheon.qualification_core.frozen_smoke as frozen_smoke

    _ = frozen_smoke

WRAPPER_SHA256 = "be719165b81130c3afb04d189fd5107c1dde6e2d6062800840aaaca3365a81e0"
SMOKE_SHA256 = "753af4536ef5a680c9b9b9c838a377cd72579c69437af60734839adb4636f997"
SMOKE = _ROOT / "benchmarks/frozen/old_planner_19d035c_smoke.json"
WRAPPER = Path(__file__).with_name("frozen_opencode_wrapper.js")
EVALUATOR_FILES = (
    Path(__file__).with_name("frozen_archive.py"),
    Path(__file__).with_name("frozen_execution.py"),
    Path(__file__).with_name("frozen_runtime.py"),
    Path(__file__).with_name("frozen_receipt.py"),
    Path(__file__).with_name("frozen_smoke.py"),
)


def wrapper_sha256() -> str:
    return _sha(WRAPPER.read_bytes())


def frozen_implementation_sha256() -> str:
    operational = (
        *(inspect.getsource(globals()[name]) for name in _OPERATIONAL_FUNCTIONS),
        *(path.read_text(encoding="utf-8") for path in EVALUATOR_FILES),
    )
    payload = {
        "schema_version": 1,
        "commit": FROZEN_COMMIT,
        "tree": FROZEN_TREE,
        "archive_sha256": ARCHIVE_SHA256,
        "wrapper_sha256": wrapper_sha256(),
        "evaluator_sha256": _sha("\n".join(operational).encode()),
        "host": "opencode",
    }
    return _sha(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def comparator_available() -> bool:
    available = bool(
        archive_available()
        and WRAPPER.is_file()
        and wrapper_sha256() == WRAPPER_SHA256
        and SMOKE.is_file()
        and _sha(SMOKE.read_bytes()) == SMOKE_SHA256
    )
    if not available:
        return False
    try:
        receipt = json.loads(SMOKE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    from cortheon.qualification_core.frozen_receipt import validate_smoke_receipt

    return validate_smoke_receipt(
        receipt,
        frozen_implementation_sha256(),
        WRAPPER_SHA256,
    )


@contextmanager
def frozen_old_planner() -> Iterator[FrozenRuntime]:
    with tempfile.TemporaryDirectory(prefix="cortheon-eval-") as temporary:
        root = Path(temporary)
        extract_verified(root)
        adapter_root = root / "adapter"
        adapter_root.mkdir()
        adapter = adapter_root / "adapter.js"
        shutil.copyfile(WRAPPER, adapter)
        shutil.copyfile(root / "src/cortheon/opencode_plugin.js", adapter_root / "program.js")
        (adapter_root / "package.json").write_text('{"type":"module"}\n', encoding="utf-8")
        runtime = start_runtime(root, adapter, os.urandom(32).hex())
        try:
            yield runtime
        finally:
            active_error = sys.exc_info()[0] is not None
            try:
                healthy = runtime.health().get("active_sessions") == 0
            except (OSError, ValueError):
                healthy = False
            unchanged = runtime.unchanged()
            try:
                stop_runtime(runtime.process)
            finally:
                if not active_error and not healthy:
                    raise ValueError("frozen runtime leaked active sessions")
                if not active_error and not unchanged:
                    raise ValueError("frozen old-planner artifact changed during execution")


_OPERATIONAL_FUNCTIONS = ("frozen_old_planner",)

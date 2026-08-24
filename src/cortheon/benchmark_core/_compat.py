"""Late-bound lookups of facade-level callables.

Tests monkeypatch callables such as ``run_job``, ``discover_benchmark_cases``,
the health probes, and ``_latest_pypi_release`` on the
``cortheon.cognitive_benchmark`` facade. Implementation modules resolve those
names through the facade namespace at call time so the patches keep working.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

_FACADE_NAME = "cortheon.cognitive_benchmark"


def facade() -> ModuleType:
    """Return the compatibility facade module, importing it if needed."""
    module = sys.modules.get(_FACADE_NAME)
    if module is None:
        module = importlib.import_module(_FACADE_NAME)
    return module

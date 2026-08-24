"""Late-bound lookups of facade-level callables.

Tests monkeypatch ``call_contender``, ``_post_json``, and ``datetime`` on the
``cortheon.benchmark`` facade. Implementation modules resolve those names
through the facade namespace at call time so the patches keep working.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

_FACADE_NAME = "cortheon.benchmark"


def facade() -> ModuleType:
    """Return the compatibility facade module, importing it if needed."""
    module = sys.modules.get(_FACADE_NAME)
    if module is None:
        module = importlib.import_module(_FACADE_NAME)
    return module


def generated_at_now() -> str:
    """Timestamp helper that honors a facade-level fake ``datetime`` clock."""
    surface = facade()
    return surface.datetime.now(surface.UTC).replace(microsecond=0).isoformat()

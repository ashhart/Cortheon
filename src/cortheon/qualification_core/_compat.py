"""Late-bound lookups of facade-level callables.

Tests and callers monkeypatch ``_run_cell``, ``_repository_fingerprint``, and
``_git_revision`` on the ``cortheon.qualification_factory`` facade, which is
what the pre-split god file resolved them from. Both call sites that consumed
those module globals resolve them through the facade namespace at call time so
the patches keep working: ``run_qualification`` in ``report``, and the pre/post
workspace fingerprint checks around a cell in ``execution``.

The lookup is deliberately dynamic: a module-level import of the facade would
close an import cycle, because the facade imports this package.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

_FACADE_NAME = "cortheon.qualification_factory"


def facade() -> ModuleType:
    """Return the compatibility facade module, importing it if needed."""
    module = sys.modules.get(_FACADE_NAME)
    if module is None:
        module = importlib.import_module(_FACADE_NAME)
    return module

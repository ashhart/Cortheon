"""Late-bound lookup of the ``cortheon.parity`` facade.

The pre-split god file read ``UNIVERSAL_SCALE_REQUIREMENTS`` from its own
module globals, so a test that rebound the name on ``cortheon.parity``
substituted a reduced test-scale policy for the whole evaluation. Both
consumers -- the release-scale predicate and the evidence it reports --
resolve the name through the facade namespace at call time so those
rebindings keep working after the split.

The lookup is deliberately dynamic: a module-level import of the facade would
close an import cycle, because the facade imports this package.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

_FACADE_NAME = "cortheon.parity"


def facade() -> ModuleType:
    """Return the compatibility facade module, importing it if needed."""
    module = sys.modules.get(_FACADE_NAME)
    if module is None:
        module = importlib.import_module(_FACADE_NAME)
    return module

"""Late-bound lookup of the ``cortheon.parity_pack`` facade.

The pre-split god file read ``datetime`` from its own module globals, so a
test that rebound the name on ``cortheon.parity_pack`` froze the sealing
clock for the whole tool -- which is how the campaign suite seals packs in
2026 and lets them expire before the real wall clock. The clock helpers
resolve the name through the facade namespace at call time so those
rebindings keep working after the split.

The lookup is deliberately dynamic: a module-level import of the facade would
close an import cycle, because the facade imports this package.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

_FACADE_NAME = "cortheon.parity_pack"


def facade() -> ModuleType:
    """Return the compatibility facade module, importing it if needed."""
    module = sys.modules.get(_FACADE_NAME)
    if module is None:
        module = importlib.import_module(_FACADE_NAME)
    return module

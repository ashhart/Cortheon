from __future__ import annotations

import importlib
import sys
from types import ModuleType

_FACADE_NAME = "cortheon.source_planner"


def facade() -> ModuleType:
    module = sys.modules.get(_FACADE_NAME)
    if module is None:
        module = importlib.import_module(_FACADE_NAME)
    return module

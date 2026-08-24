"""Late-bound access to the stable decision facade."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def facade() -> ModuleType:
    return import_module("cortheon.decision")

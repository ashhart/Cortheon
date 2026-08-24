"""Late access to the public GitHub connector facade and patch points."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def facade() -> ModuleType:
    return import_module("cortheon.connectors.github")

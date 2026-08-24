"""Late access to the public docs-reader facade and its patch points."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def facade() -> ModuleType:
    return import_module("cortheon.docs_reader")

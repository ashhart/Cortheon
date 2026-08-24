"""Keep a facade monkeypatch reaching the module that resolves the name.

The pre-split ``cognitive_mcp`` resolved every lookup through its own globals,
so patching that module changed what the running code saw and assigning the
original back restored it. After the split each lookup resolves in the
``cognitive_mcp_core`` module that owns the caller, so a facade assignment
alone would change nothing. The bridge mirrors an assignment into every
implementation module still holding the object being replaced, which makes
undo the same operation with the two objects swapped. Identity decides, not
the name: a module that rebound the name itself keeps its own binding.
See tests/test_cognitive_mcp_compat.py for what this preserves, seam by seam.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any


def install_facade_patch_bridge(facade: ModuleType) -> None:
    """Mirror facade assignment into the implementation package."""

    missing = object()
    prefix = f"{__name__.rpartition('.')[0]}."

    class _PatchBridgeModule(ModuleType):
        """Facade module type whose assignments reach the owning modules."""

        def __setattr__(self, name: str, value: Any) -> None:
            replaced = self.__dict__.get(name, missing)
            if replaced is not missing:
                for owner, module in list(sys.modules.items()):
                    if not owner.startswith(prefix) or not isinstance(module, ModuleType):
                        continue
                    if module.__dict__.get(name, missing) is replaced:
                        ModuleType.__setattr__(module, name, value)
            ModuleType.__setattr__(self, name, value)

    facade.__class__ = _PatchBridgeModule

"""Installer constants, errors, and result values."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any

SUPPORTED_HOSTS = ("opencode", "pi", "codex", "generic")
MARKETPLACE_NAME = "cortheon-local"
LEGACY_PACKAGE_NAMES = frozenset({"learn_layer", "cortheon"})


class InstallError(RuntimeError):
    """A host integration could not be installed safely."""


@dataclass(frozen=True, slots=True)
class InstallResult:
    host: str
    status: str
    target: str | None
    details: dict[str, Any]

    def public(self) -> dict[str, Any]:
        return asdict(self)


def install_facade_patch_bridge(facade: ModuleType) -> None:
    missing = object()
    prefix = f"{__name__.rpartition('.')[0]}."

    class _PatchBridgeModule(ModuleType):
        def __setattr__(self, name: str, value: Any) -> None:
            replaced = self.__dict__.get(name, missing)
            if replaced is not missing:
                for owner, module in list(sys.modules.items()):
                    if (
                        owner.startswith(prefix)
                        and isinstance(module, ModuleType)
                        and module.__dict__.get(name, missing) is replaced
                    ):
                        ModuleType.__setattr__(module, name, value)
            ModuleType.__setattr__(self, name, value)

    facade.__class__ = _PatchBridgeModule


InstallError.__module__ = "cortheon.cognitive_install"
InstallResult.__module__ = "cortheon.cognitive_install"
InstallResult.public.__module__ = "cortheon.cognitive_install"

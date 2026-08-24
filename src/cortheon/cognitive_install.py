"""Idempotent Cortheon host installers."""

from __future__ import annotations

import json  # noqa: F401
import os  # noqa: F401
import shlex  # noqa: F401
import shutil  # noqa: F401
import subprocess  # noqa: F401
import sys
import tempfile  # noqa: F401
from collections.abc import Iterable  # noqa: F401
from dataclasses import asdict, dataclass  # noqa: F401
from importlib.resources import files  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any  # noqa: F401

from cortheon.cognitive_install_core.config import (  # noqa: F401
    _atomic_json,
    _configured_codex_plugins,
    _is_packaged_adapter_reference,
    _load_json_config,
    _pi_config_home,
    _xdg_config_home,
    _xdg_data_home,
    package_asset,
)
from cortheon.cognitive_install_core.hosts import (  # noqa: F401
    _configured_codex_marketplaces,
    _normalize_hosts,
    _preflight_hosts,
    _preflight_json_string_list,
    _run,
    generic_mcp_config,
    install_codex,
    install_hosts,
    install_opencode,
    install_pi,
)
from cortheon.cognitive_install_core.lifecycle import (  # noqa: F401
    _uninstall_adapter,
    _uninstall_codex,
    host_installation_status,
    uninstall_hosts,
)
from cortheon.cognitive_install_core.model import (  # noqa: F401
    LEGACY_PACKAGE_NAMES,
    MARKETPLACE_NAME,
    SUPPORTED_HOSTS,
    InstallError,
    InstallResult,
    install_facade_patch_bridge,
)

install_facade_patch_bridge(sys.modules[__name__])
del install_facade_patch_bridge

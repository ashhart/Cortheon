from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cortheon.cognitive_install import (
    InstallError,
    install_opencode,
)


def test_installer_refuses_to_rewrite_malformed_host_config():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = root / "opencode.json"
        original = "{ // jsonc remains user-owned\n"
        config.write_text(original)

        with pytest.raises(InstallError, match="not strict JSON"):
            install_opencode(scope="project", project_dir=root, dry_run=False)

        assert config.read_text() == original


def test_installer_refuses_symlinked_host_configuration():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        owned = root / "owned.json"
        owned.write_text('{"theme":"owned"}\n')
        config = root / "opencode.json"
        config.symlink_to(owned)

        with pytest.raises(InstallError, match="symlinked"):
            install_opencode(scope="project", project_dir=root, dry_run=False)

        assert json.loads(owned.read_text()) == {"theme": "owned"}

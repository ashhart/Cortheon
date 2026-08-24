from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).parents[1]
INSTALLER = ROOT / "install"


def test_machine_installer_is_the_documented_entry_point() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert INSTALLER.stat().st_mode & 0o111
    assert "./install" in readme
    assert "source .venv/bin/activate" not in readme
    assert "python -m pip install ." not in readme
    assert "cortheon install --host pi" in readme
    assert "cortheon install --host opencode" in readme


def test_machine_installer_has_no_fixed_user_path() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    assert "/Users/" not in source
    assert "$HOME/.local/bin" in source
    assert "$HOME/.local/share" in source
    assert "uv tool install --force" in source
    assert os.access(INSTALLER, os.X_OK)

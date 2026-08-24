from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from cortheon.cognitive_install import (
    install_opencode,
)


def test_opencode_install_is_atomic_preserving_and_idempotent():
    with tempfile.TemporaryDirectory() as directory:
        config_home = Path(directory)
        config = config_home / "opencode" / "opencode.json"
        config.parent.mkdir(parents=True)
        config.write_text('{"theme":"dark","plugin":["file:///existing.js"]}\n')

        with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(config_home)}):
            first = install_opencode(
                scope="user",
                project_dir=config_home,
                dry_run=False,
            )
            second = install_opencode(
                scope="user",
                project_dir=config_home,
                dry_run=False,
            )

        payload = json.loads(config.read_text())
        assert payload["theme"] == "dark"
        assert payload["plugin"][0] == "file:///existing.js"
        assert payload["plugin"][1].endswith("/opencode_plugin.js")
        assert (config.parent / "opencode.json.cortheon.bak").exists()
        assert first.status == "installed"
        assert second.status == "present"


def test_opencode_install_replaces_stale_cortheon_adapters_without_duplicates():
    with tempfile.TemporaryDirectory() as directory:
        config_home = Path(directory)
        config = config_home / "opencode" / "opencode.json"
        config.parent.mkdir(parents=True)
        config.write_text(
            json.dumps(
                {
                    "plugin": [
                        "file:///existing.js",
                        "file:///old/cortheon/opencode_plugin.js",
                        "file:///checkout/cortheon/opencode_plugin.js",
                    ]
                }
            )
        )

        with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(config_home)}):
            result = install_opencode(
                scope="user",
                project_dir=config_home,
                dry_run=False,
            )

        configured = json.loads(config.read_text())["plugin"]
        assert configured[0] == "file:///existing.js"
        assert len(configured) == 2
        assert configured[1].endswith("/cortheon/opencode_plugin.js")
        assert result.details["replaced_stale_plugins"] == 2

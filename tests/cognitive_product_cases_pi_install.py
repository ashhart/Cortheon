from __future__ import annotations

import json
import tempfile
from pathlib import Path

from cortheon.cognitive_install import (
    install_pi,
)


def test_pi_project_install_preserves_settings_and_enables_extension():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        settings = root / ".pi" / "settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"theme": "dark", "extensions": ["./existing.ts"]}))

        result = install_pi(scope="project", project_dir=root, dry_run=False)

        payload = json.loads(settings.read_text())
        assert payload["theme"] == "dark"
        assert payload["extensions"][0] == "./existing.ts"
        assert payload["extensions"][1].endswith("/pi_extension.ts")
        assert result.status == "installed"


def test_pi_install_replaces_legacy_adapter_without_loading_both():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        settings = root / ".pi" / "settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(
            json.dumps(
                {
                    "extensions": [
                        "./existing.ts",
                        "/tools/learn-layer/site-packages/learn_layer/pi_extension.ts",
                        "!/tools/cortheon/site-packages/cortheon/pi_extension.ts",
                    ]
                }
            )
        )

        result = install_pi(scope="project", project_dir=root, dry_run=False)

        configured = json.loads(settings.read_text())["extensions"]
        assert configured[0] == "./existing.ts"
        assert len(configured) == 2
        assert configured[1].endswith("/cortheon/pi_extension.ts")
        assert result.details["replaced_stale_extensions"] == 2

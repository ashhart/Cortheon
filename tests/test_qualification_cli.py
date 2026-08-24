"""``cortheon-qualify`` command line: machine-readable, path-free validation."""

import json

from qualification_support import _write_manifest

from cortheon.qualification_factory import main


def test_validate_only_is_machine_readable_and_has_no_repository_path(
    tmp_path,
    capsys,
):
    manifest_path = _write_manifest(tmp_path)

    result = main([str(manifest_path), "--validate-only"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert result == 0
    assert payload["valid"]
    assert payload["content_free"]
    assert str(tmp_path) not in captured.out

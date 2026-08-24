"""The build-only compressor must stay standard, deterministic, and build-only.

Zopfli is a packaging tool, not a runtime one: it searches harder for the
same DEFLATE and gzip streams zlib emits. These tests pin that the streams
decode with the standard library alone, that the same input always produces
the same bytes, that rewriting an archive preserves every member and its
metadata, and that nothing about it reaches the installed product.
"""

from __future__ import annotations

import gzip
import hashlib
import subprocess
import sys
import threading
import tomllib
import zipfile
import zlib
from pathlib import Path
from typing import Any

import pytest

import build_backend
from build_support.lean_commands import recompress_wheel
from build_support.lean_compress import deflate, gzip_bytes, zopfli_zip_streams

ROOT = Path(__file__).parents[1]
# Same seam build_support/lean_compress.py patches, reached the same way:
# the name is private, so it is read through a reference typed as the
# dynamic object it is rather than through the module's declared surface.
_ZIPFILE: Any = zipfile
SAMPLE = (
    b"".join(
        f'def f{index}(value: int) -> int:\n    """Doc."""\n    return value\n'.encode()
        for index in range(400)
    )
    + bytes(range(256)) * 8
)


def test_deflate_is_standard_and_never_worse_than_zlib() -> None:
    stream = deflate(SAMPLE)

    assert zlib.decompress(stream, -zlib.MAX_WBITS) == SAMPLE
    reference = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    assert len(stream) <= len(reference.compress(SAMPLE) + reference.flush())
    assert deflate(b"") == deflate(b"")
    assert zlib.decompress(deflate(b""), -zlib.MAX_WBITS) == b""


def test_gzip_container_is_standard_and_carries_no_timestamp() -> None:
    blob = gzip_bytes(SAMPLE)

    assert gzip.decompress(blob) == SAMPLE
    assert blob[:2] == b"\x1f\x8b" and blob[2] == 8
    # FLG has neither FNAME nor FEXTRA, and MTIME is zero, so the container
    # is a pure function of its payload rather than of the build machine.
    assert blob[3] == 0
    assert blob[4:8] == b"\x00\x00\x00\x00"


def test_recompression_is_deterministic_in_independent_processes() -> None:
    script = (
        f"import hashlib,sys;"
        f"sys.path.insert(0, {str(ROOT)!r});"
        f"from build_support.lean_compress import deflate, gzip_bytes;"
        f"data=open({str(ROOT / 'pyproject.toml')!r},'rb').read();"
        f"print(hashlib.sha256(deflate(data)).hexdigest(),"
        f"hashlib.sha256(gzip_bytes(data)).hexdigest())"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout.strip()
        for _attempt in range(2)
    }
    payload = (ROOT / "pyproject.toml").read_bytes()
    in_process = (
        f"{hashlib.sha256(deflate(payload)).hexdigest()} "
        f"{hashlib.sha256(gzip_bytes(payload)).hexdigest()}"
    )

    assert len(runs) == 1
    assert runs == {in_process}


def test_rewriting_an_archive_preserves_members_and_their_metadata(tmp_path: Path) -> None:
    archive_path = tmp_path / "sample-0.1.0-py3-none-any.whl"
    payloads = {
        "package/module.py": SAMPLE,
        "package/data.json": b'{"key": "value"}',
        "sample-0.1.0.dist-info/RECORD": b"package/module.py,,\n",
    }
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for index, (name, payload) in enumerate(payloads.items()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1 + index, 3, 4, 6))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o644 | 0o100000) << 16
            info.create_system = 3
            archive.writestr(info, payload)
    with zipfile.ZipFile(archive_path) as archive:
        before = {
            info.filename: (info.date_time, info.external_attr, info.create_system, info.CRC)
            for info in archive.infolist()
        }
    original_size = archive_path.stat().st_size

    recompress_wheel(archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert {name: archive.read(name) for name in archive.namelist()} == payloads
        after = {
            info.filename: (info.date_time, info.external_attr, info.create_system, info.CRC)
            for info in archive.infolist()
        }
        assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in archive.infolist())
    assert after == before
    assert archive_path.stat().st_size <= original_size


def test_the_compressor_is_declared_for_builds_only() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requires = configuration["build-system"]["requires"]
    test_requires = configuration["project"]["optional-dependencies"]["test"]

    assert any(item.startswith("zopfli") for item in requires)
    assert any(item.startswith("setuptools") for item in test_requires)
    assert any(item.startswith("zopfli") for item in test_requires)
    assert configuration["project"]["dependencies"] == []
    shipped = {
        path
        for path in (ROOT / "src/cortheon").rglob("*.py")
        if "zopfli" in path.read_text(encoding="utf-8")
    }
    assert shipped == set(), "the installed product must never reference the build compressor"


def test_the_in_tree_backend_supports_editable_developer_installs() -> None:
    assert callable(build_backend.build_editable)
    assert callable(build_backend.get_requires_for_build_editable)
    assert callable(build_backend.prepare_metadata_for_build_editable)


def test_wheel_compression_fails_closed_if_python_removes_the_private_seam(
    monkeypatch,
) -> None:
    monkeypatch.delattr(zipfile, "_get_compressor")

    with pytest.raises(SystemExit, match="no compressor seam"), zopfli_zip_streams():
        raise AssertionError("the context must not open without its compressor")


# Import the module setup.py imports, with zopfli hidden from the import
# system the way a --no-build-isolation environment without it would leave it.
WITHOUT_ZOPFLI = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})


class _Absent:
    def find_spec(self, name, path=None, target=None):
        if name == "zopfli" or name.startswith("zopfli."):
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
        return None


sys.meta_path.insert(0, _Absent())
for cached in [n for n in sys.modules if n == "zopfli" or n.startswith("zopfli.")]:
    del sys.modules[cached]
import build_support.lean_commands
print("FELL BACK TO A QUIETER COMPRESSOR")
"""


def test_a_build_without_the_compressor_fails_loudly_instead_of_falling_back() -> None:
    """A missing build dependency must stop the build with an actionable
    message, never silently produce a larger artifact whose size then depends
    on which machine ran the build."""

    completed = subprocess.run(
        [sys.executable, "-c", WITHOUT_ZOPFLI],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 1
    assert "FELL BACK" not in completed.stdout
    assert "zopfli compressor listed in [build-system].requires" in completed.stderr
    assert "pip install zopfli" in completed.stderr


def test_concurrent_entry_cannot_leak_the_patched_compressor() -> None:
    """The seam is one process-global name. Threads that entered and left in
    non-LIFO order would each restore what they saw on the way in, leaving the
    wrapper installed for the rest of the process; entry is serialized so that
    interleaving cannot happen."""

    original = _ZIPFILE._get_compressor
    patched = threading.Event()
    release = threading.Event()
    entered_second = threading.Event()

    def holder() -> None:
        with zopfli_zip_streams():
            patched.set()
            release.wait(30)

    def follower() -> None:
        patched.wait(30)
        with zopfli_zip_streams():
            entered_second.set()

    first, second = threading.Thread(target=holder), threading.Thread(target=follower)
    first.start()
    second.start()
    try:
        assert patched.wait(30)
        assert not entered_second.wait(0.5), "the second entry was not serialized"
    finally:
        release.set()
        first.join(30)
        second.join(30)

    assert not first.is_alive() and not second.is_alive()
    assert entered_second.is_set()
    assert _ZIPFILE._get_compressor is original

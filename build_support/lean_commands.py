"""Wheel- and data-level packaging commands for the lean Cortheon build.

Kept out of setup.py so the entry point stays small; everything here
operates only on build output.
"""

from __future__ import annotations

import io
import json
import os
import stat
import tarfile
import uuid
import zipfile
from pathlib import Path

from setuptools.command.bdist_wheel import bdist_wheel

from build_support.lean_compress import gzip_bytes, zopfli_zip_streams

# Permission bits an artifact is allowed to record. Whether a file is
# executable is real information about it; the remaining bits only record
# the umask the build happened to run under, so they are pinned. Every
# member is a regular file or a directory: archives here carry no devices,
# symlinks, or hard links.
FILE_MODE = 0o644
EXECUTABLE_MODE = 0o755


def minify_json(path: Path) -> None:
    """Rewrite a shipped JSON document compactly, preserving its value."""
    value = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _stable_mode(mode: int, *, directory: bool) -> int:
    """Return ``mode`` with only its executable bit and file kind kept."""

    if directory:
        return EXECUTABLE_MODE
    return EXECUTABLE_MODE if mode & 0o111 else FILE_MODE


def recompress_wheel(path: Path) -> None:
    """Rewrite a finished wheel with searched DEFLATE and a pinned mode."""

    # Members, their order, and their recorded timestamps carry over
    # unchanged, so RECORD hashes stay valid; each member's DEFLATE stream is
    # re-searched, and its permission bits are reduced to the executable bit
    # the installer acts on, which is the one part of the mode that is not
    # just the build machine's umask. The rewrite lands via a unique
    # temporary path and os.replace.
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with (
            zopfli_zip_streams(),
            zipfile.ZipFile(path, "r") as source,
            zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as target,
        ):
            for info in source.infolist():
                recorded = info.external_attr >> 16
                if recorded:
                    directory = info.is_dir()
                    kind = stat.S_IFDIR if directory else stat.S_IFREG
                    info.external_attr = (
                        _stable_mode(recorded, directory=directory) | kind
                    ) << 16 | (info.external_attr & 0xFFFF)
                target.writestr(info, source.read(info.filename))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def repack_sdist(path: Path, timestamp: int) -> None:
    """Rewrite a finished sdist as a function of its contents alone."""

    # ``tarfile`` takes float ``os.stat`` mtimes, and a float mtime forces a
    # 1 KiB pax extended header per member purely to carry sub-second
    # precision. GNU_FORMAT writes whole-second mtimes in the base header and
    # never consults pax_headers, so no extended header is emitted at all.
    #
    # Member order, names, and contents carry over unchanged, so the archive
    # still extracts to the same tree. The fields that describe the machine
    # rather than the release -- modification times, owner and group ids and
    # names, and the umask-dependent part of each mode -- are replaced by
    # fixed values, so two builds of the same sources produce the same
    # archive. The tar is assembled uncompressed so the gzip layer can be
    # searched; gzip's own optional name and mtime fields are then simply
    # absent, which is what makes the container byte-identical build to build.
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        payloads: list[bytes] = []
        with tarfile.open(path, "r:gz") as source:
            members = source.getmembers()
            for member in members:
                if not member.isreg():
                    payloads.append(b"")
                    continue
                handle = source.extractfile(member)
                if handle is None:
                    raise SystemExit(f"repack_sdist: {member.name} is unreadable")
                payloads.append(handle.read())
        assembled = io.BytesIO()
        with tarfile.open(fileobj=assembled, mode="w", format=tarfile.GNU_FORMAT) as target:
            for member, payload in zip(members, payloads, strict=True):
                member.mtime = timestamp
                member.uid = member.gid = 0
                member.uname = member.gname = ""
                member.mode = _stable_mode(member.mode, directory=member.isdir())
                target.addfile(member, io.BytesIO(payload) if member.isreg() else None)
        temporary.write_bytes(gzip_bytes(assembled.getvalue()))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class LeanWheel(bdist_wheel):
    """bdist_wheel that maximally recompresses the finished artifact."""

    def run(self) -> None:
        # Snapshot the recorded bdist_wheel artifacts before building. On a
        # repeated build the new wheel overwrites an existing file with the
        # same name, so comparing directory listings before and after cannot
        # identify it; the freshly appended dist_files entry can. Entries are
        # (command, python version, wheel path).
        wheel_files = self.distribution.dist_files
        before = [entry for entry in wheel_files if entry[0] == "bdist_wheel"]
        super().run()
        produced = [entry for entry in wheel_files if entry[0] == "bdist_wheel"]
        for entry in before:
            produced.remove(entry)
        if len(produced) != 1:
            raise SystemExit(
                "LeanWheel: expected exactly one new bdist_wheel artifact, "
                f"found {len(produced)}; refusing to recompress"
            )
        recompress_wheel(Path(produced[0][2]))

"""Exact archive identity and extraction for the frozen comparator."""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

FROZEN_COMMIT = "19d035c4e8c6df52be861636029e18a9a1d2d777"
FROZEN_TREE = "ee8a2cbc19806123ca39686c61428a0fd76f8d02"
ARCHIVE_SHA256 = "a37de1098022b3d438e9e96d10e747db76a88e902087783f1896d88e5cc407bd"
ARCHIVE_BYTES = 147_266
MEMBER_SHA256 = {
    "src/cortheon/__init__.py": "9177f19d157634ee412b5257ae48b7d1424125975f4aa15adc970a27a146e55d",
    "src/cortheon/cognitive_graph.py": "34ae000abc999a9c9fc56a8c19df94b0406498bc1b565e124ba12eac039c55e4",
    "src/cortheon/cognitive_program.py": "301dbd34f2ec7e5269f05c16e7161bbe30fc130c5a69727ef2a186d55b0ad801",
    "src/cortheon/cognitive_protocol.py": "4b2961eb62350bb40f798a5afc6d5f68cd2aa221f80228ae25d397456ea5cd58",
    "src/cortheon/cognitive_repair.py": "9074d919dbacce6261419d98d62a2e6b83563cac64df22d08ca118dbf75ee2a1",
    "src/cortheon/sanitize.py": "a7e4d323a99666c6112345ddb036be2e4a2b7289ab91debd7f11704eb34623b8",
    "src/cortheon/cognitive_hooks.py": "e174cfd56ad03db7c1f275d1e7290096a4cad702e93d8c9490d437d117e2a879",
    "src/cortheon/cognitive_runtime.py": "3027ab2b007cbe35ac340e37de0ed905724e83a6d59851822f8b47b3a88dbbc4",
    "src/cortheon/cognitive_http.py": "8ae1f7dca5f24cc220bdf90c5d95eede52ab0f269167f0577de0136c959d41e0",
    "src/cortheon/pi_extension.ts": "804ef97995bce5b013f0d8a0bab554d1224c26f2fa56777d4681185ef3d1d621",
    "src/cortheon/opencode_plugin.js": "40f52246d06fab9689178ddee38dc5a3a839ba0a5c5dfc0a93e2b88e26c65e23",
}
_DIRECTORIES = {"src", "src/cortheon"}
_ROOT = Path(__file__).parents[3]
ARCHIVE = _ROOT / "benchmarks/frozen/old_planner_19d035c.tar.gz"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive_available() -> bool:
    return bool(
        ARCHIVE.is_file()
        and ARCHIVE.stat().st_size == ARCHIVE_BYTES
        and _sha(ARCHIVE.read_bytes()) == ARCHIVE_SHA256
    )


def extract_verified(destination: Path) -> None:
    if not archive_available():
        raise ValueError("frozen old-planner archive is missing or has the wrong digest")
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        members = archive.getmembers()
        if len({member.name for member in members}) != len(members):
            raise ValueError("frozen old-planner archive has duplicate members")
        names = {member.name for member in members}
        if names != {*MEMBER_SHA256, *_DIRECTORIES}:
            raise ValueError("frozen old-planner archive membership is invalid")
        for member in members:
            if member.name in _DIRECTORIES:
                if not member.isdir():
                    raise ValueError("frozen archive directory has the wrong type")
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("frozen archive member is not a regular file")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("frozen archive member cannot be read")
            data = source.read()
            if _sha(data) != MEMBER_SHA256[member.name]:
                raise ValueError("frozen archive member digest mismatch")
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

"""Maximal standard-compression for finished Cortheon artifacts.

Build-only, and content-preserving: every byte a reader sees after
decompression is the byte the packaging commands produced.  Zopfli emits
ordinary DEFLATE and gzip streams -- the same formats ``zlib`` emits, just
searched harder -- so wheels stay readable by ``zipfile``/``unzip`` and
source archives by ``tarfile``/``gunzip``.  Nothing here is installed, and
the product keeps zero runtime dependencies.
"""

from __future__ import annotations

import threading
import zipfile
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:
    from zopfli import gzip as zopfli_gzip
    from zopfli import zlib as zopfli_zlib
except ModuleNotFoundError as missing:
    # Fail closed rather than falling back: a quieter compressor would still
    # produce a correct artifact, just a larger one, and the size contract
    # would then depend on which machine ran the build.
    raise SystemExit(
        "Cortheon's build needs the zopfli compressor listed in "
        "[build-system].requires. An isolated PEP 517 build installs it; a "
        "--no-build-isolation build has to provide it (pip install zopfli)."
    ) from missing

# Zopfli's own default is 15. Measured on this project's wheel, five
# iterations reach 222,784 member bytes and fifteen reach 222,704: a further
# 0.04% for 20% more build time, so the build stops where the curve flattens.
ITERATIONS = 5

# ``zipfile`` picks its compressor through one module-level factory. That
# name is not public API, so it is reached through a reference typed as the
# dynamic object it is; ``zopfli_zip_streams`` refuses to run if it is gone.
_ZIPFILE: Any = zipfile

# Replacing that factory mutates one process-global name, so entry has to be
# serialized. Two threads that entered and then left in non-LIFO order would
# each restore what they saw on the way in -- the later one restoring the
# earlier one's wrapper -- and the standard-library compressor would never go
# back. A re-entrant lock keeps same-thread nesting working, where the
# save/restore pairs really are LIFO.
_PATCH_LOCK = threading.RLock()


def deflate(data: bytes) -> bytes:
    """Return the smallest correct raw DEFLATE stream for ``data``.

    Zopfli only emits wrapped output, so the RFC 1950 two-byte header and
    trailing Adler-32 are removed to leave the raw RFC 1951 stream a ZIP
    member holds. The result is decoded back before it is returned, and
    ``zlib`` at level 9 is kept as a floor, so a member is never larger
    than the plain build produced.
    """

    wrapped = zopfli_zlib.compress(data, numiterations=ITERATIONS)
    searched = wrapped[2:-4]
    if zlib.decompress(searched, -zlib.MAX_WBITS) != data:
        raise SystemExit("lean_compress: Zopfli DEFLATE stream did not round-trip")
    compressor = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    baseline = compressor.compress(data) + compressor.flush()
    return searched if len(searched) <= len(baseline) else baseline


def gzip_bytes(data: bytes) -> bytes:
    """Return the smallest correct gzip container for ``data``."""

    searched = zopfli_gzip.compress(data, numiterations=ITERATIONS)
    if zlib.decompress(searched, zlib.MAX_WBITS | 16) != data:
        raise SystemExit("lean_compress: Zopfli gzip stream did not round-trip")
    return searched


class _BufferedZopfliCompressor:
    """``zlib.compressobj`` stand-in that searches the whole member at once.

    Zopfli needs the complete input, so writes are buffered and the stream
    is produced at flush time. ``zipfile`` sums the bytes returned from
    both calls, so its recorded compressed size stays correct.
    """

    def __init__(self) -> None:
        self._chunks: list[bytes] = []

    def compress(self, data: bytes) -> bytes:
        self._chunks.append(data)
        return b""

    def flush(self) -> bytes:
        return deflate(b"".join(self._chunks))


@contextmanager
def zopfli_zip_streams() -> Iterator[None]:
    """Make ``zipfile`` write Zopfli DEFLATE streams inside this block.

    Only the compressor is replaced: ``zipfile`` still writes every local
    header, CRC, size, and central-directory record itself, so the archive
    layout is exactly the one the standard library produces.
    """

    with _PATCH_LOCK:
        original = getattr(zipfile, "_get_compressor", None)
        if original is None:
            raise SystemExit(
                "lean_compress: this Python's zipfile has no compressor seam, so a "
                "wheel cannot be written with Zopfli DEFLATE streams"
            )

        def compressor(compress_type: int, compresslevel: int | None = None) -> object:
            if compress_type == zipfile.ZIP_DEFLATED:
                return _BufferedZopfliCompressor()
            return original(compress_type, compresslevel)

        _ZIPFILE._get_compressor = compressor
        try:
            yield
        finally:
            _ZIPFILE._get_compressor = original

"""One fixed timestamp for every archive a build writes."""

# An artifact should be a function of its sources, not of the minute someone
# built it. Wheels already take their member timestamps from
# ``SOURCE_DATE_EPOCH`` when it is set, so pinning that variable is the
# standard way to reach that, and the same value is stamped into the source
# archive's tar headers. An explicit setting from the caller always wins, so
# a release that pins its own epoch keeps it.
#
# What this cannot pin is the wheel's own dist-info WHEEL file, which records
# the setuptools version that wrote it. Two builds of the same sources on
# different setuptools releases therefore differ in that member's Generator
# line and in the RECORD row hashing it, and in nothing else.

from __future__ import annotations

import os

ENVIRONMENT_VARIABLE = "SOURCE_DATE_EPOCH"
# 2025-01-01T00:00:00Z. The instant carries no meaning; being constant, and
# later than the 1980 epoch ZIP timestamps start at, is the whole point.
DEFAULT_EPOCH = 1_735_689_600


def pin_source_date_epoch() -> int:
    """Return this build's timestamp, exporting it for the tools that read it."""

    raw = os.environ.get(ENVIRONMENT_VARIABLE)
    if raw is None:
        os.environ[ENVIRONMENT_VARIABLE] = str(DEFAULT_EPOCH)
        return DEFAULT_EPOCH
    try:
        stamp = int(raw)
    except ValueError:
        raise SystemExit(
            f"{ENVIRONMENT_VARIABLE}={raw!r} is not an integer number of seconds "
            "since the Unix epoch, so the build cannot date its archives"
        ) from None
    if stamp < 0:
        raise SystemExit(f"{ENVIRONMENT_VARIABLE}={raw!r} predates the Unix epoch")
    return stamp

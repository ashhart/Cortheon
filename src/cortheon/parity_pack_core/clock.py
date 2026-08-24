"""The sealing clock, read late so a frozen-clock test still steers it.

Every timestamp a pack carries is produced here. ``datetime`` is resolved
through the facade on each call rather than imported into this module,
because the campaign suite rebinds ``cortheon.parity_pack.datetime`` to a
fake class to seal packs at a chosen instant; an import-bound name would
silently ignore that and stamp real wall-clock times instead.
"""

from __future__ import annotations

from datetime import UTC, datetime

from cortheon.parity_pack_core._compat import facade


def _clock() -> type[datetime]:
    return facade().datetime


def issued_at() -> str:
    """The pack's ``created_at``: whole-second UTC, ISO-8601."""

    return _clock().now(UTC).replace(microsecond=0).isoformat()


def require_future_expiry(expires_at: str) -> None:
    """Reject an expiry that is unparseable, naive, or already past."""

    try:
        parsed_expiry = _clock().fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("expires-at must be ISO-8601") from exc
    try:
        expiry_is_future = parsed_expiry > _clock().now(UTC)
    except TypeError as exc:
        raise ValueError("expires-at must include a timezone") from exc
    if not expiry_is_future:
        raise ValueError("expires-at must be in the future")

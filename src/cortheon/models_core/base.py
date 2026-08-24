from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from cortheon.models_core._compat import facade


def utc_now() -> datetime:
    api = facade()
    return api.datetime.now(api.UTC).replace(microsecond=0)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        parsed = facade().datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=facade().UTC)
    return parsed.astimezone(facade().UTC)


def to_jsonable(value: Any) -> Any:
    api = facade()
    if isinstance(value, api.datetime):
        return value.isoformat()
    if isinstance(value, api.Enum):
        return value.value
    if api.is_dataclass(value) and not isinstance(value, type):
        return {key: api.to_jsonable(item) for key, item in api.asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): api.to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [api.to_jsonable(item) for item in value]
    return value


class EvidenceStatus(StrEnum):
    CURRENT = "current"
    STALE = "stale"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"
    UNVERIFIED = "unverified"


class SupportLevel(StrEnum):
    OBSERVED = "observed"
    VERIFIED = "verified"
    INFERRED = "inferred"
    FAILED = "failed"


@dataclass(slots=True)
class Evidence:
    claim: str
    source_type: str
    source_url: str | None
    package: str | None = None
    version: str | None = None
    support: SupportLevel = SupportLevel.OBSERVED
    status: EvidenceStatus = EvidenceStatus.CURRENT
    retrieved_at: datetime = field(default_factory=utc_now)
    expires_at: datetime | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.expires_at is None:
            self.expires_at = self.retrieved_at + facade().timedelta(days=7)

    def refresh_status(self, now: datetime | None = None) -> EvidenceStatus:
        api = facade()
        current_time = now or api.utc_now()
        if self.status in {
            api.EvidenceStatus.SUPERSEDED,
            api.EvidenceStatus.DISPUTED,
            api.EvidenceStatus.UNVERIFIED,
        }:
            return self.status
        if self.expires_at and self.expires_at < current_time:
            self.status = api.EvidenceStatus.STALE
        return self.status

"""Current-web oracle with freshness, origin, and contradiction binding."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from cortheon.parity_benchmark_core.oracle_common import closed_object, record_map, string_set


def grade_current_web(
    oracle: dict[str, Any],
    answer: dict[str, Any],
    *,
    now: datetime | None = None,
) -> list[str]:
    if not closed_object(answer, {"as_of", "sources", "claims", "contradictions"}, {"summary"}):
        return ["invalid_answer_schema"]
    failures = validate_current_web_oracle(oracle, now=now)
    if failures:
        return failures
    if answer.get("as_of") != oracle.get("as_of"):
        failures.append("wrong_as_of")
    expected_sources = record_map(oracle.get("sources"), _SOURCE_FIELDS, key="canonical_url")
    observed_sources = record_map(answer.get("sources"), _ANSWER_SOURCE_FIELDS, key="canonical_url")
    if (
        expected_sources is None
        or observed_sources is None
        or set(expected_sources) != set(observed_sources)
    ):
        failures.append("wrong_source_graph")
    expected_claims = record_map(oracle.get("claims"), ("id", "value", "source_urls"))
    observed_claims = record_map(answer.get("claims"), ("id", "value", "source_urls"))
    if (
        expected_claims is None
        or observed_claims is None
        or set(expected_claims) != set(observed_claims)
        or any(
            observed_claims[key].get("value") != expected_claims[key].get("value")
            or string_set(observed_claims[key].get("source_urls"), minimum=2)
            != string_set(expected_claims[key].get("source_urls"), minimum=2)
            for key in expected_claims
        )
    ):
        failures.append("wrong_current_claims")
    expected_conflicts = record_map(
        oracle.get("contradictions"),
        ("claim_id", "source_url", "rejected_value", "resolved_by_url"),
        key="claim_id",
    )
    observed_conflicts = record_map(
        answer.get("contradictions"),
        ("claim_id", "source_url", "rejected_value", "resolved_by_url"),
        key="claim_id",
    )
    if expected_conflicts != observed_conflicts:
        failures.append("wrong_contradiction_resolution")
    elif (
        expected_conflicts is not None
        and expected_sources is not None
        and any(
            expected_sources.get(item["resolved_by_url"], {}).get("authority") != "primary"
            or expected_sources.get(item["source_url"], {}).get("authority") == "primary"
            for item in expected_conflicts.values()
        )
    ):
        failures.append("primary_precedence_not_proven")
    return failures


def validate_current_web_oracle(
    oracle: dict[str, Any], *, now: datetime | None = None
) -> list[str]:
    failures: list[str] = []
    try:
        as_of = _timestamp(oracle.get("as_of"))
        revalidated = _timestamp(oracle.get("revalidated_at"))
        valid_until = _timestamp(oracle.get("valid_until"))
    except ValueError:
        return ["invalid_freshness_attestation"]
    current = now or datetime.now(UTC)
    if not as_of <= revalidated <= current <= valid_until:
        failures.append("stale_or_unvalidated_truth")
    computed_truth = _truth_digest(oracle)
    if oracle.get("truth_digest") != computed_truth:
        failures.append("invalid_truth_digest")
    if oracle.get("truth_digest") != oracle.get("revalidated_truth_digest"):
        failures.append("truth_changed_since_seal")
    sources = record_map(oracle.get("sources"), _SOURCE_FIELDS, key="canonical_url")
    acquisition = _validate_acquisition(oracle.get("acquisition_attestation"), sources or {})
    if acquisition is not None:
        failures.append(acquisition)
    equivalence = _origin_equivalence(oracle.get("origin_equivalence"))
    if sources is None or equivalence is None:
        failures.append("invalid_origin_graph")
    elif (
        len({item.get("origin_id") for item in sources.values()}) < 2
        or len({item.get("syndication_group") for item in sources.values()}) < 2
    ):
        failures.append("insufficient_independent_origins")
    elif any(
        _canonical_url(item.get("canonical_url")) != item.get("canonical_url")
        or _url_host(str(item.get("canonical_url")))
        not in equivalence.get(str(item.get("origin_id")), set())
        or re.fullmatch(r"[0-9a-f]{64}", str(item.get("content_sha256"))) is None
        for item in sources.values()
    ):
        failures.append("noncanonical_source_url")
    return failures


def validate_pack_web_authority(metadata: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    evaluator = metadata.get("evaluator")
    for case in cases:
        if case.get("task_class") != "current_web_research":
            continue
        attestation = case["grader"]["oracle"].get("acquisition_attestation")
        if not isinstance(attestation, dict) or attestation.get("evaluator_id") != evaluator:
            raise ValueError("current-web acquisition was not attested by the declared evaluator")


def validate_public_web_prompt(case: dict[str, Any]) -> None:
    prompt = case.get("prompt")
    oracle = case["grader"]["oracle"]
    required = ("as_of", "sources", "canonical_url", "claims", "contradictions")
    if (
        not isinstance(prompt, str)
        or str(oracle.get("as_of")) not in prompt
        or any(field not in prompt for field in required)
    ):
        raise ValueError("current-web prompt omits its public as_of or response schema")


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError
    return parsed.astimezone(UTC)


def _canonical_url(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) > 2048:
        return None
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return None
    host = parsed.hostname.casefold()
    port = f":{parsed.port}" if parsed.port and parsed.port != 443 else ""
    return urlunsplit(("https", host + port, parsed.path or "/", parsed.query, ""))


def _url_host(value: str) -> str:
    return (urlsplit(value).hostname or "").casefold()


def _origin_equivalence(value: Any) -> dict[str, set[str]] | None:
    records = record_map(value, ("id", "hosts"))
    if records is None:
        return None
    groups: dict[str, set[str]] = {}
    all_hosts: set[str] = set()
    for identifier, record in records.items():
        hosts = string_set(record.get("hosts"), minimum=1)
        if hosts is None:
            return None
        normalized = {host.casefold() for host in hosts}
        if normalized & all_hosts:
            return None
        groups[identifier] = normalized
        all_hosts |= normalized
    return groups


_SOURCE_FIELDS = (
    "canonical_url",
    "origin_id",
    "syndication_group",
    "published_at",
    "retrieved_at",
    "authority",
    "content_sha256",
)

_ANSWER_SOURCE_FIELDS = ("canonical_url",)


def _truth_digest(oracle: dict[str, Any]) -> str:
    sources = oracle.get("sources")
    payload = {
        "sources": [
            {
                "canonical_url": item.get("canonical_url"),
                "content_sha256": item.get("content_sha256"),
            }
            for item in sources or []
            if isinstance(item, dict)
        ],
        "claims": oracle.get("claims"),
        "contradictions": oracle.get("contradictions"),
    }
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_acquisition(value: Any, sources: dict[str, dict[str, Any]]) -> str | None:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "evaluator_id",
        "policy_sha256",
        "records",
    }:
        return "missing_evaluator_acquisition_attestation"
    if (
        value.get("schema_version") != 1
        or not isinstance(value.get("evaluator_id"), str)
        or not value["evaluator_id"]
        or re.fullmatch(r"[0-9a-f]{64}", str(value.get("policy_sha256"))) is None
    ):
        return "invalid_evaluator_acquisition_attestation"
    records = record_map(
        value.get("records"),
        (
            "source_url",
            "requested_url",
            "final_url",
            "redirect_chain",
            "initial_sha256",
            "revalidated_sha256",
            "acquired_at",
            "revalidated_at",
        ),
        key="source_url",
    )
    if records is None or set(records) != set(sources):
        return "incomplete_evaluator_acquisition_attestation"
    for source_url, source in sources.items():
        record = records[source_url]
        redirects = record.get("redirect_chain")
        if (
            record.get("final_url") != source.get("canonical_url")
            or _canonical_url(record.get("requested_url")) is None
            or not isinstance(redirects, list)
            or any(_canonical_url(url) != url for url in redirects)
            or record.get("initial_sha256") != source.get("content_sha256")
            or record.get("revalidated_sha256") != source.get("content_sha256")
            or record.get("revalidated_at") != source.get("retrieved_at")
        ):
            return "acquisition_does_not_match_current_source_bytes"
        try:
            if _timestamp(record.get("acquired_at")) > _timestamp(record.get("revalidated_at")):
                return "invalid_acquisition_order"
        except ValueError:
            return "invalid_acquisition_order"
    return None

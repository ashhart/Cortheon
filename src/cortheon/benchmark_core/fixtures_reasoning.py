"""Held-out abductive-synthesis and ambiguity-resolution fixtures."""

from __future__ import annotations

import hashlib
import random

from cortheon.benchmark_core.fixtures_reasoning_ambiguity import _ambiguity_definitions
from cortheon.benchmark_core.fixtures_reasoning_synthesis import _novel_synthesis_definitions
from cortheon.benchmark_core.models import ReasoningCase


def _reasoning_derived_relations(
    name: str,
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    """Return hidden relation-level obligations for a synthesis case."""

    relations: dict[str, tuple[tuple[tuple[str, ...], ...], ...]] = {
        "weekend_activation": (
            (
                ("legacy token broker", "legacy broker"),
                ("500", "capacity", "limit"),
                ("900", "burst"),
            ),
        ),
        "regional_refunds": (
            (
                ("eea", "vat"),
                ("invoice line", "per line", "line by line"),
                ("round", "one cent", "cent"),
                ("refund", "discrepancy"),
            ),
        ),
        "overnight_sensor_dropout": (
            (
                ("low power mode", "power saving"),
                ("02 00", "2 00", "overnight"),
                ("high frequency sampling", "sampling"),
                ("dropout", "samples disappear", "disabled"),
            ),
        ),
        "top_of_hour_queue_stall": (
            (
                ("hourly reconciliation", "top of hour", "minute 00"),
                ("tenant id", "tenant key"),
                ("hot partition", "saturated partition", "throttled partition"),
                ("large tenant", "queue stall"),
            ),
        ),
        "compact_nonce_collision": (
            (
                ("northstar", "android v9"),
                ("first 8", "eight character", "truncat"),
                ("household prefix", "shared prefix", "household member"),
                ("cache collision", "same cache key", "wrong member token"),
            ),
            (
                ("parallel refresh", "concurrent refresh"),
                ("cache collision", "same cache key"),
                ("authentication failure", "wrong member token", "cross account token"),
            ),
        ),
        "cold_closure_gap": (
            (
                ("quartz", "hub b"),
                ("22", "minus 22", "below 15", "cold"),
                ("p7", "polymer closure"),
                ("contract", "shrink", "seal gap"),
            ),
            (
                ("seal gap", "closure gap", "loss of compression"),
                ("integrity", "leak", "failed seal"),
            ),
        ),
        "renderer_lease_overlap": (
            (
                ("heron", "pdf export"),
                ("30", "lease"),
                ("38", "render"),
                ("expire", "retry", "duplicate"),
            ),
            (
                ("duplicate", "overlap", "second worker"),
                ("lock", "contention"),
                ("45", "timeout"),
            ),
        ),
        "orchid_rounding_drift": (
            (
                ("orchid", "annual prepaid"),
                ("micro credit", "small credit"),
                ("cad", "currency conversion", "conversion"),
                ("per allocation", "each allocation", "round"),
            ),
            (
                ("rounding", "rounded"),
                ("accumulat", "compound"),
                ("one cent", "cent", "statement mismatch"),
            ),
        ),
        "vega_namespace_mismatch": (
            (
                ("vega", "new namespace"),
                ("old namespace", "legacy namespace", "legacy", "observability"),
                ("exact match", "exact namespace"),
                ("alert", "rule"),
            ),
            (
                ("metrics", "telemetry"),
                ("still emitted", "continue emitting", "present"),
                ("namespace mismatch", "matcher mismatch", "exact namespace"),
                ("missing alert", "no alert", "fail to fire", "silently fail"),
            ),
        ),
        "inherited_deny_loss": (
            (
                ("lumen", "research workspace"),
                ("inherited deny", "parent deny"),
                ("flatten", "direct membership"),
                ("drop", "omit", "lost"),
            ),
            (
                ("deny", "restriction"),
                ("sync", "migration"),
                (
                    "unexpected access",
                    "gained access",
                    "access gained",
                    "access leak",
                    "unauthorized access",
                ),
            ),
        ),
        "stale_alias_embeddings": (
            (
                ("sparrow", "catalog revision"),
                ("alias", "synonym"),
                ("incremental index", "incremental update"),
                ("skip", "not re embed", "stale embedding"),
            ),
            (
                ("confidence", "lexical fallback"),
                ("disabled", "bypass", "not run", "suppress"),
                ("zero result", "missing result", "search failure"),
            ),
        ),
        "writer_lease_split": (
            (
                ("south", "zone c"),
                ("90", "dns", "ttl"),
                ("40", "writer lease", "lease"),
                ("old writer", "former writer"),
            ),
            (
                ("old writer", "former writer"),
                ("new writer", "promoted writer"),
                ("overlap", "simultaneous"),
                ("stale write", "write conflict", "split brain"),
            ),
        ),
    }
    return relations.get(name, ())


def discover_reasoning_cases(
    *,
    count: int,
    seed: int,
    mode: str,
) -> list[ReasoningCase]:
    """Return held-out abductive synthesis or ambiguity-resolution tasks."""

    if mode == "novel_synthesis":
        definitions = _novel_synthesis_definitions()
    elif mode == "ambiguity":
        definitions = _ambiguity_definitions()
    else:
        raise ValueError("reasoning mode must be novel_synthesis or ambiguity")
    if count > len(definitions):
        raise ValueError(f"{mode} suite has {len(definitions)} held-out cases; requested {count}")
    random.Random(seed ^ (0xABD0C7 if mode == "novel_synthesis" else 0xA6B1)).shuffle(definitions)
    cases: list[ReasoningCase] = []
    for name, case_files, expected, forbidden, required_any, prompt in definitions[:count]:
        raw = f"{seed}\0{mode}\0{name}\0{expected}".encode()
        cases.append(
            ReasoningCase(
                case_id=f"{mode}_" + hashlib.sha256(raw).hexdigest()[:12],
                mode=mode,
                files=case_files,
                expected=expected,
                forbidden_answers=forbidden,
                required_any=required_any,
                derived_relations=_reasoning_derived_relations(name),
                prompt=prompt,
            )
        )
    return cases

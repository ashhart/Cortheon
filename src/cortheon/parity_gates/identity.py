"""Contender identity gates.

The report names contenders only by opaque alias, so the first job is to map
those aliases back to the registered names and then prove that what ran is
what was pre-registered: independent families, the exact models, the official
provider endpoints, no process-local contender, and the candidate's own
pricing, compute rate, runtime digest, and release identity.
"""

from __future__ import annotations

import hmac
import urllib.parse

from cortheon.parity_gates.context import ContenderIdentities, ParityContext
from cortheon.parity_gates.errors import _TRUSTED_FRONTIER_HOSTS
from cortheon.parity_gates.values import _mapping, _number


def resolve_contenders(context: ParityContext) -> ContenderIdentities:
    """Bind report aliases to registered contenders and gate every binding."""

    candidates = context.candidates
    name_to_alias = {
        str(identity.get("name")): str(alias)
        for alias, identity in candidates.items()
        if isinstance(identity, dict) and identity.get("name")
    }
    candidate_alias = name_to_alias.get(context.candidate_name)
    candidate_identity = (
        _mapping(candidates.get(candidate_alias)) if candidate_alias is not None else {}
    )
    frontier_aliases: dict[str, str | None] = {
        name: name_to_alias.get(name) for name in context.frontier_names
    }
    observed_aliases = {
        str(alias): str(identity.get("name") or "")
        for alias, identity in candidates.items()
        if isinstance(identity, dict)
    }
    identities = ContenderIdentities(
        candidate_name=context.candidate_name,
        candidate_alias=candidate_alias,
        candidate_identity=candidate_identity,
        frontier_aliases=frontier_aliases,
        observed_aliases=observed_aliases,
    )
    min_frontiers = int(context.thresholds["min_frontiers"])
    expected_names = {context.candidate_name, *context.frontier_names}
    expected_aliases = {
        f"candidate_{index + 1}": name for index, name in enumerate(sorted(expected_names))
    }
    context.check(
        "declared_contenders_present",
        bool(
            candidate_alias
            and len(context.frontier_names) >= min_frontiers
            and all(frontier_aliases.values())
            and set(name_to_alias) == expected_names
            and observed_aliases == expected_aliases
        ),
        candidate=context.candidate_name,
        candidate_present=bool(candidate_alias),
        frontiers=frontier_aliases,
        minimum_frontiers=min_frontiers,
        expected_aliases=expected_aliases,
        observed_aliases=observed_aliases,
    )
    observed_families = {
        name: (
            str(_mapping(candidates.get(alias)).get("family") or "") if alias is not None else ""
        )
        for name, alias in frontier_aliases.items()
    }
    context.check(
        "independent_frontier_families",
        bool(
            set(context.frontier_families) == set(context.frontier_names)
            and len(set(context.frontier_families.values())) >= min_frontiers
            and observed_families == context.frontier_families
        ),
        expected=context.frontier_families,
        observed=observed_families,
    )
    _check_registered_bindings(context, identities)
    return identities


def _check_registered_bindings(
    context: ParityContext,
    identities: ContenderIdentities,
) -> None:
    """Models, endpoints, contender kinds, pricing, and the candidate's identity."""

    candidates = context.candidates
    contender_aliases = identities.contender_aliases()
    observed_models = {
        name: (str(_mapping(candidates.get(alias)).get("model") or "") if alias is not None else "")
        for name, alias in contender_aliases.items()
    }
    context.check(
        "precommitted_contender_models",
        bool(
            set(context.contender_models) == {context.candidate_name, *context.frontier_names}
            and observed_models == context.contender_models
        ),
        expected=context.contender_models,
        observed=observed_models,
    )
    observed_endpoints = {
        name: (
            str(_mapping(candidates.get(alias)).get("base_url") or "").rstrip("/")
            if alias is not None
            else ""
        )
        for name, alias in contender_aliases.items()
    }
    frontier_hosts = [
        (urllib.parse.urlparse(context.contender_endpoints.get(name, "")).hostname or "").casefold()
        for name in context.frontier_names
    ]
    trusted_provider_bindings = {
        name: (
            urllib.parse.urlparse(context.contender_endpoints.get(name, "")).hostname or ""
        ).casefold()
        in _TRUSTED_FRONTIER_HOSTS.get(
            context.frontier_families.get(name, ""),
            set(),
        )
        for name in context.frontier_names
    }
    context.check(
        "precommitted_provider_endpoints",
        bool(
            observed_endpoints == context.contender_endpoints
            and all(
                urllib.parse.urlparse(context.contender_endpoints.get(name, "")).scheme == "https"
                for name in context.frontier_names
            )
            and all(frontier_hosts)
            and len(set(frontier_hosts)) == len(frontier_hosts)
            and all(trusted_provider_bindings.values())
        ),
        expected=context.contender_endpoints,
        observed=observed_endpoints,
        frontier_hosts=frontier_hosts,
        trusted_provider_bindings=trusted_provider_bindings,
    )
    context.check(
        "no_process_local_contenders",
        bool(
            identities.candidate_identity.get("kind") == "cortheon"
            and all(_mapping(identity).get("kind") != "cli" for identity in candidates.values())
        ),
        kinds={
            str(alias): _mapping(identity).get("kind") for alias, identity in candidates.items()
        },
    )
    observed_pricing = {
        name: (
            _mapping(_mapping(candidates.get(alias)).get("pricing_per_million"))
            if alias is not None
            else {}
        )
        for name, alias in contender_aliases.items()
    }
    context.check(
        "precommitted_contender_pricing",
        observed_pricing == context.registered_pricing,
        expected=context.registered_pricing,
        observed=observed_pricing,
    )
    _check_candidate_identity(context, identities)


def _check_candidate_identity(
    context: ParityContext,
    identities: ContenderIdentities,
) -> None:
    """The candidate's compute rate, runtime digest, and bound release identity."""

    contract = context.contract
    candidate_identity = identities.candidate_identity
    candidate_compute_rate = _number(candidate_identity.get("compute_cost_per_hour"))
    context.check(
        "precommitted_candidate_compute_rate",
        candidate_compute_rate == float(contract["candidate_compute_usd_per_hour"]),
        expected=float(contract["candidate_compute_usd_per_hour"]),
        observed=candidate_compute_rate,
    )
    candidate_runtime_sha256 = str(candidate_identity.get("runtime_sha256") or "")
    context.check(
        "precommitted_candidate_runtime",
        hmac.compare_digest(
            candidate_runtime_sha256,
            str(contract["candidate_runtime_sha256"]),
        ),
        expected=contract["candidate_runtime_sha256"],
        observed=candidate_runtime_sha256,
    )
    case_bank = context.case_bank
    release_identity = _mapping(context.report.get("release_identity"))
    expected_release_identity = {
        "model": str(context.contender_models.get(context.candidate_name) or ""),
        "family": str(contract["candidate_family"]),
        "host": str(contract["candidate_host"]),
        "runtime_sha256": str(contract["candidate_runtime_sha256"]),
        "contract_sha256": context.contract_sha256,
        "pack_issuer": str(case_bank.get("issuer") or ""),
        "pack_id": str(case_bank.get("pack_id") or ""),
        "runner_id": str(case_bank.get("runner_id") or ""),
        # The pack binds the grading authority (issuer) and, when declared, a
        # distinct evaluator identity; either way both are pack-attested.
        "evaluator": str(case_bank.get("evaluator") or case_bank.get("issuer") or ""),
    }
    context.check(
        "release_identity_bound",
        bool(
            release_identity == expected_release_identity
            and candidate_identity.get("family") == contract["candidate_family"]
        ),
        expected=expected_release_identity,
        observed=release_identity,
        observed_candidate_family=candidate_identity.get("family"),
    )

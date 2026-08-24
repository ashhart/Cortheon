"""Evaluator secret material and the seal computed under it.

Two distinct secrets are required and must differ: one authenticates the
sealed pack, the other authenticates the runner that will execute it. Sharing
one key would let whoever can run the benchmark also mint the pack it is
graded against, which is the confound the whole apparatus exists to remove.

Only commitments to the keys travel in a manifest -- never the keys.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any

_MINIMUM_KEY_BYTES = 32


def _read_secret(env_name: str, description: str) -> str:
    value = os.environ.get(env_name, "")
    if len(value.encode("utf-8")) < _MINIMUM_KEY_BYTES:
        raise ValueError(
            f"{env_name} must contain at least {_MINIMUM_KEY_BYTES} bytes of {description}"
        )
    return value


def read_signing_keys(key_env: str, runner_key_env: str) -> tuple[str, str]:
    """The pack key and the runner-attestation key, validated and distinct."""

    key = _read_secret(key_env, "evaluator-owned secret")
    runner_key = _read_secret(runner_key_env, "evaluator-runner secret")
    if hmac.compare_digest(key, runner_key):
        raise ValueError("case-pack and runner-attestation keys must differ")
    return key, runner_key


def key_commitment(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def key_id(env_name: str) -> str:
    return hashlib.sha256(env_name.encode("utf-8")).hexdigest()[:16]


def runner_attestation(runner_key_env: str, runner_key: str) -> dict[str, str]:
    return {
        "algorithm": "hmac-sha256",
        "key_id": key_id(runner_key_env),
        "key_commitment": key_commitment(runner_key),
    }


def _canonical_signed_payload(manifest: dict[str, Any], cases: list[Any]) -> bytes:
    """The exact bytes the seal is computed over: manifest minus signature."""

    unsigned = dict(manifest)
    unsigned.pop("signature", None)
    return json.dumps(
        {"manifest": unsigned, "cases": cases},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def signature(key_env: str, key: str, canonical: bytes) -> dict[str, str]:
    return {
        "algorithm": "hmac-sha256",
        "key_id": key_id(key_env),
        # SHA-256 commitment to the signing key itself: a stable evaluator
        # identity across packs. HMAC under a shared secret is still symmetric
        # and requires external audit; it is not public-key proof.
        "key_commitment": key_commitment(key),
        "value": hmac.new(key.encode("utf-8"), canonical, hashlib.sha256).hexdigest(),
    }

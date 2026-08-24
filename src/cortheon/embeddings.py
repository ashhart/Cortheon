"""Dependency-free vector providers for Cortheon's hybrid evidence retrieval.

The default local provider is deliberately described as feature hashing rather
than a semantic model: it improves fuzzy and compound-term retrieval without
network access. Deployments can point the same interface at any
OpenAI-compatible embeddings endpoint for genuinely semantic vectors.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Protocol

from cortheon.connectors.http import JsonHttpClient


class EmbeddingProvider(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def dimensions(self) -> int | None: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class LocalFeatureHashEmbedder:
    """Fast local lexical-feature vectors with no model or third-party package."""

    vector_size: int = 256

    @property
    def model(self) -> str:
        return "cortheon-feature-hash-v1"

    @property
    def dimensions(self) -> int:
        return self.vector_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_feature_hash_vector(text, self.vector_size) for text in texts]


class OpenAICompatibleEmbedder:
    """Call ``POST /embeddings`` on OpenAI, Ollama, vLLM, or another provider."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        dimensions: int | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not base_url.strip() or not model.strip():
            raise ValueError("embedding base URL and model are required")
        if dimensions is not None and dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")
        self.base_url = base_url.rstrip("/")
        self._model = model.strip()
        self.api_key = api_key
        self._dimensions = dimensions
        self.client = JsonHttpClient(timeout_seconds=timeout_seconds)

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {
            "model": self._model,
            "input": [text[:32_000] for text in texts],
            "encoding_format": "float",
        }
        if self._dimensions is not None:
            payload["dimensions"] = self._dimensions
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        response = self.client.post_json(
            _embeddings_endpoint(self.base_url),
            payload,
            headers=headers,
        )
        data = response.get("data") if isinstance(response, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise ValueError("embedding endpoint returned the wrong number of vectors")
        ordered = sorted(
            data,
            key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0,
        )
        vectors: list[list[float]] = []
        for item in ordered:
            raw = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(raw, list) or not raw:
                raise ValueError("embedding endpoint returned an invalid vector")
            vector = [float(value) for value in raw]
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("embedding endpoint returned a non-finite value")
            vectors.append(_normalize(vector))
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError("embedding endpoint returned inconsistent dimensions")
        if self._dimensions is not None and dimensions != {self._dimensions}:
            raise ValueError("embedding endpoint ignored the requested dimensions")
        return vectors


def build_embedding_provider(
    mode: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    dimensions: int | None = None,
) -> EmbeddingProvider | None:
    normalized = mode.strip().casefold()
    if normalized in {"", "off", "none", "disabled"}:
        return None
    if normalized in {"local", "feature_hash", "feature-hash"}:
        return LocalFeatureHashEmbedder(vector_size=dimensions or 256)
    if normalized in {"openai", "remote"}:
        if not base_url or not model:
            raise ValueError(
                "remote embeddings require CORTHEON_EMBEDDING_BASE_URL and CORTHEON_EMBEDDING_MODEL"
            )
        return OpenAICompatibleEmbedder(
            base_url=base_url,
            model=model,
            api_key=api_key,
            dimensions=dimensions,
        )
    raise ValueError("embedding mode must be off, local, or openai")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    return max(
        -1.0,
        min(1.0, sum(a * b for a, b in zip(left, right, strict=True))),
    )


def _feature_hash_vector(text: str, dimensions: int) -> list[float]:
    values = [0.0] * dimensions
    words = re.findall(r"[a-z0-9]+", text.casefold())
    features: list[str] = list(words)
    features.extend(f"{words[index]}::{words[index + 1]}" for index in range(len(words) - 1))
    compact = " ".join(words)
    features.extend(
        compact[index : index + 4]
        for index in range(max(0, len(compact) - 3))
        if " " not in compact[index : index + 4]
    )
    for feature in features[:20_000]:
        digest = hashlib.blake2b(
            feature.encode("utf-8"),
            digest_size=8,
            person=b"cortheon-v1",
        ).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        values[index] += sign
    return _normalize(values)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _embeddings_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/embeddings"):
        return normalized
    if normalized.endswith("/v1"):
        return normalized + "/embeddings"
    return normalized + "/v1/embeddings"

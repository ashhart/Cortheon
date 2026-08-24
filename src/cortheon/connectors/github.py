# ruff: noqa: F401
"""Stable GitHub connector API.

Implementation ownership is split across ``github_core`` while this module keeps
the established import path and runtime patch seams used by hosts and tests.
"""

from __future__ import annotations

import base64
import os
import re
import urllib.parse
from urllib.parse import urlparse

from cortheon.connectors.github_core.artifacts import (
    _int_or_none,
    find_github_url,
    normalize_github_url,
    parse_owner_repo,
    repository_artifact_confidence,
    repository_item_to_artifact,
    repository_query_terms,
    repository_relevance,
)
from cortheon.connectors.github_core.client import GitHubConnector, GitHubRepositorySearch
from cortheon.connectors.github_core.constants import GITHUB_RE, REPOSITORY_SEARCH_STOPWORDS
from cortheon.connectors.github_core.normalization import (
    adjusted_repository_confidence,
    implementation_signals,
    int_or_zero,
    language_metadata,
    normalize_readme,
    readme_metadata,
    readme_text,
    repository_health_score,
    repository_metadata,
    root_content_metadata,
    split_metadata_csv,
)
from cortheon.connectors.github_core.transport import (
    github_headers,
    repository_search_url,
    safe_get_json,
)
from cortheon.connectors.http import ConnectorError, JsonHttpClient
from cortheon.models import (
    Evidence,
    GitHubRepoReport,
    ResearchArtifact,
    SupportLevel,
    parse_datetime,
    utc_now,
)

_OWNED_CALLABLES = (
    GitHubConnector,
    GitHubRepositorySearch,
    repository_search_url,
    github_headers,
    safe_get_json,
    repository_metadata,
    language_metadata,
    root_content_metadata,
    readme_text,
    readme_metadata,
    normalize_readme,
    implementation_signals,
    split_metadata_csv,
    repository_health_score,
    adjusted_repository_confidence,
    int_or_zero,
    repository_item_to_artifact,
    repository_relevance,
    repository_query_terms,
    repository_artifact_confidence,
    find_github_url,
    parse_owner_repo,
    normalize_github_url,
    _int_or_none,
)

for _callable in _OWNED_CALLABLES:
    _callable.__module__ = __name__

for _class in (GitHubConnector, GitHubRepositorySearch):
    for _member in vars(_class).values():
        if callable(_member) and getattr(_member, "__module__", "").startswith(
            "cortheon.connectors.github_core"
        ):
            _member.__module__ = __name__

del _OWNED_CALLABLES, _callable, _class, _member

"""GitHub URL parsing and repository-search vocabulary."""

from __future__ import annotations

import re

GITHUB_RE = re.compile(r"github\.com[:/](?P<owner>[^/\s]+)/(?P<repo>[^/#?\s]+)")
REPOSITORY_SEARCH_STOPWORDS = {
    "and",
    "api",
    "build",
    "code",
    "current",
    "engine",
    "for",
    "framework",
    "from",
    "libraries",
    "library",
    "new",
    "open",
    "package",
    "packages",
    "python",
    "research",
    "software",
    "the",
    "with",
}

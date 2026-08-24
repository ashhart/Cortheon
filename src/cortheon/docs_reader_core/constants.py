from __future__ import annotations

DOCS_LABELS = ("Documentation", "Docs", "Read the Docs")
CHANGELOG_LABELS = (
    "Changelog",
    "Change Log",
    "Changes",
    "Release Notes",
    "Release-Notes",
    "History",
    "News",
)
# Guide pages worth one extra fetch, best first. Matched against link paths and
# anchor text from the docs root page.
GUIDE_KEYWORDS = (
    "quickstart",
    "getting-started",
    "getting_started",
    "gettingstarted",
    "tutorial",
    "usage",
    "guide",
    "examples",
    "api",
)
# When corroborating a specific symbol, reference pages carry the signatures;
# quickstarts mostly carry prose.
API_GUIDE_KEYWORDS = ("api", "reference", "interface", *GUIDE_KEYWORDS)
MAX_PAGE_TEXT_CHARS = 8_000
MAX_CHANGELOG_HEAD_CHARS = 600
MAX_CODE_BLOCKS_PER_PAGE = 20
MAX_CODE_BLOCK_CHARS = 1_200
MAX_VERSION_PROBES = 6

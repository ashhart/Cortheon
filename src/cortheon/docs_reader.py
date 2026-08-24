"""Current documentation discovery, parsing, and bounded extraction.

Implementation ownership lives in :mod:`cortheon.docs_reader_core`. This
facade keeps the original import path, identities, defaults, and patch points.
"""

# Former module globals remain observable patch points for moved code.
# ruff: noqa: F401

from __future__ import annotations

import re
import urllib.parse
import urllib.robotparser
from html.parser import HTMLParser
from typing import ClassVar

from cortheon.connectors.docs import pick_url
from cortheon.connectors.http import (
    ConnectorError,
    JsonHttpClient,
    validate_public_http_url,
)
from cortheon.docs_reader_core.constants import (
    API_GUIDE_KEYWORDS,
    CHANGELOG_LABELS,
    DOCS_LABELS,
    GUIDE_KEYWORDS,
    MAX_CHANGELOG_HEAD_CHARS,
    MAX_CODE_BLOCK_CHARS,
    MAX_CODE_BLOCKS_PER_PAGE,
    MAX_PAGE_TEXT_CHARS,
    MAX_VERSION_PROBES,
)
from cortheon.docs_reader_core.discovery import (
    detect_version_in_url,
    fetch_robots,
    looks_like_docs,
    prefer_versioned_link,
    raw_github_url,
    resolve_docs_url,
    select_guide_links,
    version_variants,
    versioned_docs_candidates,
)
from cortheon.docs_reader_core.extraction import (
    error_docs_page,
    find_symbol_mention,
    runnable_examples,
)
from cortheon.docs_reader_core.parser import DocsHtmlParser
from cortheon.docs_reader_core.reader import DocsSiteReader
from cortheon.examples import (
    MAX_EXAMPLE_CHARS,
    doctest_to_script,
    is_runnable_candidate,
)
from cortheon.models import (
    DocsExample,
    DocsPage,
    DocsSiteReport,
    Evidence,
    PackageMetadata,
    SupportLevel,
    utc_now,
)
from cortheon.sanitize import injection_flags, scan_text
from cortheon.verifier import guess_import_name
from cortheon.web_crawler import (
    charset_from_content_type,
    domain_of,
    normalize_space,
    normalize_url,
)

_DEFINITIONS = (
    DocsHtmlParser,
    DocsSiteReader,
    detect_version_in_url,
    error_docs_page,
    fetch_robots,
    find_symbol_mention,
    looks_like_docs,
    prefer_versioned_link,
    raw_github_url,
    resolve_docs_url,
    runnable_examples,
    select_guide_links,
    version_variants,
    versioned_docs_candidates,
)

for _definition in _DEFINITIONS:
    _definition.__module__ = __name__

for _owner in (DocsHtmlParser, DocsSiteReader):
    for _member in vars(_owner).values():
        if isinstance(_member, (classmethod, staticmethod)):
            _member = _member.__func__
        if (
            callable(_member)
            and hasattr(_member, "__module__")
            and _member.__module__.startswith("cortheon.docs_reader_core")
        ):
            _member.__module__ = __name__

del _definition, _member, _owner

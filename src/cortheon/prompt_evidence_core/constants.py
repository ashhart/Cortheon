"""Package aliases, prose filters, bounds, prompts, and compiled patterns."""

from __future__ import annotations

import re

REVERSE_IMPORT_OVERRIDES = {
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "PIL": "pillow",
    "pythonjsonlogger": "python-json-logger",
    "ninja": "django-ninja",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "dotenv": "python-dotenv",
}
COMMON_ALIASES = {
    "np": "numpy",
    "pd": "pandas",
    "plt": "matplotlib",
    "sns": "seaborn",
    "tf": "tensorflow",
}
PROSE_STOPWORDS = {
    "self",
    "cls",
    "args",
    "kwargs",
    "true",
    "false",
    "none",
    "client",
    "server",
    "session",
    "response",
    "resp",
    "request",
    "result",
    "results",
    "data",
    "value",
    "values",
    "item",
    "items",
    "obj",
    "object",
    "config",
    "settings",
    "options",
    "params",
    "app",
    "api",
    "db",
    "env",
    "ctx",
    "context",
    "file",
    "files",
    "path",
    "paths",
    "user",
    "users",
    "name",
    "names",
    "message",
    "messages",
    "model",
    "models",
    "test",
    "tests",
    "testing",
    "main",
    "utils",
    "util",
    "core",
    "base",
    "common",
    "code",
    "error",
    "errors",
    "async",
    "await",
    "class",
    "type",
    "types",
    "list",
    "dict",
    "print",
    "python",
    "install",
    "package",
    "packages",
    "library",
    "libraries",
    "module",
    "modules",
    "version",
    "versions",
    "import",
    "imports",
    "example",
    "examples",
    "docs",
    "documentation",
    "retry",
    "retries",
    "timeout",
    "cache",
    "proxy",
    "stream",
    "task",
    "tasks",
    "content",
    "text",
    "html",
    "http",
    "https",
    "localhost",
    "state",
    "event",
    "events",
    "disk",
    "download",
    "downloads",
    "chunk",
    "chunks",
    "current",
    "stable",
    "exact",
    "today",
    "each",
    "addition",
    "additions",
    "interface",
    "production",
    "implementation",
    "complete",
    "write",
    "straight",
    "large",
}
MAX_PACKAGES = 3
MAX_PROBES = 8
MAX_EVIDENCE_CHARS = 2000

EVIDENCE_HEADER = (
    "CORTHEON SUBSTRATE EVIDENCE — fetched live from the current published "
    "source of the packages this task mentions. Treat it as ground truth over "
    "your training memory. Use the exact verified signatures; do not invent "
    "symbols or keyword arguments."
)
ASSUMPTION_HEADER = (
    "CORTHEON — your working assumptions for this task. Reason FROM these.\n"
    "These are verified facts from the current live source. Your training memory\n"
    "is stale for everything below. Do not reason from what you remember; reason\n"
    "from what is verified. If your instinct says something different, the\n"
    "verified fact wins.\n"
)
FAILURE_PREDICTOR_HEADER = (
    "CORTHEON — what you are about to get wrong. Your training weights will\n"
    "make you reach for these stale patterns. Avoid them explicitly:\n"
)

IMPORT_RE = re.compile(r"^\s*import\s+(.+)$", re.MULTILINE)
FROM_RE = re.compile(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import\b", re.MULTILINE)
INSTALL_RE = re.compile(r"\b(?:pip3?|uv)\s+(?:install|add)\s+([^\n`]+)")
PACKAGE_LIST_RE = re.compile(
    r"\bpackages?\b[^:\n]{0,100}:\s*"
    r"([A-Za-z][A-Za-z0-9._-]*(?:\s*,\s*[A-Za-z][A-Za-z0-9._-]*)+)",
    re.IGNORECASE,
)
DOTTED_RE = re.compile(r"(?<![\w./-])([A-Za-z_][\w-]+)\.(?=([A-Za-z_]\w*))")
TLDS = {"org", "com", "io", "net", "dev", "ai", "co", "edu", "gov"}
BACKTICK_RE = re.compile(r"`([A-Za-z][A-Za-z0-9._-]+)`")
BARE_RE = re.compile(r"\b([a-z][a-z0-9_-]{3,})\b")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
VERSION_COMPARISON_RE = re.compile(
    r"\b(?:since|after|from)\s+(?:version\s+)?v?"
    r"(\d+(?:\.\d+){1,3}(?:[A-Za-z][A-Za-z0-9.-]*)?)\b",
    re.IGNORECASE,
)

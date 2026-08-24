"""Diff receipt validation and change-establishment rules."""

from __future__ import annotations

import json
import re
from typing import Any

from cortheon.cognitive_core.models import Observation
from cortheon.cognitive_core.profiles import _has_hint
from cortheon.cognitive_core.tasks import _CODE_PATH_RE, _DOCUMENT_PATH_RE
from cortheon.cognitive_repair import changed_paths_from_diff

_SINGLE_LINE_CHANGE_RE = re.compile(r"\b(?:one|single)[- ]line\b", flags=re.IGNORECASE)


_CONCISE_CHANGE_HINTS = frozenset({"minimal", "smallest", "surgical", "concise", "tiny"})


def _diff_line_budget(goal: str) -> int | None:
    """Return a changed-line ceiling when the goal explicitly asks for a
    minimal change; ``None`` means the goal never promised concision."""

    if _SINGLE_LINE_CHANGE_RE.search(goal):
        return 6
    if _has_hint(goal, _CONCISE_CHANGE_HINTS):
        return 30
    return None


def _diff_changed_line_count(content: str) -> int:
    return sum(
        1
        for line in content.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )


def _diff_weakens_tests(content: str) -> bool:
    added = "\n".join(
        line[1:].casefold()
        for line in content.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    removed = "\n".join(
        line[1:].casefold()
        for line in content.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    disables = (
        "pytest.skip",
        "@pytest.mark.skip",
        "@pytest.mark.xfail",
        "unittest.skip",
        ".skip(",
        "test.skip",
        "xit(",
        "xdescribe(",
    )
    removed_assertion = bool(re.search(r"\b(?:assert|expect\s*\(|self\.assert)", removed))
    added_assertion = bool(re.search(r"\b(?:assert|expect\s*\(|self\.assert)", added))
    added_disable = any(marker in added for marker in disables)
    return (removed_assertion and not added_assertion) or added_disable


def _diff_establishes_change(
    observation: Observation,
    *,
    require_receipt: bool,
) -> bool:
    receipt = observation.host_receipt
    receipt_tool = str((receipt or {}).get("tool") or "").casefold()
    receipt_outcome = str((receipt or {}).get("outcome") or "").casefold()
    receipt_executor = str((receipt or {}).get("executor") or "").casefold()
    # Host-hook diffs (before/after reads around a real mutation, or the
    # host session's own diff API) arrive as "observed": no independent
    # command checked them, but a host interaction produced them. They are
    # live host evidence; only a diff the model asserted without a host
    # executor behind it fails the receipt requirement.
    host_derived = receipt_executor in {
        "mutation_hook",
        "session.diff",
        "host_bounded_edit",
        "host_bounded_multi_edit",
    }
    valid_receipt = (
        receipt_tool in {"diff", "git"}
        and receipt_outcome in {"changed", "result"}
        and (observation.status == "verified" or host_derived)
    )
    if require_receipt and not valid_receipt:
        return False
    body = observation.content
    added = any(line.startswith("+") and not line.startswith("+++") for line in body.splitlines())
    removed = any(line.startswith("-") and not line.startswith("---") for line in body.splitlines())
    if added or removed:
        return bool(changed_paths_from_diff(body)) or not require_receipt
    if not valid_receipt:
        return False

    receipt_paths = _diff_receipt_paths(receipt)
    normalized_body = body.casefold().replace("\\", "/")
    describes_change = any(
        marker in normalized_body
        for marker in ("diff", "change", "new-file", "added", "removed", "deleted")
    )
    mentions_target = any(
        candidate.rsplit("/", 1)[-1].casefold() in normalized_body
        or ("/" in candidate and candidate.split("/", 1)[0].casefold() + "/" in normalized_body)
        for candidate in receipt_paths
    )
    return describes_change and mentions_target


def _diff_receipt_paths(receipt: dict[str, Any] | None) -> set[str]:
    arguments = (receipt or {}).get("args")
    if not isinstance(arguments, dict):
        return set()
    candidates = {
        *_CODE_PATH_RE.findall(json.dumps(arguments, sort_keys=True)),
        *_DOCUMENT_PATH_RE.findall(json.dumps(arguments, sort_keys=True)),
    }
    return {
        path
        for candidate in candidates
        if (path := candidate.replace("\\", "/").strip(".,:;()[]{}'\""))
        and not path.startswith("/")
        and "://" not in path
        and ".." not in path.split("/")
        and not {".git", ".cortheon", "node_modules"} & set(path.split("/"))
        and len(path) <= 240
    }

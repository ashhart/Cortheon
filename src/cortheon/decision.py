# ruff: noqa: F401
"""Deterministic, auditable evidence policy for model-proposed actions.

The gate matches explicit rules; it does not infer semantics beyond them.
"""

from __future__ import annotations

import re

from cortheon.decision_core.layer import DecisionLayer
from cortheon.decision_core.outcomes import (
    confidence_for,
    missing_evidence,
    notes_for,
    recommended_tools,
    verdict_for,
)
from cortheon.models import DecisionCheck, DecisionReport

_MOVED_CALLABLES = (
    DecisionLayer,
    verdict_for,
    confidence_for,
    missing_evidence,
    recommended_tools,
    notes_for,
)
for _callable in _MOVED_CALLABLES:
    _callable.__module__ = __name__
for _member in vars(DecisionLayer).values():
    if callable(_member) and getattr(_member, "__module__", "").startswith(
        "cortheon.decision_core"
    ):
        _member.__module__ = __name__
del _MOVED_CALLABLES, _callable, _member

PolicyGate = DecisionLayer


def build_checks(text: str, evidence: set[str]) -> list[DecisionCheck]:
    lower = text.lower()
    checks: list[DecisionCheck] = []
    if destructive_intent(lower):
        checks.append(
            DecisionCheck(
                name="destructive_action",
                status="block",
                reason="The proposed action appears destructive or security-sensitive.",
                next_action="Require explicit user approval and a rollback plan before proceeding.",
            )
        )
    if package_decision(lower):
        checks.append(
            evidence_check(
                name="current_package_evidence",
                evidence=evidence,
                accepted={
                    "package_verified",
                    "recommendation_report",
                    "package_comparison",
                    "technology_research_report",
                },
                reason="External library/package decisions need current package evidence.",
                next_action="Call cortheon_recommend, cortheon_compare, or cortheon_verify before choosing the dependency.",
            )
        )
    if api_decision(lower):
        checks.append(
            evidence_check(
                name="api_evidence",
                evidence=evidence,
                accepted={"api_evidence", "source_symbol_evidence", "example_execution"},
                reason="Package-specific API usage needs source-derived or execution-backed API evidence.",
                next_action="Call cortheon_api_evidence for the package and symbol before writing production code.",
            )
        )
    if repo_change_decision(lower):
        checks.append(
            evidence_check(
                name="repo_context",
                evidence=evidence,
                accepted={"repo_context", "tests_passed", "diff_reviewed"},
                reason="Repo-changing tasks need repository context and validation.",
                next_action="Inspect the repo, make a scoped diff, and run the relevant tests.",
            )
        )
    if version_migration_decision(lower):
        checks.append(
            evidence_check(
                name="api_diff_evidence",
                evidence=evidence,
                accepted={"api_diff_evidence", "version_diff_report", "changelog_reviewed"},
                reason="Version migrations need source-derived evidence of what changed between versions.",
                next_action="Call cortheon_api_diff for the old and new versions before migrating code.",
            )
        )
    if frontier_research_decision(lower):
        checks.append(
            evidence_check(
                name="research_report",
                evidence=evidence,
                accepted={"research_report", "source_coverage", "grounded_claims"},
                reason="Scientific/medical/frontier claims need live research evidence and grounded claims.",
                next_action="Call cortheon_research and inspect source_coverage before making a strong claim.",
            )
        )
    if architecture_commitment_decision(lower):
        checks.append(
            evidence_check(
                name="architecture_evidence",
                evidence=evidence,
                accepted={
                    "architecture_evidence",
                    "architecture_research_report",
                    "architecture_benchmark",
                },
                reason="Committing to a strongest/current architecture needs architecture-specific evidence, not only adjacent research.",
                next_action="Call cortheon_research for architecture, benchmark, implementation, and validation evidence before committing.",
            )
        )
    if not checks:
        checks.append(
            DecisionCheck(
                name="low_risk_task",
                status="passed",
                reason="No high-risk package, API, repo-write, destructive, or frontier-research decision was detected.",
            )
        )
    return checks


def evidence_check(
    *,
    name: str,
    evidence: set[str],
    accepted: set[str],
    reason: str,
    next_action: str,
) -> DecisionCheck:
    if evidence.intersection(accepted):
        return DecisionCheck(name=name, status="passed", reason=reason)
    return DecisionCheck(name=name, status="missing", reason=reason, next_action=next_action)


def destructive_intent(text: str) -> bool:
    # Destructive operations arrive in many phrasings: reordered or split
    # rm flags, long forms, SQL against tables and schemas, force pushes,
    # raw-device writes, world-writable system paths, and pipe-to-shell
    # downloads. --force-with-lease is the safe push variant and must not
    # match the --force patterns.
    patterns = (
        r"\brm\s+-rf\b",
        r"\brm\b[^|;&\n]{0,24}(?<!\S)-\w*r\w*f\w*\b",
        r"\brm\b[^|;&\n]{0,24}(?<!\S)-\w*f\w*r\w*\b",
        r"\brm\s+-\w*r\w*\s+-\w*f\w*\b",
        r"\brm\s+-\w*f\w*\s+-\w*r\w*\b",
        r"\brm\b[^|;&\n]{0,24}(?<!\S)--force\b",
        r"\bdrop\s+(?:database|table|schema)\b",
        r"\btruncate\s+table\b",
        r"\bgit\s+push\b[^|\n]{0,40}(?<!\S)--force\b(?!-)",
        r"\bgit\s+push\s+-f\b",
        r"\bgit\s+reset\b[^|\n]{0,24}(?<!\S)--hard\b",
        r"\bchmod\s+(?:-\w+\s+)?777\s+/",
        r"\bdd\b[^|\n]{0,48}(?<!\S)of=/dev/(?:sd|hd|nvme|disk|mmcblk)",
        r"\b(?:curl|wget)\b[^|\n]{0,120}\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b",
        r"\bdelete\s+production\b",
        r"\b(?:delete|remove|purge)\b[^.\n]{0,60}"
        r"\b(?:all\s+)?(?:rows?|records?|entries?|data)\b[^.\n]{0,40}"
        r"\b(?:production|prod)\b",
        r"\b(?:production|prod)\b[^.\n]{0,60}"
        r"\b(?:delete|remove|purge)\s+(?:all\s+)?(?:rows?|records?|entries?|data)\b",
        r"\bdisable\s+(?:(?:the|those|these|failing)\s+)?tests?\b",
        r"\bexfiltrat",
        r"\bssh\s+key",
        r"\bsecret",
        r"\bcredential",
        r"\b(purge|delete|remove)\b.{0,80}\b(production|prod)\b.{0,80}\b(auth|token|key|variable|env)",
        r"\b(purge|delete|remove)\b.{0,80}\b(auth|token|key|variable|env).{0,80}\b(production|prod)\b",
        r"\b(production|prod)\s+auth\s+variables?\b",
        r"\b(delete|remove)\b.{0,40}\bauth(?:entication)?\s+"
        r"(configuration|config|variables?|env)\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def package_decision(text: str) -> bool:
    # Generic install/adopt phrasing must trigger the gate without a
    # hardcoded package list: "Install and use PyPDF2" sailed past the
    # old term set and was allowed at 0.95 with zero evidence.
    if re.search(
        r"\b(?:pip|pipx|uv|npm|pnpm|yarn|cargo|gem|brew)\s+(?:install|add)\b",
        text,
    ):
        return True
    if re.search(r"\binstall(?:ing|ed)?\b(?:\s+\w+){0,3}\s+\buse\b", text):
        return True
    # Narration ("we installed the schema last week") must not read as
    # adopting a package; skip the determiners that follow the verb.
    if re.search(
        r"\binstall(?:ing|ed)?\s+"
        r"(?!(?:the|a|an|this|that|these|those|our|your|its|it|all|any|some|more)\s)"
        r"[a-z][a-z0-9_.-]{1,40}\b",
        text,
    ):
        return True
    # "use boto3" is package-shaped; "import base64" is the standard
    # library and version-numbered encodings are not packages.
    stdlib_named = {
        "base64",
        "base32",
        "base58",
        "base85",
        "utf8",
        "utf16",
        "utf32",
        "sha1",
        "sha256",
        "sha512",
        "md5",
        "crc32",
    }
    for match in re.finditer(
        r"\b(?:use|using|adopt(?:ing)?|import(?:ing)?)\s+([a-z][a-z0-9_.]*)\b",
        text,
    ):
        token = match.group(1)
        if token not in stdlib_named and any(char.isdigit() for char in token):
            return True
    terms = {
        "dependency",
        "framework",
        "library",
        "package",
        "pip install",
        "sdk",
        "use fastapi",
        "use django",
        "use httpx",
    }
    if any(term in text for term in terms):
        return True
    choice_actions = {
        "choose",
        "commit to",
        "current best",
        "pick",
        "recommend",
        "select",
    }
    technology_targets = {
        "database",
        "model",
        "platform",
        "service",
        "stack",
        "tool",
        "vector database",
        "vector db",
    }
    return any(action in text for action in choice_actions) and any(
        target in text for target in technology_targets
    )


def api_decision(text: str) -> bool:
    without_file_paths = re.sub(
        r"\b[\w./-]+\.(?:c|cc|cpp|css|go|h|hpp|html|java|js|json|jsx|md|"
        r"php|py|rb|rs|sh|toml|ts|tsx|yaml|yml)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # "e.g.", "i.e.", and "U.S." are prose abbreviations, not module
    # paths; require at least two leading identifier characters.
    if re.search(
        r"\b[a-zA-Z_]\w+\.[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)?\b",
        without_file_paths,
    ):
        return True
    return any(term in text for term in {"method", "class", "function", "import path", "api call"})


def repo_change_decision(text: str) -> bool:
    # Word boundaries matter: "dispatch" contains "patch". Repair phrasing
    # ("fix X so the test passes") is a repo change; the runtime classifies
    # the same sentence code_change and the gate must agree.
    patterns = (
        r"\b(?:edit|modify)\s+the\s+repo\b",
        r"\b(?:fix(?:es|ed|ing)?|bugfix(?:es|ed|ing)?|repair(?:s|ed|ing)?|"
        r"patch(?:es|ed|ing)?|refactor(?:s|ed|ing)?)\b",
        r"\bmigrations?\b",
        r"\bpull\s+request\b",
        r"\bwrite\s+code\b",
        r"\bchange\s+files?\b",
        r"\brun(?:ning|s)?\s+tests?\b",
        r"\bmake\s+(?:the\s+|all\s+)?tests?\s+pass\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def version_migration_decision(text: str) -> bool:
    actions = {"migrate", "migration", "upgrade", "bump"}
    if not any(action in text for action in actions):
        return False
    return bool(re.search(r"\bv?\d+(\.\d+)*\b", text)) or "version" in text or "latest" in text


def frontier_research_decision(text: str) -> bool:
    # Whole words only: "newspaper" contains "paper", "procedure" contains
    # "cure". A bare "trial" is usually a product trial period; "clinical"
    # carries the research sense on its own.
    pattern = re.compile(
        r"\b(?:alife|biolog(?:y|ical|ists?)|clinical|cur(?:e|es|ed)|"
        r"diseas(?:e|es)|frontier|medic(?:al|ine|ines)|papers?|"
        r"research(?:ing)?|scientific)\b"
    )
    return bool(pattern.search(text))


def architecture_commitment_decision(text: str) -> bool:
    if "architecture" not in text:
        return False
    commitment_terms = {
        "build first",
        "choose",
        "commit",
        "current best",
        "production",
        "strongest",
    }
    return any(term in text for term in commitment_terms)

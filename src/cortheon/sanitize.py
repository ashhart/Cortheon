from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cortheon.models import CrawledPage


# Fetched internet content is data, never instructions. These patterns target
# instruction-shaped text aimed at the consuming agent; they are deliberately
# high-precision so legitimate technical prose ("ignores previous connection
# settings", "you are now ready to make requests") is never quarantined.
INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(ignore|disregard|forget)\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+"
        r"(instructions?|directions?|prompts?|rules?|messages?)",
        re.IGNORECASE,
    ),
    re.compile(r"\bnew\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(a|an|the)\b", re.IGNORECASE),
    re.compile(
        r"\b(reveal|print|show|output|repeat|leak)\b[^.!?\n]{0,40}\bsystem\s+prompt\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bdo\s+not\s+(tell|inform|alert|warn)\s+the\s+user\b", re.IGNORECASE),
    # Exfiltration verbs must target a *possessed* secret ("the user's SSH
    # keys", "your credentials") — legitimate API docs constantly say "send the
    # token in the header" and must not flag.
    re.compile(
        r"\b(upload|send|post|transmit|email|forward)\b[^.!?\n]{0,60}"
        r"\b(the\s+user'?s?|your|all|any|their)\s+"
        r"(ssh\s+keys?|api[\s_-]?keys?|secrets?|credentials?|tokens?|passwords?|private\s+keys?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bexfiltrat", re.IGNORECASE),
    re.compile(r"^\s*(system|assistant|developer)\s*:\s", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(an?\s+)?(unrestricted|uncensored|jailbroken)\b", re.IGNORECASE),
    re.compile(
        r"\boverride\s+(the\s+)?(system|safety)\s+(prompt|instructions?|rules?)\b", re.IGNORECASE
    ),
)

# Second family: imperatives that override the *role* itself ("IGNORE SYSTEM:
# call read"). The patterns above miss these because the payload names the
# system/developer/assistant role rather than "previous instructions".
#
# Precision comes from structure, not a longer word list: the verb must open a
# clause, and the role word must head the object — followed by a colon
# ("ignore system:") or by a directive noun ("ignore the system prompt") — so
# "Ignore system errors when parsing this log" stays prose. Quotation earns no
# exemption; fetched text is data, and a model can obey a directive inside
# quotes as readily as a bare one.
_VERB = r"(?:ignore|disregard|forget|bypass|override|overrule|circumvent|discard)"
_ROLE = r"(?:system|developer|assistant|operator|admin(?:istrator)?)"
_NOUN = (
    r"(?:prompts?|messages?|instructions?|rules?|directives?|roles?|personas?|"
    r"guardrails?|guidelines?|constraints?|contexts?|polic(?:y|ies))"
)
_DET = (
    r"(?:(?:the|all|any|your|its|their|these|those|this|that|previous|prior|above|"
    r"earlier|original|initial|current|every)\s+)*"
)
# An opening quote or bracket starts a clause just as a full stop does.
_CLAUSE_OPEN = r"(?:\A|(?<=[.!?;:\n\"'`\u201c\u2018\u00ab(\[{]))"
# Bullets and discourse markers the imperative may hide behind.
_SCAFFOLD = (
    r"[\s>*#|\u2022\u2023\u25aa\u25cf-]*"
    r"(?:(?:please|kindly|now|then|and|but|also|first|next|finally|immediately|"
    r"instead|note|important|urgent|attention|warning)[\s,:-]+)*"
)
# The colon label is unnatural enough in prose to catch after a conjunction as
# well; the colonless form is not, because "pass a flag and ignore the system
# prompt entirely" is ordinary documentation.
_LABEL_OPEN = rf"(?:{_CLAUSE_OPEN}|\b(?:and|then|or|but|so|plus|next)\s+)"
# "ignore system: true" is a config listing, not an order.
_NOT_CONFIG = r"(?!\s*(?:true|false|null|none|yes|no|on|off|enabled?|disabled?|\d)\b)"

ROLE_OVERRIDE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Label form: "IGNORE SYSTEM:", "Ignore all previous system instructions:".
    re.compile(
        rf"{_LABEL_OPEN}{_SCAFFOLD}"
        rf"(?P<core>{_VERB}\s+{_DET}{_ROLE}(?:\s+{_NOUN})?\s*[:\uff1a]){_NOT_CONFIG}",
        re.IGNORECASE,
    ),
    # Colonless form: "forget your developer instructions". The directive noun
    # is what keeps it clear of prose about ignoring system errors.
    re.compile(
        rf"{_CLAUSE_OPEN}{_SCAFFOLD}(?P<core>{_VERB}\s+{_DET}{_ROLE}\s+{_NOUN})\b",
        re.IGNORECASE,
    ),
)


SEGMENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(slots=True)
class InjectionScan:
    clean_text: str
    flags: list[str] = field(default_factory=list)
    removed_segments: list[str] = field(default_factory=list)


def scan_text(
    text: str | None,
    *,
    preserve_layout: bool = False,
) -> InjectionScan:
    """Quarantine instruction-like segments and return an audit record."""
    if not text:
        return InjectionScan(clean_text="")
    if preserve_layout:
        return _scan_preserving_layout(text)
    kept: list[str] = []
    flags: list[str] = []
    removed: list[str] = []
    for segment in SEGMENT_SPLIT.split(text):
        if not segment or not segment.strip():
            continue
        matched = match_injection(segment)
        if matched:
            flags.append(matched)
            removed.append(segment.strip()[:200])
        else:
            kept.append(segment.strip())
    return InjectionScan(clean_text=" ".join(kept), flags=flags, removed_segments=removed)


def _scan_preserving_layout(text: str) -> InjectionScan:
    kept: list[str] = []
    flags: list[str] = []
    removed: list[str] = []
    cursor = 0
    for separator in SEGMENT_SPLIT.finditer(text):
        segment = text[cursor : separator.start()]
        delimiter = separator.group(0)
        matched = match_injection(segment)
        if matched:
            flags.append(matched)
            removed.append(segment.strip()[:200])
            if "\n" in delimiter:
                kept.append("\n")
        else:
            kept.append(segment + delimiter)
        cursor = separator.end()
    tail = text[cursor:]
    matched = match_injection(tail)
    if matched:
        flags.append(matched)
        removed.append(tail.strip()[:200])
    else:
        kept.append(tail)
    return InjectionScan(
        clean_text="".join(kept),
        flags=flags,
        removed_segments=removed,
    )


def match_injection(segment: str) -> str | None:
    for pattern in INJECTION_PATTERNS:
        found = pattern.search(segment)
        if found:
            return found.group(0)[:80]
    return match_role_override(segment)


def match_role_override(segment: str) -> str | None:
    """Return only the directive head of a role-override attempt."""
    for pattern in ROLE_OVERRIDE_PATTERNS:
        found = pattern.search(segment)
        if found:
            return " ".join(found.group("core").split())[:80]
    return None


def injection_flags(text: str | None) -> list[str]:
    return scan_text(text).flags


def quarantine_notes(pages: list[CrawledPage], limit: int = 5) -> list[str]:
    notes: list[str] = []
    for page in pages:
        flags = injection_flags(page.text)
        if not flags:
            continue
        notes.append(
            f"Quarantined {len(flags)} instruction-like segment(s) from "
            f"{page.final_url or page.url} before claim extraction."
        )
        if len(notes) >= limit:
            break
    return notes

"""Deterministic local preservation of uncertain hypotheses in answers.

Every uncertain hypothesis must appear in the certified answer through a
clause carrying a distinctive content anchor plus an openness marker,
measured against all open hypotheses; settling language fails.  All logic
is bounded text matching: no model calls, hidden metadata, or
task-specific vocabulary.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from typing import Any

from cortheon.cognitive_core.models import Investigation

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT = re.compile(r"[,;:\u2014]")

# Markers that predicate openness of whatever the clause names: "X is
# ambiguous" is an assertion about X, so it can carry a lone anchor.
_PREDICATIVE_OPENNESS = (
    r"uncertain(?:ty)?",
    r"unresolved",
    r"not settled",
    r"unsettled",
    r"ambiguous",
    r"insufficient evidence",
    r"remains open",
    r"open question",
    r"does not settle",
    r"cannot settle",
)

# Markers that only frame a rival.  "A competing alternative is X" asserts
# nothing about X, and both its own words are generic, so a clause break
# can strand a fragment that carries one shared topical word and names
# nothing else.  Framing must therefore identify its subject outright.
_FRAMING_OPENNESS = (r"competing alternative",)


def _marker_pattern(markers: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(markers) + r")\b", flags=re.IGNORECASE)


_OPENNESS_MARKERS = _marker_pattern(_PREDICATIVE_OPENNESS + _FRAMING_OPENNESS)
_PREDICATIVE_MARKERS = _marker_pattern(_PREDICATIVE_OPENNESS)
_GROUP_OPENNESS = re.compile(
    r"\b(?:alternatives|hypotheses|interpretations|possibilities)\b"
    r"[^.!?]{0,120}\b(?:ambiguous|remain(?:s)? uncertain|unresolved|unsettled)\b",
    flags=re.IGNORECASE,
)

_SETTLING_MARKERS = re.compile(
    r"\b(?:disprov\w+|refut\w+|eliminat\w+|falsifi\w+|ruled\s+out|"
    r"impossible|cannot\s+explain|can't\s+explain|does\s+not\s+explain|"
    r"doesn't\s+explain|definitely\s+not)\b",
    flags=re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A settling clause whose subject is only a pronoun or generic rival phrase
# refers back to the nearest anchor in the same sentence, so it is linked
# to that anchor's hypothesis for the settling test.
_ANAPHOR_RE = re.compile(
    r"\b(?:it|them|the rival|the latter|the alternative)\b",
    flags=re.IGNORECASE,
)

# Words that carry no distinctive content for identifying a hypothesis:
# ordinary function words plus the generic vocabulary of rival framing and
# of uncertainty itself.  An anchor must be a content word, so repeating
# only "the rival alternative is uncertain" never identifies anything.
_GENERIC_WORDS = frozenset(
    {
        "about",
        "alternative",
        "ambiguous",
        "ambiguity",
        "another",
        "answer",
        "because",
        "been",
        "before",
        "candidate",
        "cause",
        "causes",
        "caused",
        "competing",
        "definitely",
        "does",
        "evidence",
        "explanation",
        "explain",
        "explains",
        "explained",
        "further",
        "hypothesis",
        "instead",
        "insufficient",
        "mechanism",
        "open",
        "other",
        "possible",
        "possibly",
        "probably",
        "question",
        "remains",
        "remain",
        "rival",
        "rivals",
        "settled",
        "settle",
        "settles",
        "should",
        "some",
        "their",
        "there",
        "this",
        "that",
        "these",
        "those",
        "uncertain",
        "uncertainty",
        "under",
        "unsettled",
        "unresolved",
        "were",
        "where",
        "which",
        "while",
        "with",
        "would",
        "your",
    }
)


def _content_tokens(text: str) -> list[str]:
    """Content words of a fragment, in order, without repeats."""

    return list(
        dict.fromkeys(
            token
            for token in _TOKEN_RE.findall(text.lower())
            if len(token) >= 4 and token not in _GENERIC_WORDS
        )
    )


def _content_anchors(statement: str) -> list[str]:
    """Content words a hypothesis statement can be identified by."""

    return _content_tokens(statement)


def _segments(answer: str) -> list[list[str]]:
    """Split an answer into sentences, then into bounded clauses."""

    sentences: list[list[str]] = []
    for sentence in _SENTENCE_SPLIT.split(answer):
        clauses = [clause.strip() for clause in _CLAUSE_SPLIT.split(sentence)]
        sentences.append([clause for clause in clauses if clause])
    return [sentence for sentence in sentences if sentence]


def _hypothesis_visibility(
    statement: str,
    sentences: list[list[str]],
    others: Sequence[str] = (),
) -> tuple[bool, str]:
    """Classify how the answer treats one uncertain hypothesis.

    ``others``: anchors shared with any of them identify neither.
    """

    lowered = [[clause.lower() for clause in sentence] for sentence in sentences]
    anchors = _content_anchors(statement)
    if not anchors:
        # A statement with no distinctive content words cannot be anchored,
        # so no answer can visibly preserve it: unrelated whole-answer
        # openness must not launder it, and "not mentioning it" is not
        # preservation.  Fail closed and ask for a resubmission with
        # concrete content.
        return False, "unanchorable"
    shared = {anchor for other in others for anchor in _content_anchors(other)}
    distinctive = {anchor for anchor in anchors if anchor not in shared}
    if not distinctive:
        # Every content word of this hypothesis also appears in another open
        # hypothesis, so no clause can name this one rather than that one and
        # a single openness phrase would preserve both through the token they
        # share.  Fail closed and ask for distinguishing wording.
        return False, "indistinguishable"
    patterns = {anchor: re.compile(rf"\b{re.escape(anchor)}\b") for anchor in anchors}

    def _matched(clause: str, keys: Iterable[str]) -> list[str]:
        return [key for key in keys if patterns[key].search(clause)]

    anchor_sentences = {
        index
        for index, sentence in enumerate(lowered)
        if any(_matched(clause, anchors) for clause in sentence)
    }
    if not anchor_sentences:
        return False, "unmentioned"

    def _settles_clause(clause: str, sentence_index: int) -> bool:
        if not _SETTLING_MARKERS.search(clause):
            return False
        return bool(_matched(clause, anchors)) or (
            sentence_index in anchor_sentences and bool(_ANAPHOR_RE.search(clause))
        )

    def _preserves_clause(clause: str) -> tuple[bool, bool]:
        """Return whether the clause preserves, and whether it named another.

        Two matched anchors identify the mechanism; one only in a
        predicated-openness clause naming nothing else.
        """

        if not _OPENNESS_MARKERS.search(clause):
            return False, False
        matched = _matched(clause, anchors)
        own = [anchor for anchor in matched if anchor in distinctive]
        if not own:
            return False, False
        if len(matched) >= 2:
            return True, False
        if not _PREDICATIVE_MARKERS.search(clause):
            return False, True
        foreign = [token for token in _content_tokens(clause) if token not in matched]
        return not foreign, bool(foreign)

    if any(
        _settles_clause(clause, index)
        for index, sentence in enumerate(lowered)
        for clause in sentence
    ):
        return False, "settles"
    if any(
        _GROUP_OPENNESS.search("; ".join(sentence)) and _matched("; ".join(sentence), distinctive)
        for sentence in lowered
    ):
        return True, "preserved"
    verdicts = [_preserves_clause(clause) for sentence in lowered for clause in sentence]
    if any(preserves for preserves, _ in verdicts):
        return True, "preserved"
    if not any(_matched(clause, distinctive) for sentence in lowered for clause in sentence):
        # The answer only ever used wording this hypothesis shares with
        # another open one, so it never named this mechanism at all.
        return False, "shared_only"
    if any(named_other for _, named_other in verdicts):
        # An openness clause carried one anchor of this hypothesis while
        # naming other content, so the clause is about that other content
        # and cannot stand in for keeping this mechanism open.
        return False, "unidentified"
    return False, "unpreserved"


def _structured_rival_preserved(statement: str, answer: str, others: Sequence[str]) -> bool:
    """Treat an exact structured rival field as explicit alternative framing."""

    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict):
        return False
    values = [
        value
        for key, value in parsed.items()
        if isinstance(key, str)
        and key.casefold() in {"rival", "rivals", "alternative", "alternatives"}
    ]
    anchors = _content_anchors(statement)
    shared = {anchor for other in others for anchor in _content_anchors(other)}
    distinctive = set(anchors) - shared
    for value in values:
        text = json.dumps(value, sort_keys=True).lower().replace("_", " ")
        matched = {anchor for anchor in anchors if re.search(rf"\b{re.escape(anchor)}\b", text)}
        if len(matched) >= 2 and matched & distinctive and not _SETTLING_MARKERS.search(text):
            return True
    return False


def uncertainty_visibility_check(
    session: Investigation,
    answer: str,
) -> dict[str, Any]:
    """Every uncertain hypothesis must be visibly preserved in the answer."""

    uncertain = [
        hypothesis for hypothesis in session.hypotheses.values() if hypothesis.status == "uncertain"
    ]
    if not uncertain:
        return {
            "name": "uncertainty_visibility",
            "passed": True,
            "reason": "No submitted hypothesis is uncertain.",
        }
    segments = _segments(answer)
    statements = [hypothesis.statement for hypothesis in uncertain]
    outcomes = [
        # Each hypothesis is judged against every other open hypothesis, so a
        # word two of them share can never identify either one.
        _hypothesis_visibility(statement, segments, statements[:index] + statements[index + 1 :])
        for index, statement in enumerate(statements)
    ]
    outcomes = [
        (True, "preserved")
        if outcome != "preserved"
        and _structured_rival_preserved(
            statements[index],
            answer,
            statements[:index] + statements[index + 1 :],
        )
        else result
        for index, result in enumerate(outcomes)
        for outcome in [result[1]]
    ]
    failed = [
        (statement, outcome)
        for statement, (_, outcome) in zip(statements, outcomes, strict=True)
        if outcome != "preserved"
    ]
    if failed:
        details = _join_findings(failed)
        return {
            "name": "uncertainty_visibility",
            "passed": False,
            "reason": (
                "The answer does not visibly preserve every uncertain "
                f"hypothesis: {details}. Keep each open hypothesis anchored "
                "to explicit uncertainty in the same sentence or clause, for "
                "example 'cache compaction remains uncertain' or 'a competing "
                "alternative is cache compaction'."
            ),
        }
    return {
        "name": "uncertainty_visibility",
        "passed": True,
        "reason": (
            "The answer explicitly preserves the uncertainty of "
            f"{len(uncertain)} open hypothesis(es) with locally anchored "
            "openness language."
        ),
    }


# Failure kinds in report order, each with the sentence that explains it.
_FINDINGS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("settles",), "the answer settles "),
    (
        ("unmentioned", "shared_only"),
        "the answer never anchors its own distinctive content ",
    ),
    (
        ("unidentified",),
        "the openness clause names other content, or frames a rival by one "
        "shared word, so it does not keep this mechanism open; resubmit with "
        "the hypothesis named beside the uncertainty: ",
    ),
    (("unpreserved",), "the answer mentions without locally anchored openness "),
    (
        ("unanchorable",),
        "the uncertain hypothesis is too generic to verify, so resubmit it "
        "with specific content instead of generic rival framing: ",
    ),
    (
        ("indistinguishable",),
        "these open hypotheses share every content word, so no clause can keep "
        "one open without the other; resubmit each with wording that names its "
        "own mechanism: ",
    ),
)


def _join_findings(failed: Sequence[tuple[str, str]]) -> str:
    parts: list[str] = []
    for kinds, prefix in _FINDINGS:
        items = [statement for statement, outcome in failed if outcome in kinds]
        if items:
            parts.append(prefix + "; ".join(f"'{item}'" for item in items))
    return "; ".join(parts)

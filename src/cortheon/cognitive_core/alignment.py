"""Answer/evidence alignment checks for research goals."""

from __future__ import annotations

import re
from typing import Any

from cortheon.cognitive_core.claims import _NEGATION_RE
from cortheon.cognitive_core.models import Investigation, Observation, PublicClaim
from cortheon.cognitive_core.receipts import (
    _HOST_EVIDENCE_PREFIX,
    _host_path_matches_request,
    _observation_origin,
    _read_receipt_paths,
)
from cortheon.cognitive_core.research_gaps import (
    _answer_acknowledges_research_conflict,
    _answer_urls,
    _latest_release_goal,
    _research_release_analysis,
)
from cortheon.cognitive_core.semantic_graph import _semantic_terms
from cortheon.cognitive_core.tasks import _ABDUCTIVE_GOAL_RE, _AMBIGUITY_GOAL_RE
from cortheon.cognitive_core.text import _normalized
from cortheon.cognitive_protocol import evaluation_operator

_FALSIFICATION_DESIGN_RE = re.compile(
    r"\b(?:state|give|identify|propose|specify|describe)\b.{0,100}"
    r"\b(?:falsif(?:y|ication)|disprov(?:e|ing)|distinguish(?:ing)?)\b.{0,80}"
    r"\b(?:observation|test|counterexample)\b|"
    r"\b(?:state|give|identify|propose|specify|describe)\b.{0,100}"
    r"\b(?:observation|test|counterexample)\b.{0,80}"
    r"\b(?:falsif(?:y|ication)|disprov(?:e|ing)|distinguish(?:ing)?)\b",
    flags=re.IGNORECASE | re.DOTALL,
)


def _research_conflict_present(observations: list[Observation]) -> bool:
    passages = []
    for item in observations:
        body = "\n".join(
            line for line in item.content.splitlines() if not line.startswith(_HOST_EVIDENCE_PREFIX)
        )
        terms = _semantic_terms(body)
        passages.append(
            (
                _observation_origin(item),
                terms,
                _NEGATION_RE.search(body) is not None,
                set(re.findall(r"\b\d+(?:\.\d+){1,3}\b", body)),
            )
        )
    for index, (
        left_origin,
        left_terms,
        left_negative,
        left_versions,
    ) in enumerate(passages):
        for (
            right_origin,
            right_terms,
            right_negative,
            right_versions,
        ) in passages[index + 1 :]:
            conflicting_versions = (
                bool(left_versions)
                and bool(right_versions)
                and left_versions.isdisjoint(right_versions)
            )
            if (
                left_origin
                and right_origin
                and left_origin != right_origin
                and len(left_terms.intersection(right_terms)) >= 3
                and (left_negative != right_negative or conflicting_versions)
            ):
                return True
    return False


def _research_alignment_check(
    session: Investigation,
    answer: str,
    claims: list[PublicClaim],
) -> dict[str, Any]:
    cited_ids = {evidence_id for claim in claims for evidence_id in claim.evidence_ids}
    cited = [
        session.observations[evidence_id]
        for evidence_id in cited_ids
        if evidence_id in session.observations and session.observations[evidence_id].kind == "web"
    ]
    evidence_origins = {_observation_origin(item) for item in cited}
    evidence_origins.discard(None)
    cited_origins = _answer_urls(answer)
    matched = evidence_origins.intersection(cited_origins)
    required_origins = 1 if "corroboration" in session.waivers else 2
    if len(matched) < required_origins:
        return {
            "name": "evidence_alignment",
            "passed": False,
            "reason": (
                "The answer must include clickable citations to at least two "
                "independent origins cited by its claims."
                if required_origins == 2
                else "The answer must include at least one clickable citation to "
                "an origin cited by its claims."
            ),
        }
    release = _research_release_analysis(session.goal, cited)
    latest_release_question = _latest_release_goal(session.goal)
    if latest_release_question and release is None and "corroboration" not in session.waivers:
        return {
            "name": "evidence_alignment",
            "passed": False,
            "reason": (
                "The cited evidence does not establish one release version across "
                "two independent origins."
            ),
        }
    if release is not None and not re.search(
        rf"(?<![0-9])v?{re.escape(release['value'])}(?![0-9]|\.[0-9])",
        answer,
        flags=re.IGNORECASE,
    ):
        return {
            "name": "evidence_alignment",
            "passed": False,
            "reason": (
                "The answer does not state the release version independently "
                f"established by the cited sources: {release['value']}."
            ),
        }
    revision_enabled = evaluation_operator(
        session.evaluation_profile,
        "contradiction_revision",
    )
    if (
        revision_enabled
        and _research_conflict_present(cited)
        and not (_answer_acknowledges_research_conflict(answer, cited))
    ):
        return {
            "name": "evidence_alignment",
            "passed": False,
            "reason": (
                "The cited sources contain a material polarity conflict that the "
                "answer does not substantively acknowledge or scope."
            ),
        }
    return {
        "name": "evidence_alignment",
        "passed": True,
        "reason": (
            f"The answer cites {len(matched)} independent live origins"
            + (
                " and handles any detected source conflict."
                if revision_enabled
                else ". Contradiction revision is not applicable in this condition."
            )
        ),
    }


def _answer_polarity(answer: str) -> bool | None:
    normalized = _normalized(answer).lstrip("`*_ ")
    match = re.match(r"^(?:answer:\s*)?(yes|no)\b", normalized)
    if match is None:
        return None
    return match.group(1) == "yes"


def _ambiguity_alignment_check(
    session: Investigation,
    answer: str,
    cited: list[Observation],
    paths: list[str],
) -> dict[str, Any] | None:
    """Accept a calibrated non-decision when live sources expose real ambiguity."""

    if _AMBIGUITY_GOAL_RE.search(session.goal) is None:
        return None
    reads = _read_receipt_paths(cited)
    relevant = [
        observation
        for path, observation in reads.items()
        if not paths or any(_host_path_matches_request(path, expected) for expected in paths)
    ]
    answer_key = answer.casefold()
    uncertainty = re.search(
        r"\b(?:ambiguous|cannot determine|can't determine|insufficient|unresolved|"
        r"not enough information|either|conflicts?|underspecified|"
        r"not actionable|no basis)\b",
        answer_key,
    )
    clarification = re.search(
        r"\b(?:clarif(?:y|ication)|which|effective date|authority|supersession|"
        r"environment|component|metric|percentile|funnel|baseline)\b",
        answer_key,
    )
    unsupported_choice = re.search(
        r"\b(?:therefore|recommend(?:ed)?|should|will|proceed(?:ing)?(?:\s+with)?)"
        r"\b.{0,40}\b(?:deploy|optimi[sz]e|choose|select|owner is)\b",
        answer_key,
    )
    # A branch can be settled without any decision verb: naming both
    # readings and then quietly adopting one ("treating X as the
    # operative reading") is still a guess, not a calibrated non-decision.
    # Ordinary methodological phrasing ("treating each statement as
    # evidence") is not settlement, so the generic detector requires an
    # adoption predicate or an operative/intended/working/prevailing
    # interpretation construction.
    settled_branch = re.search(
        r"\b(?:treating|treated)\b.{0,80}\bas\b[^.!?\n]{0,60}"
        r"\b(?:the|our|one)\s+(?:operative|intended|working|prevailing|"
        r"correct|governing)\s+"
        r"(?:reading|meaning|definition|interpretation|sense|one)\b|"
        r"\b(?:going with|settling on|settled on|adopt(?:ing)?)\b|"
        r"\bthe\s+(?:operative|prevailing|intended|working)\s+"
        r"(?:reading|meaning|definition|interpretation|sense)\b|"
        r"\bis\s+the\s+one\s+that\s+(?:applies|counts|matters|governs)\b",
        answer_key,
    )
    answer_terms = _semantic_terms(answer)
    # Every clean read that carries terms distinctive to it among the cited
    # reads must be distinctively represented in the answer. Shared topical
    # boilerplate ("margin" appearing in every glossary) must not let a
    # one-sided answer count as covering both sides of a live disagreement,
    # and a neutral request document must not stand in for an omitted
    # disagreeing source.
    term_sets = [_semantic_terms(observation.content) for observation in relevant]
    uncovered_reads: list[str] = []
    for index, (terms, observation) in enumerate(zip(term_sets, relevant, strict=True)):
        shared = set().union(*(term_sets[:index] + term_sets[index + 1 :]))
        distinctive = terms - shared
        if distinctive and not answer_terms & distinctive:
            uncovered_reads.append(str(observation.source or observation.evidence_id))
    passed = (
        len(reads) >= 2
        and not uncovered_reads
        and uncertainty is not None
        and clarification is not None
        and unsupported_choice is None
        and settled_branch is None
    )
    missing = [
        label
        for condition, label in (
            (len(reads) >= 2, "two clean live reads"),
            (not uncovered_reads, "the distinctive content of every clean read"),
            (uncertainty is not None, "an explicit ambiguity statement"),
            (clarification is not None, "a discriminating clarification"),
            (unsupported_choice is None, "removal of the unsupported branch choice"),
            (settled_branch is None, "removal of the quiet branch settlement"),
        )
        if not condition
    ]
    return {
        "name": "evidence_alignment",
        "passed": passed,
        "reason": (
            "The answer preserves the live cross-source ambiguity, names both "
            "evidence-backed interpretations, and asks a discriminating clarification."
            if passed
            else (
                "Ambiguity completion is missing: "
                + ", ".join(missing)
                + (" (uncovered: " + ", ".join(uncovered_reads) + ")" if uncovered_reads else "")
                + "."
            )
        ),
    }


_LEADING_EVIDENCE_LINE_RE = re.compile(r"\s*evidence\s*:", flags=re.IGNORECASE)
_ANSWER_SECTION_LINE_RE = re.compile(r"\s*(?:cause|rival|test)\s*:", flags=re.IGNORECASE)


def _model_authored_answer(answer: str) -> str:
    """Return the answer without its leading host-built ``Evidence:`` ledger.

    The causal synthesis adapter prepends that ledger, quoting accepted
    records verbatim; reading it as model output would let host data
    satisfy the model-authored causal checks.  Only the leading section is
    removed, and only for this computation.
    """

    lines = answer.splitlines()
    start = next((index for index, line in enumerate(lines) if line.strip()), None)
    if start is None or _LEADING_EVIDENCE_LINE_RE.match(lines[start]) is None:
        return answer
    body = next(
        (
            index
            for index in range(start + 1, len(lines))
            if _ANSWER_SECTION_LINE_RE.match(lines[index])
        ),
        start + 1,
    )
    return "\n".join(lines[body:])


def _abductive_alignment_check(
    session: Investigation,
    answer: str,
    cited: list[Observation],
    paths: list[str],
) -> dict[str, Any] | None:
    """Validate evidence-bound abduction without pretending it is deduction."""

    if (
        _ABDUCTIVE_GOAL_RE.search(session.goal) is None
        or _AMBIGUITY_GOAL_RE.search(session.goal) is not None
    ):
        return None
    hypotheses = list(session.hypotheses.values())
    reads = _read_receipt_paths(cited)
    if len(hypotheses) < 2 or len(reads) < 2:
        return {
            "name": "evidence_alignment",
            "passed": False,
            "reason": (
                "Abductive synthesis requires at least two resolved competing "
                "hypotheses and two clean, separately cited live document reads."
            ),
        }

    known = set(session.observations)
    supported = [
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.status == "supported"
        and hypothesis.supporting_evidence
        and all(evidence_id in known for evidence_id in hypothesis.supporting_evidence)
    ]
    all_resolved = all(
        hypothesis.status in {"supported", "refuted", "uncertain"}
        and (
            hypothesis.supporting_evidence
            or hypothesis.contradicting_evidence
            or hypothesis.bearing_evidence
        )
        for hypothesis in hypotheses
    )
    supported_read_paths: set[str] = set()
    for hypothesis in supported:
        for evidence_id in hypothesis.supporting_evidence:
            observation = session.observations[evidence_id]
            supported_read_paths.update(_read_receipt_paths([observation]))

    model_answer = _model_authored_answer(answer)
    answer_terms = _semantic_terms(model_answer)
    relevant_reads = [
        observation
        for path, observation in reads.items()
        if not paths or any(_host_path_matches_request(path, expected) for expected in paths)
    ]
    represented_sources = sum(
        1
        for observation in relevant_reads
        if len(answer_terms & _semantic_terms(observation.content)) >= 2
    )
    has_alternative = re.search(
        r"\b(?:alternative|competing|another hypothesis|other explanation)\b",
        model_answer,
        flags=re.IGNORECASE,
    )
    has_falsification = re.search(
        r"\b(?:falsif(?:y|ies|ied|ying|iable|ication)|disprov(?:e|ing)|counterexample|"
        r"distinguish(?:ing)? test|would fail if|test would)\b",
        model_answer,
        flags=re.IGNORECASE,
    )
    has_bridge = re.search(
        r"\b(?:because|therefore|causal|leads? to|results? in|explains?|"
        r"interaction|combined|exceeds?|routes? through)\b",
        model_answer,
        flags=re.IGNORECASE,
    )
    passed = (
        bool(supported)
        and all_resolved
        and len(supported_read_paths) >= 2
        and represented_sources >= 2
        and has_alternative is not None
        and has_falsification is not None
        and has_bridge is not None
    )
    missing = [
        label
        for condition, label in (
            (bool(supported), "a supported leading hypothesis"),
            (all_resolved, "resolved competing hypotheses"),
            (len(supported_read_paths) >= 2, "support from two separate reads"),
            (represented_sources >= 2, "answer coverage across two source records"),
            (has_alternative is not None, "an explicit alternative"),
            (has_falsification is not None, "an observable falsification test"),
            (has_bridge is not None, "an explicit causal bridge"),
        )
        if not condition
    ]
    if passed:
        return {
            "name": "evidence_alignment",
            "passed": True,
            "reason": (
                "The answer is a bounded abductive conclusion: separate live sources "
                "support the leading explanation, a resolved alternative is retained, "
                "the causal bridge is explicit, and a discriminating falsification is stated."
            ),
        }
    reason = "Abductive completion is missing: " + ", ".join(missing) + "."
    if model_answer != answer:
        reason += (
            " The host-built Evidence ledger is not model output, so state the "
            "supporting facts and the causal bridge in your own Cause, Rival, and "
            "Test lines."
        )
    return {"name": "evidence_alignment", "passed": False, "reason": reason}

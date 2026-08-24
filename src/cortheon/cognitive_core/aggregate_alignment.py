"""Aggregate read-many and evidence alignment checks."""

from __future__ import annotations

import re
from typing import Any

from cortheon.cognitive_core.alignment import (
    _abductive_alignment_check,
    _ambiguity_alignment_check,
    _answer_polarity,
    _research_alignment_check,
)
from cortheon.cognitive_core.claims import _NEGATION_RE
from cortheon.cognitive_core.models import EvidenceRequest, Investigation, Observation, PublicClaim
from cortheon.cognitive_core.semantic_graph import (
    _affirmatively_mentions,
    _phrase_mentioned,
    _semantic_key,
)
from cortheon.cognitive_core.semantic_join import _semantic_join_analysis
from cortheon.cognitive_core.tasks import (
    _INTEGER_TOKEN,
    _answer_integer_assertions,
    _goal_code_paths,
    _parse_integer,
)
from cortheon.cognitive_core.text import _lookup_target_match, _normalized
from cortheon.cognitive_protocol import evaluation_operator


def _evidence_alignment_check(
    session: Investigation,
    answer: str,
    claims: list[PublicClaim],
) -> dict[str, Any]:
    """Check high-confidence atomic lookups against deterministic host receipts."""

    read_many = next(
        (request for request in session.requests.values() if request.capability == "read_many"),
        None,
    )
    if session.deliverable == "research_answer":
        return _research_alignment_check(session, answer, claims)
    if (
        session.deliverable in {"code_understanding", "document_synthesis"}
        and read_many is not None
    ):
        return _read_many_alignment_check(session, answer, claims, read_many)

    target_match = _lookup_target_match(session.goal)
    if session.deliverable != "code_understanding" or target_match is None:
        return {
            "name": "evidence_alignment",
            "passed": True,
            "reason": "No deterministic atomic-lookup contract applies to this task.",
        }

    target = target_match.group(1).lower()
    paths = _goal_code_paths(session.goal)
    expected_scope = paths[0].lower() if paths else None
    cited_ids = {evidence_id for claim in claims for evidence_id in claim.evidence_ids}
    aligned: list[tuple[Observation, dict[str, Any]]] = []
    for evidence_id in cited_ids:
        observation = session.observations.get(evidence_id)
        if observation is None:
            continue
        receipt = observation.host_receipt
        if receipt is None or receipt.get("tool") != "grep":
            continue
        arguments = receipt["args"]
        pattern = str(arguments.get("pattern") or "").lower()
        scope = str(arguments.get("path") or "").lower()
        if target not in pattern:
            continue
        if expected_scope and not (expected_scope in scope or (scope and scope in expected_scope)):
            continue
        aligned.append((observation, receipt))

    if not aligned:
        return {
            "name": "evidence_alignment",
            "passed": False,
            "reason": (
                f"The lookup target '{target}' lacks an exact grep receipt in the named scope."
            ),
        }

    outcomes = {receipt["outcome"] for _, receipt in aligned}
    if outcomes == {"no_match"}:
        predicate_present = False
    elif outcomes == {"match"}:
        predicate_present = True
    else:
        return {
            "name": "evidence_alignment",
            "passed": False,
            "reason": "The targeted host receipts are ambiguous or mutually inconsistent.",
        }

    verb = target_match.group(0).split()[0].lower()
    if verb.startswith("import") and predicate_present:
        import_pattern = re.compile(
            rf"(?:\bfrom\s+{re.escape(target)}\b|\bimport\b[^\n]*\b"
            rf"{re.escape(target)}\b)",
            flags=re.IGNORECASE,
        )
        if not any(import_pattern.search(item.content) for item, _ in aligned):
            return {
                "name": "evidence_alignment",
                "passed": False,
                "reason": (
                    f"The search matched '{target}' but did not show a static import statement."
                ),
            }

    answer_polarity = _answer_polarity(answer)
    if answer_polarity is not None and answer_polarity != predicate_present:
        result = "a match" if predicate_present else "zero matches"
        return {
            "name": "evidence_alignment",
            "passed": False,
            "reason": (f"The answer polarity contradicts the deterministic grep result: {result}."),
        }

    if (
        expected_scope
        and re.search(
            r"\b(?:cite|citation|evidence|source)\b",
            session.goal,
            flags=re.IGNORECASE,
        )
        and expected_scope not in answer.casefold()
    ):
        return {
            "name": "evidence_alignment",
            "passed": False,
            "reason": (
                "The user requested evidence in the answer, but the answer does not "
                f"name the verified scope '{expected_scope}'."
            ),
        }

    relevant_claims = [claim for claim in claims if target in _normalized(claim.claim)]
    if not relevant_claims:
        return {
            "name": "evidence_alignment",
            "passed": False,
            "reason": f"No explicit answer claim addresses the lookup target '{target}'.",
        }
    for claim in relevant_claims:
        claim_is_positive = _NEGATION_RE.search(claim.claim) is None
        if claim_is_positive != predicate_present:
            result = "a match" if predicate_present else "zero matches"
            return {
                "name": "evidence_alignment",
                "passed": False,
                "reason": (
                    f"Claim '{claim.claim}' contradicts the deterministic result: {result}."
                ),
            }

    result = "a scoped match" if predicate_present else "a scoped zero-match result"
    return {
        "name": "evidence_alignment",
        "passed": True,
        "reason": f"The answer and claims agree with {result} for '{target}'.",
    }


def _read_many_alignment_check(
    session: Investigation,
    answer: str,
    claims: list[PublicClaim],
    request: EvidenceRequest,
) -> dict[str, Any]:
    """Bind a cross-file conclusion to every requested live read.

    Numeric joins receive an additional deterministic check. This catches the
    common small-model failure where the right values are retrieved but the
    arithmetic in the final answer is wrong.
    """

    paths = [str(item) for item in request.parameters.get("paths", ()) if isinstance(item, str)]
    symbols = [str(item) for item in request.parameters.get("symbols", ()) if isinstance(item, str)]
    cited_ids = {evidence_id for claim in claims for evidence_id in claim.evidence_ids}
    cited = [
        session.observations[evidence_id]
        for evidence_id in cited_ids
        if evidence_id in session.observations
    ]

    covered_paths: set[str] = set()
    for observation in cited:
        receipt = observation.host_receipt
        if receipt is None or receipt.get("tool") != "read":
            continue
        file_path = str(receipt.get("args", {}).get("filePath") or "")
        for path in paths:
            if file_path.casefold() == path.casefold():
                covered_paths.add(path)
    # A requested path whose only live reads were quarantined or failed is
    # still missing evidence: the answer is certified only from reads the
    # runtime can actually trust, and a three-document request is never
    # satisfied by two clean documents.
    missing_paths = [path for path in paths if path not in covered_paths]
    if missing_paths and not bool(request.parameters.get("discovered")):
        return {
            "name": "evidence_alignment",
            "passed": False,
            "reason": (
                "The cross-file answer lacks cited host-read receipts for: "
                + ", ".join(missing_paths)
                + "."
            ),
        }

    operation = request.parameters.get("operation")
    if operation in {"semantic_join", "sum"} and not evaluation_operator(
        session.evaluation_profile,
        "cross_source_derivation",
    ):
        return {
            "name": "evidence_alignment",
            "passed": True,
            "applicable": False,
            "reason": (
                "Cross-source derivation is not applicable in this condition; "
                "the answer still cites every requested live source."
            ),
        }
    if operation == "semantic_join":
        ambiguity = _ambiguity_alignment_check(
            session,
            answer,
            cited,
            paths,
        )
        if ambiguity is not None:
            return ambiguity
        derivation = _semantic_join_analysis(
            session.goal,
            paths,
            cited,
            require_all_documents=not bool(request.parameters.get("discovered")),
        )
        if derivation is not None:
            if derivation.get("status") == "conflicted":
                conflicts = derivation.get("conflicts", ())
                entities = [
                    str(item.get("entity"))
                    for item in conflicts
                    if isinstance(item, dict) and item.get("entity")
                ]
                return {
                    "name": "evidence_alignment",
                    "passed": False,
                    "reason": (
                        "The cited documents contain an unresolved source conflict"
                        + (f" for {', '.join(entities)}" if entities else "")
                        + ". Cortheon will not choose a branch without explicit "
                        "current or authoritative disambiguation."
                    ),
                }
            if derivation.get("status") == "ordered_plan":
                answer_key = _semantic_key(answer)
                positions = [answer_key.find(_semantic_key(node)) for node in derivation["nodes"]]
                if any(position < 0 for position in positions):
                    missing = [
                        node
                        for node, position in zip(
                            derivation["nodes"],
                            positions,
                            strict=True,
                        )
                        if position < 0
                    ]
                    return {
                        "name": "evidence_alignment",
                        "passed": False,
                        "reason": (
                            "The plan omits evidence-bound steps: " + ", ".join(missing) + "."
                        ),
                    }
                if positions != sorted(positions) or len(set(positions)) != len(positions):
                    return {
                        "name": "evidence_alignment",
                        "passed": False,
                        "reason": (
                            "The stated plan order contradicts the deterministic dependency graph."
                        ),
                    }
                missing_owners = [
                    owner
                    for owner in derivation.get("owners", {}).values()
                    if not _phrase_mentioned(answer, owner)
                ]
                if missing_owners:
                    return {
                        "name": "evidence_alignment",
                        "passed": False,
                        "reason": (
                            "The plan omits evidence-bound owners: "
                            + ", ".join(missing_owners)
                            + "."
                        ),
                    }
                return {
                    "name": "evidence_alignment",
                    "passed": True,
                    "reason": (
                        "The answer follows the deterministic dependency order and "
                        "names every evidence-bound owner."
                    ),
                }
            missing_nodes = [
                node for node in derivation["nodes"] if not _phrase_mentioned(answer, node)
            ]
            if missing_nodes:
                return {
                    "name": "evidence_alignment",
                    "passed": False,
                    "reason": (
                        "The answer omits evidence-linked nodes from the deterministic "
                        "cross-document chain: " + ", ".join(missing_nodes) + "."
                    ),
                }
            unrelated = [
                node
                for node in derivation["excluded_nodes"]
                if _affirmatively_mentions(answer, node)
            ]
            if unrelated:
                return {
                    "name": "evidence_alignment",
                    "passed": False,
                    "reason": (
                        "The answer affirmatively introduces unrelated branches instead "
                        "of the unique evidence-linked chain: " + ", ".join(unrelated) + "."
                    ),
                }
            return {
                "name": "evidence_alignment",
                "passed": True,
                "reason": (
                    "The answer states every node in the deterministic cross-document "
                    f"chain across {len(set(derivation['sources']))} sources."
                ),
            }

        abductive = _abductive_alignment_check(
            session,
            answer,
            cited,
            paths,
        )
        if abductive is not None:
            return abductive
        return {
            "name": "evidence_alignment",
            "passed": False,
            "reason": (
                "The cited documents do not yield a deterministic evidence-constrained "
                "derivation anchored to the question. Cortheon will not substitute "
                "lexical overlap for cross-document reasoning."
            ),
        }
    if operation != "sum":
        return {
            "name": "evidence_alignment",
            "passed": True,
            "reason": (
                f"The answer cites separate live host reads for all {len(paths)} requested files."
            ),
        }

    values: dict[str, int] = {}
    for symbol in symbols:
        matcher = re.compile(rf"\b{re.escape(symbol)}\b(?:\s*:[^=\n]+)?\s*=\s*({_INTEGER_TOKEN})")
        matches: set[int] = set()
        for observation in cited:
            for token in matcher.findall(observation.content):
                try:
                    matches.add(_parse_integer(token))
                except ValueError:
                    continue
        if len(matches) == 1:
            values[symbol] = next(iter(matches))
        elif not matches:
            return {
                "name": "evidence_alignment",
                "passed": False,
                "reason": f"The live reads do not expose a numeric value for {symbol}.",
            }
        else:
            return {
                "name": "evidence_alignment",
                "passed": False,
                "reason": f"The live reads expose conflicting values for {symbol}.",
            }

    if not values:
        return {
            "name": "evidence_alignment",
            "passed": False,
            "reason": "The requested sum has no evidence-bound numeric operands.",
        }
    expected = sum(values.values())
    asserted = _answer_integer_assertions(answer)
    if not asserted:
        return {
            "name": "evidence_alignment",
            "passed": False,
            "reason": "The answer does not state a deterministic integer result for the sum.",
        }
    if asserted != {expected}:
        stated = ", ".join(str(item) for item in sorted(asserted))
        return {
            "name": "evidence_alignment",
            "passed": False,
            "reason": (
                f"The answer states result value(s) {stated}, but the evidence-bound "
                f"sum is {expected}."
            ),
        }
    return {
        "name": "evidence_alignment",
        "passed": True,
        "reason": (
            f"All requested files are covered and the stated sum agrees with "
            f"{len(values)} evidence-bound operands."
        ),
    }

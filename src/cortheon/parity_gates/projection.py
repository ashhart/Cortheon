"""The public task surface and the evaluator-committed execution schedule.

Two hashes anchor a blind run. ``public_task_hash`` covers exactly the task
material a contender runner may possess -- prompts and documents, never
labels or expected verdicts -- so a runner that saw more can be detected.
``evaluation_schedule_hash`` covers the exact order of case, contender, and
repetition cells, so the executed order can be replayed and compared against
what was pre-registered.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from cortheon.parity_gates.errors import ParityContractError


def public_case_projection(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the exact task material a contender runner may possess."""

    return [
        {
            "id": str(case["id"]),
            "category": str(case.get("category") or "custom"),
            "domain": str(case.get("domain") or case.get("category") or "custom"),
            "difficulty": str(case.get("difficulty") or "medium"),
            "prompt": str(case["prompt"]),
            "documents": [
                {
                    "uri": str(document.get("uri") or ""),
                    "title": str(document.get("title") or ""),
                    "source_type": str(document.get("source_type") or "benchmark_document"),
                    "text": str(document.get("text") or ""),
                }
                for document in case.get("documents") or []
                if isinstance(document, dict)
            ],
        }
        for case in cases
    ]


def public_task_hash(cases: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        sorted(public_case_projection(cases), key=lambda item: item["id"]),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def evaluation_schedule(
    case_ids: list[str],
    contender_names: list[str],
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build the exact evaluator-committed execution order."""

    normalized_cases = sorted({str(value) for value in case_ids})
    normalized_contenders = sorted({str(value) for value in contender_names})
    if (
        not normalized_cases
        or not normalized_contenders
        or len(normalized_cases) != len(case_ids)
        or len(normalized_contenders) != len(contender_names)
        or repetitions < 1
    ):
        raise ParityContractError(
            "evaluation schedule needs unique cases, contenders, and repetitions"
        )
    aliases = {name: f"candidate_{index + 1}" for index, name in enumerate(normalized_contenders)}
    cells = [
        {
            "repetition": repetition,
            "case_id": case_id,
            "candidate": aliases[name],
            "contender_name": name,
        }
        for repetition in range(1, repetitions + 1)
        for case_id in normalized_cases
        for name in normalized_contenders
    ]
    random.Random(seed).shuffle(cells)
    return [{"run": index, **cell} for index, cell in enumerate(cells, start=1)]


def evaluation_schedule_hash(
    case_ids: list[str],
    contender_names: list[str],
    repetitions: int,
    seed: int,
) -> str:
    """Hash the exact case/contender/repetition execution schedule."""

    canonical = json.dumps(
        evaluation_schedule(case_ids, contender_names, repetitions, seed),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()

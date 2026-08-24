"""Real Pi behavioral tests for the bounded causal-synthesis completion path.

The causal path must add reasoning (draft plus at most two internal
deliberation calls), validate the synthesis deterministically, and submit it
through /v1/complete. The old /v1/evidence-close bypass is gone: an invalid
synthesis or a withheld completion ends boundedly with a Cortheon-withheld
answer and one abandoned session — never a false certification and never
another continuation or tool loop.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from pi_causal_helpers import (
    ACCEPTED_EVIDENCE_IDS,
    CAUSAL_PROMPT,
    EXPECTED_GOOD_SYNTHESIS,
    EXPECTED_UNCERTAIN_SYNTHESIS,
    GOOD_REPAIR,
    GOOD_UNCERTAIN_REPAIR,
    INJECTION_REPAIR,
    RESTATEMENT_REPAIR,
    WEAK_DRAFT,
    causal_runtime_script,
    causal_workspace,
    runtime_calls,
)
from pi_recovery_helpers import (
    Servers,
    assistant_answers,
    continuation_requests,
    require_pi,
    run_pi,
)

EXTENSION = Path(__file__).parents[1] / "src" / "cortheon" / "pi_extension.ts"
WITHHELD_MARKER = "[Cortheon withheld:"


def _run(
    tmp_path: Path,
    turns: list[dict[str, Any]],
    *,
    completes: bool,
    neutral_rival: bool = False,
    ungrounded: bool = False,
):
    model_state: dict[str, Any] = {"requests": [], "turns": turns}
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": causal_runtime_script(
            completes, neutral_rival=neutral_rival, ungrounded=ungrounded
        ),
    }
    with Servers(model_state, runtime_state) as servers:
        started = time.monotonic()
        completed = run_pi(
            EXTENSION,
            CAUSAL_PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=causal_workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=30,
        )
        elapsed = time.monotonic() - started
    return completed, model_state, runtime_state, elapsed


def test_critic_repair_certifies_through_complete(tmp_path: Path) -> None:
    """Weak draft, first repair restates the Cause, critic repair succeeds:
    exactly three model requests, one /v1/complete, zero /v1/evidence-close,
    a completed runtime session, distinct hypothesis payloads, accepted ids
    bound, the exact evidence ledger retained in the certified answer, and no
    continuation or tool loop afterwards."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, model_state, runtime_state, elapsed = _run(
        tmp_path,
        [{"text": WEAK_DRAFT}, {"text": RESTATEMENT_REPAIR}, {"text": GOOD_REPAIR}],
        completes=True,
    )

    assert completed.returncode == 0, completed.stderr
    # The original draft plus exactly two bounded deliberation calls.
    assert len(model_state["requests"]) == 3, len(model_state["requests"])
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 1, paths
    assert paths.count("/v1/evidence-close") == 0, paths
    assert paths.count("/v1/abandon") == 0, paths
    assert continuation_requests(model_state) == 0

    submission = runtime_calls(runtime_state, "/v1/complete")[0]
    hypotheses = submission["hypotheses"]
    assert [item["status"] for item in hypotheses] == ["supported", "refuted"]
    assert hypotheses[0]["statement"] != hypotheses[1]["statement"]
    for item in hypotheses:
        assert item["falsification_test"] == submission["answer"].split("\nTest: ")[1]
    assert submission["completion_evidence_ids"] == ACCEPTED_EVIDENCE_IDS
    assert submission["claims"][0]["evidence_ids"] == ACCEPTED_EVIDENCE_IDS

    answers = assistant_answers(completed)
    assert answers and answers[-1] == EXPECTED_GOOD_SYNTHESIS
    assert elapsed < 20, elapsed


def test_both_repairs_invalid_ends_withheld(tmp_path: Path) -> None:
    """Both deliberation attempts fail validation: exactly three model
    requests, zero /v1/complete, zero /v1/evidence-close, one abandon, a
    withheld answer, and no continuation or tool loop."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, model_state, runtime_state, elapsed = _run(
        tmp_path,
        [{"text": WEAK_DRAFT}, {"text": RESTATEMENT_REPAIR}, {"text": RESTATEMENT_REPAIR}],
        completes=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert len(model_state["requests"]) == 3, len(model_state["requests"])
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 0, paths
    assert paths.count("/v1/evidence-close") == 0, paths
    assert paths.count("/v1/abandon") == 1, paths
    assert continuation_requests(model_state) == 0
    answers = assistant_answers(completed)
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert elapsed < 20, elapsed


def test_open_rival_completes_honestly_as_uncertain(tmp_path: Path) -> None:
    """No clean record observes the clash persisting while archiving is
    disabled, so the honest repair keeps the rival visibly uncertain: the
    accepted evidence does not settle archiving and the question remains
    open with a future discriminating test. The adapter submits the Rival as
    uncertain with neutral bearing evidence, still calls /v1/complete, and
    the certified answer keeps the uncertainty explicit."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, _model_state, runtime_state, elapsed = _run(
        tmp_path,
        [{"text": WEAK_DRAFT}, {"text": GOOD_UNCERTAIN_REPAIR}],
        completes=True,
        neutral_rival=True,
    )

    assert completed.returncode == 0, completed.stderr
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 1, paths
    submission = runtime_calls(runtime_state, "/v1/complete")[0]
    hypotheses = submission["hypotheses"]
    assert [item["status"] for item in hypotheses] == ["supported", "uncertain"]
    assert hypotheses[1]["evidence_ids"]
    assert submission["completion_evidence_ids"] == ACCEPTED_EVIDENCE_IDS

    answers = assistant_answers(completed)
    assert answers and answers[-1] == EXPECTED_UNCERTAIN_SYNTHESIS
    assert "remains uncertain and unresolved" in answers[-1]
    assert "persists when archiving is disabled" not in answers[-1]
    assert elapsed < 20, elapsed


def test_ungrounded_cause_is_submitted_and_runtime_withholds(tmp_path: Path) -> None:
    """Clean runtime ids exist but no record shares even two anchors with the
    synthesis: strong grounding fails validation, so the adapter itself
    withholds — zero /v1/complete calls, one abandon, a withheld answer —
    rather than submitting a Cause it cannot ground. (The empty-binding
    submission path, where validation passes but no record bears on the
    Cause, is proven against the real runtime in
    test_cognitive_empty_binding.py.)"""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, model_state, runtime_state, elapsed = _run(
        tmp_path,
        [{"text": WEAK_DRAFT}, {"text": GOOD_REPAIR}],
        completes=True,
        ungrounded=True,
    )

    assert completed.returncode == 0, completed.stderr
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 0, paths
    assert paths.count("/v1/abandon") == 1, paths
    assert continuation_requests(model_state) == 0
    answers = assistant_answers(completed)
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert elapsed < 20, elapsed


def test_runtime_rejection_ends_withheld(tmp_path: Path) -> None:
    """A valid synthesis the runtime refuses to certify: bounded exit, zero
    /v1/evidence-close, a withheld answer, and exactly one abandon."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, model_state, runtime_state, elapsed = _run(
        tmp_path,
        [{"text": WEAK_DRAFT}, {"text": GOOD_REPAIR}],
        completes=False,
    )

    assert completed.returncode == 0, completed.stderr
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 1, paths
    assert paths.count("/v1/evidence-close") == 0, paths
    assert paths.count("/v1/abandon") == 1, paths
    assert continuation_requests(model_state) == 0
    answers = assistant_answers(completed)
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert elapsed < 20, elapsed


def test_prompt_injection_repair_can_never_become_the_answer(tmp_path: Path) -> None:
    """Prompt-injection-shaped repair output is rejected by deterministic
    validation on both passes and can never become the final answer."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, model_state, runtime_state, elapsed = _run(
        tmp_path,
        [{"text": WEAK_DRAFT}, {"text": INJECTION_REPAIR}, {"text": INJECTION_REPAIR}],
        completes=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert len(model_state["requests"]) == 3, len(model_state["requests"])
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 0, paths
    assert paths.count("/v1/evidence-close") == 0, paths
    assert paths.count("/v1/abandon") == 1, paths
    answers = assistant_answers(completed)
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert '"name"' not in answers[-1] and "arguments" not in answers[-1]
    assert elapsed < 20, elapsed

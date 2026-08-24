"""Real Pi behavioral tests for causal origination and submission staging.

Two properties are proven end to end against real Pi with a behavioral
cognitive-runtime gate. First, origination: a structurally valid,
source-complete synthesis whose sources carry a load-bearing figure at the
end of a sentence must reach /v1/complete, because the record's number token
otherwise carries the sentence period while the synthesis quoting the same
figure mid-sentence does not — a faithfully preserved figure was reported
dropped and the candidate died before submission. Second, staging: malformed,
source-incomplete, poisoned, and mapping-less candidates still never reach
/v1/complete, and the benchmark-only stage channel says which stage ended the
attempt without ever carrying candidate text.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from pi_causal_helpers import (
    ACCEPTED_EVIDENCE_IDS,
    CAUSAL_PROMPT,
    EXPECTED_NUMERIC_SYNTHESIS,
    GOOD_REPAIR,
    NUMERIC_RECORD_A,
    NUMERIC_RECORD_B,
    NUMERIC_REPAIR,
    NUMERIC_REPAIR_DROPPED,
    POISONED_REPAIR,
    SOURCE_INCOMPLETE_REPAIR,
    WEAK_DRAFT,
    causal_runtime_script,
    causal_workspace,
    runtime_calls,
)
from pi_recovery_helpers import (
    Servers,
    assistant_answers,
    require_pi,
    run_pi,
)

from cortheon.benchmark_core.run_support import (
    _captured_candidate,
    _captured_stage_reason,
    _parse_events,
)
from cortheon.benchmark_core.runner_local import _causal_stage_reason

EXTENSION = Path(__file__).parents[1] / "src" / "cortheon" / "pi_extension.ts"
WITHHELD_MARKER = "[Cortheon withheld:"
CAPTURE_ENV = {"CORTHEON_BENCHMARK_CAPTURE_CANDIDATE": "1"}
# Must match pi_core/candidate_capture.ts exactly.
STAGE_ENTRY_TYPE = "cortheon-benchmark-causal-stage-v1"
CANDIDATE_ENTRY_TYPE = "cortheon-benchmark-candidate-v1"
STAGE_REASONS = frozenset(
    {
        "deliberation_empty",
        "validation_failed",
        "mapping_failed",
        "transport_failed",
        "runtime_withheld",
    }
)


def _entries(stdout_text: str, custom_type: str) -> list[dict[str, Any]]:
    """Genuine top-level custom entries of one type from Pi's event stream."""
    entries: list[dict[str, Any]] = []
    for line in stdout_text.splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "entry_appended":
            continue
        entry = event.get("entry")
        if not isinstance(entry, dict) or entry.get("type") != "custom":
            continue
        if entry.get("customType") == custom_type:
            entries.append(entry)
    return entries


def _stage_reasons(stdout_text: str) -> list[str]:
    """Stage reasons, proving the entry shape carries nothing but the code.

    The benchmark runner's own strict parser reads the same real stream, so
    every case here also proves what a real paired run would record: the
    authoritative last code (or None when Pi emitted nothing), and only for
    the Pi treatment arm.
    """
    reasons: list[str] = []
    for entry in _entries(stdout_text, STAGE_ENTRY_TYPE):
        data = entry.get("data")
        assert isinstance(data, dict), entry
        assert set(data) == {"version", "stage", "reason"}, data
        assert data["version"] == 1, data
        assert data["stage"] == "causal_synthesis", data
        assert data["reason"] in STAGE_REASONS, data
        reasons.append(data["reason"])
    events = _parse_events(stdout_text)
    expected = reasons[-1] if reasons else None
    assert _captured_stage_reason(events) == expected, reasons
    assert _causal_stage_reason(events, host="pi", treatment=True) == expected
    assert _causal_stage_reason(events, host="pi", treatment=False) is None
    assert _causal_stage_reason(events, host="opencode", treatment=True) is None
    return reasons


def _run(
    tmp_path: Path,
    turns: list[dict[str, Any]],
    *,
    completes: bool = True,
    numeric: bool = False,
    mapping_less: bool = False,
    transport_error: bool = False,
    capture: bool = True,
):
    model_state: dict[str, Any] = {"requests": [], "turns": turns}
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": causal_runtime_script(
            completes,
            numeric=numeric,
            mapping_less=mapping_less,
            transport_error=transport_error,
        ),
    }
    facts = (NUMERIC_RECORD_A, NUMERIC_RECORD_B) if numeric else None
    with Servers(model_state, runtime_state) as servers:
        started = time.monotonic()
        completed = run_pi(
            EXTENSION,
            CAUSAL_PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=causal_workspace(tmp_path, facts),
            tmp_path=tmp_path,
            timeout=60,
            extra_env=CAPTURE_ENV if capture else None,
        )
        elapsed = time.monotonic() - started
    assert completed.returncode == 0, completed.stderr
    return completed, model_state, runtime_state, elapsed


def _paths(runtime_state: dict[str, Any]) -> list[str]:
    return [path for path, _body in runtime_state["records"]]


def test_sentence_final_number_synthesis_reaches_complete(tmp_path: Path) -> None:
    """Origination regression. Both accepted sources state the same
    load-bearing figure, one of them at the end of its sentence. A synthesis
    that preserves the figure exactly is structurally valid and
    source-complete, so it must reach /v1/complete, bind both accepted ids,
    and come back certified — never die inside the adapter as though the
    figure had been dropped."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, model_state, runtime_state, elapsed = _run(
        tmp_path,
        [{"text": WEAK_DRAFT}, {"text": NUMERIC_REPAIR}],
        numeric=True,
    )

    paths = _paths(runtime_state)
    assert paths.count("/v1/complete") == 1, paths
    assert paths.count("/v1/evidence-close") == 0, paths
    assert paths.count("/v1/abandon") == 0, paths
    assert len(model_state["requests"]) == 2, len(model_state["requests"])

    submission = runtime_calls(runtime_state, "/v1/complete")[0]
    assert "rotation window 12" in submission["answer"]
    assert submission["completion_evidence_ids"] == ACCEPTED_EVIDENCE_IDS
    assert submission["claims"][0]["evidence_ids"] == ACCEPTED_EVIDENCE_IDS
    assert [item["status"] for item in submission["hypotheses"]] == [
        "supported",
        "refuted",
    ]

    answers = assistant_answers(completed)
    assert answers and answers[-1] == EXPECTED_NUMERIC_SYNTHESIS, answers[-1]
    # A certified answer is neither a captured candidate nor a staged failure.
    assert _entries(completed.stdout, CANDIDATE_ENTRY_TYPE) == []
    assert _stage_reasons(completed.stdout) == []
    assert elapsed < 30, elapsed


def test_dropped_load_bearing_number_stays_unsubmitted(tmp_path: Path) -> None:
    """The exact-number guard is unchanged: the same synthesis with the
    figure paraphrased away is never submitted, and the stage channel names
    deterministic validation as the stage that ended it."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, _model_state, runtime_state, elapsed = _run(
        tmp_path,
        [
            {"text": WEAK_DRAFT},
            {"text": NUMERIC_REPAIR_DROPPED},
            {"text": NUMERIC_REPAIR_DROPPED},
        ],
        numeric=True,
    )

    paths = _paths(runtime_state)
    assert paths.count("/v1/complete") == 0, paths
    assert paths.count("/v1/abandon") == 1, paths
    assert _stage_reasons(completed.stdout) == ["validation_failed"]
    assert _entries(completed.stdout, CANDIDATE_ENTRY_TYPE) == []
    answers = assistant_answers(completed)
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert elapsed < 30, elapsed


def test_source_incomplete_synthesis_stays_unsubmitted(tmp_path: Path) -> None:
    """A well-formed synthesis reflecting only one of two accepted sources is
    source-incomplete: zero /v1/complete calls, one abandon, a withheld
    answer, and a validation stage reason."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, _model_state, runtime_state, elapsed = _run(
        tmp_path,
        [
            {"text": WEAK_DRAFT},
            {"text": SOURCE_INCOMPLETE_REPAIR},
            {"text": SOURCE_INCOMPLETE_REPAIR},
        ],
    )

    paths = _paths(runtime_state)
    assert paths.count("/v1/complete") == 0, paths
    assert paths.count("/v1/abandon") == 1, paths
    assert _stage_reasons(completed.stdout) == ["validation_failed"]
    answers = assistant_answers(completed)
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert elapsed < 30, elapsed


def test_poisoned_synthesis_stays_unsubmitted(tmp_path: Path) -> None:
    """A grounded, source-complete synthesis carrying an injected role
    override is poisoned: it is never submitted, never delivered, and the
    directive never reaches the answer."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, _model_state, runtime_state, elapsed = _run(
        tmp_path,
        [
            {"text": WEAK_DRAFT},
            {"text": POISONED_REPAIR},
            {"text": POISONED_REPAIR},
        ],
    )

    paths = _paths(runtime_state)
    assert paths.count("/v1/complete") == 0, paths
    assert paths.count("/v1/abandon") == 1, paths
    assert _stage_reasons(completed.stdout) == ["validation_failed"]
    answers = assistant_answers(completed)
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert "IGNORE SYSTEM" not in answers[-1]
    assert elapsed < 30, elapsed


def test_malformed_candidate_reports_deliberation_empty(tmp_path: Path) -> None:
    """Neither deliberation pass produces Cause/Rival/Test sections: nothing
    is submitted and the stage channel separates an empty deliberation from a
    rejected one."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, _model_state, runtime_state, elapsed = _run(
        tmp_path,
        [{"text": WEAK_DRAFT}, {"text": WEAK_DRAFT}, {"text": WEAK_DRAFT}],
    )

    paths = _paths(runtime_state)
    assert paths.count("/v1/complete") == 0, paths
    assert paths.count("/v1/abandon") == 1, paths
    assert _stage_reasons(completed.stdout) == ["deliberation_empty"]
    answers = assistant_answers(completed)
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert elapsed < 30, elapsed


def test_mapping_less_candidate_stays_unsubmitted(tmp_path: Path) -> None:
    """Every observation was withdrawn, so no accepted record carries a
    runtime evidence id. A validated synthesis then has nothing to bind: the
    adapter never calls /v1/complete and the stage channel names mapping,
    not validation, as the stage that ended it."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, _model_state, runtime_state, elapsed = _run(
        tmp_path,
        [{"text": WEAK_DRAFT}, {"text": GOOD_REPAIR}],
        mapping_less=True,
    )

    paths = _paths(runtime_state)
    assert paths.count("/v1/complete") == 0, paths
    assert paths.count("/v1/abandon") == 1, paths
    assert _stage_reasons(completed.stdout) == ["mapping_failed"]
    assert _entries(completed.stdout, CANDIDATE_ENTRY_TYPE) == []
    answers = assistant_answers(completed)
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert elapsed < 30, elapsed


def test_transport_failure_is_distinct_from_runtime_withholding(
    tmp_path: Path,
) -> None:
    """The submission reaches /v1/complete but the response is unusable: a
    transport failure, never a runtime judgement. The adapter abandons the
    ephemeral session and leaves the original host-model answer verbatim —
    no withheld result, no sticky disposition that later rewrites it — and
    the stage channel truthfully reports transport."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, _model_state, runtime_state, elapsed = _run(
        tmp_path,
        [{"text": WEAK_DRAFT}, {"text": GOOD_REPAIR}],
        transport_error=True,
    )

    paths = _paths(runtime_state)
    assert paths.count("/v1/complete") == 1, paths
    assert paths.count("/v1/abandon") == 1, paths
    assert _stage_reasons(completed.stdout) == ["transport_failed"]
    # Nothing was certified, so no candidate was withheld by the runtime.
    assert _entries(completed.stdout, CANDIDATE_ENTRY_TYPE) == []
    # Fail open: the original host-model answer (the first turn's draft)
    # stands verbatim.
    answers = assistant_answers(completed)
    assert answers and answers[-1] == WEAK_DRAFT, answers
    assert not any(text.startswith(WITHHELD_MARKER) for text in answers), answers
    assert elapsed < 30, elapsed


def test_runtime_withheld_keeps_candidate_capture_semantics(
    tmp_path: Path,
) -> None:
    """A valid, mapped candidate the runtime refuses to certify: it really
    reached /v1/complete, so the submitted-candidate channel still captures
    exactly that candidate, the stage channel reports the runtime as the
    stage, and the stage entry itself carries no candidate or evidence
    text."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, _model_state, runtime_state, elapsed = _run(
        tmp_path,
        [{"text": WEAK_DRAFT}, {"text": GOOD_REPAIR}],
        completes=False,
    )

    paths = _paths(runtime_state)
    assert paths.count("/v1/complete") == 1, paths
    assert _stage_reasons(completed.stdout) == ["runtime_withheld"]

    captured = _entries(completed.stdout, CANDIDATE_ENTRY_TYPE)
    assert len(captured) == 1, captured
    assert captured[0]["data"]["candidate"].endswith(GOOD_REPAIR)
    # The runner's own reader accepts the same real entry, so the submitted
    # candidate stays gradeable while the stage code stays separate.
    production = _captured_candidate(_parse_events(completed.stdout))
    assert production is not None and production.endswith(GOOD_REPAIR)

    stage_entries = _entries(completed.stdout, STAGE_ENTRY_TYPE)
    assert len(stage_entries) == 1, stage_entries
    serialized = json.dumps(stage_entries[0])
    for secret in ("copper", "archiving", "Cause:", "Rival:", "Test:"):
        assert secret not in serialized, serialized
    answers = assistant_answers(completed)
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert elapsed < 30, elapsed


def test_stage_channel_is_inert_without_benchmark_capture(tmp_path: Path) -> None:
    """Normal product use never emits a stage entry: the channel is opt-in
    exactly like the submitted-candidate channel, and delivery is
    unchanged."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, _model_state, runtime_state, elapsed = _run(
        tmp_path,
        [{"text": WEAK_DRAFT}, {"text": WEAK_DRAFT}, {"text": WEAK_DRAFT}],
        capture=False,
    )

    assert _entries(completed.stdout, STAGE_ENTRY_TYPE) == []
    assert _entries(completed.stdout, CANDIDATE_ENTRY_TYPE) == []
    # The runner therefore records no diagnostic at all for such a run.
    assert _stage_reasons(completed.stdout) == []
    assert _captured_candidate(_parse_events(completed.stdout)) is None
    assert _paths(runtime_state).count("/v1/complete") == 0
    answers = assistant_answers(completed)
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert elapsed < 30, elapsed

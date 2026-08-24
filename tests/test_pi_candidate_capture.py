"""Benchmark-only pre-block candidate capture over Pi's custom entry channel.

The adapter emits the exact withheld candidate through
``pi.appendEntry(cortheon-benchmark-candidate-v1, ...)`` only when the
benchmark opts in; the runner parses genuine ``entry_appended`` events, grades
the captured candidate with the hidden grader only when the terminal answer
was blocked, and never persists the candidate text anywhere.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from pi_causal_helpers import (
    EXPECTED_GOOD_SYNTHESIS,
    GOOD_REPAIR,
    RESTATEMENT_REPAIR,
    WEAK_DRAFT,
    causal_runtime_script,
    causal_workspace,
)
from pi_doom_loop_helpers import TOOL_TURN
from pi_recovery_helpers import (
    Servers,
    assistant_answers,
    require_pi,
    run_pi,
)

from cortheon.benchmark_core.run_support import CANDIDATE_ENTRY_TYPE

EXTENSION = Path(__file__).parents[1] / "src" / "cortheon" / "pi_extension.ts"
CAPTURE_ENV = {"CORTHEON_BENCHMARK_CAPTURE_CANDIDATE": "1"}
WITHHELD_MARKER = "[Cortheon withheld:"
CAUSAL_PROMPT = (
    "Diagnose the causal explanation for the clash between the two ledgers, "
    "disprove the rival hypothesis, and give a discriminating test."
)
PLAIN_PROMPT = "Summarize the relationship between the two fact files."
PLAIN_ANSWER = "Both ledgers reuse the shard key copper."
REVISED_ANSWER = PLAIN_ANSWER + " Both files also agree on the rotation window."
CERTIFIED_ANSWER = "CORTHEON CERTIFIED: the summary is complete and grounded."


def _candidate_entry(candidate: str, *, stage: str = "completion") -> dict[str, Any]:
    return {
        "type": "entry_appended",
        "entry": {
            "type": "custom",
            "customType": CANDIDATE_ENTRY_TYPE,
            "id": "entry-1",
            "timestamp": "2026-08-22T00:00:00.000Z",
            "data": {"version": 1, "stage": stage, "candidate": candidate},
        },
    }


def _candidate_events(stdout_text: str) -> list[dict[str, Any]]:
    return [
        event
        for event in (json.loads(line) for line in stdout_text.splitlines() if line.startswith("{"))
        if isinstance(event, dict)
        and event.get("type") == "entry_appended"
        and isinstance(event.get("entry"), dict)
        and event["entry"].get("customType") == CANDIDATE_ENTRY_TYPE
    ]


def _run_causal(
    tmp_path: Path,
    turns: list[dict[str, Any]],
    *,
    completes: bool,
    capture: bool,
):
    model_state: dict[str, Any] = {"requests": [], "turns": turns}
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": causal_runtime_script(completes),
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
            timeout=60,
            extra_env=CAPTURE_ENV if capture else None,
        )
        elapsed = time.monotonic() - started
    assert completed.returncode == 0, completed.stderr
    return completed, model_state, runtime_state, elapsed


def _plain_script(completes: bool):
    def script(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "plain-1",
                    "status": "observing",
                    "session": {"deliverable": "document_synthesis"},
                    "context": {"goal": body.get("goal")},
                    "next_action": {"type": "finish"},
                },
            )
        if path == "/v1/observe":
            return (
                200,
                {
                    "session_id": "plain-1",
                    "status": "observing",
                    "accepted_evidence_ids": [],
                    "next_action": {"type": "finish"},
                },
            )
        if path == "/v1/complete":
            if completes:
                return (
                    200,
                    {"session_id": "plain-1", "status": "complete", "answer": CERTIFIED_ANSWER},
                )
            return (
                200,
                {
                    "session_id": "plain-1",
                    "status": "needs_evidence",
                    "verification": {"gaps": ["more evidence required"]},
                    "next_action": {"type": "finish"},
                },
            )
        return 200, {"status": "ok"}

    return script


def _run_plain(
    tmp_path: Path,
    *,
    completes: bool,
    capture: bool,
    turns: list[dict[str, Any]] | None = None,
    script: Any = None,
):
    workspace = causal_workspace(tmp_path)
    model_state: dict[str, Any] = {
        "requests": [],
        "turns": turns if turns is not None else [{"text": PLAIN_ANSWER}],
    }
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": script if script is not None else _plain_script(completes),
    }
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            EXTENSION,
            PLAIN_PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace,
            tmp_path=tmp_path,
            timeout=60,
            extra_env=CAPTURE_ENV if capture else None,
        )
    assert completed.returncode == 0, completed.stderr
    return completed, model_state, runtime_state


# --- Live Pi behavior (skipped when pi is not installed) ---


def test_causal_withheld_candidate_is_captured_only_when_opted_in(
    tmp_path: Path,
) -> None:
    if not require_pi():
        pytest.skip("Pi is not installed")
    turns = [{"text": WEAK_DRAFT}, {"text": RESTATEMENT_REPAIR}, {"text": GOOD_REPAIR}]
    completed, _model, runtime, _elapsed = _run_causal(
        tmp_path, turns, completes=False, capture=True
    )
    answers = assistant_answers(completed)
    assert answers[-1].startswith(WITHHELD_MARKER)
    entries = _candidate_events(completed.stdout)
    assert len(entries) == 1, completed.stdout
    entry = entries[0]["entry"]
    assert entry["type"] == "custom"
    assert entry["data"]["version"] == 1
    assert entry["data"]["stage"] == "causal_synthesis"
    # The exact validated deliberated text, never the earlier weak draft.
    assert entry["data"]["candidate"] == EXPECTED_GOOD_SYNTHESIS
    submitted = [body for path, body in runtime["records"] if path == "/v1/complete"]
    assert submitted and submitted[-1]["answer"] == entry["data"]["candidate"]

    completed_off, _model2, _runtime2, _elapsed2 = _run_causal(
        tmp_path / "off", turns, completes=False, capture=False
    )
    assert assistant_answers(completed_off)[-1].startswith(WITHHELD_MARKER)
    assert _candidate_events(completed_off.stdout) == []


def test_causal_certified_run_emits_no_candidate(tmp_path: Path) -> None:
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, _model, _runtime, _elapsed = _run_causal(
        tmp_path,
        [{"text": WEAK_DRAFT}, {"text": RESTATEMENT_REPAIR}, {"text": GOOD_REPAIR}],
        completes=True,
        capture=True,
    )
    answers = assistant_answers(completed)
    assert not answers[-1].startswith(WITHHELD_MARKER)
    assert _candidate_events(completed.stdout) == []


def test_causal_without_validated_candidate_emits_nothing(tmp_path: Path) -> None:
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, _model, runtime, _elapsed = _run_causal(
        tmp_path,
        [{"text": WEAK_DRAFT}, {"text": WEAK_DRAFT}, {"text": WEAK_DRAFT}],
        completes=False,
        capture=True,
    )
    assert assistant_answers(completed)[-1].startswith(WITHHELD_MARKER)
    assert _candidate_events(completed.stdout) == []
    assert not [body for path, body in runtime["records"] if path == "/v1/complete"]


def _fail_open_second_script():
    calls = {"complete": 0}
    withholding = _plain_script(False)

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/complete":
            calls["complete"] += 1
            if calls["complete"] >= 2:
                return "connection-reset"
        return withholding(path, body)

    return script


# Terminal-scoped, memory-only candidate capture on the regular path: a
# provisional withhold only RETAINS (or replaces) the pending candidate; the
# entry is appended exactly once, with the latest retained text, when the
# path actually becomes terminal. A tool-only continuation emits the
# retained FIRST candidate; a later transport fail-open clears the pending
# candidate and emits NONE while the host answer stands verbatim.
@pytest.mark.parametrize(
    ("label", "turns", "script", "expected_candidate", "expected_completes"),
    [
        ("same-candidate-twice", [{"text": PLAIN_ANSWER}], None, PLAIN_ANSWER, 2),
        (
            "changed-second-candidate",
            [{"text": PLAIN_ANSWER}, {"text": REVISED_ANSWER}],
            None,
            REVISED_ANSWER,
            2,
        ),
        ("tool-only-continuation", [{"text": PLAIN_ANSWER}, TOOL_TURN], None, PLAIN_ANSWER, 1),
        ("transport-fail-open", [{"text": PLAIN_ANSWER}], _fail_open_second_script, None, 2),
    ],
)
def test_terminal_scoped_candidate_capture(
    tmp_path: Path,
    label: str,
    turns: list[dict[str, Any]],
    script: Any,
    expected_candidate: str | None,
    expected_completes: int,
) -> None:
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, _model, runtime = _run_plain(
        tmp_path / label,
        completes=False,
        capture=True,
        turns=turns,
        script=script() if script is not None else None,
    )
    entries = _candidate_events(completed.stdout)
    submitted = [body for path, body in runtime["records"] if path == "/v1/complete"]
    answers = assistant_answers(completed)
    assert len(submitted) == expected_completes, submitted
    if expected_candidate is None:
        # Fail open: the last visible answer is verbatim and the only
        # withheld text is the earlier provisional withhold.
        assert answers[-1] == PLAIN_ANSWER, answers
        assert sum(1 for t in answers if t.startswith(WITHHELD_MARKER)) == 1
        assert entries == []
        return
    assert answers[-1].startswith(WITHHELD_MARKER), answers
    assert len(entries) == 1, completed.stdout
    data = entries[0]["entry"]["data"]
    assert data["stage"] == "completion"
    assert data["candidate"] == expected_candidate
    assert submitted[-1]["answer"] == expected_candidate


def test_regular_path_certified_run_emits_nothing(tmp_path: Path) -> None:
    if not require_pi():
        pytest.skip("Pi is not installed")
    completed, _model, _runtime = _run_plain(tmp_path, completes=True, capture=True)
    answers = assistant_answers(completed)
    assert answers[-1] == CERTIFIED_ANSWER
    assert _candidate_events(completed.stdout) == []

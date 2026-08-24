"""Real Pi regressions for the repeated-withhold terminal state.

The round-24 live signature: an ambiguity-resolution treatment had its
completion withheld twice, then the model wandered into irrelevant tool
calls (a `.com.openai.codex...` path) with no pending runtime request
until the harness wall clock killed the run — session started, never
completed, never blocked, no terminal answer.

These tests reproduce that lifecycle domain-independently: after the
bounded continuation budget is exhausted, further tool persistence is
blocked, the last answer is replaced with a single sticky withheld result
carrying a truthful terminal disposition, the blocked candidate stays
classifiable, every completion went through /v1/complete (no bypass), and
no active runtime state survives. Removing the guard reopens the
wandering, and the fail-open controls are untouched.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from pi_doom_loop_helpers import TOOL_TURN, workspace
from pi_recovery_helpers import (
    Servers,
    assistant_answers,
    continuation_requests,
    host_executions,
    parse_events,
    require_pi,
    run_pi,
)
from pi_terminal_helpers import (
    AMBIGUITY_ANSWER,
    AMBIGUITY_PROMPT,
    CANDIDATE_ENTRY_TYPE,
    EXTENSION,
    TERMINAL_STATUS_MARKER,
    TERMINAL_STATUS_VERSION,
    WITHHELD_MARKER,
    custom_entry_data,
    mutated_source,
    terminal_status_messages,
    varying_action_withholding_script,
    withhold_then_finish_script,
    withholding_ambiguity_script,
)

from cortheon.benchmark_core.transport_outcomes import parse_transport_outcome

CONTINUATION_PREFIX = "[CORTHEON_CONTINUE]"
REPAIR_CONTINUATION_HEAD = "[CORTHEON_CONTINUE] Completion was withheld."


def _fresh_continuations(model_state: dict[str, Any], head: str = CONTINUATION_PREFIX) -> int:
    """Model requests triggered by a Cortheon follow-up as THIS request's
    prompt (the last message), not old continuation messages still present
    in the conversation history from earlier prompts."""
    count = 0
    for request in model_state["requests"]:
        messages = request.get("messages", [])
        last = messages[-1] if messages else None
        if not isinstance(last, dict):
            continue
        content = last.get("content")
        texts = (
            [content]
            if isinstance(content, str)
            else [block.get("text", "") for block in content or [] if isinstance(block, dict)]
            if isinstance(content, list)
            else []
        )
        if any(text.startswith(head) for text in texts):
            count += 1
    return count


def _run(tmp_path: Path, turns: list[dict[str, Any]], script: Any = None):
    model_state: dict[str, Any] = {"requests": [], "turns": turns}
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": script or withholding_ambiguity_script(),
    }
    with Servers(model_state, runtime_state) as servers:
        started = time.monotonic()
        completed = run_pi(
            EXTENSION,
            AMBIGUITY_PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=60,
            extra_env={"CORTHEON_BENCHMARK_CAPTURE_CANDIDATE": "1"},
        )
    return completed, model_state, runtime_state, time.monotonic() - started


def test_twice_withheld_then_wandering_ends_in_one_sticky_withheld(
    tmp_path: Path,
) -> None:
    """The exact live signature, bounded to ONE follow-up total: the
    candidate is withheld twice through /v1/complete, the second withhold
    spends the single unified continuation budget, a sticky terminal
    disposition is held, the session is abandoned exactly once, and no
    wandering tool execution (or any third model operation) ever happens
    after the second withhold."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    turns = [
        TOOL_TURN,
        {"text": AMBIGUITY_ANSWER},
        {"text": AMBIGUITY_ANSWER},
        TOOL_TURN,
        {"text": AMBIGUITY_ANSWER},
    ]
    completed, model_state, runtime_state, elapsed = _run(tmp_path / "run", turns)
    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    # The uncertified candidate is never delivered raw.
    assert AMBIGUITY_ANSWER not in answers, answers
    assert answers[-1].startswith(WITHHELD_MARKER), answers
    withheld = [text for text in answers if text.startswith(WITHHELD_MARKER)]
    assert answers[-1] == withheld[-1], answers
    # Exactly ONE automatic follow-up total — the repair continuation. The
    # budget is unified: no separate answer-only continuation may follow
    # the cap terminal.
    assert _fresh_continuations(model_state) == 1, model_state["requests"]
    assert _fresh_continuations(model_state, REPAIR_CONTINUATION_HEAD) == 1
    # Intended trace: initial tool turn, initial withheld answer, one repair
    # continuation whose answer is withheld again — then the terminal. The
    # wandering TOOL_TURN and the final text turn are never requested.
    assert len(model_state["requests"]) == 3, len(model_state["requests"])
    events = parse_events(completed.stdout)
    assert len(host_executions(events)) == 2, events
    # The exhausted budget ended with one host-visible Cortheon terminal
    # status and no third model operation.
    statuses = terminal_status_messages(completed)
    assert len(statuses) == 1, statuses
    # Every completion went through the runtime; no bypass, no cert.
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 2, paths
    # No active runtime state survives: one abandon, nothing after it.
    assert paths.count("/v1/abandon") == 1, paths
    assert paths[-1] == "/v1/abandon", paths
    # The blocked candidate stays classifiable on the completion channel.
    candidates = custom_entry_data(completed, CANDIDATE_ENTRY_TYPE)
    assert candidates, "no classifiable candidate captured"
    assert all(item["stage"] == "completion" for item in candidates), candidates
    assert candidates[-1]["candidate"] == AMBIGUITY_ANSWER, candidates
    assert elapsed < 30, elapsed


def test_withhold_then_tool_only_continuation_terminates_visibly(
    tmp_path: Path,
) -> None:
    """A provisional nonterminal withhold must not count as delivery: the
    initial ordinary withhold (visible, uncertified) earns the one repair
    continuation; that continuation executes its pending observation and
    then persists tool-only into the finish boundary. Because the unified
    budget is already spent, that boundary is terminal — not a reason to
    schedule an answer-only follow-up. The run emits exactly one final
    terminal status, abandons once, leaves no active runtime state, and
    every later raw answerable text would be intercepted."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    single_tool_turn = {"tool_calls": [("read", {"path": "facts/a.txt"})]}
    turns = [{"text": AMBIGUITY_ANSWER}] + [single_tool_turn] * 4
    model_state: dict[str, Any] = {"requests": [], "turns": turns}
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": withhold_then_finish_script(),
    }
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            EXTENSION,
            AMBIGUITY_PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=60,
            extra_env={"CORTHEON_BENCHMARK_CAPTURE_CANDIDATE": "1"},
        )
    assert completed.returncode == 0, completed.stderr
    # Intended trace: the initial withheld answer, the one repair
    # continuation (whose read is admitted and observed), and the model turn
    # whose tool-only persistence hits the spent-budget finish boundary and
    # is blocked — no fourth model operation ever runs.
    assert len(model_state["requests"]) == 3, len(model_state["requests"])
    assert _fresh_continuations(model_state) == 1, model_state["requests"]
    assert _fresh_continuations(model_state, REPAIR_CONTINUATION_HEAD) == 1, model_state["requests"]
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 1, paths
    # Exactly one final terminal status, host-visible, fixed shape: only the
    # withheld status and the bounded reason — never candidate or evidence
    # text — with the closed type and version.
    statuses = terminal_status_messages(completed)
    assert len(statuses) == 1, statuses
    message = statuses[0]
    assert message.get("display") is True, message
    content = message.get("content")
    text = (
        content
        if isinstance(content, str)
        else "".join(block.get("text", "") for block in content or [])
    )
    assert text.startswith(WITHHELD_MARKER), text
    assert TERMINAL_STATUS_MARKER in text, text
    assert AMBIGUITY_ANSWER not in text, text
    details = message.get("details", {})
    assert details.get("version") == TERMINAL_STATUS_VERSION, details
    assert details.get("status") == "withheld", details
    assert details.get("reason"), details
    # No raw escape: the uncertified candidate never reached the host raw.
    answers = assistant_answers(completed)
    assert AMBIGUITY_ANSWER not in answers, answers
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    # Abandoned exactly once by the exhausted fallback, and no active
    # runtime state survives it.
    assert paths.count("/v1/abandon") == 1, paths
    assert paths[-1] == "/v1/abandon", paths


def test_repeated_identical_withhold_within_one_investigation_is_terminal(
    tmp_path: Path,
) -> None:
    """The progress key is per investigation: within ONE prompt, a withheld
    completion that repeats the identical request/action of the already
    granted continuation terminates immediately — exactly one repair
    continuation, two completions, one abandon, and a withheld terminal."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    turns = [{"text": AMBIGUITY_ANSWER}] * 3
    completed, model_state, runtime_state, _elapsed = _run(tmp_path / "run", turns)
    assert completed.returncode == 0, completed.stderr
    assert _fresh_continuations(model_state, REPAIR_CONTINUATION_HEAD) == 1, model_state["requests"]
    assert len(model_state["requests"]) == 2, len(model_state["requests"])
    assert _fresh_continuations(model_state) == 1, model_state["requests"]
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 2, paths
    assert paths.count("/v1/abandon") == 1, paths
    assert paths[-1] == "/v1/abandon", paths
    assert len(terminal_status_messages(completed)) == 1
    outcome = parse_transport_outcome(parse_events(completed.stdout), host="pi").outcome
    assert outcome.terminal_status == "withheld"
    assert outcome.terminal_provenance == "pi_custom_terminal"
    answers = assistant_answers(completed)
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert AMBIGUITY_ANSWER not in answers, answers


def test_two_identical_prompts_each_earn_their_own_continuation(
    tmp_path: Path,
) -> None:
    """No state from task A may change task B: two identical independent
    prompts in one Pi session each receive their own single allowed repair
    continuation — the continuation fingerprint is per investigation, so
    task B's identical withhold is never blocked by task A's."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    turns = [{"text": AMBIGUITY_ANSWER}] * 6
    model_state: dict[str, Any] = {"requests": [], "turns": turns}
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": withholding_ambiguity_script(),
    }
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            EXTENSION,
            [AMBIGUITY_PROMPT, AMBIGUITY_PROMPT],
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=60,
        )
    assert completed.returncode == 0, completed.stderr
    # Each prompt ran its own full bounded lifecycle with its own fresh
    # unified budget: withhold -> its own one repair continuation -> withhold
    # -> cap terminal. Two prompts, two follow-ups total, never more.
    assert _fresh_continuations(model_state, REPAIR_CONTINUATION_HEAD) == 2, model_state["requests"]
    assert _fresh_continuations(model_state) == 2, model_state["requests"]
    assert len(model_state["requests"]) == 4, len(model_state["requests"])
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/complete") == 4, paths
    assert paths.count("/v1/abandon") == 2, paths
    assert paths[-1] == "/v1/abandon", paths
    assert len(terminal_status_messages(completed)) == 2
    answers = assistant_answers(completed)
    assert answers and answers[-1].startswith(WITHHELD_MARKER), answers
    assert AMBIGUITY_ANSWER not in answers, answers


def test_mutation_removing_continuation_guard_reopens_wandering(
    tmp_path: Path,
) -> None:
    """Removing the agent_end continuation cap reinstates the live loop: a
    runtime whose every withhold carries genuine new request/action progress
    (so the fingerprint gate cannot stop it) keeps earning continuations and
    the wandering persists through more completions and host tool calls
    until the host tool budget finally ends it."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    facade = mutated_source(
        tmp_path / "mutation",
        {
            "protocol": (
                "export const MAX_AUTOMATIC_CONTINUATIONS = 1;",
                "export const MAX_AUTOMATIC_CONTINUATIONS = 2;",
            )
        },
    )
    turns = [
        {"text": AMBIGUITY_ANSWER},
        {"text": AMBIGUITY_ANSWER},
        {"text": AMBIGUITY_ANSWER},
        {"text": AMBIGUITY_ANSWER},
        TOOL_TURN,
    ]
    model_state: dict[str, Any] = {"requests": [], "turns": turns}
    runtime_state: dict[str, Any] = {
        "records": [],
        "script": varying_action_withholding_script(),
    }
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            facade,
            AMBIGUITY_PROMPT,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path / "run"),
            tmp_path=tmp_path / "run",
            timeout=90,
            extra_env={"CORTHEON_MAX_HOST_TOOL_CALLS": "8"},
        )
    assert completed.returncode == 0, completed.stderr
    paths = [path for path, _body in runtime_state["records"]]
    # The unmutated adapter stops at exactly two completions; without the
    # cap every varying-action withhold earned another continuation.
    assert paths.count("/v1/complete") > 2, paths
    assert continuation_requests(model_state) > 1, model_state["requests"]
    answers = assistant_answers(completed)
    assert answers[-1].startswith(WITHHELD_MARKER), answers


def test_mutation_repeating_fingerprint_earns_another_continuation(
    tmp_path: Path,
) -> None:
    """The fingerprint gate is independently load-bearing WITHIN one
    investigation: with the continuation budget temporarily widened to two
    (test-only mutation; the shipped cap stays 1), the intact fingerprint
    still stops the identical repeated withhold at two completions, and
    disabling the fingerprint comparison lets that identical retry earn
    another continuation — a third completion of the same work. The proof
    never relies on state persisting across tasks."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    turns = [{"text": AMBIGUITY_ANSWER}] * 4

    def _run_facade(label: str, mutations: dict[str, tuple[str, str]]) -> None:
        facade = mutated_source(tmp_path / f"mutation-{label}", mutations)
        model_state: dict[str, Any] = {"requests": [], "turns": turns}
        runtime_state: dict[str, Any] = {
            "records": [],
            "script": withholding_ambiguity_script(),
        }
        with Servers(model_state, runtime_state) as servers:
            completed = run_pi(
                facade,
                AMBIGUITY_PROMPT,
                model_port=servers.model.server_port,
                runtime_port=servers.runtime.server_port,
                workspace=workspace(tmp_path / f"run-{label}"),
                tmp_path=tmp_path / f"run-{label}",
                timeout=60,
            )
        assert completed.returncode == 0, completed.stderr
        paths = [path for path, _body in runtime_state["records"]]
        repairs = _fresh_continuations(model_state, REPAIR_CONTINUATION_HEAD)
        answers = assistant_answers(completed)
        assert answers[-1].startswith(WITHHELD_MARKER), (label, answers)
        recorded[label] = (paths.count("/v1/complete"), repairs)

    recorded: dict[str, tuple[int, int]] = {}
    widen_cap = {
        "protocol": (
            "export const MAX_AUTOMATIC_CONTINUATIONS = 1;",
            "export const MAX_AUTOMATIC_CONTINUATIONS = 2;",
        )
    }
    # Control: cap widened but the fingerprint intact — the identical
    # repeated withhold still cannot earn a second continuation.
    _run_facade("cap-only", widen_cap)
    # Mutation: fingerprint comparison disabled — the identical retry
    # earns another continuation and a third completion of the same work.
    _run_facade(
        "no-fingerprint",
        {
            **widen_cap,
            "session_events": (
                "fingerprint === getContinuationFingerprint()",
                "false && fingerprint === getContinuationFingerprint()",
            ),
        },
    )
    assert recorded["cap-only"][0] == 2, recorded
    assert recorded["no-fingerprint"] == (3, 2), recorded


def test_withheld_wander_fail_open_controls_unchanged(tmp_path: Path) -> None:
    """The terminal path never overrides fail-open: an explicit disable and
    a transport failure at /v1/start deliver the host model's ordinary
    answer verbatim with no withheld result."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    for name, extra in (
        ("disable", {"CORTHEON_AUTO_ENABLE": "0"}),
        ("transport", {}),
    ):
        model_state: dict[str, Any] = {
            "requests": [],
            "turns": [{"text": AMBIGUITY_ANSWER}],
        }
        runtime_state: dict[str, Any] = {"records": []}

        def script(path: str, _body: dict[str, Any], _name: str = name) -> Any:
            if path == "/healthz":
                return 200, {"status": "ok"}
            if path == "/v1/start":
                return "invalid-json" if _name == "transport" else (200, {})
            return 200, {"status": "ok"}

        runtime_state["script"] = script
        with Servers(model_state, runtime_state) as servers:
            completed = run_pi(
                EXTENSION,
                AMBIGUITY_PROMPT,
                model_port=servers.model.server_port,
                runtime_port=servers.runtime.server_port,
                workspace=workspace(tmp_path / f"{name}-run"),
                tmp_path=tmp_path / f"{name}-run",
                timeout=45,
                extra_env=extra,
            )
        assert completed.returncode == 0, (name, completed.stderr)
        answers = assistant_answers(completed)
        assert answers and answers[-1] == AMBIGUITY_ANSWER, (name, answers)

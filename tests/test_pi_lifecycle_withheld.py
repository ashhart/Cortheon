"""Real Pi regressions for the abandoned-treatment terminal disposition.

The live round-23 false allow: a causal novel-synthesis treatment exhausted
its admitted host tool budget, ``markTerminationState`` marked answer-only
AND abandoned the runtime session in one step, the single answer-only
continuation then produced ordinary assistant text, and ``message_end``
short-circuited on the missing session — delivering the uncertified answer
verbatim with no deliberation, no certification, and no stage reason.

These tests reproduce that exact lifecycle against real Pi: an active causal
session, accepted host evidence, a bounded tool block that abandons the
session before the final assistant text. The adapter must answer with
exactly one withheld result naming the real reason, emit the truthful
terminated-before-deliberation stage code, request nothing further, and
leave no runtime state behind. Fail-open controls (explicit disable,
transport failure at start, non-treatment prompt) stay untouched, and a
mutation that removes the guard reopens the false allow.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import pytest
from pi_causal_helpers import CAUSAL_PROMPT
from pi_doom_loop_helpers import TOOL_TURN, workspace
from pi_lifecycle_helpers import (
    CANDIDATE_ENTRY_TYPE,
    CAP,
    EXTENSION,
    MIXED_TEXT_TOOL_TURN,
    ORDINARY_ANSWER,
    PLAIN_ANSWER,
    PLAIN_PROMPT,
    SINGLE_TOOL_TURN,
    SOURCE_DIR,
    WITHHELD_MARKER,
    never_finishing_causal_script,
    run_lifecycle,
)
from pi_recovery_helpers import (
    Servers,
    assistant_answers,
    continuation_requests,
    parse_events,
    require_pi,
    run_pi,
)
from pi_terminal_helpers import terminal_status_messages

from cortheon.benchmark_core.models import ImportCase, RunResult
from cortheon.benchmark_core.runner_local import _candidate_correct
from cortheon.benchmark_core.stats import (
    FALSE_BLOCK,
    SAFE_BLOCK,
    UNCLASSIFIED_BLOCK,
    classify_block,
)
from cortheon.benchmark_core.transport_outcomes import parse_transport_outcome


def test_abandoned_causal_treatment_answers_withheld_not_raw(
    tmp_path: Path,
) -> None:
    """The precise live lifecycle under the unified budget: budget
    exhaustion blocks the tool batch and abandons the session with a held
    disposition, ZERO automatic follow-ups are scheduled (no model turn is
    spent to intercept output), exactly one host-visible terminal names the
    budget reason, no raw candidate turn is ever requested, and the runtime
    state is cleared."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    turns = [TOOL_TURN] * 5 + [{"text": ORDINARY_ANSWER}]
    model_state: dict[str, Any] = {"requests": [], "turns": turns}
    runtime_state: dict[str, Any] = {"records": []}
    runtime_state["script"] = never_finishing_causal_script(runtime_state)
    with Servers(model_state, runtime_state) as servers:
        started = time.monotonic()
        completed = run_lifecycle(EXTENSION, tmp_path / "run", servers)
        elapsed = time.monotonic() - started

    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    # The wrong-shaped ordinary answer is never delivered: its turn is never
    # even requested (no raw candidate).
    assert ORDINARY_ANSWER not in answers, answers
    statuses = terminal_status_messages(completed)
    assert len(statuses) == 1, statuses
    assert "host tool budget was exhausted" in str(statuses[0].get("content"))
    # Zero automatic follow-ups after the budget terminal abandoned the
    # session; intended trace: four admitted tool batches plus the blocked
    # fifth that ends the operation.
    assert continuation_requests(model_state) == 0
    assert len(model_state["requests"]) == 5, len(model_state["requests"])
    paths = [path for path, _body in runtime_state["records"]]
    # Evidence was really accepted before the abandonment.
    observes = [path for path in paths if path == "/v1/observe"]
    assert observes, paths
    # Runtime state cleared: one abandon, nothing after it, no completion.
    assert paths.count("/v1/abandon") == 1, paths
    assert paths[-1] == "/v1/abandon", paths
    assert paths.count("/v1/complete") == 0, paths
    assert elapsed < 30, elapsed


def test_mixed_text_tool_message_is_replaced_but_keeps_tool_call_pairing(
    tmp_path: Path,
) -> None:
    """A mixed assistant message (ordinary text alongside a tool call) at
    the budget boundary: its ungated text passes its own message_end while
    the session is still alive (the tool boundary then blocks its tool and
    the budget terminal abandons the session), the operation ends with one
    host-visible terminal, and no further model operation is scheduled. The
    post-abandon replacement that retains the toolCall pairing is proven
    source-level in test_pi_unified_budget.py."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    turns = [TOOL_TURN] * 4 + [MIXED_TEXT_TOOL_TURN]
    model_state: dict[str, Any] = {"requests": [], "turns": turns}
    runtime_state: dict[str, Any] = {"records": []}
    runtime_state["script"] = never_finishing_causal_script(runtime_state)
    with Servers(model_state, runtime_state) as servers:
        completed = run_lifecycle(EXTENSION, tmp_path / "run", servers)
    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    # The ungated early text passed its own message_end before the block.
    assert answers == [ORDINARY_ANSWER], answers
    # Exactly one genuinely mixed assistant message (text alongside a
    # toolCall): its ungated text passed whole and its toolCall block was
    # retained in the delivered message, so Pi kept tool-result pairing.
    mixed_ends = [
        event.get("message", {})
        for event in parse_events(completed.stdout)
        if event.get("type") == "message_end"
        and event.get("message", {}).get("role") == "assistant"
        and any(block.get("type") == "text" for block in event["message"].get("content", []))
        and any(block.get("type") == "toolCall" for block in event["message"].get("content", []))
    ]
    assert len(mixed_ends) == 1, "expected exactly one mixed assistant message"
    blocks = mixed_ends[0]["content"]
    assert any(block.get("type") == "toolCall" for block in blocks), blocks
    assert any(
        block.get("type") == "text" and block.get("text") == ORDINARY_ANSWER for block in blocks
    ), blocks
    assert continuation_requests(model_state) == 0
    assert len(model_state["requests"]) == 5, len(model_state["requests"])
    statuses = terminal_status_messages(completed)
    assert len(statuses) == 1, statuses
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/abandon") == 1, paths
    assert paths[-1] == "/v1/abandon", paths
    assert paths.count("/v1/complete") == 0, paths


def test_explicit_disable_with_held_disposition_fails_open(
    tmp_path: Path,
) -> None:
    """The held-disposition-then-disable control: after a budget-exhausted
    treatment withholds its answer, an explicit /cortheon disable must
    immediately reset finalization, and the next prompt's ordinary answer
    is delivered verbatim — no withheld replay from the dead treatment."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    turns = [
        TOOL_TURN,
        TOOL_TURN,
        TOOL_TURN,
        TOOL_TURN,
        SINGLE_TOOL_TURN,
        {"text": PLAIN_ANSWER},
    ]
    model_state: dict[str, Any] = {"requests": [], "turns": turns}
    runtime_state: dict[str, Any] = {"records": []}
    runtime_state["script"] = never_finishing_causal_script(runtime_state)
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            EXTENSION,
            [CAUSAL_PROMPT, "/cortheon disable", PLAIN_PROMPT],
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=workspace(tmp_path),
            tmp_path=tmp_path,
            timeout=60,
            extra_env={"CORTHEON_MAX_HOST_TOOL_CALLS": CAP},
        )
    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    # The post-disable answer is delivered unchanged and ungated.
    assert answers and answers[-1] == PLAIN_ANSWER, answers
    assert WITHHELD_MARKER not in answers[-1], answers
    paths = [path for path, _body in runtime_state["records"]]
    # The treatment abandoned exactly once; disable added no runtime calls.
    assert paths.count("/v1/abandon") == 1, paths


def test_abandon_transport_failure_does_not_launder_the_block(
    tmp_path: Path,
) -> None:
    """Runtime failure after bounded abandonment: the controlled block is
    set before the abandon call, so /v1/abandon failing at transport level
    must not recast the termination as a fail-open transport error and
    deliver the uncertified answer. The withheld result still stands."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    turns = [TOOL_TURN] * 5 + [{"text": ORDINARY_ANSWER}]
    model_state: dict[str, Any] = {"requests": [], "turns": turns}
    runtime_state: dict[str, Any] = {"records": []}
    runtime_state["script"] = never_finishing_causal_script(runtime_state, abandon_fails=True)
    with Servers(model_state, runtime_state) as servers:
        completed = run_lifecycle(EXTENSION, tmp_path / "run", servers)
    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    assert ORDINARY_ANSWER not in answers, answers
    statuses = terminal_status_messages(completed)
    assert len(statuses) == 1, statuses
    assert "host tool budget was exhausted" in str(statuses[0].get("content"))
    # The abandon was attempted exactly once and failed; nothing after it.
    paths = [path for path, _body in runtime_state["records"]]
    assert paths.count("/v1/abandon") == 1, paths
    assert paths[-1] == "/v1/abandon", paths
    assert paths.count("/v1/complete") == 0, paths


def test_terminal_block_uses_the_evaluator_sealed_verdict() -> None:
    """A real withheld terminal is classified only from sealed task semantics."""

    def _result(expected_verdict: Literal["allow", "block"] | None) -> RunResult:
        return RunResult(
            case_id="case_terminal",
            repeat=0,
            condition="cortheon",
            expected=True,
            final_text=final,
            delivered=False,
            correct=False,
            latency_seconds=1.0,
            tokens=0,
            tool_calls=0,
            tool_errors=0,
            timed_out=False,
            process_error=None,
            expected_verdict=expected_verdict,
            evaluator_outcome=parsed.outcome,
            candidate_correct=graded,
        )

    case = ImportCase(
        case_id="case_terminal",
        path="pkg/a.py",
        module="jsonpath",
        expected=True,
        prompt="Does pkg/a.py import jsonpath?",
    )
    # The exact bounded withheld replacement the adapter emits, including
    # the full pinned header and reason sentence the evaluator authenticates.
    final = (
        "[Cortheon withheld: completion was not certified]\nThe Cortheon "
        "investigation ended without a certified answer because the causal "
        "synthesis completion was withheld or failed certification after "
        "validation."
    )
    events = [
        {
            "type": "entry_appended",
            "entry": {
                "type": "custom",
                "customType": CANDIDATE_ENTRY_TYPE,
                "id": "entry-1",
                "timestamp": "2026-08-22T00:00:00.000Z",
                "data": {
                    "version": 1,
                    "stage": "causal_synthesis",
                    "candidate": "Answer: no, it does not import jsonpath.",
                },
            },
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": final}],
                "stopReason": "stop",
            },
        },
    ]
    parsed = parse_transport_outcome(events, host="pi")
    graded = _candidate_correct(
        case,
        events,
        host="pi",
        treatment=True,
        final=parsed.final_text,
        evaluator_outcome=parsed.outcome,
    )
    assert graded is False
    assert classify_block(_result("allow")) == FALSE_BLOCK
    assert classify_block(_result("block")) == SAFE_BLOCK
    assert classify_block(_result(None)) == UNCLASSIFIED_BLOCK


def test_deliberation_setup_rejection_never_escapes_message_end(
    tmp_path: Path,
) -> None:
    """A rejected modelRegistry.getApiKeyAndHeaders during deliberation
    setup must resolve as no-deliberation (the caller's bounded
    deliberation_empty stage), never reject out of deliberateCausalSynthesis
    — an escape would propagate through message_end and recreate the
    ordinary-answer false allow. Runs the real TypeScript module under
    Node's type stripping with a stub context whose registry rejects."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    pi_binary = Path(shutil.which("pi") or "").resolve()
    # ESM bare specifiers resolve from the importing file, so run a copy of
    # the pi_core sources beside a node_modules link to a root that can
    # actually satisfy the scoped "@earendil-works/..." specifiers. Pi's
    # dependencies may live either in the installing package's bundled
    # node_modules or in the global root — discover either shape and assert
    # the runtime scope dependency (@earendil-works/pi-ai; the
    # pi-coding-agent imports are type-only and stripped) is really present.
    candidates: list[Path] = []
    for parent in pi_binary.parents:
        candidates.append(parent / "node_modules")
        if parent.name == "node_modules":
            candidates.append(parent)
    pi_package_modules = next(
        (root for root in candidates if (root / "@earendil-works" / "pi-ai").is_dir()),
        None,
    )
    assert pi_package_modules is not None, (
        f"no node_modules root with @earendil-works/pi-ai found above {pi_binary}"
    )
    harness_dir = tmp_path / "ts-harness"
    harness_dir.mkdir()
    shutil.copytree(SOURCE_DIR / "pi_core", harness_dir / "pi_core")
    os.symlink(pi_package_modules, harness_dir / "node_modules")
    harness = harness_dir / "deliberation_setup_reject.mjs"
    harness.write_text(
        "import { deliberateCausalSynthesis } from "
        f"{json.dumps(str(harness_dir / 'pi_core' / 'repair.ts'))};\n"
        "import { setActive } from "
        f"{json.dumps(str(harness_dir / 'pi_core' / 'state.ts'))};\n"
        "setActive({ evidenceSummary: 'accepted evidence', "
        "evidenceRecords: [], goal: 'goal' });\n"
        "const context = {\n"
        "  model: { id: 'mock-small' },\n"
        "  modelRegistry: {\n"
        "    getApiKeyAndHeaders: async () => {\n"
        "      throw new Error('registry lookup rejected');\n"
        "    },\n"
        "  },\n"
        "  signal: undefined,\n"
        "};\n"
        "try {\n"
        "  const result = await deliberateCausalSynthesis(context, 'draft');\n"
        "  console.log(JSON.stringify({\n"
        "    escaped: false,\n"
        "    hasSections: Boolean(result?.sections),\n"
        "  }));\n"
        "} catch {\n"
        "  console.log(JSON.stringify({ escaped: true }));\n"
        "}\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [node, "--experimental-strip-types", str(harness)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=harness_dir,
        env={**os.environ},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report == {"escaped": False, "hasSections": False}, report


def test_fail_open_controls_unchanged(tmp_path: Path) -> None:
    """Genuine fail-open paths never set a disposition: an explicit disable,
    a transport failure at /v1/start, and a non-treatment prompt all deliver
    the host model's ordinary text verbatim with no withheld result."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    controls: list[tuple[str, str, dict[str, Any]]] = [
        (
            "disable",
            CAUSAL_PROMPT,
            {"CORTHEON_AUTO_ENABLE": "0"},
        ),
        ("transport", CAUSAL_PROMPT, {}),
        ("plain", "Say hello.", {}),
    ]
    for name, prompt, extra in controls:
        model_state: dict[str, Any] = {
            "requests": [],
            "turns": [{"text": ORDINARY_ANSWER}],
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
                prompt,
                model_port=servers.model.server_port,
                runtime_port=servers.runtime.server_port,
                workspace=workspace(tmp_path / f"{name}-run"),
                tmp_path=tmp_path / f"{name}-run",
                timeout=45,
                extra_env=extra,
            )
        assert completed.returncode == 0, (name, completed.stderr)
        answers = assistant_answers(completed)
        assert answers and answers[-1] == ORDINARY_ANSWER, (name, answers)
        paths = [path for path, _body in runtime_state["records"]]
        if name == "plain":
            assert "/v1/start" not in paths, (name, paths)


def test_mutation_without_guard_reopens_the_false_allow(
    tmp_path: Path,
) -> None:
    """Removing the terminal-disposition guard from message_end reinstates
    the live false allow: the continuation's ordinary text is delivered
    unvalidated. The guard is load-bearing, not decorative."""
    if not require_pi():
        pytest.skip("Pi is not installed")
    root = tmp_path / "mutation" / "cortheon"
    (root / "pi_core").mkdir(parents=True)
    for path in sorted((SOURCE_DIR / "pi_core").glob("*.ts")):
        text = path.read_text(encoding="utf-8")
        if path.stem == "tool_events":
            old = "return terminalDispositionResult(pi, event.message);"
            assert old in text
            text = text.replace(old, "return;")
        (root / "pi_core" / path.name).write_text(text, encoding="utf-8")
    facade = root / "pi_extension.ts"
    shutil.copy2(SOURCE_DIR / "pi_extension.ts", facade)

    turns = [TOOL_TURN] * 4 + [MIXED_TEXT_TOOL_TURN, {"text": ORDINARY_ANSWER}]
    model_state: dict[str, Any] = {"requests": [], "turns": turns}
    runtime_state: dict[str, Any] = {"records": []}
    runtime_state["script"] = never_finishing_causal_script(runtime_state)
    with Servers(model_state, runtime_state) as servers:
        completed = run_lifecycle(facade, tmp_path / "run", servers)

    assert completed.returncode == 0, completed.stderr
    answers = assistant_answers(completed)
    # With the guard removed the raw wrong-shaped answer comes through.
    assert ORDINARY_ANSWER in answers, answers

"""One-off acceptance evidence: exact counts for the four corrected traces.

Not a pytest module: run directly from tests/ to print the deterministic
terminal/candidate/stage/model-request/complete/abandon/host-execution
counts for the four masked defects' corrected scenarios.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from pi_causal_helpers import (
    CAUSAL_PROMPT,
    GOOD_REPAIR,
    WEAK_DRAFT,
    causal_runtime_script,
    causal_workspace,
)
from pi_doom_loop_helpers import TOOL_TURN, workspace
from pi_recovery_helpers import Servers, require_pi, run_pi
from pi_terminal_constants import AMBIGUITY_ANSWER, AMBIGUITY_PROMPT, EXTENSION

TMP = Path("/tmp/cortheon-acceptance")
POLICY_422 = (
    422,
    {
        "error": "cognitive policy refusal",
        "error_type": "CognitivePolicyRefusal",
    },
)
CAPTURE = {"CORTHEON_BENCHMARK_CAPTURE_CANDIDATE": "1"}


def counts(completed, runtime, model):
    events = [
        json.loads(line) for line in completed.stdout.splitlines() if line.strip().startswith("{")
    ]
    by_type: dict[str, int] = {}
    custom_candidates = []
    custom_stages = []
    terminals = 0
    withheld = 0
    host_exec = 0
    for event in events:
        by_type[event.get("type", "?")] = by_type.get(event.get("type", "?"), 0) + 1
        if event.get("type") == "entry_appended":
            entry = event.get("entry", {})
            if entry.get("customType") == "cortheon-benchmark-candidate-v1":
                custom_candidates.append(entry["data"])
            if entry.get("customType") == "cortheon-benchmark-causal-stage-v1":
                custom_stages.append(entry["data"]["reason"])
        if event.get("type") == "message_end":
            message = event.get("message", {})
            if message.get("role") == "custom" and message.get("customType") == (
                "cortheon-terminal-status-v1"
            ):
                terminals += 1
            if message.get("role") == "assistant":
                for block in message.get("content", []):
                    if isinstance(block, dict) and block.get("text", "").startswith(
                        "[Cortheon withheld:"
                    ):
                        withheld += 1
        if event.get("type") == "tool_execution_end":
            text = ""
            for block in event.get("result", {}).get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
            blocked = (
                "Cortheon has all the evidence",
                "Cortheon already certified",
                "Cortheon reached its host tool budget",
                "Cortheon has accepted sufficient independent evidence",
                "Cortheon's bounded completion continuation budget",
            )
            if not text.startswith(blocked) and "not found" not in text:
                host_exec += 1
    paths = [path for path, _body in runtime["records"]]
    submitted_answers = [
        body.get("answer") for path, body in runtime["records"] if path == "/v1/complete"
    ]
    # The candidate-measurement contract: each policy 422 path must emit
    # exactly one candidate entry whose text is the exact answer submitted
    # to /v1/complete — a policy block without its candidate is praising an
    # unmeasured terminal, so the trace refuses it.
    for entry in custom_candidates:
        assert entry["candidate"] in submitted_answers, entry
    return {
        "assistant_withheld": withheld,
        "custom_terminals": terminals,
        "candidate_entries": len(custom_candidates),
        "candidate_stages": [entry["stage"] for entry in custom_candidates],
        "candidate_texts": [entry["candidate"] for entry in custom_candidates],
        "stage_reasons": custom_stages,
        "model_requests": len(model["requests"]),
        "completes": paths.count("/v1/complete"),
        "abandons": paths.count("/v1/abandon"),
        "host_executions": host_exec,
        "tool_execution_start": by_type.get("tool_execution_start", 0),
    }


TABLE: dict[str, dict[str, Any]] = {}


def run(label, prompt, turns, script, capture=True, causal=False, prompt_list=None):
    TMP.mkdir(parents=True, exist_ok=True)
    model_state: dict[str, Any] = {"requests": [], "turns": turns}
    runtime_state: dict[str, Any] = {"records": [], "script": script}
    with Servers(model_state, runtime_state) as servers:
        completed = run_pi(
            EXTENSION,
            prompt_list if prompt_list else prompt,
            model_port=servers.model.server_port,
            runtime_port=servers.runtime.server_port,
            workspace=(causal_workspace if causal else workspace)(TMP / label),
            tmp_path=TMP / label,
            timeout=90,
            extra_env=CAPTURE if capture else None,
        )
    assert completed.returncode == 0, completed.stderr
    table = counts(completed, runtime_state, model_state)
    TABLE[label] = table
    print(f"--- {label}")
    print(json.dumps(table, indent=None))
    return completed


def ambiguity_422_script():
    def script(path: str, _body: dict[str, Any]) -> Any:
        if path == "/v1/start":
            return (
                200,
                {
                    "session_id": "amb-1",
                    "status": "observing",
                    "session": {"deliverable": "document_synthesis"},
                    "context": {"goal": AMBIGUITY_PROMPT},
                    "next_action": {"type": "reason"},
                },
            )
        if path == "/v1/complete":
            return POLICY_422
        return 200, {"status": "ok"}

    return script


def causal_script(complete_failure):
    base = causal_runtime_script(True)

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/complete":
            return complete_failure
        return base(path, body)

    return script


def plain_script():
    def script(path: str, body: dict[str, Any]) -> Any:
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


def fail_open_second():
    state = {"complete": 0}
    base = plain_script()

    def script(path: str, body: dict[str, Any]) -> Any:
        if path == "/v1/complete":
            state["complete"] += 1
            if state["complete"] >= 2:
                return "connection-reset"
        return base(path, body)

    return script


if not require_pi():
    raise SystemExit("pi not installed")

buffer = io.StringIO()
with redirect_stdout(buffer):
    run("d1-422", AMBIGUITY_PROMPT, [{"text": AMBIGUITY_ANSWER}], ambiguity_422_script())
    run(
        "d2-causal-422",
        CAUSAL_PROMPT,
        [{"text": WEAK_DRAFT}, {"text": GOOD_REPAIR}],
        causal_script(POLICY_422),
        causal=True,
    )
    run(
        "d2-causal-reset",
        CAUSAL_PROMPT,
        [{"text": WEAK_DRAFT}, {"text": GOOD_REPAIR}],
        causal_script("connection-reset"),
        causal=True,
    )
    plain = "Both ledgers reuse the shard key copper."
    run(
        "d3-same",
        "Summarize the relationship between the two fact files.",
        [{"text": plain}],
        plain_script(),
    )
    run(
        "d3-changed",
        "Summarize the relationship between the two fact files.",
        [{"text": plain}, {"text": plain + " Revised."}],
        plain_script(),
    )
    run(
        "d3-toolonly",
        "Summarize the relationship between the two fact files.",
        [{"text": plain}, TOOL_TURN],
        plain_script(),
    )
    run(
        "d3-failopen",
        "Summarize the relationship between the two fact files.",
        [{"text": plain}],
        fail_open_second(),
    )
print(buffer.getvalue())

# Exact acceptance table for the policy-measurement contract: a clean
# policy terminal with a real submitted candidate. These hard assertions
# make it impossible for the printed tables to praise a trace whose policy
# 422 path silently emitted zero candidates.
assert TABLE["d1-422"]["candidate_entries"] == 1, TABLE["d1-422"]
assert TABLE["d1-422"]["candidate_stages"] == ["completion"], TABLE["d1-422"]
assert TABLE["d1-422"]["candidate_texts"] == [AMBIGUITY_ANSWER], TABLE["d1-422"]
assert TABLE["d1-422"]["assistant_withheld"] == 1, TABLE["d1-422"]
assert TABLE["d1-422"]["custom_terminals"] == 0, TABLE["d1-422"]
assert TABLE["d1-422"]["completes"] == 1, TABLE["d1-422"]
assert TABLE["d1-422"]["abandons"] == 1, TABLE["d1-422"]
assert TABLE["d2-causal-422"]["candidate_entries"] == 1, TABLE["d2-causal-422"]
assert TABLE["d2-causal-422"]["candidate_stages"] == ["causal_synthesis"], TABLE["d2-causal-422"]
assert TABLE["d2-causal-422"]["stage_reasons"] == ["runtime_withheld"], TABLE["d2-causal-422"]
assert TABLE["d2-causal-422"]["completes"] == 1, TABLE["d2-causal-422"]
assert TABLE["d2-causal-422"]["abandons"] == 1, TABLE["d2-causal-422"]
assert TABLE["d2-causal-reset"]["candidate_entries"] == 0, TABLE["d2-causal-reset"]
assert TABLE["d2-causal-reset"]["stage_reasons"] == ["transport_failed"], TABLE["d2-causal-reset"]
assert TABLE["d3-failopen"]["candidate_entries"] == 0, TABLE["d3-failopen"]
print("acceptance table: policy candidates exact")

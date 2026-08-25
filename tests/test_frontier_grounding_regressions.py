from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from cortheon.cognitive_runtime import CognitiveRuntime

GOAL = (
    "Implement a production-ready async HTTP client. Inspect the exact installed "
    "runtime and dependency versions, then use current official documentation, "
    "scientific papers where relevant, and strong maintained repositories that "
    "are compatible with this project."
)


def _request(payload: dict[str, object]) -> dict[str, object]:
    action = payload["next_action"]
    assert isinstance(action, dict)
    request = action["request"]
    assert isinstance(request, dict)
    return request


def _operation(payload: dict[str, object]) -> str | None:
    parameters = _request(payload).get("parameters")
    assert isinstance(parameters, dict)
    operation = parameters.get("operation")
    return operation if isinstance(operation, str) else None


def _observe(
    runtime: CognitiveRuntime,
    payload: dict[str, object],
    observation: dict[str, object],
) -> dict[str, object]:
    session = payload["session"]
    assert isinstance(session, dict)
    request = _request(payload)
    return runtime.observe(
        str(session["session_id"]),
        [observation],
        request_id=str(request["request_id"]),
    )


def test_quick_frontier_code_task_reserves_patch_and_test_evidence() -> None:
    runtime = CognitiveRuntime()
    payload = runtime.start(
        "Implement a production-ready payment API client.",
        effort="quick",
    )
    session_id = str(payload["session"]["session_id"])
    retrieved_at = datetime.now(UTC).isoformat()

    for observation in (
        {
            "kind": "documentation",
            "content": "Python 3.13 and httpx 0.28.1 are pinned in pyproject.toml.",
            "source": "pyproject.toml",
        },
        {
            "kind": "web",
            "content": "Current compatible official guidance and implementation leads.",
            "source": "https://example.org/discovery",
            "url": "https://example.org/discovery",
            "retrieved_at": retrieved_at,
            "purpose": "discovery",
        },
        {
            "kind": "web",
            "content": "The primary documentation confirms the supported client API.",
            "source": "https://example.org/primary",
            "url": "https://example.org/primary",
            "retrieved_at": retrieved_at,
            "purpose": "primary_fetch",
        },
        {
            "kind": "web",
            "content": "A maintained compatible repository includes tests and a license.",
            "source": "https://github.com/example/payment-client",
            "url": "https://github.com/example/payment-client",
            "retrieved_at": retrieved_at,
            "purpose": "implementation_reference",
        },
        {
            "kind": "web",
            "content": "The main limitation is explicit transport cleanup.",
            "source": "https://example.org/limitation",
            "url": "https://example.org/limitation",
            "retrieved_at": retrieved_at,
            "purpose": "contradiction_check",
        },
    ):
        payload = _observe(runtime, payload, observation)

    discovery = _request(payload)
    assert discovery["parameters"]["operation"] == "code_discovery"
    payload = _observe(
        runtime,
        payload,
        {
            "kind": "code",
            "content": (
                '[CORTHEON_HOST_EVIDENCE] {"tool":"grep","outcome":"match",'
                '"args":{"command":"rg payment"}}\n'
                "src/payment_client.py:1: class PaymentClient\n"
                "tests/test_payment_client.py:1: def test_payment_client"
            ),
            "source": "project search",
        },
    )
    context = _request(payload)
    assert context["parameters"]["operation"] == "code_context"
    payload = runtime.observe(
        session_id,
        [
            {
                "kind": "code",
                "content": "class PaymentClient: pass",
                "source": "src/payment_client.py",
            },
            {
                "kind": "code",
                "content": "def test_payment_client(): pass",
                "source": "tests/test_payment_client.py",
            },
        ],
        request_id=str(context["request_id"]),
    )
    payload = runtime.observe(
        session_id,
        [
            {
                "kind": "diff",
                "content": "diff --git a/src/payment_client.py b/src/payment_client.py\n+fixed",
                "source": "git diff",
            },
            {
                "kind": "test",
                "content": "pytest: 1 passed",
                "source": "pytest",
                "status": "verified",
            },
        ],
    )
    assert payload["session"]["observations_used"] == 10


def test_host_receipted_scoped_null_advances_frontier_review() -> None:
    runtime = CognitiveRuntime(require_host_receipts=True)
    payload = runtime.start(GOAL, effort="deep")
    retrieved_at = datetime.now(UTC).isoformat()
    payload = _observe(
        runtime,
        payload,
        {
            "kind": "documentation",
            "content": (
                '[CORTHEON_HOST_EVIDENCE] {"tool":"read","outcome":"result",'
                '"args":{"filePath":"pyproject.toml"}}\n'
                "Python 3.13 and httpx 0.28.1 are pinned in the live project."
            ),
            "source": "pyproject.toml",
        },
    )
    for purpose, suffix in (
        ("discovery", "discovery"),
        ("primary_fetch", "primary"),
    ):
        payload = _observe(
            runtime,
            payload,
            {
                "kind": "web",
                "content": f"Attributable compatible evidence for {purpose}.",
                "source": f"https://example.org/{suffix}",
                "url": f"https://example.org/{suffix}",
                "retrieved_at": retrieved_at,
                "purpose": purpose,
            },
        )

    assert _operation(payload) == "scholarly_source_review"
    payload = _observe(
        runtime,
        payload,
        {
            "kind": "web",
            "content": (
                '[CORTHEON_HOST_EVIDENCE] {"tool":"websearch","outcome":"no_match",'
                '"args":{"query":"scoped scholarly search"}}\n'
                "No directly relevant paper was found in the scoped search."
            ),
            "source": "opencode:websearch:unattributed",
            "retrieved_at": retrieved_at,
            "purpose": "scholarly_validation",
        },
    )
    assert _operation(payload) == "repository_source_review"


def test_opencode_websearch_marks_a_scoped_null() -> None:
    root = Path(__file__).parents[1]
    script = r"""
import {webEvidenceBatch} from './src/cortheon/opencode_core/evidence.js';
const observations = webEvidenceBatch(
  'websearch',
  {query: 'directly relevant primary paper'},
  'No results found.',
  {request: {parameters: {purpose: 'scholarly_validation'}}},
);
const first = observations[0];
const receipt = JSON.parse(
  first.content.split('\n')[0].slice('[CORTHEON_HOST_EVIDENCE] '.length),
);
console.log(JSON.stringify({first, receipt}));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["receipt"]["outcome"] == "no_match"
    assert result["first"]["purpose"] == "scholarly_validation"
    assert "url" not in result["first"]

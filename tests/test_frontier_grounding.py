from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from cortheon.cognitive_core.receipts import _read_only_shell_receipt
from cortheon.cognitive_program import compile_program
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
    request = _request(payload)
    parameters = request.get("parameters")
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


def test_compatibility_sensitive_code_task_starts_with_environment_grounding() -> None:
    started = CognitiveRuntime().start(GOAL, effort="deep")

    request = _request(started)
    assert request["capability"] == "inspect"
    assert _operation(started) == "environment_grounding"
    assert request["parameters"]["required_facts"] == [
        "runtime_versions",
        "dependency_versions",
        "manifests_and_lockfiles",
        "available_api_surface",
    ]
    program = compile_program(
        goal=GOAL,
        task_kind="code",
        deliverable="code_change",
        effort="deep",
        requirements=(),
        max_turns=15,
        max_observations=32,
    )
    operators = {item["operator_id"] for item in program["operators"]}
    assert {"ground_environment", "discover_frontier", "filter_compatibility"} <= operators


def test_repository_routing_requires_source_intent_not_bare_mentions() -> None:
    from cortheon.cognitive_core.frontier_policy import needs_repository_sources

    # Bare mentions are not repository-inspection intent.
    assert not needs_repository_sources("Write a report about GitHub's acquisition strategy.")
    assert not needs_repository_sources("Write a survey of source-code privacy laws.")
    assert not needs_repository_sources("Summarize what GitHub allows in a code of conduct.")
    # Source intent triggers repository review.
    assert needs_repository_sources("Review the maintenance and license of the fastapi repository.")
    assert needs_repository_sources("Inspect the source code of the httpx client project.")
    assert needs_repository_sources("Compare reference implementations of CRDT libraries.")
    assert needs_repository_sources("Check the tests and CI of github.com/encode/httpx.")


def test_grounding_progresses_to_current_frontier_sources_then_primary_fetch() -> None:
    runtime = CognitiveRuntime()
    started = runtime.start(GOAL, effort="deep")

    grounded = _observe(
        runtime,
        started,
        {
            "kind": "documentation",
            "content": (
                "Python 3.13; httpx 0.28.1 from pyproject.toml and uv.lock; "
                "AsyncClient includes the required transport hooks."
            ),
            "source": "pyproject.toml",
            "status": "observed",
        },
    )
    discovery = _request(grounded)
    assert discovery["capability"] == "search_or_fetch"
    assert _operation(grounded) == "frontier_discovery"
    assert discovery["parameters"]["environment_evidence_ids"] == ["ev1"]
    assert "official_documentation" in discovery["parameters"]["source_classes"]
    assert "maintained_reference_repository" in discovery["parameters"]["source_classes"]
    assert "primary_research" in discovery["parameters"]["source_classes"]

    retrieved_at = datetime.now(UTC).isoformat()
    discovered = _observe(
        runtime,
        grounded,
        {
            "kind": "web",
            "content": (
                "Official httpx documentation for 0.28.1 describes AsyncClient transport "
                "hooks. A maintained reference implementation exercises the same API."
            ),
            "source": "https://www.python-httpx.org/advanced/transports/",
            "url": "https://www.python-httpx.org/advanced/transports/",
            "retrieved_at": retrieved_at,
            "purpose": "discovery",
            "status": "observed",
        },
    )
    primary = _request(discovered)
    assert primary["capability"] == "fetch"
    assert _operation(discovered) == "primary_source_fetch"
    assert primary["parameters"]["purpose"] == "primary_fetch"


def test_frontier_grounding_checks_counterevidence_before_returning_to_local_code() -> None:
    runtime = CognitiveRuntime()
    payload = runtime.start(GOAL, effort="deep")
    payload = _observe(
        runtime,
        payload,
        {
            "kind": "documentation",
            "content": "Python 3.13 and httpx 0.28.1 are pinned in the live project.",
            "source": "uv.lock",
        },
    )
    retrieved_at = datetime.now(UTC).isoformat()
    payload = _observe(
        runtime,
        payload,
        {
            "kind": "web",
            "content": "Current official transport documentation and compatible examples.",
            "source": "https://www.python-httpx.org/advanced/transports/",
            "url": "https://www.python-httpx.org/advanced/transports/",
            "retrieved_at": retrieved_at,
            "purpose": "discovery",
        },
    )
    payload = _observe(
        runtime,
        payload,
        {
            "kind": "web",
            "content": "The official documentation confirms the 0.28.1 API surface.",
            "source": "https://www.python-httpx.org/advanced/transports/",
            "url": "https://www.python-httpx.org/advanced/transports/",
            "retrieved_at": retrieved_at,
            "purpose": "primary_fetch",
        },
    )
    assert _operation(payload) == "scholarly_source_review"
    scholarly = _request(payload)
    assert scholarly["parameters"]["deduplicate_by"] == [
        "doi",
        "arxiv_id",
        "normalized_title",
    ]
    assert scholarly["parameters"]["rank_by"][:2] == ["direct_relevance", "method_quality"]

    payload = _observe(
        runtime,
        payload,
        {
            "kind": "web",
            "content": (
                '[CORTHEON_HOST_EVIDENCE] {"tool":"webfetch","outcome":"result",'
                '"args":{"url":"https://doi.org/10.0000/example"}}\n'
                "Primary paper method, result, transfer conditions, and limitations."
            ),
            "source": "https://doi.org/10.0000/example",
            "url": "https://doi.org/10.0000/example",
            "retrieved_at": retrieved_at,
            "purpose": "scholarly_validation",
            "source_record": {
                "identifier": "10.0000/example",
                "method": "controlled benchmark with held-out splits",
                "limitations": "single domain",
            },
        },
    )
    assert _operation(payload) == "repository_source_review"
    repository = _request(payload)
    assert "tests_and_ci" in repository["parameters"]["required_signals"]
    assert "implementation_files" in repository["parameters"]["required_signals"]

    payload = _observe(
        runtime,
        payload,
        {
            "kind": "web",
            "content": (
                '[CORTHEON_HOST_EVIDENCE] {"tool":"webfetch","outcome":"result",'
                '"args":{"url":"https://github.com/example/http-client"}}\n'
                "Maintained compatible repository with release, license, tests, and code."
            ),
            "source": "https://github.com/example/http-client",
            "url": "https://github.com/example/http-client",
            "retrieved_at": retrieved_at,
            "purpose": "implementation_reference",
            "source_record": {
                "repository_url": "https://github.com/example/http-client",
                "maintenance": "commits within the last month",
                "license": "MIT",
                "tests": "CI test suite passes on the current release",
                "compatibility": "matches the pinned httpx runtime",
            },
        },
    )
    assert _operation(payload) == "counterevidence_search"

    payload = _observe(
        runtime,
        payload,
        {
            "kind": "web",
            "content": "Known limitation: custom transports must implement cleanup explicitly.",
            "source": "https://github.com/encode/httpx/issues/0000",
            "url": "https://github.com/encode/httpx/issues/0000",
            "retrieved_at": retrieved_at,
            "purpose": "contradiction_check",
        },
    )
    assert _operation(payload) == "code_discovery"
    local = _request(payload)
    assert local["parameters"]["frontier_grounded"] is True
    assert local["parameters"]["environment_evidence_ids"] == ["ev1"]
    assert local["parameters"]["external_evidence_ids"] == [
        "ev2",
        "ev3",
        "ev4",
        "ev5",
        "ev6",
    ]


def test_simple_local_fix_does_not_pay_for_frontier_research() -> None:
    started = CognitiveRuntime().start(
        "Fix the off-by-one bug in src/counter.py and run its existing test.",
        effort="standard",
    )

    assert _operation(started) != "environment_grounding"


def test_external_technology_build_infers_frontier_grounding_without_magic_words() -> None:
    started = CognitiveRuntime().start(
        "Implement an async client for the payment API using the project's HTTP library.",
        effort="standard",
    )

    assert _operation(started) == "environment_grounding"


def test_explicit_research_task_keeps_the_existing_research_protocol() -> None:
    started = CognitiveRuntime().start(
        "Research the latest stable Python release from current primary sources.",
        effort="deep",
    )

    request = _request(started)
    assert request["capability"] == "search"
    assert request["parameters"]["purpose"] == "contradiction_check"


def test_environment_grounding_accepts_identity_probes_but_not_runtime_execution() -> None:
    assert _read_only_shell_receipt({"command": "python3 --version"}) is True
    assert _read_only_shell_receipt({"command": "python3 -m pip show httpx"}) is True
    assert _read_only_shell_receipt({"command": "node --version && git status --short"}) is True
    assert _read_only_shell_receipt({"command": "python3 -c 'print(1)'"}) is False
    assert _read_only_shell_receipt({"command": "npm install example"}) is False


def test_opencode_code_task_preserves_frontier_web_provenance() -> None:
    root = Path(__file__).parents[1]
    script = r"""
import {createToolAfterHook} from './src/cortheon/opencode_core/hook_output.js';
import {investigations, permittedHostTool, safeHostArguments} from './src/cortheon/opencode_core/state.js';
let captured;
const state = {
  automatic: true, active: true, cortheonSessionID: 'vx_frontier',
  deliverable: 'code_change', requestID: 'req2', evidenceIDs: [],
  request: {
    request_id: 'req2', capability: 'search_or_fetch',
    parameters: {operation: 'frontier_discovery', purpose: 'discovery'},
  },
};
investigations.set('s', state);
const hook = createToolAfterHook({
  debug: async () => {}, captureMutationAfter: async () => {},
  runRequestedTest: async (_id, value) => value,
  patchHygieneIssue: async () => undefined,
  certifyCodeChange: async (_id, value) => value,
  submitAutomaticObservation: async (_id, value) => {
    captured = value.hostEvidenceBatch;
    return value;
  },
  submitPassiveObservations: async (_id, value) => value,
});
await hook['tool.execute.after'](
  {sessionID: 's', tool: 'websearch', args: {query: 'current compatible API'}},
  {output: 'Official docs https://docs.example.org/api Version-compatible API details.'},
);
console.log(JSON.stringify({
  captured,
  webAllowed: permittedHostTool('websearch', state),
  localBlocked: permittedHostTool('read', state),
  bashArgs: safeHostArguments('bash', {command: 'python3 --version'}),
}));
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
    assert result["webAllowed"] is True
    assert result["localBlocked"] is False
    assert result["bashArgs"] == {"command": "python3 --version"}
    assert result["captured"][0]["kind"] == "web"
    assert result["captured"][0]["url"] == "https://docs.example.org/api"
    assert result["captured"][0]["purpose"] == "discovery"

import json
import urllib.error
import urllib.request

import pytest
from cognitive_http_cases_common import post, running_server

from cortheon.cognitive_http import build_server


def test_optional_bearer_token_protects_mutating_routes():
    with running_server(token="secret") as (_server, base):
        with pytest.raises(urllib.error.HTTPError) as error:
            post(base, "/v1/start", {"goal": "Inspect code"})
        assert error.value.code == 401

        with post(
            base,
            "/v1/start",
            {"goal": "Inspect code"},
            token="secret",
        ) as response:
            assert json.load(response)["session"]["storage"] == "memory_only"

        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + "/metrics", timeout=2)
        assert error.value.code == 401

        request = urllib.request.Request(
            base + "/metrics",
            headers={"Authorization": "Bearer secret"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            assert json.load(response)["ok"] is True


def test_invalid_http_concurrency_limit_is_rejected():
    with pytest.raises(ValueError, match="max_concurrent_requests"):
        build_server("127.0.0.1", 0, max_concurrent_requests=0)


def test_http_native_adapter_lease_can_be_renewed():
    with running_server() as (server, base):
        with post(
            base,
            "/v1/start",
            {"goal": "Inspect code", "lease_seconds": 30},
        ) as response:
            started = json.load(response)
        session_id = started["session"]["session_id"]
        assert started["session"]["lease_seconds"] == 30.0

        with post(
            base,
            "/v1/heartbeat",
            {"session_id": session_id},
        ) as response:
            heartbeat = json.load(response)

        assert heartbeat["ok"] is True
        assert heartbeat["lease_seconds"] == 30.0
        assert server.runtime.active_sessions == 1


def test_trusted_http_adapter_can_append_passive_host_evidence():
    with running_server() as (_server, base):
        with post(
            base,
            "/v1/start",
            {"goal": "Fix the parser and run its tests", "effort": "quick"},
        ) as response:
            started = json.load(response)
        session_id = started["session"]["session_id"]
        request_id = started["next_action"]["request"]["request_id"]
        with post(
            base,
            "/v1/observe",
            {
                "session_id": session_id,
                "request_id": request_id,
                "observations": [
                    {
                        "kind": "code",
                        "content": (
                            '[CORTHEON_HOST_EVIDENCE] {"tool":"read",'
                            '"outcome":"result","args":{"filePath":"parser.py"}}\n'
                            "def parse(value): return value"
                        ),
                        "source": "host:read:parser.py",
                    }
                ],
            },
        ):
            pass

        with post(
            base,
            "/v1/observe",
            {
                "session_id": session_id,
                "observations": [
                    {
                        "kind": "diff",
                        "content": (
                            '[CORTHEON_HOST_EVIDENCE] {"tool":"diff",'
                            '"outcome":"changed","args":{"path":"parser.py"}}\n'
                            "- return value\n+ return normalize(value)"
                        ),
                        "source": "host:session-diff",
                    },
                    {
                        "kind": "test",
                        "content": (
                            '[CORTHEON_HOST_EVIDENCE] {"tool":"test",'
                            '"outcome":"passed","args":{"command":'
                            '"pytest tests/test_parser.py"}}\n'
                            "pytest tests/test_parser.py: 3 passed"
                        ),
                        "source": "host:bash",
                        "status": "verified",
                    },
                ],
            },
        ) as response:
            observed = json.load(response)

        assert observed["accepted_evidence_ids"] == ["ev2", "ev3"]

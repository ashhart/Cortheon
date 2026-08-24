import json

from cognitive_http_cases_common import post, running_server


def test_http_reasoning_routes_drive_hypothesis_and_challenge_passes():
    with running_server() as (_server, base):
        with post(
            base,
            "/v1/start",
            {
                "goal": "Determine why the rollout stalled from the live notes.",
                "effort": "standard",
            },
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
                        "kind": "documentation",
                        "content": (
                            '[CORTHEON_HOST_EVIDENCE] {"tool":"read",'
                            '"outcome":"result","args":{"filePath":"rollout.md"}}\n'
                            "The canary gate is waiting for schema approval."
                        ),
                        "source": "pi:read:rollout.md",
                    }
                ],
            },
        ) as response:
            observed = json.load(response)
        assert observed["next_action"]["submit_via"] == "cortheon_step"
        assert observed["cognition"]["stage"] == "frame"
        with post(base, "/v1/resume", {}) as response:
            resumed = json.load(response)
        assert resumed["sessions"][0]["next_action"] == observed["next_action"]

        with post(
            base,
            "/v1/step",
            {
                "session_id": session_id,
                "hypotheses": [
                    {
                        "statement": "Schema approval is blocking the rollout.",
                        "falsification_test": "Inspect the schema approval record.",
                    },
                    {
                        "statement": "The canary itself failed.",
                        "falsification_test": "Inspect the canary result.",
                    },
                ],
            },
        ) as response:
            stepped = json.load(response)
        assert stepped["next_action"]["request"]["hypothesis_id"] == "h1"
        hypothesis_request = stepped["next_action"]["request"]

        with post(
            base,
            "/v1/observe",
            {
                "session_id": session_id,
                "request_id": hypothesis_request["request_id"],
                "observations": [
                    {
                        "kind": "documentation",
                        "content": (
                            '[CORTHEON_HOST_EVIDENCE] {"tool":"read",'
                            '"outcome":"result","args":{"filePath":"approval.md"}}\n'
                            "Schema approval remains pending."
                        ),
                        "source": "pi:read:approval.md",
                        "status": "verified",
                    }
                ],
            },
        ) as response:
            classified = json.load(response)
        assert classified["next_action"]["submit_via"] == "cortheon_step"
        assert classified["next_action"]["required_fields"] == ["hypothesis_updates"]
        assert classified["cognition"]["stage"] == "update"
        assert "ev2" in classified["guidance"]
        with post(base, "/v1/resume", {}) as response:
            resumed = json.load(response)
        assert resumed["sessions"][0]["next_action"] == classified["next_action"]

        with post(
            base,
            "/v1/challenge",
            {
                "session_id": session_id,
                "draft": "Schema approval is blocking the rollout.",
                "claims": [
                    {
                        "claim": "The rollout is waiting for schema approval.",
                        "evidence_ids": ["ev1"],
                    }
                ],
            },
        ) as response:
            challenged = json.load(response)
        assert "attacks" in challenged
        assert challenged["next_action"]["type"] in {"harness_tool", "reason"}


def test_retract_route_withdraws_evidence():
    with running_server() as (_server, base):
        with post(
            base,
            "/v1/start",
            {"goal": "Explain why the rollout stalled this afternoon."},
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
                            '"outcome":"result","args":{"path":"notes/rollout.md"}}\n'
                            "The canary gate is still waiting."
                        ),
                    }
                ],
            },
        ) as response:
            observed = json.load(response)
        assert observed["accepted_evidence_ids"] == ["ev1"]

        with post(
            base,
            "/v1/retract",
            {
                "session_id": session_id,
                "evidence_ids": ["ev1"],
                "reason": "The note was from the wrong environment.",
            },
        ) as response:
            retracted = json.load(response)
        assert retracted["retracted_evidence_ids"] == ["ev1"]

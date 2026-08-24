import json

from cognitive_http_cases_common import post, running_server


def test_http_transport_completes_and_erases_atomic_lookup():
    with running_server() as (server, base):
        with post(
            base,
            "/v1/start",
            {"goal": "Does src/example.py import pathlib?", "effort": "quick"},
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
                            '[CORTHEON_HOST_EVIDENCE] {"tool":"grep",'
                            '"outcome":"no_match","args":{"pattern":"pathlib",'
                            '"path":"src/example.py"}}\nNo matches found.'
                        ),
                        "source": "test-harness:grep",
                        "status": "verified",
                    }
                ],
            },
        ) as response:
            observed = json.load(response)
        assert observed["accepted_evidence_ids"] == ["ev1"]

        with post(
            base,
            "/v1/complete",
            {
                "session_id": session_id,
                "answer": "No.",
                "claims": [
                    {
                        "claim": "src/example.py does not import pathlib.",
                        "evidence_ids": ["ev1"],
                    }
                ],
                "hypotheses": [
                    {
                        "statement": "src/example.py has no pathlib import.",
                        "falsification_test": "Search the file for pathlib.",
                        "status": "supported",
                        "evidence_ids": ["ev1"],
                    }
                ],
                "completion_evidence_ids": ["ev1"],
            },
        ) as response:
            completed = json.load(response)

        assert completed["status"] == "complete"
        assert completed["answer"] == "No."
        assert completed["retained_project_data"] is False
        assert server.runtime.active_sessions == 0


def test_start_accepts_a_strictness_profile():
    with running_server() as (_server, base):
        with post(
            base,
            "/v1/start",
            {"goal": "Inspect code", "strictness": "assist"},
        ) as response:
            started = json.load(response)
        assert started["session"]["strictness"] == "assist"


def test_http_evidence_close_discards_synthesis_without_answer_certification():
    with running_server() as (server, base):
        with post(
            base,
            "/v1/start",
            {
                "goal": (
                    "Synthesize competing records and give a falsification test. "
                    "Do not modify files."
                ),
                "effort": "quick",
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
                            '[CORTHEON_HOST_EVIDENCE] {"tool":"search",'
                            '"outcome":"result","args":{"query":"records"}}\n'
                            "Record A establishes the live cohort boundary."
                        ),
                        "source": "test-harness:read",
                        "status": "verified",
                    }
                ],
            },
        ):
            pass

        with post(
            base,
            "/v1/evidence-close",
            {"session_id": session_id},
        ) as response:
            closed = json.load(response)

        assert closed["status"] == "evidence_closed"
        assert closed["answer_certified"] is False
        assert closed["retained_project_data"] is False
        assert server.runtime.active_sessions == 0
        assert server.runtime.metrics["sessions_evidence_closed"] == 1


def test_resume_route_lists_active_sessions():
    with running_server() as (_server, base):
        with post(
            base,
            "/v1/start",
            {"goal": "Explain why the rollout stalled this afternoon."},
        ) as response:
            started = json.load(response)
        with post(base, "/v1/resume", {}) as response:
            resumed = json.load(response)
        assert resumed["sessions"][0]["session_id"] == started["session"]["session_id"]
        assert "rollout" in resumed["sessions"][0]["goal"]

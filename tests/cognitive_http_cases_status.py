import json
import urllib.request

from cognitive_http_cases_common import post, running_server


def test_health_is_memory_only_and_not_cached():
    with running_server() as (_server, base):
        with urllib.request.urlopen(base + "/healthz", timeout=2) as response:
            payload = json.load(response)

        assert payload["ok"] is True
        assert payload["storage"] == "memory_only"
        assert payload["protocol_version"] == "1.0.0"
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Cortheon-Protocol"] == "1.0.0"


def test_capabilities_and_content_free_metrics_are_machine_readable():
    with running_server() as (_server, base):
        with urllib.request.urlopen(base + "/v1/capabilities", timeout=2) as response:
            capabilities = json.load(response)
        with urllib.request.urlopen(base + "/metrics", timeout=2) as response:
            metrics = json.load(response)

        assert capabilities["owns_project_files"] is False
        assert capabilities["native_adapter_leases"] is True
        assert capabilities["adaptive_cognition"]["cross_source_derivation"] is True
        assert capabilities["adaptive_cognition"]["explicit_alias_resolution"] is True
        assert capabilities["adaptive_cognition"]["conjunctive_rule_derivation"] is True
        assert capabilities["adaptive_cognition"]["requirement_level_completion_coverage"] is True
        assert capabilities["adaptive_cognition"]["contradiction_driven_replanning"] is True
        assert capabilities["adaptive_cognition"]["host_owned_tools"] is True
        assert capabilities["adaptive_cognition"]["substrate_abductive_origination"] is True
        assert capabilities["adaptive_cognition"]["originated_hypothesis_provenance"] is True
        assert capabilities["adaptive_cognition"]["ephemeral_cognitive_graph"] is True
        assert capabilities["adaptive_cognition"]["information_gain_planning"] is True
        assert capabilities["adaptive_cognition"]["task_program_compilation"] is True
        assert {"frame", "update"} <= set(capabilities["adaptive_cognition"]["stages"])
        assert capabilities["evidence_assurance"]["pi"] == "enforced_adapter"
        assert capabilities["evidence_assurance"]["codex"].startswith("enforced")
        assert metrics["storage"] == "memory_only"
        assert metrics["active_hook_turns"] == 0
        assert "goal" not in metrics


def test_codex_hook_routes_enforce_a_content_free_lifecycle():
    identity = {
        "host": "codex",
        "host_session_id": "secret-session-id",
        "turn_id": "secret-turn-id",
    }
    with running_server() as (server, base):
        with post(base, "/v1/hooks/register", identity) as response:
            registered = json.load(response)
        with post(
            base,
            "/v1/hooks/pre-tool",
            {**identity, "tool_name": "Bash"},
        ) as response:
            nudged = json.load(response)
        assert registered["started"] is False
        assert nudged["allow"] is True
        assert "cortheon_start" in nudged["guidance"]

        for tool_name, certified in (
            ("mcp__cortheon__cortheon_start", False),
            ("mcp__cortheon__cortheon_observe", False),
            ("mcp__cortheon__cortheon_complete", True),
        ):
            with post(
                base,
                "/v1/hooks/post-tool",
                {
                    **identity,
                    "tool_name": tool_name,
                    "succeeded": True,
                    "certified": certified,
                },
            ):
                pass

        with post(base, "/v1/hooks/stop", identity) as response:
            stopped = json.load(response)
        assert stopped["allow"] is True
        assert server.hook_tracker.active_turns == 0
        assert "secret-session-id" not in json.dumps(server.hook_tracker.metrics)

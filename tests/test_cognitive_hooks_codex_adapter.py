"""The Codex host adapter: bounded argv, recovery guidance, and sandboxing."""

import json

from cortheon.codex_plugins.cortheon.hooks import cortheon_hook


def test_codex_host_adapter_accepts_only_scheduled_diff_and_test_commands():
    test_result = {
        "automatic": True,
        "next_action": {
            "type": "harness_tool",
            "request": {
                "request_id": "hook_test",
                "capability": "test",
                "parameters": {
                    "command": [
                        "python3",
                        "-m",
                        "pytest",
                        "-q",
                        "test_calculator.py",
                    ]
                },
            },
        },
    }
    assert cortheon_hook._host_adapter_argv(
        test_result,
        "python3 -m pytest -q test_calculator.py",
    ) == ["python3", "-m", "pytest", "-q", "test_calculator.py"]
    assert (
        cortheon_hook._host_adapter_argv(
            test_result,
            "python3 -m pytest -q test_calculator.py; touch owned",
        )
        is None
    )
    assert (
        cortheon_hook._host_adapter_argv(
            test_result,
            "python3 -m pytest -q ../test_calculator.py",
        )
        is None
    )

    diff_result = {
        "automatic": True,
        "next_action": {
            "type": "harness_tool",
            "request": {
                "request_id": "hook_diff",
                "capability": "diff",
                "parameters": {"path": "calculator.py"},
            },
        },
    }
    assert cortheon_hook._host_adapter_argv(
        diff_result,
        "git diff -- calculator.py",
    ) == ["git", "diff", "--", "calculator.py"]
    assert (
        cortheon_hook._host_adapter_argv(
            diff_result,
            "git diff -- ../calculator.py",
        )
        is None
    )


def test_codex_hook_tells_small_models_to_recover_after_a_failed_host_action(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        cortheon_hook,
        "_post",
        lambda path, payload: {
            "automatic": True,
            "next_action": {
                "type": "harness_tool",
                "request": {
                    "request_id": "req1",
                    "capability": "read_many",
                    "query": "Read rtc.html.",
                },
            },
        },
    )

    cortheon_hook._post_tool_use(
        {
            "session_id": "session",
            "turn_id": "turn",
            "tool_name": "Bash",
            "tool_response": {"exit_code": 1, "output": "No such file"},
        }
    )

    emitted = json.loads(capsys.readouterr().out)
    context = emitted["hookSpecificOutput"]["additionalContext"]
    assert "host action failed" in context
    assert "do not repeat the identical command" in context


def test_codex_host_adapter_runs_scheduled_test_in_workspace_sandbox(
    monkeypatch,
    tmp_path,
):
    result = {
        "automatic": True,
        "allow": False,
        "next_action": {
            "type": "harness_tool",
            "request": {
                "request_id": "hook_test",
                "capability": "test",
                "parameters": {
                    "command": [
                        "python3",
                        "-m",
                        "pytest",
                        "-q",
                        "test_calculator.py",
                    ]
                },
            },
        },
    }
    observed = []
    executions = []

    def fake_post(path, payload):
        if path == "/v1/hooks/pre-tool":
            return {
                "allow": True,
                "updated_input": {"command": "python3 -m pytest -q test_calculator.py"},
            }
        if path == "/v1/hooks/post-tool":
            observed.append(payload)
            return {"tracked": True}
        raise AssertionError(path)

    def fake_run(command, **kwargs):
        executions.append((command, kwargs))
        return cortheon_hook.subprocess.CompletedProcess(
            command,
            0,
            "1 passed in 0.01s\n",
            "",
        )

    monkeypatch.setattr(cortheon_hook, "_post", fake_post)
    monkeypatch.setattr(cortheon_hook.shutil, "which", lambda name: "/opt/bin/codex")
    monkeypatch.setattr(cortheon_hook.subprocess, "run", fake_run)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-tests")

    assert cortheon_hook._run_host_adapter_step(
        {
            "session_id": "session",
            "turn_id": "turn",
            "cwd": str(tmp_path),
            "permission_mode": "dontAsk",
        },
        result,
    )
    command, options = executions[0]
    assert command == [
        "/opt/bin/codex",
        "sandbox",
        "--permission-profile",
        ":workspace",
        "--cd",
        str(tmp_path),
        "--",
        "python3",
        "-m",
        "pytest",
        "-q",
        "test_calculator.py",
    ]
    assert "shell" not in options
    assert "OPENAI_API_KEY" not in options["env"]
    assert observed[0]["succeeded"] is True
    assert "1 passed" in observed[0]["tool_output"]

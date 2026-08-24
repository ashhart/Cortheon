import json
import os
import subprocess
from pathlib import Path

from cognitive_http_cases_common import running_server


def test_bundled_codex_hook_drives_live_http_tracker():
    hook = (
        Path(__file__).parents[1]
        / "src"
        / "cortheon"
        / "codex_plugins"
        / "cortheon"
        / "hooks"
        / "cortheon_hook.py"
    )

    def run_hook(base: str, payload: dict) -> dict | None:
        completed = subprocess.run(
            [os.environ.get("PYTHON", "python3"), str(hook)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=3,
            env={**os.environ, "CORTHEON_RUNTIME_URL": base},
            check=True,
        )
        return json.loads(completed.stdout) if completed.stdout else None

    common = {
        "session_id": "session",
        "turn_id": "turn",
        "cwd": "/tmp",
    }
    with running_server() as (server, base):
        prompt = run_hook(
            base,
            {
                **common,
                "hook_event_name": "UserPromptSubmit",
                "prompt": (
                    "Read pyproject.toml and report which console command maps to "
                    "cortheon.cognitive_cli:main."
                ),
            },
        )
        assert (
            "CORTHEON AUTOMATIC SESSION IS ACTIVE"
            in prompt["hookSpecificOutput"][  # pyright: ignore[reportOptionalSubscript]
                "additionalContext"
            ]
        )
        assert "ADAPTIVE COGNITION" in prompt["hookSpecificOutput"]["additionalContext"]  # pyright: ignore[reportOptionalSubscript]

        # Non-shell tools are allowed through during investigation; the shim
        # prints nothing for a plain allow without an updated input.
        assert (
            run_hook(
                base,
                {
                    **common,
                    "hook_event_name": "PreToolUse",
                    "tool_name": "view_image",
                    "tool_input": {"path": "cortheon.cognitive_cli:main"},
                },
            )
            is None
        )

        investigation_input = {
            "command": ("rg -n --fixed-strings -- cortheon.cognitive_cli:main pyproject.toml")
        }
        assert (
            run_hook(
                base,
                {
                    **common,
                    "hook_event_name": "PreToolUse",
                    "tool_name": "Bash",
                    "tool_input": investigation_input,
                },
            )
            is None
        )

        assert (
            run_hook(
                base,
                {
                    **common,
                    "hook_event_name": "PostToolUse",
                    "tool_name": "Bash",
                    "tool_input": investigation_input,
                    "tool_response": ('25:cortheon = "cortheon.cognitive_cli:main"'),
                },
            )
            is None
        )

        assert (
            run_hook(
                base,
                {
                    **common,
                    "hook_event_name": "Stop",
                    "last_assistant_message": ("cortheon maps to cortheon.cognitive_cli:main."),
                },
            )
            is None
        )
        assert server.hook_tracker.active_turns == 0
        assert server.runtime.active_sessions == 0


def test_bundled_codex_hook_blocks_incomplete_automatic_answer():
    hook = (
        Path(__file__).parents[1]
        / "src"
        / "cortheon"
        / "codex_plugins"
        / "cortheon"
        / "hooks"
        / "cortheon_hook.py"
    )

    def run_hook(base: str, payload: dict) -> dict | None:
        completed = subprocess.run(
            [os.environ.get("PYTHON", "python3"), str(hook)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=3,
            env={**os.environ, "CORTHEON_RUNTIME_URL": base},
            check=True,
        )
        return json.loads(completed.stdout) if completed.stdout else None

    common = {"session_id": "blocked-session", "turn_id": "turn", "cwd": "/tmp"}
    with running_server() as (_server, base):
        run_hook(
            base,
            {
                **common,
                "hook_event_name": "UserPromptSubmit",
                "prompt": (
                    "Read pyproject.toml and report which console command maps to "
                    "cortheon.cognitive_cli:main."
                ),
            },
        )
        blocked = run_hook(
            base,
            {
                **common,
                "hook_event_name": "Stop",
                "last_assistant_message": "I guessed cortheon.",
            },
        )
        assert blocked["decision"] == "block"  # pyright: ignore[reportOptionalSubscript]
        assert "withheld" in blocked["reason"].lower()  # pyright: ignore[reportOptionalSubscript]

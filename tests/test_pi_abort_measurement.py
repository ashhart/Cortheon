"""Fail-closed recognition of Pi's post-abort lifecycle envelope."""

from __future__ import annotations

import pytest

from cortheon.benchmark_core.execution_provenance import execution_facts


def _assistant() -> dict:
    return {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "provider": "Local",
            "model": "small-model",
            "content": [{"type": "text", "text": "answer"}],
            "usage": {"totalTokens": 12, "cost": {"total": 0}},
            "stopReason": "stop",
        },
    }


def _terminal() -> dict:
    reason = "the host ended its bounded execution before Cortheon certified an answer"
    return {
        "type": "message_end",
        "message": {
            "role": "custom",
            "customType": "cortheon-terminal-status-v1",
            "content": (
                "[Cortheon withheld: completion was not certified]\n"
                f"The Cortheon investigation ended without a certified answer because {reason}."
            ),
            "display": True,
            "details": {
                "version": 1,
                "status": "withheld",
                "reason": reason,
                "causal": True,
            },
            "timestamp": 1,
        },
    }


def _abort(**changes) -> dict:
    message = {
        "role": "assistant",
        "content": [],
        "api": "openai-completions",
        "provider": "Local",
        "model": "small-model",
        "usage": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 0,
            "cost": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "total": 0,
            },
        },
        "stopReason": "error",
        "errorMessage": "This operation was aborted",
        "timestamp": 1,
    }
    message.update(changes)
    return {"type": "message_end", "message": message}


def test_exact_authenticated_abort_envelope_is_not_an_inference_step() -> None:
    facts = execution_facts([_assistant(), _terminal(), _abort()], host="pi")

    assert facts.steps == 1
    assert facts.identity_valid is True


@pytest.mark.parametrize(
    "poison",
    [
        _abort(stopReason="stop", errorMessage=""),
        _abort(content=[{"type": "thinking", "thinking": "hidden"}]),
        _abort(model="forged-model"),
        _abort(usage={"totalTokens": 0, "cost": {"total": 0}}),
    ],
)
def test_near_abort_envelopes_remain_counted(poison: dict) -> None:
    assert execution_facts([_assistant(), _terminal(), poison], host="pi").steps == 2


def test_only_one_immediately_adjacent_exact_abort_is_ignored() -> None:
    exact = _abort()
    second = execution_facts([_assistant(), _terminal(), exact, exact], host="pi")
    nonadjacent = execution_facts(
        [
            _assistant(),
            _terminal(),
            {"type": "message_end", "message": {"role": "toolResult"}},
            exact,
        ],
        host="pi",
    )

    assert second.steps == 2
    assert nonadjacent.steps == 2

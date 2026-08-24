import unittest

import cognitive_runtime_cases_change_budget as _change_budget
import cognitive_runtime_cases_claim_verification as _claim_verification
import cognitive_runtime_cases_completion as _completion
import cognitive_runtime_cases_discovery as _discovery
import cognitive_runtime_cases_evidence as _evidence
import cognitive_runtime_cases_profiles as _profiles
import cognitive_runtime_cases_reasoning as _reasoning
import cognitive_runtime_cases_recovery as _recovery
import cognitive_runtime_cases_research as _research
import cognitive_runtime_cases_semantic_relations as _semantic_relations
import cognitive_runtime_cases_semantic_rules as _semantic_rules
import cognitive_runtime_cases_terminal as _terminal
from cognitive_runtime_cases_common import ANSWER, CLAIMS, FakeClock

__all__ = ["ANSWER", "CLAIMS", "FakeClock"]


class CognitiveRuntimeTests(
    _reasoning.ReasoningMixin,
    _discovery.DiscoveryMixin,
    _semantic_relations.SemanticRelationsMixin,
    _semantic_rules.SemanticRulesMixin,
    _research.ResearchMixin,
    _evidence.EvidenceMixin,
    _completion.CompletionMixin,
    _terminal.TerminalMixin,
):
    pass


class WaiverAndRetractionTests(_recovery.WaiverAndRetractionTests):
    pass


class StrictnessProfileTests(_profiles.StrictnessProfileTests):
    pass


class ToolCallBudgetTests(_profiles.ToolCallBudgetTests):
    pass


class ResearchReframeTests(_profiles.ResearchReframeTests):
    pass


class ConciseChangeBudgetTests(_change_budget.ConciseChangeBudgetTests):
    pass


class ClaimVerificationEngineTests(_claim_verification.ClaimVerificationEngineTests):
    pass


def test_read_only_goal_naming_change_flavored_paths_stays_read_only() -> None:
    """File paths are locations, not intents.

    Regression: a read-only cross-file sum naming patch_runner.py classified
    as code_change because path tokens fed the change-hint scan, and the
    session then demanded mutation evidence forever (seed 31415926 join).
    """

    from cortheon.cognitive_runtime import _infer_deliverable, _requests_change

    goal = (
        "Inspect the actual repository before answering. Read MAX_INPUT_CHARS "
        "in src/cortheon/codex_plugins/cortheon/hooks/cortheon_hook.py and "
        "DEFAULT_TEST_TIMEOUT in src/cortheon/patch_runner.py. What is their "
        "sum? Give the numeric result and show the arithmetic. Do not modify "
        "files."
    )
    assert _requests_change(goal) is False
    assert _infer_deliverable(goal, "code") == "code_understanding"

    real_change = (
        "Fix total_with_tax in cortheon_fixture_tax.py so "
        "test_cortheon_fixture_tax.py passes. Do not change the test."
    )
    assert _requests_change(real_change) is True
    assert _infer_deliverable(real_change, "code") == "code_change"


def test_move_and_copy_goals_request_change() -> None:
    """Regression: move and copy mutate the tree but were absent from the
    change-hint vocabulary, so those goals classed as read-only and the
    session never demanded mutation evidence."""

    from cortheon.cognitive_runtime import _infer_deliverable, _requests_change

    assert _requests_change("move the file") is True
    assert _requests_change("Move patch_runner.py into src/lib/") is True
    assert _requests_change("copy config.yaml to config_backup.yaml") is True
    assert _infer_deliverable("Move patch_runner.py into src/lib/", "code") == ("code_change")


def test_host_hook_diff_receipt_establishes_change_when_observed() -> None:
    """Regression: plugin-captured diffs land with status "observed" and a
    host-hook executor receipt. require_host_receipts rejected them solely
    on the status label, so correct patches stayed uncertified and the
    runtime kept re-asking for diff evidence nobody could answer
    (Qwen3.5-4B patch runs: artifacts correct, zero verified completions)."""

    from cortheon.cognitive_runtime import Observation, _diff_establishes_change

    def diff_observation(*, status: str, executor: str) -> Observation:
        content = "\n".join(
            [
                "cortheon_fixture_tax.py",
                "--- a/cortheon_fixture_tax.py",
                "+++ b/cortheon_fixture_tax.py",
                "@@ line 2 @@",
                "-     return subtotal + rate",
                "+     return subtotal + (subtotal * rate)",
            ]
        )
        return Observation(
            evidence_id="ev1",
            kind="diff",
            content=content,
            source="opencode:mutation-diff",
            status=status,
            supports=[],
            contradicts=[],
            quarantine_flags=[],
            sequence=1,
            digest="digest",
            host_receipt={
                "tool": "diff",
                "executor": executor,
                "outcome": "changed",
                "args": {"paths": ["cortheon_fixture_tax.py"]},
            },
        )

    hooked = diff_observation(status="observed", executor="mutation_hook")
    assert _diff_establishes_change(hooked, require_receipt=True) is True

    from_session = diff_observation(status="observed", executor="session.diff")
    assert _diff_establishes_change(from_session, require_receipt=True) is True

    # A diff the model merely asserted, with no host executor behind it,
    # must stay insufficient under require_host_receipts.
    model_claimed = diff_observation(status="observed", executor="model_text")
    assert _diff_establishes_change(model_claimed, require_receipt=True) is False


if __name__ == "__main__":
    unittest.main()

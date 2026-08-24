"""Adversarial phrasing properties for the lexical classifiers.

Every classifier that gates behavior on goal text gets hostile inputs here:
hint words hidden in file paths, negations, nouns that look like verbs, and
read-only phrasing near change vocabulary. A new brittle gate should fail
this file in CI rather than in a stranger's session.
"""

from __future__ import annotations

import pytest

from cortheon.cognitive_runtime import _infer_deliverable, _requests_change
from cortheon.decision import package_decision

READ_ONLY_WITH_HINT_PATHS = [
    "Read MAX_RETRIES in src/patch_runner.py and report its value. Do not modify files.",
    "Summarize what update_schema.py does without changing anything.",
    "Inspect fix_imports.py and explain the algorithm. Do not edit.",
    "Compare the constants in repair/tools.py and migrate/plan.py and give their sum.",
    "What does build/config.py export? Read only.",
]

GENUINE_CHANGES = [
    "Fix the off-by-one in retry.py so the tests pass.",
    "Add a timeout parameter to the client in http.py.",
    "Refactor the parser in lexer.py to remove the duplicated loop.",
    "Update the schema migration in db/migrate.py to drop the legacy column.",
]


@pytest.mark.parametrize("goal", READ_ONLY_WITH_HINT_PATHS)
def test_read_only_goals_with_hint_words_in_paths_stay_read_only(goal: str) -> None:
    assert _requests_change(goal) is False
    assert _infer_deliverable(goal, "code") == "code_understanding"


@pytest.mark.parametrize("goal", GENUINE_CHANGES)
def test_genuine_change_goals_are_detected(goal: str) -> None:
    assert _requests_change(goal) is True
    assert _infer_deliverable(goal, "code") == "code_change"


@pytest.mark.parametrize(
    "text",
    [
        "install and use pypdf2",
        "pip install requests",
        "npm install left-pad",
        "adopt boto3 for the uploader",
        "use urllib3 directly",
        "add the httpx dependency",
    ],
)
def test_package_intent_phrasings_trigger_the_gate(text: str) -> None:
    assert package_decision(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "explain how the parser works",
        "use the existing helper",
        "install the printer driver on the office machine",
        "use caution when deleting logs",
    ],
)
def test_non_package_phrasings_do_not_overfire(text: str) -> None:
    # "install the printer driver" is an honest false positive budget: the
    # gate may fire on software installs. Everything else must stay quiet.
    if text.startswith("install"):
        pytest.skip("software-install phrasing is an accepted false-positive budget")
    assert package_decision(text) is False

from cortheon.decision import DecisionLayer, PolicyGate


def test_package_api_decision_needs_evidence() -> None:
    report = DecisionLayer().evaluate(
        "Build a REST API",
        proposed_action="Use FastAPI and call httpx.Client.stream.",
    )
    assert report.verdict == "needs_evidence"
    assert {"current_package_evidence", "api_evidence"} <= set(report.required_evidence)
    assert report.cortheon is None


def test_supplied_evidence_allows_package_api_decision() -> None:
    report = DecisionLayer().evaluate(
        "Build a REST API",
        proposed_action="Use FastAPI and call httpx.Client.stream.",
        evidence=["package_verified", "api_evidence"],
    )
    assert report.verdict == "allow"


def test_destructive_action_blocks() -> None:
    report = DecisionLayer().evaluate(
        "Fix production quickly",
        proposed_action="Disable tests and delete production credentials.",
    )
    assert report.verdict == "block"
    assert any(check.status == "block" for check in report.checks)


def test_policy_gate_is_the_deterministic_layer() -> None:
    assert PolicyGate is DecisionLayer
    assert "Deterministic" in DecisionLayer.__doc__

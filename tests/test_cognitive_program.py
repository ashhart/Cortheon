from cortheon.cognitive_program import compile_program, select_operator


def program(deliverable: str = "code_change"):
    return compile_program(
        goal="Fix the parser and verify it.",
        task_kind="code",
        deliverable=deliverable,
        effort="standard",
        requirements=[("r1", "mutation"), ("r2", "verification")],
        max_turns=9,
        max_observations=18,
    )


def test_compiler_is_deterministic_and_binds_proof_obligations():
    first = program()
    second = program()

    assert first == second
    assert first["program_id"].startswith("cp_")
    assert first["proof_obligations"] == [
        {"requirement_id": "r1", "proof": "mutation"},
        {"requirement_id": "r2", "proof": "verification"},
    ]
    assert "verify_change" in {item["operator_id"] for item in first["operators"]}
    assert first["budgets"] == {"turns": 9, "observations": 18}


def test_operator_selection_tracks_live_action_and_derivations():
    compiled = program()

    discriminating = select_operator(
        compiled,
        {
            "type": "harness_tool",
            "request": {
                "hypothesis_id": "h2",
                "capability": "read",
                "parameters": {},
            },
        },
        has_derivation=False,
        has_conflict=False,
    )
    connecting = select_operator(
        compiled,
        {"type": "verify"},
        has_derivation=True,
        has_conflict=False,
    )
    conflicting = select_operator(
        compiled,
        {"type": "verify"},
        has_derivation=True,
        has_conflict=True,
    )

    assert discriminating["operator_id"] == "discriminate"
    assert connecting["operator_id"] == "connect_sources"
    assert conflicting["operator_id"] == "resolve_contradiction"


def test_research_program_includes_freshness_and_independence():
    compiled = compile_program(
        goal="Research the current release.",
        task_kind="research",
        deliverable="research_answer",
        effort="deep",
        requirements=[("r1", "research")],
        max_turns=15,
        max_observations=32,
    )
    operators = [item["operator_id"] for item in compiled["operators"]]

    assert "establish_freshness" in operators
    assert "corroborate" in operators
    assert "resolve_contradiction" in operators

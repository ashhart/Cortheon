"""Goal decomposition.

The risk being managed is not under-splitting, it is over-splitting: fragmenting
one coherent task into pieces that each look satisfiable alone is exactly how an
agent reports completion having done a third of the work. Most of these tests
assert that a split does *not* happen.
"""

from __future__ import annotations

from cortheon.cognitive_program import compile_program
from cortheon.decomposition import decompose, max_sub_goals_for, ready_sub_goals


class TestConservatism:
    def test_a_plain_goal_is_left_whole(self) -> None:
        plan = decompose("fix the failing parser test")
        assert plan.split is False
        assert plan.strategy == "none"
        assert len(plan.sub_goals) == 1

    def test_and_joining_objects_is_not_two_tasks(self) -> None:
        # One action, two objects. Splitting this would invent work.
        plan = decompose("update the parser and the lexer")
        assert plan.split is False

    def test_and_joining_imperatives_is_two_tasks(self) -> None:
        plan = decompose("update the parser and run the integration tests")
        assert plan.split is True
        assert plan.strategy == "conjunction"
        assert len(plan.sub_goals) == 2

    def test_then_before_a_non_action_is_one_task(self) -> None:
        # "tell me" is not an imperative we act on; this is a single request.
        plan = decompose("read the config file then tell me what it does")
        assert plan.split is False

    def test_then_before_an_action_is_two_tasks(self) -> None:
        plan = decompose("read the config file then run the migration")
        assert plan.split is True
        assert plan.sequential is True

    def test_fragments_below_the_minimum_are_ignored(self) -> None:
        assert decompose("fix it and run").split is False


class TestStructure:
    def test_numbered_lists_decompose(self) -> None:
        plan = decompose(
            "Do the following:\n"
            "1. extract the retry logic into a helper\n"
            "2. add a unit test for the backoff\n"
            "3. update the module docstring"
        )
        assert plan.strategy == "enumeration"
        assert len(plan.sub_goals) == 3
        assert "retry logic" in plan.sub_goals[0].statement

    def test_bullets_decompose(self) -> None:
        plan = decompose("Work items:\n- migrate the session store\n- remove the old adapter")
        assert plan.split is True
        assert len(plan.sub_goals) == 2

    def test_ordered_markers_create_a_dependency_chain(self) -> None:
        plan = decompose(
            "First extract the parser module.\n"
            "1. extract the parser into its own module\n"
            "2. update every import site\n"
            "3. run the full test suite"
        )
        assert plan.sequential is True
        assert plan.sub_goals[1].depends_on == ("sg1",)
        assert plan.sub_goals[2].depends_on == ("sg2",)

    def test_unordered_work_has_no_dependencies(self) -> None:
        plan = decompose("Items:\n- document the cache layer\n- document the queue layer")
        assert plan.sequential is False
        assert all(goal.depends_on == () for goal in plan.sub_goals)

    def test_every_sub_goal_carries_a_proof_obligation(self) -> None:
        plan = decompose("Steps:\n1. add the flag\n2. wire it into the parser")
        assert all(goal.proof for goal in plan.sub_goals)


class TestRequirementsWin:
    def test_host_requirements_are_used_over_text_structure(self) -> None:
        plan = decompose(
            "1. one thing\n2. another thing\n3. a third thing",
            requirements=[("r1", "test passes"), ("r2", "diff applied")],
        )
        assert plan.strategy == "requirements"
        assert [goal.statement for goal in plan.sub_goals] == ["r1", "r2"]
        assert plan.sub_goals[0].proof == "test passes"

    def test_a_single_requirement_does_not_force_a_split(self) -> None:
        plan = decompose("fix the bug", requirements=[("r1", "test passes")])
        assert plan.split is False


class TestBounds:
    def test_quick_never_decomposes(self) -> None:
        plan = decompose(
            "1. first task here\n2. second task here\n3. third task here",
            effort="quick",
        )
        assert plan.split is False
        assert max_sub_goals_for("quick") == 1

    def test_deep_allows_more_parts_than_standard(self) -> None:
        assert max_sub_goals_for("deep") > max_sub_goals_for("standard")

    def test_excess_parts_are_merged_not_dropped(self) -> None:
        goal = "\n".join(f"{i}. do the numbered task {i} properly" for i in range(1, 8))
        plan = decompose(goal, effort="standard")
        assert len(plan.sub_goals) == max_sub_goals_for("standard")
        assert any("exceeded" in note for note in plan.notes)
        # Nothing may be silently lost.
        assert "7" in plan.sub_goals[-1].statement

    def test_unknown_effort_falls_back_to_standard(self) -> None:
        assert max_sub_goals_for("wildly-unknown") == max_sub_goals_for("standard")


class TestScheduling:
    def test_only_dependency_satisfied_sub_goals_are_ready(self) -> None:
        plan = decompose("read the file then run the migration")
        ready = ready_sub_goals(plan, completed=[])
        assert [goal.sub_goal_id for goal in ready] == ["sg1"]

        after = ready_sub_goals(plan, completed=["sg1"])
        assert [goal.sub_goal_id for goal in after] == ["sg2"]

    def test_independent_sub_goals_are_all_ready(self) -> None:
        plan = decompose("Items:\n- document the cache\n- document the queue")
        assert len(ready_sub_goals(plan, completed=[])) == 2


class TestCompilerIntegration:
    def _program(self, goal: str, effort: str, deliverable: str = "code_change"):
        return compile_program(
            goal=goal,
            task_kind="auto",
            deliverable=deliverable,
            effort=effort,
            requirements=[],
            max_turns=8,
            max_observations=8,
        )

    def test_effort_now_changes_the_program(self) -> None:
        quick = self._program("fix the parser", "quick")
        deep = self._program("fix the parser", "deep")
        assert [o["operator_id"] for o in quick["operators"]] != [
            o["operator_id"] for o in deep["operators"]
        ]

    def test_quick_drops_the_expensive_operators(self) -> None:
        operators = [o["operator_id"] for o in self._program("fix it", "quick")["operators"]]
        assert "challenge" not in operators
        # Verification is never dropped: a cheap task can still be answered wrong.
        assert "verify" in operators
        assert "synthesize" in operators

    def test_deep_adds_an_independent_source_lineage(self) -> None:
        operators = [o["operator_id"] for o in self._program("fix it", "deep")["operators"]]
        assert "corroborate" in operators
        assert operators.index("corroborate") < operators.index("synthesize")

    def test_decompose_is_available_but_not_yet_wired(self) -> None:
        # The operator is implemented and tested, but does not ship: the 150 KB
        # runtime budget had 63 bytes of headroom. Compiler wiring is parked
        # pending that decision, so the program must not reference it.
        program = self._program(
            "1. extract the parser module\n2. update all import sites", "standard"
        )
        assert "decompose" not in [o["operator_id"] for o in program["operators"]]
        assert decompose("1. extract the parser module\n2. update all import sites").split is True

    def test_programs_remain_content_addressed(self) -> None:
        first = self._program("fix the parser", "deep")
        second = self._program("fix the parser", "deep")
        assert first["program_id"] == second["program_id"]
        assert first["program_id"] != self._program("fix the parser", "quick")["program_id"]

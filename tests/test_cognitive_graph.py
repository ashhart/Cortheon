from cortheon.cognitive_graph import CognitiveGraph, content_id, rank_information_gain


def test_graph_exposes_provenance_path_and_stable_digest():
    graph = CognitiveGraph()
    graph.add_node("a", "fact", "Service A requires token B")
    graph.add_node("b", "fact", "Token B is issued by service C")
    graph.add_node("c", "conclusion", "Service A depends on service C")
    graph.add_edge("a", "implies", "b", evidence_id="ev1")
    graph.add_edge("b", "implies", "c", evidence_id="ev2")

    first = graph.snapshot()
    second = graph.snapshot()

    assert first == second
    assert first["digest"].startswith("cg_")
    assert graph.proof_path("a", "c") == [
        {"from": "a", "relation": "implies", "to": "b", "evidence_id": "ev1"},
        {"from": "b", "relation": "implies", "to": "c", "evidence_id": "ev2"},
    ]


def test_graph_detects_conflicting_functional_propositions():
    graph = CognitiveGraph()
    source, first = graph.proposition("release", "has_version", "1.0")
    _source, second = graph.proposition("release", "has_version", "2.0")

    snapshot = graph.snapshot()

    assert first != second
    assert snapshot["contradictions"] == [
        {
            "source": source,
            "relation": "has_version",
            "targets": sorted([first, second]),
            "evidence_ids": [],
        }
    ]


def test_information_gain_prefers_discriminating_low_cost_evidence():
    ranked = rank_information_gain(
        {"h1": 1.0, "h2": 1.0, "h3": 1.0, "h4": 1.0},
        [
            {"action_id": "confirm", "resolves": ["h1"], "cost": 1.0},
            {
                "action_id": "split",
                "partitions": [["h1", "h2"], ["h3", "h4"]],
                "cost": 0.5,
            },
            {"action_id": "expensive", "resolves": ["h1", "h2", "h3"], "cost": 10.0},
        ],
    )

    assert ranked[0]["action_id"] == "split"
    assert ranked[0]["information_gain_bits"] == 1.0
    assert ranked[0]["expected_utility"] == 2.0
    assert ranked[-1]["action_id"] == "expensive"


def test_content_ids_are_canonical():
    assert content_id("x", {"a": 1, "b": 2}) == content_id("x", {"b": 2, "a": 1})

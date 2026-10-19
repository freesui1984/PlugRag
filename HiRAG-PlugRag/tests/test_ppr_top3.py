import networkx as nx

from hirag.ppr_top3 import _build_numeric_ppr_graph, _score_from_vector_hit


def test_parallel_relation_weights_are_preserved_for_ppr():
    graph = nx.MultiGraph()
    graph.add_edge("a", "b", weight=2)
    graph.add_edge("a", "b", weight=3)
    graph.add_edge("a", "c", weight=1)

    weighted = _build_numeric_ppr_graph(graph)

    assert weighted["a"]["b"]["weight"] == 5
    assert weighted["a"]["c"]["weight"] == 1


def test_missing_vector_score_uses_rank_without_zeroing_seed():
    assert _score_from_vector_hit({"entity_name": "a"}, 0, 20) == 1
    assert _score_from_vector_hit({"entity_name": "b"}, 1, 20) == 0.95


def test_negative_or_invalid_relation_weight_does_not_bias_ppr():
    graph = nx.MultiGraph()
    graph.add_edge("a", "b", weight=-3)
    graph.add_edge("a", "b", weight="invalid")

    weighted = _build_numeric_ppr_graph(graph)

    assert weighted["a"]["b"]["weight"] == 1

import networkx as nx

from hirag.pathrag_evidence import collect_flow_pruned_paths
from hirag.pathrag_evidence_aligned import rank_aligned_entity_pool


def test_uses_all_semantic_candidates_as_path_endpoints():
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("seed_a", "middle"),
            ("middle", "late_candidate"),
            ("seed_b", "other"),
        ]
    )

    paths = collect_flow_pruned_paths(
        graph,
        ["seed_a", "seed_b", "late_candidate"],
        semantic_scores={"seed_a": 1.0, "seed_b": 0.8, "late_candidate": 0.6},
        alpha=0.9,
        theta=0.01,
        max_hops=3,
        max_paths=10,
        relation_limit=50,
    )

    assert any(path.nodes == ("seed_a", "middle", "late_candidate") for path in paths)


def test_traverses_directed_graph_as_a_retrieval_connectivity_graph():
    graph = nx.DiGraph()
    graph.add_edge("candidate_b", "bridge")
    graph.add_edge("bridge", "candidate_a")

    paths = collect_flow_pruned_paths(
        graph,
        ["candidate_a", "candidate_b"],
        semantic_scores={"candidate_a": 1.0, "candidate_b": 0.8},
        alpha=0.9,
        theta=0.01,
        max_hops=3,
        max_paths=10,
        relation_limit=50,
    )

    assert any(set(path.nodes) == {"candidate_a", "bridge", "candidate_b"} for path in paths)


def test_low_resource_branch_is_pruned_before_it_reaches_a_candidate():
    graph = nx.Graph()
    graph.add_edge("seed", "good")
    graph.add_edge("good", "target")
    for index in range(40):
        graph.add_edge("seed", f"noise_{index}")
        graph.add_edge("target", f"target_noise_{index}")

    paths = collect_flow_pruned_paths(
        graph,
        ["seed", "target"],
        semantic_scores={"seed": 1.0, "target": 0.8},
        alpha=0.8,
        theta=0.05,
        max_hops=3,
        max_paths=10,
        relation_limit=100,
    )

    assert paths == []


def test_path_results_are_limited_to_highest_reliability_paths():
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("a", "high"),
            ("high", "b"),
            ("a", "low_1"),
            ("low_1", "low_2"),
            ("low_2", "c"),
        ]
    )

    paths = collect_flow_pruned_paths(
        graph,
        ["a", "b", "c"],
        semantic_scores={"a": 1.0, "b": 0.9, "c": 0.7},
        alpha=0.9,
        theta=0.01,
        max_hops=3,
        max_paths=1,
        relation_limit=50,
    )

    assert len(paths) == 1
    assert set(paths[0].nodes) == {"a", "high", "b"}


def test_aligned_pool_locks_three_semantic_entities_then_promotes_flow_entities():
    ranked = rank_aligned_entity_pool(
        semantic_names=["s1", "s2", "s3", "s4", "s5"],
        semantic_scores={"s1": 1.0, "s2": 0.9, "s3": 0.8, "s4": 0.7, "s5": 0.6},
        flow_scores={"x1": 0.95, "x2": 0.85, "s4": 0.8, "x3": 0.3},
        lock_top_n=3,
        priority_flow_n=2,
        entity_pool_size=8,
    )

    assert ranked[:5] == ["s1", "s2", "s3", "x1", "x2"]
    assert len(ranked) == len(set(ranked)) == 8


def test_aligned_pool_never_promotes_locked_entities_as_flow_slots():
    ranked = rank_aligned_entity_pool(
        semantic_names=["s1", "s2", "s3", "s4"],
        semantic_scores={"s1": 1.0, "s2": 0.9, "s3": 0.8, "s4": 0.7},
        flow_scores={"s1": 0.99, "s2": 0.98, "x1": 0.9, "x2": 0.8},
        lock_top_n=3,
        priority_flow_n=2,
        entity_pool_size=6,
    )

    assert ranked[:5] == ["s1", "s2", "s3", "x1", "x2"]

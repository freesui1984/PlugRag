"""Deterministic relation-agnostic graph projection, extracted from the experiment workspace."""
from typing import Any
from collections import defaultdict
import networkx as nx
from ._utils import split_string_by_multi_markers
from .prompt import GRAPH_FIELD_SEP

def _single_edge_key(source: Any, target: Any) -> tuple[str, str]:
    """Return a deterministic undirected edge key without changing node IDs."""
    source_text, target_text = str(source), str(target)
    return (
        (source_text, target_text)
        if source_text <= target_text
        else (target_text, source_text)
    )

def _coerce_edge_weight(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0

def build_single_graph_from_multi_graph(
    multi_graph: nx.MultiGraph,
) -> tuple[nx.Graph, list[dict[str, Any]]]:
    """Project a relation-aware MultiGraph into a relation-agnostic Graph.

    Parallel edges between the same entities are merged deterministically:
    weights are summed; descriptions and source chunks are deduplicated; the
    earliest extraction order is retained.  ``relation_type`` is intentionally
    excluded from the resulting graph, while retained in the returned audit.
    This function is pure and does not make any model or embedding request.
    """
    from ._op import _merge_edge_attributes

    if not multi_graph.is_multigraph():
        raise ValueError("Multi-to-Single projection requires an nx.MultiGraph.")

    single_graph = nx.Graph()
    single_graph.graph.update(dict(multi_graph.graph))
    single_graph.add_nodes_from(
        (node_id, dict(attributes))
        for node_id, attributes in multi_graph.nodes(data=True)
    )

    grouped_edges: dict[tuple[str, str], list[tuple[Any, Any, Any, dict[str, Any]]]] = (
        defaultdict(list)
    )
    for source, target, edge_key, attributes in multi_graph.edges(keys=True, data=True):
        grouped_edges[_single_edge_key(source, target)].append(
            (source, target, edge_key, dict(attributes))
        )

    audit_rows: list[dict[str, Any]] = []
    for pair in sorted(grouped_edges):
        entries = grouped_edges[pair]
        source, target = entries[0][0], entries[0][1]
        normalized_attributes = []
        relation_types: set[str] = set()
        source_ids: set[str] = set()

        for _, _, _, attributes in entries:
            normalized = dict(attributes)
            normalized["weight"] = _coerce_edge_weight(normalized.get("weight", 1.0))
            normalized_attributes.append(normalized)

            relation_type = str(attributes.get("relation_type", "")).strip()
            if relation_type:
                relation_types.add(relation_type)
            source_ids.update(
                filter(
                    None,
                    split_string_by_multi_markers(
                        str(attributes.get("source_id", "")), [GRAPH_FIELD_SEP]
                    ),
                )
            )

        merged_attributes = _merge_edge_attributes(normalized_attributes)
        single_graph.add_edge(source, target, **merged_attributes)
        audit_rows.append(
            {
                "source": str(source),
                "target": str(target),
                "multi_edge_count": len(entries),
                "multi_edge_keys": sorted(str(edge_key) for _, _, edge_key, _ in entries),
                "relation_types_removed": sorted(relation_types),
                "multi_total_weight": merged_attributes["weight"],
                "single_weight": merged_attributes["weight"],
                "source_chunk_count": len(source_ids),
            }
        )

    return single_graph, audit_rows

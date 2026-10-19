"""Same-graph PathRAG-style retrieval with path-derived evidence.

This is an isolated comparison baseline.  It shares HiRAG's pre-built graph,
entity vector index, community reports, and source chunks, but does not invoke
R-GCN expansion, PlugRag reranking, or PlugRag traceback paths.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import networkx as nx

from .ppr_top3 import _score_from_vector_hit
from ._utils import logger, split_string_by_multi_markers, truncate_list_by_token_size
from .base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage, QueryParam
from .prompt import GRAPH_FIELD_SEP, PROMPTS


DEFAULT_CONFIG = {
    "candidate_top_k": 20,
    "alpha": 0.85,
    "theta": 0.005,
    "max_hops": 4,
    "max_paths": 15,
    "relation_limit": 200,
}


@dataclass(frozen=True)
class FlowPath:
    nodes: tuple[str, ...]
    score: float
    source: str
    target: str


def resolve_config(global_config: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config.update((global_config or {}).get("pathrag_evidence") or {})
    config["candidate_top_k"] = max(2, int(config["candidate_top_k"]))
    config["alpha"] = min(1.0, max(0.0, float(config["alpha"])))
    config["theta"] = max(0.0, float(config["theta"]))
    config["max_hops"] = max(1, int(config["max_hops"]))
    config["max_paths"] = max(1, int(config["max_paths"]))
    config["relation_limit"] = max(1, int(config["relation_limit"]))
    return config


def _edge_attributes(graph: nx.Graph, source: str, target: str) -> dict[str, Any]:
    data = graph.get_edge_data(source, target)
    if data is None:
        data = graph.get_edge_data(target, source)
    if not data:
        return {}
    if graph.is_multigraph():
        values = [value for value in data.values() if isinstance(value, dict)]
        if values:
            return max(values, key=lambda item: float(item.get("weight", 1) or 1))
    return data if isinstance(data, dict) else {}


def _neighbors(graph: nx.Graph, node: str, relation_limit: int) -> list[str]:
    """Traverse a directed KG as a connectivity graph, retaining edge metadata."""
    if graph.is_directed():
        names = set(graph.successors(node)) | set(graph.predecessors(node))
    else:
        names = set(graph.neighbors(node))

    records: list[tuple[str, float]] = []
    for neighbor in names:
        data = _edge_attributes(graph, node, neighbor)
        try:
            weight = float(data.get("weight", 1.0) or 1.0)
        except (TypeError, ValueError):
            weight = 1.0
        records.append((str(neighbor), max(weight, 0.0)))
    records.sort(key=lambda item: (-item[1], item[0]))
    return [name for name, _ in records[:relation_limit]]


def _recover_path(
    parents: dict[tuple[int, str], str | None], depth: int, target: str
) -> tuple[str, ...]:
    recovered: list[str] = []
    current: str | None = target
    while current is not None and depth >= 0:
        recovered.append(current)
        current = parents.get((depth, current))
        depth -= 1
    return tuple(reversed(recovered))


def _paths_from_source(
    graph: nx.Graph,
    source: str,
    candidates: set[str],
    semantic_scores: dict[str, float],
    *,
    alpha: float,
    theta: float,
    max_hops: int,
    relation_limit: int,
) -> list[FlowPath]:
    initial_resource = max(float(semantic_scores.get(source, 0.0)), 1.0e-6)
    frontier: dict[str, float] = {source: initial_resource}
    resources: dict[int, dict[str, float]] = {0: {source: initial_resource}}
    parents: dict[tuple[int, str], str | None] = {(0, source): None}
    seen_nodes = {source}
    paths: list[FlowPath] = []

    for depth in range(max_hops):
        next_resources: defaultdict[str, float] = defaultdict(float)
        best_parent: dict[str, tuple[float, str]] = {}
        for current, mass in frontier.items():
            neighbors = [name for name in _neighbors(graph, current, relation_limit) if name not in seen_nodes]
            if not neighbors:
                continue
            share = mass * alpha / len(neighbors)
            if share < theta:
                continue
            for neighbor in neighbors:
                next_resources[neighbor] += share
                if neighbor not in best_parent or share > best_parent[neighbor][0]:
                    best_parent[neighbor] = (share, current)

        next_frontier = {name: mass for name, mass in next_resources.items() if mass >= theta}
        if not next_frontier:
            break
        next_depth = depth + 1
        resources[next_depth] = next_frontier
        for name in next_frontier:
            parents[(next_depth, name)] = best_parent[name][1]
            if name in candidates and name != source:
                nodes = _recover_path(parents, next_depth, name)
                if len(nodes) < 2:
                    continue
                mean_resource = sum(
                    resources[index].get(node, 0.0) for index, node in enumerate(nodes)
                ) / len(nodes)
                endpoint_score = (semantic_scores.get(source, 0.0) + semantic_scores.get(name, 0.0)) / 2
                paths.append(FlowPath(nodes, mean_resource * endpoint_score, source, name))
        seen_nodes.update(next_frontier)
        frontier = next_frontier
    return paths


def collect_flow_pruned_paths(
    graph: nx.Graph,
    candidate_names: list[str],
    *,
    semantic_scores: dict[str, float],
    alpha: float = 0.85,
    theta: float = 0.005,
    max_hops: int = 4,
    max_paths: int = 15,
    relation_limit: int = 200,
) -> list[FlowPath]:
    """Retrieve reliable paths between all semantically retrieved candidates."""
    candidates = [name for name in dict.fromkeys(candidate_names) if name in graph]
    candidate_set = set(candidates)
    best_paths: dict[tuple[str, ...], FlowPath] = {}
    for source in candidates:
        for path in _paths_from_source(
            graph,
            source,
            candidate_set,
            semantic_scores,
            alpha=alpha,
            theta=theta,
            max_hops=max_hops,
            relation_limit=relation_limit,
        ):
            key = min(path.nodes, tuple(reversed(path.nodes)))
            previous = best_paths.get(key)
            if previous is None or path.score > previous.score:
                best_paths[key] = path
    return sorted(
        best_paths.values(), key=lambda item: (-item.score, len(item.nodes), item.nodes)
    )[:max_paths]


def _csv(rows: list[list[Any]]) -> str:
    stream = io.StringIO()
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue().rstrip("\n")


def _path_text(graph: nx.Graph, path: FlowPath) -> str:
    parts = []
    for index, name in enumerate(path.nodes):
        node = graph.nodes.get(name, {}) or {}
        description = str(node.get("description", "")).strip() or "No entity description available."
        parts.append(f"{name}: {description}")
        if index < len(path.nodes) - 1:
            target = path.nodes[index + 1]
            edge = _edge_attributes(graph, name, target)
            relation = edge.get("relation_type", edge.get("label", "RELATED"))
            edge_description = str(edge.get("description", "")).strip()
            parts.append(f"--[{relation}: {edge_description}]-->")
    return " ".join(parts)


async def _path_communities(
    node_data: list[dict[str, Any]],
    node_scores: dict[str, float],
    community_reports: BaseKVStorage,
    query_param: QueryParam,
    limit: int,
) -> list[dict[str, Any]]:
    scores: defaultdict[str, float] = defaultdict(float)
    for node in node_data:
        try:
            clusters = json.loads(node.get("clusters", "[]"))
        except (TypeError, json.JSONDecodeError):
            continue
        for cluster in clusters:
            if cluster.get("level", 0) <= query_param.level:
                scores[str(cluster.get("cluster"))] += node_scores.get(node["entity_name"], 0.0)
    report_ids = sorted(scores, key=lambda item: (-scores[item], item))[: max(limit * 4, limit)]
    reports = await asyncio.gather(*[community_reports.get_by_id(item) for item in report_ids])
    valid = [
        (report_id, report)
        for report_id, report in zip(report_ids, reports)
        if report
    ]
    valid.sort(
        key=lambda item: (
            -scores[item[0]],
            -float(item[1].get("report_json", {}).get("rating", 0) or 0),
        )
    )
    selected = truncate_list_by_token_size(
        [report for _, report in valid],
        key=lambda item: item.get("report_string", ""),
        max_token_size=query_param.max_token_for_community_report,
    )
    return selected[:1] if query_param.community_single_one else selected


async def _path_source_units(
    node_data: list[dict[str, Any]],
    node_scores: dict[str, float],
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
) -> list[dict[str, Any]]:
    chunk_scores: defaultdict[str, float] = defaultdict(float)
    for node in node_data:
        for chunk_id in split_string_by_multi_markers(str(node.get("source_id", "")), [GRAPH_FIELD_SEP]):
            if chunk_id:
                chunk_scores[chunk_id] += node_scores.get(node["entity_name"], 0.0)
    chunk_ids = sorted(chunk_scores, key=lambda item: (-chunk_scores[item], item))
    chunks = await asyncio.gather(*[text_chunks_db.get_by_id(item) for item in chunk_ids])
    rows = [
        {"id": chunk_id, "data": chunk, "path_score": chunk_scores[chunk_id]}
        for chunk_id, chunk in zip(chunk_ids, chunks)
        if chunk
    ]
    return truncate_list_by_token_size(
        rows,
        key=lambda item: item["data"].get("content", ""),
        max_token_size=query_param.max_token_for_text_unit,
    )


async def build_pathrag_evidence_context(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, Any],
) -> str | None:
    graph = getattr(knowledge_graph_inst, "_graph", None)
    if graph is None or graph.number_of_nodes() == 0:
        return None
    config = resolve_config(global_config)
    results = await entities_vdb.query(query, top_k=config["candidate_top_k"])
    candidates: list[dict[str, Any]] = []
    semantic_scores: dict[str, float] = {}
    for rank, hit in enumerate(results):
        name = str(hit.get("entity_name", "")).strip()
        if not name or name in semantic_scores or name not in graph:
            continue
        node = await knowledge_graph_inst.get_node(name)
        if node is None:
            continue
        semantic_scores[name] = _score_from_vector_hit(hit, rank, len(results))
        candidates.append({**node, "entity_name": name, "semantic_rank": rank})
        if len(candidates) >= config["candidate_top_k"]:
            break
    if len(candidates) < 2:
        return None

    paths = collect_flow_pruned_paths(
        graph,
        [item["entity_name"] for item in candidates],
        semantic_scores=semantic_scores,
        alpha=config["alpha"],
        theta=config["theta"],
        max_hops=config["max_hops"],
        max_paths=config["max_paths"],
        relation_limit=config["relation_limit"],
    )
    selected_paths = paths
    if not selected_paths:
        return None

    node_scores: defaultdict[str, float] = defaultdict(float)
    for path in selected_paths:
        for name in path.nodes:
            node_scores[name] = max(node_scores[name], path.score)
    nodes_by_name = {item["entity_name"]: item for item in candidates}
    node_names = sorted(node_scores, key=lambda item: (-node_scores[item], item))
    node_data: list[dict[str, Any]] = []
    for name in node_names:
        node = nodes_by_name.get(name) or await knowledge_graph_inst.get_node(name)
        if node:
            node_data.append({**node, "entity_name": name})

    communities = await _path_communities(node_data, node_scores, community_reports, query_param, limit=20)
    source_units = await _path_source_units(node_data, node_scores, text_chunks_db, query_param)

    path_rows = [["id", "reliability", "source", "target", "hops", "path"]]
    for index, path in enumerate(sorted(selected_paths, key=lambda item: item.score)):
        path_rows.append([index, f"{path.score:.6f}", path.source, path.target, len(path.nodes) - 1, _path_text(graph, path)])
    entity_rows = [["id", "entity", "type", "description", "path_score", "semantic_rank"]]
    for index, node in enumerate(node_data):
        entity_rows.append([index, node["entity_name"], node.get("entity_type", "UNKNOWN"), node.get("description", ""), f"{node_scores[node['entity_name']]:.6f}", node.get("semantic_rank", "path")])
    community_rows = [["id", "content"]] + [[index, report.get("report_string", "")] for index, report in enumerate(communities)]
    source_rows = [["id", "content"]] + [[index, item["data"].get("content", "")] for index, item in enumerate(source_units)]

    context = f"""
-----Backgrounds-----
```csv
{_csv(community_rows)}
```
-----Reasoning Path-----
```csv
{_csv(path_rows)}
```
-----Detail Entity Information-----
```csv
{_csv(entity_rows)}
```
-----Source Documents-----
```csv
{_csv(source_rows)}
```
"""
    logger.info(
        "PathRAG evidence context: candidates=%d paths=%d path_nodes=%d communities=%d sources=%d",
        len(candidates), len(selected_paths), len(node_data), len(communities), len(source_units),
    )
    return context


async def pathrag_evidence_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, Any],
) -> str:
    context = await build_pathrag_evidence_context(
        query, knowledge_graph_inst, entities_vdb, community_reports, text_chunks_db, query_param, global_config
    )
    if query_param.only_need_context:
        return context or ""
    if context is None:
        return PROMPTS["fail_response"]
    system_prompt = PROMPTS["local_rag_response"].format(
        context_data=context,
        response_type=query_param.response_type,
    )
    return await global_config["best_model_func"](query, system_prompt=system_prompt)

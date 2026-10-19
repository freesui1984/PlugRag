"""PathRAG-style retrieval with a PlugRag-aligned entity evidence budget.

This comparison mode keeps the same semantic seed count and entity-pool size
as PlugRag full, while replacing only R-GCN expansion and traceback retrieval
with query-conditioned flow propagation and parent-pointer path recovery.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import networkx as nx

from .ppr_top3 import _score_from_vector_hit
from ._utils import logger, split_string_by_multi_markers, truncate_list_by_token_size
from .base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage, QueryParam
from .pathrag_evidence import _csv, _edge_attributes, _neighbors
from .prompt import GRAPH_FIELD_SEP, PROMPTS


DEFAULT_CONFIG = {
    "candidate_top_k": 20,
    "entity_pool_size": 40,
    "lock_top_n": 3,
    "priority_flow_n": 2,
    "display_top_k": 20,
    "alpha": 0.85,
    "theta": 0.005,
    "max_hops": 4,
    "max_paths": 15,
    "relation_limit": 200,
}


@dataclass(frozen=True)
class FlowTrace:
    nodes: tuple[str, ...]
    score: float
    source: str
    target: str


def resolve_config(global_config: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config.update((global_config or {}).get("pathrag_evidence_aligned") or {})
    config["candidate_top_k"] = max(2, int(config["candidate_top_k"]))
    config["entity_pool_size"] = max(config["candidate_top_k"], int(config["entity_pool_size"]))
    config["lock_top_n"] = max(1, min(int(config["lock_top_n"]), config["candidate_top_k"]))
    config["priority_flow_n"] = max(0, int(config["priority_flow_n"]))
    config["display_top_k"] = max(1, min(int(config["display_top_k"]), config["entity_pool_size"]))
    config["alpha"] = min(1.0, max(0.0, float(config["alpha"])))
    config["theta"] = max(0.0, float(config["theta"]))
    config["max_hops"] = max(1, int(config["max_hops"]))
    config["max_paths"] = max(1, int(config["max_paths"]))
    config["relation_limit"] = max(1, int(config["relation_limit"]))
    return config


def _normalise(scores: dict[str, float]) -> dict[str, float]:
    maximum = max(scores.values(), default=0.0)
    if maximum <= 0:
        return {name: 0.0 for name in scores}
    return {name: max(0.0, score) / maximum for name, score in scores.items()}


def rank_aligned_entity_pool(
    *,
    semantic_names: list[str],
    semantic_scores: dict[str, float],
    flow_scores: dict[str, float],
    lock_top_n: int,
    priority_flow_n: int,
    entity_pool_size: int,
) -> list[str]:
    """Lock semantic anchors, then place flow expansions where context is decided."""
    semantic_names = list(dict.fromkeys(semantic_names))
    locked = semantic_names[:lock_top_n]
    locked_set = set(locked)
    semantic_set = set(semantic_names)

    flow_only = [name for name in flow_scores if name not in locked_set and name not in semantic_set]
    flow_only.sort(key=lambda name: (-flow_scores[name], name))
    priority_flow = flow_only[:priority_flow_n]
    selected = locked + priority_flow
    selected_set = set(selected)

    semantic_norm = _normalise({name: semantic_scores.get(name, 0.0) for name in semantic_names})
    flow_norm = _normalise(flow_scores)
    remainder = list(dict.fromkeys(semantic_names + list(flow_scores)))
    remainder = [name for name in remainder if name not in selected_set]
    remainder.sort(
        key=lambda name: (
            -(0.5 * semantic_norm.get(name, 0.0) + 0.5 * flow_norm.get(name, 0.0)),
            name,
        )
    )
    return (selected + remainder)[:entity_pool_size]


def collect_flow_expansion_traces(
    graph: nx.Graph,
    candidate_names: list[str],
    *,
    semantic_scores: dict[str, float],
    alpha: float,
    theta: float,
    max_hops: int,
    relation_limit: int,
) -> dict[str, FlowTrace]:
    """Propagate query-conditioned resource and retain the best parent trace per node."""
    candidates = [name for name in dict.fromkeys(candidate_names) if name in graph]
    traces: dict[str, FlowTrace] = {}
    for source in candidates:
        initial = max(float(semantic_scores.get(source, 0.0)), 1.0e-6)
        frontier: dict[str, float] = {source: initial}
        parents: dict[str, str | None] = {source: None}
        paths: dict[str, tuple[str, ...]] = {source: (source,)}
        visited = {source}
        for _depth in range(max_hops):
            next_resource: defaultdict[str, float] = defaultdict(float)
            best_parent: dict[str, tuple[float, str]] = {}
            for current, mass in frontier.items():
                neighbors = [name for name in _neighbors(graph, current, relation_limit) if name not in visited]
                if not neighbors:
                    continue
                share = mass * alpha / len(neighbors)
                if share < theta:
                    continue
                for neighbor in neighbors:
                    next_resource[neighbor] += share
                    if neighbor not in best_parent or share > best_parent[neighbor][0]:
                        best_parent[neighbor] = (share, current)
            frontier = {name: mass for name, mass in next_resource.items() if mass >= theta}
            if not frontier:
                break
            for name, mass in frontier.items():
                parent = best_parent[name][1]
                parents[name] = parent
                paths[name] = paths[parent] + (name,)
                score = mass * initial
                previous = traces.get(name)
                if previous is None or score > previous.score:
                    traces[name] = FlowTrace(paths[name], score, source, name)
            visited.update(frontier)
    return traces


async def _load_nodes(
    names: list[str], candidates_by_name: dict[str, dict[str, Any]], knowledge_graph_inst: BaseGraphStorage
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for name in names:
        node = candidates_by_name.get(name) or await knowledge_graph_inst.get_node(name)
        if node:
            nodes.append({**node, "entity_name": name})
    return nodes


async def _communities_from_entities(
    node_data: list[dict[str, Any]],
    scores: dict[str, float],
    community_reports: BaseKVStorage,
    query_param: QueryParam,
) -> list[dict[str, Any]]:
    cluster_scores: defaultdict[str, float] = defaultdict(float)
    for node in node_data:
        try:
            clusters = json.loads(node.get("clusters", "[]"))
        except (TypeError, json.JSONDecodeError):
            continue
        for cluster in clusters:
            if cluster.get("level", 0) <= query_param.level:
                cluster_scores[str(cluster.get("cluster"))] += scores.get(node["entity_name"], 0.0)
    report_ids = sorted(cluster_scores, key=lambda item: (-cluster_scores[item], item))
    reports = await asyncio.gather(*[community_reports.get_by_id(item) for item in report_ids])
    rows = [report for report in reports if report]
    rows = truncate_list_by_token_size(
        rows,
        key=lambda item: item.get("report_string", ""),
        max_token_size=query_param.max_token_for_community_report,
    )
    return rows[:1] if query_param.community_single_one else rows


async def _sources_from_entities(
    node_data: list[dict[str, Any]],
    scores: dict[str, float],
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
) -> list[dict[str, Any]]:
    chunk_scores: defaultdict[str, float] = defaultdict(float)
    for node in node_data:
        for chunk_id in split_string_by_multi_markers(str(node.get("source_id", "")), [GRAPH_FIELD_SEP]):
            if chunk_id:
                chunk_scores[chunk_id] += scores.get(node["entity_name"], 0.0)
    chunk_ids = sorted(chunk_scores, key=lambda item: (-chunk_scores[item], item))
    chunks = await asyncio.gather(*[text_chunks_db.get_by_id(item) for item in chunk_ids])
    rows = [{"id": chunk_id, "data": chunk} for chunk_id, chunk in zip(chunk_ids, chunks) if chunk]
    return truncate_list_by_token_size(
        rows,
        key=lambda item: item["data"].get("content", ""),
        max_token_size=query_param.max_token_for_text_unit,
    )


def _path_rows(graph: nx.Graph, traces: dict[str, FlowTrace], pool_names: list[str], max_paths: int) -> list[list[Any]]:
    selected = [traces[name] for name in pool_names if name in traces and len(traces[name].nodes) > 1]
    selected.sort(key=lambda item: (-item.score, len(item.nodes), item.nodes))
    rows = [["id", "flow_score", "source_seed", "target", "hops", "path"]]
    seen = set()
    for trace in selected:
        if trace.nodes in seen:
            continue
        seen.add(trace.nodes)
        edges = []
        for source, target in zip(trace.nodes, trace.nodes[1:]):
            edge = _edge_attributes(graph, source, target)
            relation = edge.get("relation_type", edge.get("label", "RELATED"))
            description = str(edge.get("description", "")).strip()
            edges.append(f"{source} --[{relation}: {description}]--> {target}")
        rows.append([len(rows) - 1, f"{trace.score:.6f}", trace.source, trace.target, len(trace.nodes) - 1, " ".join(edges)])
        if len(rows) - 1 >= max_paths:
            break
    return rows


async def build_pathrag_evidence_aligned_context(
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

    traces = collect_flow_expansion_traces(
        graph,
        [item["entity_name"] for item in candidates],
        semantic_scores=semantic_scores,
        alpha=config["alpha"],
        theta=config["theta"],
        max_hops=config["max_hops"],
        relation_limit=config["relation_limit"],
    )
    flow_scores = {name: trace.score for name, trace in traces.items()}
    pool_names = rank_aligned_entity_pool(
        semantic_names=[item["entity_name"] for item in candidates],
        semantic_scores=semantic_scores,
        flow_scores=flow_scores,
        lock_top_n=config["lock_top_n"],
        priority_flow_n=config["priority_flow_n"],
        entity_pool_size=config["entity_pool_size"],
    )
    candidates_by_name = {item["entity_name"]: item for item in candidates}
    pool_nodes = await _load_nodes(pool_names, candidates_by_name, knowledge_graph_inst)
    if not pool_nodes:
        return None

    semantic_norm = _normalise(semantic_scores)
    flow_norm = _normalise(flow_scores)
    pool_scores = {
        name: 1.0 - index / max(1, len(pool_names)) + 0.1 * flow_norm.get(name, 0.0)
        for index, name in enumerate(pool_names)
    }
    communities = await _communities_from_entities(pool_nodes, pool_scores, community_reports, query_param)
    source_units = await _sources_from_entities(pool_nodes, pool_scores, text_chunks_db, query_param)

    display_nodes = pool_nodes[:config["display_top_k"]]
    entity_rows = [["id", "entity", "type", "description", "entity_source", "semantic_rank", "flow_score", "path_support"]]
    semantic_names = set(semantic_scores)
    for index, node in enumerate(display_nodes):
        name = node["entity_name"]
        entity_rows.append([
            index,
            name,
            node.get("entity_type", "UNKNOWN"),
            node.get("description", ""),
            "semantic_seed" if name in semantic_names else "flow_expansion",
            node.get("semantic_rank", ""),
            f"{flow_scores.get(name, 0.0):.6f}",
            "true" if name in traces else "false",
        ])
    path_rows = _path_rows(graph, traces, pool_names, config["max_paths"])
    community_rows = [["id", "content"]] + [[index, item.get("report_string", "")] for index, item in enumerate(communities)]
    source_rows = [["id", "content"]] + [[index, item["data"].get("content", "")] for index, item in enumerate(source_units)]
    logger.info(
        "Aligned PathRAG context: seeds=%d pool=%d display=%d paths=%d communities=%d sources=%d",
        len(candidates), len(pool_nodes), len(display_nodes), len(path_rows) - 1, len(communities), len(source_units),
    )
    return f"""
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


async def pathrag_evidence_aligned_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, Any],
) -> str:
    context = await build_pathrag_evidence_aligned_context(
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

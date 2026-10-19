import networkx as nx

from ._op import (
    _build_local_query_context,
    _find_most_related_community_from_entities,
    _find_most_related_text_unit_from_entities,
    timer,
)
from ._utils import logger, list_of_list_to_csv, truncate_list_by_token_size
from .base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage, CommunitySchema, QueryParam, TextChunkSchema
from .prompt import PROMPTS


def _get_ppr_config(global_config: dict) -> dict:
    cfg = {}
    cfg.update(global_config.get("addon_params", {}).get("ppr_graph_retrieval", {}) or {})
    cfg.update(global_config.get("ppr_graph_retrieval", {}) or {})
    return {
        "seed_multiplier": int(cfg.get("seed_multiplier", 5)),
        "damping": float(cfg.get("damping", 0.85)),
        "max_iter": int(cfg.get("max_iter", 100)),
        "tol": float(cfg.get("tol", 1.0e-6)),
        "min_seed_weight": float(cfg.get("min_seed_weight", 1.0e-6)),
        "relation_limit": int(cfg.get("relation_limit", 200)),
        "lock_top_n": int(cfg.get("lock_top_n", 3)),
        "semantic_weight": float(cfg.get("semantic_weight", 0.75)),
        "ppr_weight": float(cfg.get("ppr_weight", 0.25)),
        "degree_penalty": float(cfg.get("degree_penalty", 0.03)),
    }


def _score_from_vector_hit(hit: dict, rank: int, total: int) -> float:
    for key in ("score", "similarity", "cosine", "distance"):
        if key not in hit:
            continue
        try:
            value = float(hit[key])
        except Exception:
            continue
        if key == "distance":
            return max(1.0 - value, 0.0)
        return max(value, 0.0)
    return 1.0 - (rank / max(total, 1))


def _build_numeric_ppr_graph(graph: nx.Graph) -> nx.Graph:
    ppr_graph = nx.Graph()
    ppr_graph.add_nodes_from(graph.nodes())
    if graph.is_multigraph():
        edge_iter = graph.edges(data=True, keys=True)
        for src, tgt, _key, data in edge_iter:
            if src == tgt:
                continue
            try:
                weight = float((data or {}).get("weight", 1.0))
            except Exception:
                weight = 1.0
            old_weight = ppr_graph[src][tgt]["weight"] if ppr_graph.has_edge(src, tgt) else 0.0
            ppr_graph.add_edge(src, tgt, weight=old_weight + max(weight, 0.0))
    else:
        for src, tgt, data in graph.edges(data=True):
            if src == tgt:
                continue
            try:
                weight = float((data or {}).get("weight", 1.0))
            except Exception:
                weight = 1.0
            old_weight = ppr_graph[src][tgt]["weight"] if ppr_graph.has_edge(src, tgt) else 0.0
            ppr_graph.add_edge(src, tgt, weight=old_weight + max(weight, 0.0))
    return ppr_graph


async def _find_ppr_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
) -> list[dict]:
    graph = getattr(knowledge_graph_inst, "_graph", None)
    if graph is None:
        return []
    entity_names = [node.get("entity_name") for node in node_datas if node.get("entity_name")]
    seen = set()
    edges = []
    for name in entity_names:
        if name not in graph:
            continue
        if graph.is_multigraph():
            incident_edges = graph.edges(name, keys=True)
        else:
            incident_edges = graph.edges(name)
        for edge in incident_edges:
            if graph.is_multigraph():
                src, tgt, edge_key = edge
                edge_data = await knowledge_graph_inst.get_edge(src, tgt, key=edge_key)
                fingerprint = (src, tgt, edge_key)
            else:
                src, tgt = edge
                edge_data = await knowledge_graph_inst.get_edge(src, tgt)
                fingerprint = tuple(sorted((src, tgt)))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            if edge_data is None:
                edge_data = {}
            try:
                weight = float(edge_data.get("weight", 1.0))
            except Exception:
                weight = 1.0
            try:
                rank = await knowledge_graph_inst.edge_degree(src, tgt)
            except Exception:
                rank = 0
            edges.append(
                {
                    "src_tgt": (src, tgt),
                    "relation_type": edge_data.get("relation_type", edge_data.get("label", "RELATED")),
                    "description": edge_data.get("description", ""),
                    "weight": weight,
                    "rank": rank,
                }
            )
    edges.sort(key=lambda item: (item.get("rank", 0), item.get("weight", 0.0)), reverse=True)
    relation_limit = int(
        knowledge_graph_inst.global_config.get("ppr_graph_retrieval", {}).get(
            "relation_limit",
            knowledge_graph_inst.global_config.get("addon_params", {})
            .get("ppr_graph_retrieval", {})
            .get("relation_limit", 200),
        )
    )
    if relation_limit > 0:
        edges = edges[:relation_limit]
    return truncate_list_by_token_size(
        edges,
        key=lambda x: x.get("description", ""),
        max_token_size=query_param.max_token_for_local_context,
    )



async def _build_ppr_top3_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    """Clean PPR retrieval baseline.

    Only the semantic top-3 entities are used as personalization seeds. All
    remaining selected entities come from graph propagation, so this mode
    compares PPR-style entity retrieval against PlugRag/full without reusing
    PlugRag's R-GCN expansion or reranker.
    """
    cfg = _get_ppr_config(global_config)
    lock_top_n = max(1, min(int(cfg.get("lock_top_n", 3)), query_param.top_k))
    seed_results = await entities_vdb.query(query, top_k=max(query_param.top_k, lock_top_n))
    if not seed_results:
        return None

    raw_graph = getattr(knowledge_graph_inst, "_graph", None)
    if raw_graph is None or raw_graph.number_of_nodes() == 0:
        return await _build_local_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
        )
    graph = _build_numeric_ppr_graph(raw_graph)

    locked_hits = []
    locked_names = []
    for rank, hit in enumerate(seed_results):
        name = hit.get("entity_name")
        if not name or name not in graph or name in locked_names:
            continue
        locked_hits.append((rank, hit))
        locked_names.append(name)
        if len(locked_names) >= lock_top_n:
            break

    if not locked_names:
        return await _build_local_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
        )

    total_hits = len(seed_results)
    personalization = {}
    for rank, hit in locked_hits:
        name = hit.get("entity_name")
        weight = _score_from_vector_hit(hit, rank, total_hits)
        if weight > cfg["min_seed_weight"]:
            personalization[name] = max(personalization.get(name, 0.0), weight)
    if not personalization:
        personalization = {name: 1.0 for name in locked_names}

    weight_sum = sum(personalization.values()) or 1.0
    personalization = {name: weight / weight_sum for name, weight in personalization.items()}
    try:
        ppr_scores = nx.pagerank(
            graph,
            alpha=cfg["damping"],
            personalization=personalization,
            max_iter=cfg["max_iter"],
            tol=cfg["tol"],
            weight="weight",
        )
    except Exception as exc:
        logger.warning(f"Top3-PPR retrieval failed, falling back to local HiRAG retrieval: {exc}")
        return await _build_local_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
        )

    ranked_ppr_names = [
        name
        for name, _score in sorted(ppr_scores.items(), key=lambda item: item[1], reverse=True)
        if name not in locked_names
    ]
    ranked_names = (locked_names + ranked_ppr_names)[: query_param.top_k]

    semantic_scores = {
        hit.get("entity_name"): _score_from_vector_hit(hit, rank, total_hits)
        for rank, hit in enumerate(seed_results)
        if hit.get("entity_name")
    }

    node_datas = []
    for name in ranked_names:
        node = await knowledge_graph_inst.get_node(name)
        if node is None:
            continue
        degree = await knowledge_graph_inst.node_degree(name)
        node_datas.append(
            {
                **node,
                "entity_name": name,
                "rank": degree,
                "_ppr_score": float(ppr_scores.get(name, 0.0)),
                "_semantic_score": float(semantic_scores.get(name, 0.0)),
                "_ppr_source": "semantic_top3" if name in locked_names else "ppr_propagation",
            }
        )

    if not node_datas:
        return None

    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_ppr_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    missing_edge_desc = sum(1 for edge in use_relations if not edge.get("description"))

    logger.info(
        "====== References (Top3-Seeded PPR Retrieval) ======\n"
        f"Locked semantic entities: {locked_names}\n"
        f"PPR entities ({len(node_datas)}): {[n['entity_name'] for n in node_datas]}\n"
        f"Communities: {len(use_communities)}, Relations: {len(use_relations)}, "
        f"Relations without description: {missing_edge_desc}, Chunks: {len(use_text_units)}\n"
    )

    entities_section_list = [["id", "entity", "type", "description", "rank", "semantic_score", "ppr_score", "source"]]
    for i, n in enumerate(node_datas):
        entities_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
                f"{n.get('_semantic_score', 0.0):.6f}",
                f"{n.get('_ppr_score', 0.0):.6f}",
                n.get("_ppr_source", "ppr_propagation"),
            ]
        )
    entities_context = list_of_list_to_csv(entities_section_list)

    reasoning_path_section_list = [["id", "source", "target", "relation_type", "description", "weight", "rank"]]
    for i, e in enumerate(use_relations):
        src, tgt = e.get("src_tgt", ("", ""))
        reasoning_path_section_list.append(
            [
                i,
                src,
                tgt,
                e.get("relation_type", "UNKNOWN"),
                e.get("description", ""),
                e.get("weight", 1.0),
                e.get("rank", 0),
            ]
        )
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)

    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"].replace("\n", " ")])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return f"""
-----Backgrounds-----
```csv
{communities_context}
```
-----Reasoning Path-----
```csv
{reasoning_path_context}
```
-----Detail Entity Information-----
```csv
{entities_context}
```
-----Source Documents-----
```csv
{text_units_context}
```
"""


async def ppr_graph_top3_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    use_model_func = global_config["best_model_func"]
    with timer():
        context = await _build_ppr_top3_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
            global_config,
        )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response

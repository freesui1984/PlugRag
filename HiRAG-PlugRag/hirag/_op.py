import re
import os
import json
import asyncio
import tiktoken
import networkx as nx
import time
import logging
import torch
import numpy as np
from contextlib import contextmanager
from typing import Union
from collections import Counter, defaultdict
from ._splitter import SeparatorSplitter
from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
import asyncio
# ... 其他原本的 import ...

# 🔥 新增这一行：导入你刚才写的工具文件
# PlugRag path tracing and verification helpers.
from .plugrag_path_utils import find_path_with_required_nodes_detailed, save_hirag_verification_log,save_rgcn_metrics_log,extract_data_snapshot
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
    CommunitySchema,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from ._cluster_utils import Hierarchical_Clustering


_RGCN_STATE = {"names": None, "emb": None}

def _load_rgcn_state(global_config: dict):
    if not global_config.get("rgcn_enabled"):
        return None
    ef = global_config.get("rgcn_embeddings_file")
    mf = global_config.get("rgcn_mapping_file")
    if not ef or not mf:
        return None
    if _RGCN_STATE["emb"] is None:
        try:
            d = torch.load(ef, map_location="cpu")
            e = d.get("embeddings")
            if isinstance(e, torch.Tensor):
                e = e.detach().cpu().numpy()
            with open(mf, "r", encoding="utf-8") as f:
                m = json.load(f)
            idx_to_node = m.get("idx_to_node") or {}
            names = []
            for i in range(len(e)):
                v = idx_to_node.get(str(i), idx_to_node.get(i))
                names.append(v if isinstance(v, str) else None)
            _RGCN_STATE["names"] = names
            _RGCN_STATE["emb"] = e
        except Exception:
            _RGCN_STATE["names"] = None
            _RGCN_STATE["emb"] = None
    return _RGCN_STATE if _RGCN_STATE["emb"] is not None else None

# async def _rgcn_expand_entities(node_datas: list[dict], knowledge_graph_inst: BaseGraphStorage, query_param: QueryParam, global_config: dict) -> tuple[list[dict], list[tuple[str,str,float,str]]]:
#     state = _load_rgcn_state(global_config)
#     if state is None:
#         return node_datas, []
#     names = state["names"]
#     emb = state["emb"]
#     base_names = [n["entity_name"] for n in node_datas[:query_param.top_k]]
#     name_to_idx = {n: i for i, n in enumerate(names) if isinstance(n, str)}
#     sims_rows = []
#     added = {}
#     for bn in base_names:
#         idx = name_to_idx.get(bn)
#         if idx is None:
#             continue
#         try:
#             v = emb[idx]
#             s = emb @ v
#         except Exception:
#             continue
#         idxs = np.argsort(-s)[:global_config.get("rgcn_neighbors_k", 10)]
#         for j in idxs:
#             nn = names[j]
#             if not isinstance(nn, str) or nn == bn:
#                 continue
#             sim = float(s[j])
#             sims_rows.append((bn, nn, sim, "RGCN"))
#             if nn in added:
#                 continue
#             added[nn] = True
#     added_names = list(added.keys())
#     if len(added_names) > global_config.get("rgcn_cluster_limit", 50):
#         added_names = added_names[:global_config.get("rgcn_cluster_limit", 50)]
#     return_node_datas = node_datas.copy()
#     if added_names:
#         nodes = await asyncio.gather(*[knowledge_graph_inst.get_node(n) for n in added_names])
#         degs = await asyncio.gather(*[knowledge_graph_inst.node_degree(n) for n in added_names])
        
#         extra = [{**nd, "entity_name": nm, "rank": dg} for nm, nd, dg in zip(added_names, nodes, degs) if nd is not None]
#         exist = {d["entity_name"] for d in return_node_datas}
#         for x in extra:
#             if x["entity_name"] in exist:
#                 continue
#             return_node_datas.append(x)
#     return return_node_datas, sims_rows
async def _rgcn_expand_entities(node_datas: list[dict], knowledge_graph_inst: BaseGraphStorage, query_param: QueryParam, global_config: dict) -> tuple[list[dict], list[tuple[str,str,float,str]]]:
    
    # [检查点 1]：确认函数是否被调用
    print("\n=================== [DEBUG START] ===================", flush=True)
    print(f"[DEBUG 1] 进入 _rgcn_expand_entities 函数", flush=True)

    state = _load_rgcn_state(global_config)
    if state is None:
        print(f"[DEBUG END] R-GCN 状态未加载 (state is None)。可能 rgcn_enabled=False 或文件路径错误。", flush=True)
        return node_datas, []
    
    print(f"[DEBUG 2] R-GCN 状态加载成功。Names count: {len(state['names'])}", flush=True)

    names = state["names"]
    emb = state["emb"]
    base_names = [n["entity_name"] for n in node_datas[:query_param.top_k]]
    
    print(f"[DEBUG 3] 基础实体 (Seeds): {base_names}", flush=True)

    name_to_idx = {n: i for i, n in enumerate(names) if isinstance(n, str)}
    sims_rows = []
    added = {}
    
    for bn in base_names:
        idx = name_to_idx.get(bn)
        if idx is None:
            # print(f"[DEBUG] 警告：基础实体 '{bn}' 不在 R-GCN 映射表中", flush=True)
            continue
        try:
            v = emb[idx]
            s = emb @ v
        except Exception as e:
            print(f"[DEBUG] 向量计算出错: {e}", flush=True)
            continue
            
        idxs = np.argsort(-s)[:global_config.get("rgcn_neighbors_k", 10)]
        for j in idxs:
            nn = names[j]
            if not isinstance(nn, str) or nn == bn:
                continue
            sim = float(s[j])
            sims_rows.append((bn, nn, sim, "RGCN"))
            if nn in added:
                continue
            added[nn] = True
            
    added_names = list(added.keys())
    print(f"[DEBUG 4] R-GCN 推荐了 {len(added_names)} 个新实体", flush=True)
    
    if len(added_names) > global_config.get("rgcn_cluster_limit", 50):
        added_names = added_names[:global_config.get("rgcn_cluster_limit", 50)]
        
    return_node_datas = node_datas.copy()
    
    if added_names:
        # 1. 尝试去图谱中获取详细信息
        nodes = await asyncio.gather(*[knowledge_graph_inst.get_node(n) for n in added_names])
        degs = await asyncio.gather(*[knowledge_graph_inst.node_degree(n) for n in added_names])
        
        # ==============================================================================
        # 🕵️‍♂️ [排查幽灵节点] 使用 print 强制输出
        # ==============================================================================
        found_ghost = False
        for name, node_data in zip(added_names, nodes):
            # 如果 node_data 是 None，说明图谱里没找到这个节点
            if node_data is None:
                found_ghost = True
                print(f"🚨 [严重警告] 图谱中找不到实体: {repr(name)}", flush=True)
                print(f"   -> 名字长度: {len(name)}", flush=True)

                # 尝试暴力修复测试：去掉两边的引号看看？
                clean_name = name.strip('"')
                
                # 只有当去掉引号后名字确实变了，才去试
                if clean_name != name:
                    test_node = await knowledge_graph_inst.get_node(clean_name)
                    if test_node:
                        print(f"   -> ✅ 破案了！图谱里存的是不带引号的: '{clean_name}'", flush=True)
                        print(f"   -> 建议：请检查 R-GCN 的 mapping.json 是否多存了引号。", flush=True)
                    else:
                        print(f"   -> ❌ 去掉引号后依然找不到。可能是图谱文件版本不一致（读取了旧缓存）。", flush=True)
        
        if not found_ghost:
            print(f"[DEBUG 5] 完美！所有推荐的实体都在图谱中找到了。", flush=True)
        # ==============================================================================

        # 2. 过滤掉 None (幽灵节点)，只保留合法的
        extra = [{**nd, "entity_name": nm, "rank": dg} for nm, nd, dg in zip(added_names, nodes, degs) if nd is not None]
        
        print(f"[DEBUG 6] 实际加入 overall 的实体数: {len(extra)} / {len(added_names)}", flush=True)

        exist = {d["entity_name"] for d in return_node_datas}
        for x in extra:
            if x["entity_name"] in exist:
                continue
            return_node_datas.append(x)
    
    print("=================== [DEBUG END] ===================\n", flush=True)        
    return return_node_datas, sims_rows
@contextmanager
def timer():
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logging.info(f"[Retrieval Time: {elapsed_time:.6f} seconds]")


def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):
  # tokenizer
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):

            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def chunking_by_seperators(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):

    splitter = SeparatorSplitter(
        separators=[
            tiktoken_model.encode(s) for s in PROMPTS["default_text_separator"]
        ],
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_token]

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def get_chunks(new_docs, chunk_func=chunking_by_token_size, **chunk_func_params):
    inserting_chunks = {}

    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]

    ENCODER = tiktoken.encoding_for_model("gpt-4o")
    tokens = ENCODER.encode_batch(docs, num_threads=16)
    chunks = chunk_func(
        tokens, doc_keys=doc_keys, tiktoken_model=ENCODER, **chunk_func_params
    )

    for chunk in chunks:
        inserting_chunks.update(
            {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
        )

    return inserting_chunks


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """Summarize the entity or relation description,is used during entity extraction and when merging nodes or edges in the knowledge graph

    Args:
        entity_or_relation_name: entity or relation name
        description: description
        global_config: global configuration
    """
    use_llm_func: callable = global_config["cheap_model_func"]
    llm_max_tokens = global_config["cheap_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 6 or record_attributes[0] != '"relationship"':
        return None
    # # add this record as edge
    # source = clean_str(record_attributes[1].upper())
    # target = clean_str(record_attributes[2].upper())
    # edge_description = clean_str(record_attributes[3])
    # edge_source_id = chunk_key
    # weight = (
    #     float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    # )
    # return dict(
    #     src_id=source,
    #     tgt_id=target,
    #     weight=weight,
    #     description=edge_description,
    #     source_id=edge_source_id,
    # )
    # 正确地从 record_attributes[3] 获取关系类型
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    relation_type = clean_str(record_attributes[3])
    
    # 正确地从 record_attributes[4] 获取关系描述
    edge_description = clean_str(record_attributes[4])
    
    # 正确地从 record_attributes[5] 获取关系强度/权重
    weight_str = record_attributes[5]
    weight = float(weight_str) if is_float_regex(weight_str) else 1.0
    
    edge_source_id = chunk_key

    # 返回结构清晰的字典
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
        relation_type=relation_type,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:                                            # already exist
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knwoledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data
# # 导入必要的库
# import numpy as np
# from sklearn.cluster import KMeans
# from typing import Dict, List

# # 添加到文件的适当位置（比如在_merge_edges_then_upsert函数之前）
# async def cluster_and_summarize_relation_types(
#     relation_types: List[str],
#     embedding_func,
#     llm_func,
#     max_clusters: int = None,  # 改为可选参数
#     global_config: dict = None
# ) -> Dict[str, str]:
#     """
#     对关系类型进行聚类并使用大模型生成语义总结
#     """
#     # 如果没有传入max_clusters，则从全局配置中获取
#     if max_clusters is None and global_config is not None:
#         max_clusters = global_config.get("max_clusters", 10)  # 默认值为10
#     elif max_clusters is None:
#         max_clusters = 10  # 如果没有全局配置，使用默认值
#     # 去重处理
#     unique_relation_types = list(set(relation_types))
#     logger.info(f"开始聚类关系类型，共 {len(unique_relation_types)} 种唯一类型")
    
#     if len(unique_relation_types) <= 1:
#         # 如果只有一种或没有关系类型，直接返回原类型
#         logger.info(f"只有 {len(unique_relation_types)} 种关系类型，不执行聚类")
#         return {rt: rt for rt in relation_types}
    
#     # 确定聚类数量，最多50个簇
#     n_clusters = min(len(unique_relation_types), max_clusters)
#     logger.info(f"使用 {n_clusters} 个簇进行聚类")
    
#     # 生成关系类型的向量嵌入 - 添加分批处理逻辑
#     try:
#         # 定义批次大小 (API限制为64)
#         batch_size = 64
#         embeddings = []
        
#         # 分批处理
#         for i in range(0, len(unique_relation_types), batch_size):
#             batch = unique_relation_types[i:i + batch_size]
#             print(f"[处理] 正在生成第 {i//batch_size + 1}/{(len(unique_relation_types) + batch_size - 1)//batch_size} 批嵌入向量，批次大小: {len(batch)}")
#             batch_embeddings = await embedding_func(batch)
#             embeddings.extend(batch_embeddings)
            
#         if len(embeddings) != len(unique_relation_types):
#             raise ValueError(f"嵌入向量数量与关系类型数量不匹配: {len(embeddings)} vs {len(unique_relation_types)}")
#         print(f"[成功] 成功生成 {len(embeddings)} 个嵌入向量")
#     except Exception as e:
#         print(f"[错误] 生成嵌入时出错: {str(e)}")
#         # 出错时返回原关系类型
#         return {rt: rt for rt in relation_types}
    
#     # 使用K-means进行聚类
#     try:
#         from sklearn.cluster import KMeans
#         kmeans = KMeans(n_clusters=n_clusters, random_state=224)
#         clusters = kmeans.fit_predict(embeddings)
#         logger.info(f"成功完成K-means聚类")
#     except Exception as e:
#         logger.error(f"聚类时出错: {str(e)}")
#         # 出错时返回原关系类型
#         return {rt: rt for rt in relation_types}
    
#     # 为每个簇收集关系类型
#     cluster_to_relations = defaultdict(list)
#     for i, rt in enumerate(unique_relation_types):
#         cluster_to_relations[clusters[i]].append(rt)
    
#     # 为每个簇生成语义总结
#     relation_to_summary = {}
#     for cluster_id, relations in cluster_to_relations.items():
#         prompt = f"""
#         You are an expert Knowledge Graph construction AI. Your task is to analyze a list of similar or semantically related relationship types, and summarize them into a **single canonical relationship type** that best represents their shared meaning. This canonical relation should be suitable for unified knowledge base queries and reasoning.

#         ---

#         ### **Strict Output Rules**
#         1. **Output only the final canonical relationship name.** Do not include any explanation, reasoning, or extra text.
#         2. The canonical name must be written in **UPPER_SNAKE_CASE**.
#         3. The canonical name must be a **verb or verb-based phrase** that captures the essential action or relationship between two entities.
#         4. The resulting relationship should be **broad enough** to encompass all the listed relations, yet **specific enough** to be meaningful in query contexts (e.g., “Who LEADS Microsoft?”).

#         ---

#         ### **Examples**

#         **[Example 1]**
#         Raw Relations List: "IS_CEO_OF", "MANAGES", "HEADS", "SUPERVISES", "CONTROLS"  
#         Canonical Relation: LEADS

#         **[Example 2]**
#         Raw Relations List: "FATHER_OF", "MOTHER_OF", "BROTHER_OF", "SPOUSE_OF", "CHILD_OF"  
#         Canonical Relation: HAS_FAMILY_RELATIONSHIP_WITH

#         **[Example 3]**
#         Raw Relations List: "INVESTED_IN", "ACQUIRED_BY", "OWNS", "PARTNERED_WITH"  
#         Canonical Relation: HAS_BUSINESS_RELATIONSHIP_WITH

#         **[Example 4]**
#         Raw Relations List: "LOCATED_IN", "RESIDES_IN", "WAS_BORN_IN", "TRAVELS_TO", "WORKS_AT"  
#         Canonical Relation: IS_LOCATED_IN

#         ---

#         **[Task]**
#         Raw Relations List: {', '.join(relations)}     
#         Canonical Relation:
#         """
#         try:
#             # 使用大模型生成总结
#             summary = await llm_func(prompt)
#             summary = summary.strip()
            
#             # 处理可能的前缀
#             if '类别名称：' in summary:
#                 summary = summary.split('类别名称：')[-1].strip()
            
#             # 移除可能的括号内容
#             import re
#             summary = re.sub(r'\(.*?\)', '', summary)
#             summary = summary.strip()
            
#             # 确保结果不为空
#             if not summary:  # 如果生成失败，使用第一个关系类型作为默认值
#                 summary = relations[0]
            
#             logger.debug(f"簇 {cluster_id} 的关系类型: {relations} -> 总结: {summary}")
#         except Exception as e:
#             logger.error(f"生成簇总结时出错: {str(e)}")
#             # 出错时使用第一个关系类型作为默认值
#             summary = relations[0]
        
#         # 更新映射关系
#         for rt in relations:
#             relation_to_summary[rt] = summary
    
#     logger.info(f"成功完成关系类型聚类，生成 {len(relation_to_summary)} 个映射关系")
#     return relation_to_summary

# async def _merge_edges_then_upsert(
#     src_id: str,
#     tgt_id: str,
#     edges_data: list[dict],
#     knwoledge_graph_inst: BaseGraphStorage,
#     global_config: dict,
#     global_relation_map: dict = None
#     ) -> None:
#     """
#     合并多条边然后插入到图数据库中
    
#     Args:
#         src_id: 源节点ID
#         tgt_id: 目标节点ID
#         edges_data: 边数据列表
#         knwoledge_graph_inst: 图数据库实例
#         global_config: 全局配置
#         global_relation_map: 全局关系类型映射，可选
#     """
#     already_weights = []
#     already_source_ids = []
#     already_description = []
#     already_order = []
#     already_relation_types = []
#     if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
#         already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
#         already_weights.append(already_edge["weight"])
#         already_source_ids.extend(
#             split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
#         )
#         already_description.append(already_edge["description"])
#         already_order.append(already_edge.get("order", 1))
#         already_relation_types.append(already_edge.get("relation_type", "RELATED"))     
#     # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
#     order = min([dp.get("order", 1) for dp in edges_data] + already_order)
#     weight = sum([dp["weight"] for dp in edges_data] + already_weights)
#     description = GRAPH_FIELD_SEP.join(
#         sorted(set([dp["description"] for dp in edges_data] + already_description))
#     )
#     source_id = GRAPH_FIELD_SEP.join(
#         set([dp["source_id"] for dp in edges_data] + already_source_ids)
#     )
#     # 处理关系类型：选择出现频率最高的类型
#     all_relation_types = [dp.get("relation_type", "RELATED") for dp in edges_data] + already_relation_types
#     relation_type = Counter(all_relation_types).most_common(1)[0][0] if all_relation_types else "RELATED"
    
#     # 应用全局关系类型映射（修复缩进错误）
#     if global_relation_map and relation_type in global_relation_map:
#         original_relation_type = relation_type
#         relation_type = global_relation_map[relation_type]
#         logger.debug(f"将关系类型 '{original_relation_type}' 映射为 '{relation_type}'")
#     elif global_relation_map:
#         logger.debug(f"关系类型 '{relation_type}' 在全局映射中未找到，保持不变")
#     else:
#         logger.debug(f"未提供全局关系类型映射")

#     for need_insert_id in [src_id, tgt_id]:
#         if not (await knwoledge_graph_inst.has_node(need_insert_id)):
#             await knwoledge_graph_inst.upsert_node(
#                 need_insert_id,
#                 node_data={
#                     "source_id": source_id,
#                     "description": description,
#                     "entity_type": '"UNKNOWN"',
#                 },
#             )
#     description = await _handle_entity_relation_summary(
#         (src_id, tgt_id), description, global_config
#     )
#     await knwoledge_graph_inst.upsert_edge(
#         src_id,
#         tgt_id,
#         edge_data=dict(
#             weight=weight, description=description, source_id=source_id, order=order, relation_type=relation_type,
#         ),
#     )

# async def _merge_edges_then_upsert(
#     src_id: str,
#     tgt_id: str,
#     edges_data: list[dict],
#     knwoledge_graph_inst: BaseGraphStorage,
#     global_config: dict,
#     global_relation_map: dict = None
# ) -> None:
#     """
#     智能合并边并插入到图数据库中。
    
#     核心逻辑：
#     1. 先应用全局关系类型映射，将相似的关系类型统一
#     2. 对于映射后相同的关系类型，合并其属性（权重、描述等）
#     3. 对于不同的关系类型，创建独立的平行边
#     4. 支持多图存储，每种关系类型都有独立的边
    
#     Args:
#         src_id: 源节点ID
#         tgt_id: 目标节点ID
#         edges_data: 边数据列表
#         knwoledge_graph_inst: 图数据库实例
#         global_config: 全局配置
#         global_relation_map: 全局关系类型映射字典，用于统一相似的关系类型
#     """
#     if not edges_data:
#         return 
    
#     # 1. 应用全局关系类型映射，统一相似的关系类型
#     mapped_edges = []
#     for edge in edges_data:
#         original_type = edge.get("relation_type", "RELATED")
#         # 应用全局映射，如果没有映射则保持原类型
#         mapped_type = global_relation_map.get(original_type, original_type) if global_relation_map else original_type
        
#         # 创建新的边数据，使用映射后的关系类型
#         mapped_edge = edge.copy()
#         mapped_edge["relation_type"] = mapped_type
#         mapped_edge["original_relation_type"] = original_type  # 保留原始类型用于追溯
#         mapped_edges.append(mapped_edge)
    
#     # 2. 按映射后的关系类型对边进行分组
#     grouped_edges = defaultdict(list)
#     for edge in mapped_edges:
#         relation_type = edge["relation_type"]
#         grouped_edges[relation_type].append(edge)
    
#     # 3. 获取现有的所有边数据，用于智能合并
#     existing_edges_by_type = {}
#     if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
#         all_existing_edges = await knwoledge_graph_inst.get_edge_data(src_id, tgt_id)
#         if all_existing_edges:
#             for edge_key, edge_data in all_existing_edges.items():
#                 existing_relation_type = edge_data.get("relation_type", "RELATED")
#                 if existing_relation_type not in existing_edges_by_type:
#                     existing_edges_by_type[existing_relation_type] = []
#                 existing_edges_by_type[existing_relation_type].append((edge_key, edge_data))
    
#     # 4. 对每种映射后的关系类型分别处理
#     for relation_type, edge_group in grouped_edges.items():
#         # 收集新边的属性
#         new_weights = [edge.get("weight", 1.0) for edge in edge_group]
#         new_source_ids = []
#         new_descriptions = []
#         new_orders = [edge.get("order", 1) for edge in edge_group]
        
#         for edge in edge_group:
#             source_id = edge.get("source_id", "")
#             if source_id:
#                 new_source_ids.extend(split_string_by_multi_markers(source_id, [GRAPH_FIELD_SEP]))
            
#             description = edge.get("description", "")
#             if description:
#                 new_descriptions.append(description)
        
#         # 检查是否存在相同关系类型的边，如果存在则合并
#         existing_weights = []
#         existing_source_ids = []
#         existing_descriptions = []
#         existing_orders = []
#         existing_edge_key = None
        
#         if relation_type in existing_edges_by_type:
#             # 找到第一个相同关系类型的边进行合并（如果有多个相同类型的边，合并到第一个）
#             existing_edge_key, existing_edge_data = existing_edges_by_type[relation_type][0]
            
#             existing_weights.append(existing_edge_data.get("weight", 1.0))
#             existing_orders.append(existing_edge_data.get("order", 1))
            
#             existing_source_id = existing_edge_data.get("source_id", "")
#             if existing_source_id:
#                 existing_source_ids.extend(split_string_by_multi_markers(existing_source_id, [GRAPH_FIELD_SEP]))
            
#             existing_description = existing_edge_data.get("description", "")
#             if existing_description:
#                 existing_descriptions.append(existing_description)
        
#         # 合并属性
#         final_weight = sum(new_weights + existing_weights)
#         final_order = min(new_orders + existing_orders) if (new_orders + existing_orders) else 1
        
#         # 去重并合并source_id
#         all_source_ids = list(set(new_source_ids + existing_source_ids))
#         final_source_id = GRAPH_FIELD_SEP.join(filter(None, all_source_ids))
        
#         # 去重并合并描述
#         all_descriptions = list(set(new_descriptions + existing_descriptions))
#         final_description = GRAPH_FIELD_SEP.join(filter(None, sorted(all_descriptions)))
        
#         # 确保源节点和目标节点存在
#         for node_id in [src_id, tgt_id]:
#             if not await knwoledge_graph_inst.has_node(node_id):
#                 await knwoledge_graph_inst.upsert_node(
#                     node_id,
#                     node_data={
#                         "source_id": final_source_id,
#                         "description": final_description,
#                         "entity_type": '"UNKNOWN"',
#                     },
#                 )
        
#         # 处理关系描述摘要
#         if final_description:
#             final_description = await _handle_entity_relation_summary(
#                 (src_id, tgt_id), final_description, global_config
#             )
        
#         # 构建最终的边数据
#         edge_data = {
#             "weight": final_weight,
#             "description": final_description,
#             "source_id": final_source_id,
#             "order": final_order,
#             "relation_type": relation_type,
#         }
        
#         # 如果存在相同关系类型的边，更新它；否则创建新边
#         if existing_edge_key is not None:
#             # 更新现有边
#             await knwoledge_graph_inst.upsert_edge(
#                 src_id, tgt_id, edge_data=edge_data, key=existing_edge_key
#             )
#         else:
#             # 创建新边，使用关系类型作为key确保不同类型的边独立存在
#             await knwoledge_graph_inst.upsert_edge(
#                 src_id, tgt_id, edge_data=edge_data, key=relation_type
#             )
import asyncio
import logging
import re
from collections import defaultdict
from typing import List, Dict, Any, Optional

# 在您的文件顶部设置一个 logger
logger = logging.getLogger(__name__)

# ==============================================================================
# 辅助函数: 专门负责合并边属性 (这个函数无需修改，它工作得很好)
# ==============================================================================
def _merge_edge_attributes(edges_to_merge: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    将一个边属性字典列表合并成一个单一的字典。
    """
    if not edges_to_merge:
        return {}

    # 合并权重 (求和)
    final_weight = sum(e.get("weight", 1.0) for e in edges_to_merge)

    # 合并描述 (去重、排序、拼接)
    all_descs = {e.get("description", "") for e in edges_to_merge}
    final_description = GRAPH_FIELD_SEP.join(sorted(list(filter(None, all_descs))))

    # --- ✨ 新增且完整的 source_id 合并逻辑 ✨ ---
    all_sources = set()
    for e in edges_to_merge:
        # 先获取 source_id 字符串，然后用分隔符拆分成列表
        sources = split_string_by_multi_markers(e.get("source_id", ""), [GRAPH_FIELD_SEP])
        # 将拆分后的ID更新到集合中，实现自动去重
        all_sources.update(filter(None, sources))
    # 将所有不重复的ID排序后，用分隔符重新拼接成最终的字符串
    final_source_id = GRAPH_FIELD_SEP.join(sorted(list(all_sources)))
    # --- ✨ 逻辑结束 ✨ ---

    # 合并排序值 (取最小值)
    final_order = min(e.get("order", 1) for e in edges_to_merge)

    return {
        "weight": final_weight,
        "description": final_description,
        "source_id": final_source_id,
        "order": final_order,
    }


# ==============================================================================
# 主函数: (⭐️ [已修改] 移除了全局映射逻辑 ⭐️)
# ==============================================================================
async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: List[Dict[str, Any]], # 从 LLM 抽取器传来的新边
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,#为什么全局没有
    # [修改] 移除 global_relation_map 参数，我们不再需要它
    # global_relation_map: Optional[Dict[str, str]] = None 
) -> None:
    """
    针对 MultiGraph，严格遵循“按类型分组”的逻辑，合并或创建平行边。
    此版本已移除全局映射，直接使用边上定义的 relation_type。
    """
    if not edges_data:
        return
    
    # [修改] 不再使用 "canonical_type" (规范类型)
    # 而是直接按边上自带的 "relation_type" 分组
    new_edges_grouped_by_type = defaultdict(list)
    for edge in edges_data:
        # 直接从边数据中获取 relation_type
        # 如果 LLM 抽取失败没有提供，则使用一个默认值
        relation_type = edge.get("relation_type", "RELATED") 
        new_edges_grouped_by_type[relation_type].append(edge)
    
    
    logger.info(f"[调试] 实体 ({src_id}, {tgt_id}) 的新边被分组为: {list(new_edges_grouped_by_type.keys())}")

    
    # 这部分逻辑保持不变，它工作得很好
    # 它会获取图谱中 (src, tgt) 之间所有已存在的平行边，并按类型分组
    existing_edges_by_type = defaultdict(list)
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        all_parallel_edges = await knwoledge_graph_inst.get_edge_data(src_id, tgt_id) or {}
        
        logger.info(f"[调试] 从图中为 ({src_id}, {tgt_id}) 读取到的原始 all_parallel_edges: {all_parallel_edges}")
        
        if isinstance(all_parallel_edges, dict):
            for key, edge_attrs in all_parallel_edges.items():
                if isinstance(edge_attrs, dict):
                    edge_type = edge_attrs.get("relation_type")
                    if edge_type:
                        existing_edges_by_type[edge_type].append({"key": key, "data": edge_attrs})

    
    if existing_edges_by_type:
        logger.info(f"[调试] 整理后，已存在的边按类型分组为: {list(existing_edges_by_type.keys())}")
    else:
        logger.info(f"[调试] 整理后，实体 ({src_id}, {tgt_id}) 之间没有任何已存在的、格式正确的边。")
    

    # [修改] 遍历的变量名从 canonical_type 改为 relation_type，逻辑更清晰
    for relation_type, new_edges_list in new_edges_grouped_by_type.items():
        
        # 检查是否已存在 *同类型* 的边
        if relation_type in existing_edges_by_type:
            
            # [合并逻辑]：已存在此类型（例如 IS_A），合并新旧 IS_A 边
            logger.info(f"[合并与清理] 实体 ({src_id}, {tgt_id}) 类型为 '{relation_type}' 的边...")
            
            existing_edges = existing_edges_by_type[relation_type]
            existing_data_list = [edge_info["data"] for edge_info in existing_edges]
            
            # 合并所有 *已存在* 的此类型边和所有 *新传来* 的此类型边
            all_edges_to_merge = existing_data_list + new_edges_list
            merged_attributes = _merge_edge_attributes(all_edges_to_merge)
            merged_attributes["relation_type"] = relation_type # 确保关系类型被正确设置
            
            # 更新第一条已存在的边
            update_key = existing_edges[0]["key"]
            await knwoledge_graph_inst.upsert_edge(src_id, tgt_id, edge_data=merged_attributes, key=update_key)
            
            # [清理逻辑] 如果之前图谱中存在多条同类型的平行边，删除多余的
            delete_tasks = []
            for edge_info in existing_edges[1:]:
                delete_tasks.append(knwoledge_graph_inst.remove_edge(src_id, tgt_id, key=edge_info["key"]))
            if delete_tasks:
                await asyncio.gather(*delete_tasks)

        else:
            
            # [新建逻辑]：不存在此类型（例如 SON_OF），创建一条新的平行边
            logger.info(f"[新建] 实体 ({src_id}, {tgt_id}) 不存在类型为 '{relation_type}' 的边，创建新平行边...")
            
            # 只合并新传入的边 (可能 LLM 在同一个 chunk 里找到了 2 条 SON_OF)
            merged_attributes = _merge_edge_attributes(new_edges_list)
            merged_attributes["relation_type"] = relation_type
            
            # [关键] 使用 relation_type (例如 "SON_OF") 本身作为新平行边的唯一 key
            # 这能确保 (A, B, SON_OF) 和 (A, B, MEMBER_OF_SAME_CLAN) 作为两条独立的边共存
            await knwoledge_graph_inst.upsert_edge(src_id, tgt_id, edge_data=merged_attributes, key=relation_type)
# extract entities with normal and attribute entities
async def extract_hierarchical_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
)-> Union[BaseGraphStorage, None]:
    """Extract entities and relations from text chunks

    Args:
        chunks: text chunks
        knowledge_graph_inst: knowledge graph instance
        entity_vdb: entity vector database
        global_config: global configuration

    Returns:
        Union[BaseGraphStorage, None]: knowledge graph instance
    """
    use_llm_func: callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    entity_extract_prompt = PROMPTS["hi_entity_extraction"]        # give 3 examples in the prompt context
    relation_extract_prompt = PROMPTS["hi_relation_extraction"]

    context_base_entity = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["META_ENTITY_TYPES"])
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]     # means low quality in the last extraction
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]       # judge if there are still entities still need to be extracted

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content_entity(chunk_key_dp: tuple[str, TextChunkSchema]):           # for each chunk, run the func
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base_entity, input_text=content)      # fill in the parameter
        final_result = await use_llm_func(hint_prompt)                                      # feed into LLM with the prompt

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)               # set as history
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)      # add to history
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(                                       # judge if we still need the next iteration
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(                                            # split entities from result --> list of entities
            final_result,
            [context_base_entity["record_delimiter"], context_base_entity["completion_delimiter"]],
        )
        # resolve the entities
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(          # split entity
                record, [context_base_entity["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(       # get the name, type, desc, source_id of entity--> dict
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1                                      # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][                     # for visualization
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)
    
    # extract entities
    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    entity_results = await asyncio.gather(
        *[_process_single_content_entity(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar

    # fetch all entities from results
    all_entities = {}
    for item in entity_results:
        for k, v in item[0].items():
            value = v[0]
            all_entities[k] = v[0]
    context_entities = {key[0]: list(x[0].keys()) for key, x in zip(ordered_chunks, entity_results)}
    
    # fetch embeddings
    entity_discriptions = [v["description"] for k, v in all_entities.items()]
    entity_sequence_embeddings = []
    embeddings_batch_size = 64
    num_embeddings_batches = (len(entity_discriptions) + embeddings_batch_size - 1) // embeddings_batch_size
    for i in range(num_embeddings_batches):
        start_index = i * embeddings_batch_size
        end_index = min((i + 1) * embeddings_batch_size, len(entity_discriptions))
        batch = entity_discriptions[start_index:end_index]
        result = await entity_vdb.embedding_func(batch)
        entity_sequence_embeddings.extend(result)
    entity_embeddings = entity_sequence_embeddings
    for (k, v), x in zip(all_entities.items(), entity_embeddings):
        value = v
        value["embedding"] = x
        all_entities[k] = value
    
    already_processed = 0
    async def _process_single_content_relation(chunk_key_dp: tuple[str, TextChunkSchema]):           # for each chunk, run the func
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        entities = context_entities[chunk_key]
        context_base_relation = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entities=",".join(entities),
            relation_types=",".join(PROMPTS["DEFAULT_RELATIONSHIP_TYPES"]),
            )
        hint_prompt = relation_extract_prompt.format(**context_base_relation, input_text=content)      # fill in the parameter
        final_result = await use_llm_func(hint_prompt)                                      # feed into LLM with the prompt

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)               # set as history
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)      # add to history
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(                                       # judge if we still need the next iteration
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(                                            # split entities from result --> list of entities
            final_result,
            [context_base_relation["record_delimiter"], context_base_relation["completion_delimiter"]],
        )
        # resolve the entities
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(          # split entity
                record, [context_base_relation["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(       # get the name, type, desc, source_id of entity--> dict
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1                                      # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][                     # for visualization
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # extract entities
    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    relation_results = await asyncio.gather(
        *[_process_single_content_relation(c) for c in ordered_chunks]
    )
    print()

    # fetch all relations from results
    all_relations = {}
    for item in relation_results:
        for k, v in item[1].items():
            all_relations[k] = v
    
    # TODO: hierarchical clustering
    logger.info(f"[Hierarchical Clustering]")
    hierarchical_cluster = Hierarchical_Clustering()
    hierarchical_clustered_entities_relations = await hierarchical_cluster.perform_clustering(entity_vdb=entity_vdb, global_config=global_config, entities=all_entities)
    hierarchical_clustered_entities = [[x for x in y if "entity_name" in x.keys()] for y in hierarchical_clustered_entities_relations]
    hierarchical_clustered_relations = [[x for x in y if "src_id" in x.keys()] for y in hierarchical_clustered_entities_relations]
    
    maybe_nodes = defaultdict(list)                     # for all chunks
    maybe_edges = defaultdict(list)
    # extracted entities and relations
    for m_nodes, m_edges in zip(entity_results, relation_results):
        for k, v in m_nodes[0].items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges[1].items():
            # it's undirected graph
            maybe_edges[tuple(sorted(k))].extend(v)
    # clustered entities
    for cluster_layer in hierarchical_clustered_entities:
        for item in cluster_layer:
            maybe_nodes[item['entity_name']].extend([item])
    # clustered relations
    for cluster_layer in hierarchical_clustered_relations:
        for item in cluster_layer:
            maybe_edges[tuple(sorted((item["src_id"], item["tgt_id"])))].extend([item])
    #----------------------------------------------------------------------------------------------
    # 添加关系类型聚类逻辑
    print("[统计] 开始收集所有关系类型")
    all_relation_types = []
    missing_relation_type_count = 0
    for k, v_list in maybe_edges.items():
        for v in v_list:
            if 'relation_type' in v:
                all_relation_types.append(v['relation_type'])
            else:
                missing_relation_type_count += 1
                # 为缺少关系类型的边添加默认值
                v['relation_type'] = 'Related'
                all_relation_types.append('Related')
    
    print(f"[统计] 收集到 {len(all_relation_types)} 个关系类型，其中 {missing_relation_type_count} 个缺少关系类型键")
    
    # relation_to_summary = {}
    # if len(all_relation_types) > 0:
    #     try:
    #         print("[处理] 开始对关系类型进行聚类...")
    #         # 调用聚类函数
    #         relation_to_summary = await cluster_and_summarize_relation_types(
    #             relation_types=all_relation_types,
    #             embedding_func=entity_vdb.embedding_func if entity_vdb else None,
    #             llm_func=global_config["best_model_func"],
    #             max_clusters=global_config.get("max_clusters", 10),  # 从全局配置中获取max_clusters值，默认为5
    #             global_config=global_config  # 传递全局配置
    #         )
    #         print(f"[成功] 关系类型聚类完成，生成了 {len(relation_to_summary)} 个类别")
    #         print(f"[调试] 关系类型映射: {relation_to_summary}")
    #     except Exception as e:
    #         print(f"[错误] 关系类型聚类失败: {str(e)}")
    #         # 失败时使用原始关系类型
    #         relation_to_summary = {rt: rt for rt in set(all_relation_types)}
    # else:
    #     print("[警告] 没有收集到任何关系类型，跳过聚类")
        #----------------------------------------------------------------------------------------------
    # store the nodes

    # all_entities_data = await asyncio.gather(           
    #     *[
    #         _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
    #         for k, v in maybe_nodes.items()
    #     ]
    # )
    # # store the edges
    # await asyncio.gather(                               
    #     *[
    #         _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config, relation_to_summary)
    #         for k, v in maybe_edges.items()
    #     ]
    # )
    #-------------------------------------------------------------------------------------
    # store the nodes (先存储所有节点)
    print(f"[处理] 开始合并和存储 {len(maybe_nodes)} 个唯一的实体...")
    # 异步并行地创建或更新所有节点
    all_entities_data_results = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    # 过滤掉可能失败的 None 结果 (健壮性)
    all_entities_data = [data for data in all_entities_data_results if data is not None]
    print(f"[成功] {len(all_entities_data)} 个实体被成功创建或更新。")


    # --- ✨✨✨ 你要的修复逻辑在这里！✨✨✨ ---
    
    # 1. 拿到所有被正确创建和合并的节点名称 "白名单"
    print(f"[处理] 正在从 {len(all_entities_data)} 个已存储实体中创建“白名单”...")
    clean_node_names = {dp["entity_name"] for dp in all_entities_data}
    
    if not clean_node_names:
        print("[警告] 节点白名单为空！没有任何实体被成功存储。将跳过所有边的创建。")
        # 此时可以根据你的逻辑选择返回
    else:
        print(f"[处理] 白名单创建完毕，共 {len(clean_node_names)} 个有效节点。")

    # 2. store the edges (存储边时进行过滤)
    print(f"[处理] 准备过滤和存储 {len(maybe_edges)} 组平行边...")
    edge_tasks = []
    dropped_edge_count = 0

    for k, v in maybe_edges.items():
        # 健壮性检查：确保 key 是一个元组且至少有两个元素
        if not (isinstance(k, tuple) and len(k) >= 2):
            print(f"[警告] 丢弃数据：边的键 {k} 不是一个有效的 (src, tgt) 元组。")
            dropped_edge_count += len(v)
            continue
            
        src_id, tgt_id = k[0], k[1]
        
        # 3. 检查边的两个端点是否都在我们的“白名单”中
        if src_id in clean_node_names and tgt_id in clean_node_names:
            # 两个节点都合法，添加任务
            edge_tasks.append(
                _merge_edges_then_upsert(src_id, tgt_id, v, knowledge_graph_inst, global_config)
            )
        else:
            # 至少有一个节点不合法，丢弃这组边
            # 打印详细的日志，告诉你“幽灵节点”是哪个
            if src_id not in clean_node_names:
                print(f"[警告] 丢弃边：源节点 '{src_id}' 不在白名单中。 (幽灵节点)")
            if tgt_id not in clean_node_names:
                print(f"[警告] 丢弃边：目标节点 '{tgt_id}' 不在白名单中。 (幽灵节点)")
            
            dropped_edge_count += len(v) # 累加被丢弃的边的数量

    if dropped_edge_count > 0:
        print(f"[统计] 共丢弃了 {dropped_edge_count} 条无效的边，因为它们连接了未知的“幽灵”或“格式错误”的节点。")
    
    # 4. 只执行合法的边任务
    if edge_tasks:
        print(f"[处理] 正在并行存储 {len(edge_tasks)} 组合法的边...")
        await asyncio.gather(*edge_tasks)
        print("[成功] 边存储完成。")
    else:
        print("[警告] 经过滤后，没有合法的边需要存储。")
     #-------------------------------------------------------------------------------------
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {                                # key is the md5 hash of the entity name string
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],   # entity name and description construct the content
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    return knowledge_graph_inst

async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())                       # chunks

    entity_extract_prompt = PROMPTS["entity_extraction"]        # give 3 examples in the prompt context
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]     # means low quality in the last extraction
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]       # judge if there are still entities still need to be extracted

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):           # for each chunk, run the func
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)      # fill in the parameter
        final_result = await use_llm_func(hint_prompt)                                      # feed into LLM with the prompt

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)               # set as history
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)      # add to history
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(                                       # judge if we still need the next iteration
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(                                            # split entities from result --> list of entities
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(          # split entity
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(       # get the name, type, desc, source_id of entity--> dict
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1                                      # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][                     # for visualization
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)                     # for all chunks
    maybe_edges = defaultdict(list)
    # for m_nodes, m_edges in results:
    #     for k, v in m_nodes.items():
    #         maybe_nodes[k].extend(v)
    #     for k, v in m_edges.items():
    #         # it's undirected graph
    #         maybe_edges[tuple(sorted(k))].extend(v)
    # # 收集所有关系类型
    # all_relation_types = []
    # edge_without_relation_type_count = 0
    # for m_nodes, m_edges in results:
    #     for k, v in m_nodes.items():
    #         maybe_nodes[k].extend(v)
    #     for k, v in m_edges.items():
    #         # it's undirected graph
    #         sorted_k = tuple(sorted(k))
    #         maybe_edges[sorted_k].extend(v)
    #         # 收集关系类型
    #         for edge in v:
    #             if 'relation_type' in edge:
    #                 all_relation_types.append(edge['relation_type'])
    #             else:
    #                 # 添加默认关系类型并记录缺失情况
    #                 edge_without_relation_type_count += 1
    #                 all_relation_types.append("RELATED")
    #                 print(f"[警告] 边数据中缺少'relation_type'键，已添加默认值 'RELATED'")

    # # 记录统计信息
    # print(f"[统计] 总共收集了 {len(all_relation_types)} 个关系类型，其中 {edge_without_relation_type_count} 个边数据缺少'relation_type'键")
    # # 执行全局关系类型聚类
    # relation_to_summary = {}
    # if len(all_relation_types) > 0:
    #     try:
    #         # 调用聚类函数进行全局关系类型聚类
    #         relation_to_summary = await cluster_and_summarize_relation_types(
    #             all_relation_types,
    #             global_config['embedding_func'],
    #             global_config['best_model_func'],
    #             max_clusters=global_config.get("max_clusters", 10),  # 从全局配置中获取max_clusters值，默认为10
    #             global_config=global_config
    #         )
    #         print(f"[统计] 成功完成全局关系类型聚类，共聚类 {len(set(all_relation_types))} 种关系类型")
            
    #         # 记录聚类结果样本
    #         if relation_to_summary:
    #             sample_size = min(5, len(relation_to_summary))
    #             sample_relations = list(relation_to_summary.items())[:sample_size]
    #             print(f"[统计] 聚类结果样本: {sample_relations}")
                
    #             # 添加调试输出
    #             print("\n" + "="*20 + " [DEBUG] 全局关系类型映射表 " + "="*20)
    #             print(json.dumps(relation_to_summary, indent=2, ensure_ascii=False))
    #             print("="*66 + "\n")
    #         else:
    #             print(f"[警告] 聚类函数返回了空的关系类型映射表")
    #             # 创建默认映射
    #             relation_to_summary = {rt: rt for rt in set(all_relation_types)}
    #     except Exception as e:
    #         logger.error(f"全局关系类型聚类失败: {str(e)}")
    #         # 出错时使用原始关系类型
    #         relation_to_summary = {rt: rt for rt in set(all_relation_types)}
    #         logger.info(f"使用原始关系类型，共 {len(relation_to_summary)} 种")
    # else:
    #     # 如果没有关系类型，使用空映射
    #     relation_to_summary = {}
    #     logger.warning("没有收集到任何关系类型")
    # -----------------------------------------------------------------
    all_entities_data = await asyncio.gather(           # store the nodes
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    await asyncio.gather(                               # store the edges
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)#添加关系类型
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {                                # key is the md5 hash of the entity name string
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],   # entity name and description construct the content
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    return knwoledge_graph_inst


def _pack_single_community_by_sub_communities(
    community: SingleCommunitySchema,
    max_token_size: int,
    already_reports: dict[str, CommunitySchema],
) -> tuple[str, int]:
    # TODO
    all_sub_communities = [
        already_reports[k] for k in community["sub_communities"] if k in already_reports
    ]
    all_sub_communities = sorted(
        all_sub_communities, key=lambda x: x["occurrence"], reverse=True
    )
    may_trun_all_sub_communities = truncate_list_by_token_size(
        all_sub_communities,
        key=lambda x: x["report_string"],
        max_token_size=max_token_size,
    )
    sub_fields = ["id", "report", "rating", "importance"]
    sub_communities_describe = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,
                c["report_string"],
                c["report_json"].get("rating", -1),
                c["occurrence"],
            ]
            for i, c in enumerate(may_trun_all_sub_communities)
        ]
    )
    already_nodes = []
    already_edges = []
    for c in may_trun_all_sub_communities:
        already_nodes.extend(c["nodes"])
        already_edges.extend([tuple(e) for e in c["edges"]])
    return (
        sub_communities_describe,
        len(encode_string_by_tiktoken(sub_communities_describe)),
        set(already_nodes),
        set(already_edges),
    )


async def _pack_single_community_describe(
    knwoledge_graph_inst: BaseGraphStorage,
    community: SingleCommunitySchema,
    max_token_size: int = 8000,
    already_reports: dict[str, CommunitySchema] = {},
    global_config: dict = {},
) -> str:
    nodes_in_order = sorted(community["nodes"])
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    nodes_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_node(n) for n in nodes_in_order]
    )
    edges_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
    )
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    
    nodes_list_data = [
        [
            i,
            node_name,
            node_data.get("entity_type", "UNKNOWN"),
            node_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.node_degree(node_name),
        ]
        for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
    ]
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )
    edges_list_data = [
        [
            i,
            edge_name[0],
            edge_name[1],
            edge_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.edge_degree(*edge_name),
        ]
        for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
    ]
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

    # If context is exceed the limit and have sub-communities:
    report_describe = ""
    need_to_use_sub_communities = (
        truncated and len(community["sub_communities"]) and len(already_reports)
    )
    force_to_use_sub_communities = global_config["addon_params"].get(
        "force_to_use_sub_communities", False
    )
    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(
            f"Community {community['title']} exceeds the limit or you set force_to_use_sub_communities to True, using its sub-communities"
        )
        report_describe, report_size, contain_nodes, contain_edges = (
            _pack_single_community_by_sub_communities(
                community, max_token_size, already_reports
            )
        )
        report_exclude_nodes_list_data = [
            n for n in nodes_list_data if n[1] not in contain_nodes
        ]
        report_include_nodes_list_data = [
            n for n in nodes_list_data if n[1] in contain_nodes
        ]
        report_exclude_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) not in contain_edges
        ]
        report_include_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) in contain_edges
        ]
        # if report size is bigger than max_token_size, nodes and edges are []
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_nodes_list_data + report_include_nodes_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
        edges_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_edges_list_data + report_include_edges_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
#     return f"""-----Reports-----
# ```csv
# {report_describe}
# ```
# -----Entities-----
# ```csv
# {nodes_describe}
# ```
# -----Relationships-----
# ```csv
# {edges_describe}
# ```"""
# 构建完整的描述
    full_describe = f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""
    
    # 检查最终的令牌数量，如果超过限制，则进一步截断
    final_tokens = encode_string_by_tiktoken(full_describe)
    if len(final_tokens) > max_token_size:
        logger.warning(f"Community description exceeds token limit ({len(final_tokens)} > {max_token_size}), further truncating")
        
        # 计算当前各部分的令牌数
        report_tokens = len(encode_string_by_tiktoken(report_describe))
        nodes_tokens = len(encode_string_by_tiktoken(nodes_describe))
        edges_tokens = len(encode_string_by_tiktoken(edges_describe))
        overhead_tokens = len(final_tokens) - (report_tokens + nodes_tokens + edges_tokens)
        
        # 重新分配令牌，确保总和不超过max_token_size
        available_tokens = max_token_size - overhead_tokens
        nodes_ratio = nodes_tokens / (nodes_tokens + edges_tokens) if (nodes_tokens + edges_tokens) > 0 else 0.5
        edges_ratio = edges_tokens / (nodes_tokens + edges_tokens) if (nodes_tokens + edges_tokens) > 0 else 0.5
        
        nodes_max_tokens = int(available_tokens * nodes_ratio * 0.8)  # 留出一些缓冲
        edges_max_tokens = int(available_tokens * edges_ratio * 0.8)
        
        # 重新截断节点和边的数据
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            nodes_list_data, key=lambda x: x[3], max_token_size=nodes_max_tokens
        )
        edges_may_truncate_list_data = truncate_list_by_token_size(
            edges_list_data, key=lambda x: x[3], max_token_size=edges_max_tokens
        )
        
        # 重新生成描述
        nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
        edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
        
        full_describe = f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""
        
        # 最终检查
        final_tokens = encode_string_by_tiktoken(full_describe)
        if len(final_tokens) > max_token_size:
            logger.warning(f"Still exceeds token limit after truncation ({len(final_tokens)} > {max_token_size}), using minimal content")
            # 如果还是超过限制，使用最小内容
            minimal_nodes = nodes_may_truncate_list_data[:1] if nodes_may_truncate_list_data else []
            minimal_edges = edges_may_truncate_list_data[:1] if edges_may_truncate_list_data else []
            nodes_describe = list_of_list_to_csv([node_fields] + minimal_nodes)
            edges_describe = list_of_list_to_csv([edge_fields] + minimal_edges)
            
            full_describe = f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""
    
    return full_describe


def _community_report_json_to_str(parsed_output: dict) -> str:
    """refer official graphrag: index/graph/extractors/community_reports"""
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    def finding_summary(finding: dict):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")

    def finding_explanation(finding: dict):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}"


async def generate_community_report(
    community_report_kv: BaseKVStorage[CommunitySchema],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func: callable = global_config["best_model_func"]
    use_string_json_convert_func: callable = global_config[
        "convert_response_to_json_func"
    ]

    community_report_prompt = PROMPTS["community_report"]

    communities_schema = await knwoledge_graph_inst.community_schema()
    community_keys, community_values = list(communities_schema.keys()), list(
        communities_schema.values()
    )
    already_processed = 0

    async def _form_single_community_report(
        community: SingleCommunitySchema, already_reports: dict[str, CommunitySchema]
    ):
        nonlocal already_processed
        safe_max_token_size = min(7000, global_config["best_model_max_token_size"]) # 限制最大令牌数
        describe = await _pack_single_community_describe(
            knwoledge_graph_inst,
            community,
            max_token_size=safe_max_token_size,
            already_reports=already_reports,
            global_config=global_config,
        )
        prompt = community_report_prompt.format(input_text=describe)
        response = await use_llm_func(prompt, **llm_extra_kwargs)
        data = use_string_json_convert_func(response)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} communities\r",
            end="",
            flush=True,
        )
        return data

    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(community_keys, community_values)
                if v["level"] == level
            ]
        )
        this_level_communities_reports = await asyncio.gather(
            *[
                _form_single_community_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                    this_level_community_keys,
                    this_level_communities_reports,
                    this_level_community_values,
                )
            }
        )
    print()  # clear the progress bar
    await community_report_kv.upsert(community_datas)


async def _find_most_related_community_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    community_reports: BaseKVStorage[CommunitySchema],
):
    related_communities = []
    for node_d in node_datas:
        if "clusters" not in node_d:
            continue
        related_communities.extend(json.loads(node_d["clusters"]))
    related_community_dup_keys = [
        str(dp["cluster"])
        for dp in related_communities
        if dp["level"] <= query_param.level
    ]
    related_community_keys_counts = dict(Counter(related_community_dup_keys))
    _related_community_datas = await asyncio.gather(        # get community reports
        *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
    )
    related_community_datas = {
        k: v
        for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
        if v is not None
    }
    related_community_keys = sorted(            # sort by ratings
        related_community_keys_counts.keys(),
        key=lambda k: (
            related_community_keys_counts[k],
            related_community_datas[k]["report_json"].get("rating", -1),
        ),
        reverse=True,
    )
    sorted_community_datas = [                  # community reports sorted by ratings
        related_community_datas[k] for k in related_community_keys
    ]

    use_community_reports = truncate_list_by_token_size(    # in case community reprot is longer than token limitation
        sorted_community_datas,
        key=lambda x: x["report_string"],
        max_token_size=query_param.max_token_for_community_report,
    )
    if query_param.community_single_one:
        use_community_reports = use_community_reports[:1]
    return use_community_reports


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [      # the entities related to the retrieved entities
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(   # get relations related to the retrieved entities
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )                               # where the source entities are the retrieved entities
    all_one_hop_nodes = set()       # find the one hop neighbors
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(  # get node information from storage
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    # 恢复原始代码并添加错误处理
    all_one_hop_text_units_lookup = {               # find the text chunks of the 1-hop neighbors entities
        k: set(split_string_by_multi_markers(v.get("source_id", ""), [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and v.get("source_id")
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,     # count of relations related to the chunk
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(        # sort by relation counts
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        if not this_edges:
            continue
        # 对于MultiGraph，边可能包含key信息：(source, target) 或 (source, target, key)
        for edge in this_edges:
            if len(edge) == 2:  # 简单边 (source, target)
                all_edges.add(edge)
            elif len(edge) == 3:  # MultiGraph边 (source, target, key)
                all_edges.add(edge)
            else:
                # 兼容其他格式，保留原有逻辑
                all_edges.add(tuple(sorted(edge[:2])))
    
    all_edges = list(all_edges)
    
    # 获取边数据，支持MultiGraph的key参数
    all_edges_pack = []
    all_edges_degree = []
    
    for edge in all_edges:
        if len(edge) == 2:  # 简单边
            source, target = edge
            edge_data = await knowledge_graph_inst.get_edge(source, target)
            degree = await knowledge_graph_inst.edge_degree(source, target)
        elif len(edge) == 3:  # MultiGraph边
            source, target, key = edge
            edge_data = await knowledge_graph_inst.get_edge(source, target, key=key)
            degree = await knowledge_graph_inst.edge_degree(source, target)
        else:
            # 兼容处理
            source, target = edge[:2]
            edge_data = await knowledge_graph_inst.get_edge(source, target)
            degree = await knowledge_graph_inst.edge_degree(source, target)
        
        all_edges_pack.append(edge_data)
        all_edges_degree.append(degree)
    
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    
    # # 确保每条边都有weight字段，如果没有则提供默认值5
    # for edge_data in all_edges_data:
    #     if "weight" not in edge_data:
    #         edge_data["weight"] = 5  # 设置默认权重
    #     # 确保每条边都有description字段，如果没有则提供默认空字符串
    #     if "description" not in edge_data:
    #         edge_data["description"] = ""  # 设置默认空描述
    
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )
    return all_edges_data


# async def _find_most_related_edges_from_paths(
#     path_datas: list[dict],
#     path: list[str],
#     query_param: QueryParam,
#     knowledge_graph_inst: BaseGraphStorage,
# ):
#     # 获取子图
#     subgraph = knowledge_graph_inst._graph.subgraph(path)
    
#     # 获取所有边，包括平行边
#     all_edges = []
#     for u, v in subgraph.edges():
#         # 对于MultiGraph，获取所有平行边
#         if hasattr(subgraph, 'get_edge_data'):
#             edge_data = subgraph.get_edge_data(u, v)
#             if isinstance(edge_data, dict):
#                 # 检查是否是MultiGraph（edge_data是字典的字典）
#                 if any(isinstance(k, str) for k in edge_data.keys()):
#                     # 这是MultiGraph，遍历所有平行边
#                     for key in edge_data.keys():
#                         all_edges.append((u, v, key))
#                 else:
#                     # 这是简单图，只有一条边
#                     all_edges.append((u, v, None))
#             else:
#                 # 其他情况，添加边
#                 all_edges.append((u, v, None))
#         else:
#             all_edges.append((u, v, None))
    
#     # 获取每条边的详细数据
#     all_edges_pack = await asyncio.gather(
#         *[knowledge_graph_inst.get_edge(e[0], e[1], key=e[2]) for e in all_edges]
#     )
    
#     # 获取每条边的度数
#     all_edges_degree = await asyncio.gather(
#         *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
#     )
    
#     # 组合边数据
#     all_edges_data = [
#         {"src_tgt": (e[0], e[1]), "key": e[2], "rank": d, **v}
#         for e, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
#         if v is not None
#     ]
    
#     # 按rank和weight排序
#     all_edges_data = sorted(
#         all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
#     )
    
#     # 截断结果
#     all_edges_data = truncate_list_by_token_size(
#         all_edges_data,
#         key=lambda x: x["description"],
#         max_token_size=query_param.max_token_for_bridge_knowledge,
#     )
    
#     return all_edges_data
async def _find_most_related_edges_from_paths(
    path_datas: list[dict],
    path: list[str],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    """
    修正后的版本，能正确、高效地提取所有平行边信息，并安全处理缺失的 'weight' 键。
    """
    # 获取子图
    subgraph = knowledge_graph_inst._graph.subgraph(path)
    
    # --- 
    # 🐛 修复问题 1: 彻底修正平行边提取逻辑
    # 你原来的 `for u, v in subgraph.edges():` 循环从一开始就丢弃了平行边。
    # 正确的方式是使用 `edges(keys=True, data=True)` 来迭代，
    # 这样我们一次性就能拿到所有边的 (u, v, key, data)
    # ---
    all_edges_to_process = []
    if subgraph.is_multigraph():
        # 完整获取 (u, v, key, data)
        for u, v, key, data in subgraph.edges(keys=True, data=True):
            # data 已经是属性字典了，比如: {'weight': 10.0, 'description': ...}
            # 或者... 可能是 {'description': ...} (缺少 weight!)
            all_edges_to_process.append(
                {"src_tgt": (u, v), "key": key, **data}
            )
    else:
        # 简单图
        for u, v, data in subgraph.edges(data=True):
            all_edges_to_process.append(
                {"src_tgt": (u, v), "key": None, **data}
            )
    
    # ---
    # 逻辑简化: 既然我们已经有了所有数据 (data)，就不再需要 all_edges_pack
    # 只需要为获取度数准备数据
    # ---
    
    # 获取每条边的度数
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e["src_tgt"][0], e["src_tgt"][1]) for e in all_edges_to_process]
    )
    
    # 组合边数据
    all_edges_data = []
    for edge_data, degree in zip(all_edges_to_process, all_edges_degree):
        edge_data["rank"] = degree
        
        # ---
        # 🐛 修复问题 2 (添加防护): 
        # 确保 "weight" 和 "description" 键一定存在，如果图谱里没有，就给一个默认值
        # ---
        edge_data.setdefault("weight", 0.0) # 如果 "weight" 不存在，设置为 0.0
        edge_data.setdefault("description", "") # 如果 "description" 不存在，设置为空字符串
        
        all_edges_data.append(edge_data)
    
    # ---
    # 🐛 修复问题 2 (安全排序): 
    # 使用 .get() 进行安全访问，这是防止崩溃的最后一道防线
    # ---
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x.get("weight", 0.0)), reverse=True
    )
    
    # 截断结果
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x.get("description", ""), # 同样使用 .get() 保证健壮性
        max_token_size=query_param.max_token_for_bridge_knowledge,
    )
    
    return all_edges_data 
# context functions
async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)          # find the top-k(20) related entities
    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    node_datas = node_datas[:query_param.top_k]
    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Reports-----
```csv
{communities_context}
```
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""

async def _build_hibridge_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k * 10)          # find the top-k(20) related entities

    if not len(results):    # results just with entity name
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
  
    node_datas = [                          # add rank, which is the degree
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    overall_node_datas = node_datas
    node_datas = node_datas[:query_param.top_k]

    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    # use_relations = await _find_most_related_edges_from_entities(
    #     node_datas, query_param, knowledge_graph_inst
    # )

    def find_path_with_required_nodes(graph, source, target, required_nodes):
        # inital final path
        final_path = []
        # 起点设置为当前节点
        current_node = source

        # 遍历必经节点
        for next_node in required_nodes:
            # 找到从当前节点到下一个必经节点的最短路径
            try:
                sub_path = nx.shortest_path(graph, source=current_node, target=next_node)
            except nx.NetworkXNoPath:
                # raise ValueError(f"No path between {current_node} and {next_node}.")
                final_path.extend([next_node])
                current_node = next_node
                continue
            
            # 合并路径（避免重复添加当前节点）
            if final_path:
                final_path.extend(sub_path[1:])  # 从第二个节点开始添加，避免重复
            else:
                final_path.extend(sub_path)
            
            # 更新当前节点为下一个必经节点
            current_node = next_node

        # 最后，从最后一个必经节点到目标节点的路径
        try:
            sub_path = nx.shortest_path(graph, source=current_node, target=target)
            final_path.extend(sub_path[1:])  # 从第二个节点开始添加，避免重复
        except nx.NetworkXNoPath:
            # raise ValueError(f"No path between {current_node} and {target}.")
            final_path.extend([target])

        return final_path

    # find some top-k entities in each communities in use_communities
    key_entities = []
    max_entity_num = query_param.top_m
    if use_communities:
        for c in use_communities:
            cur_community_key_entities = []
            community_entities = c['nodes']
            # find the top-k entities in this community
            cur_community_key_entities.extend(
                [e for e in overall_node_datas if e['entity_name'] in community_entities][:max_entity_num]
            )
            key_entities.append(cur_community_key_entities)
    else:
        key_entities = [overall_node_datas[:max_entity_num]]
    # unique key entities
    key_entities = [[e['entity_name'] for e in k] for k in key_entities]
    key_entities = list(set([k for kk in key_entities for k in kk]))
    # find the shortest path between the key entities
    try:
        path = find_path_with_required_nodes(knowledge_graph_inst._graph, key_entities[0], key_entities[-1], key_entities[1:-1])
        # path = list(set(path))
        path_datas = await asyncio.gather(      # get full information of retrieved entities
            *[knowledge_graph_inst.get_node(r) for r in path]
        )
        path_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r) for r in path]
        )
        path_datas = [                          # add rank, which is the degree
            {**n, "entity_name": k, "rank": d}
            for k, n, d in zip(path, path_datas, path_degrees)
            if n is not None
        ]
        use_reasoning_path = await _find_most_related_edges_from_paths(
                                path_datas, path, query_param, knowledge_graph_inst
                            )
    except ValueError as e:
        print(e)

    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_reasoning_path)} reasoning path items, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    
    reasoning_path_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_reasoning_path):
        reasoning_path_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)
    
    # reasoning_path_context = list_of_list_to_csv([["id", "content"]] + [[i, p] for i, p in enumerate(reasoning_path)])
    
    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Reasoning Path-----
```csv
{reasoning_path_context}
```
-----Source Documents-----
```csv
{text_units_context}
```
"""


async def _build_higlobal_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k * 10)          # find the top-k(20) related entities

    if not len(results):    # results just with entity name
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [                          # add rank, which is the degree
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    overall_node_datas = node_datas
    node_datas = node_datas[:query_param.top_k]

    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )


    logger.info(
        f"Using {len(use_communities)} communities, {len(use_text_units)} text units"
    )
    
    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
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
-----Source Documents-----
```csv
{text_units_context}
```
"""


async def _build_rgcn_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    base_node_datas = node_datas[:query_param.top_k]
    overall_node_datas, rgcn_sims = await _rgcn_expand_entities(base_node_datas, knowledge_graph_inst, query_param, global_config)
    
    rgcn_counts = Counter()
    for _, target_name, _, _ in rgcn_sims:
        rgcn_counts[target_name] += 1
    name_to_node_map = {n["entity_name"]: n for n in overall_node_datas}
    rgcn_candidate_names = list(rgcn_counts.keys())
    rgcn_candidate_names.sort(
        key=lambda name: (rgcn_counts[name], name_to_node_map.get(name, {}).get("rank", 0)),
        reverse=True,
    )
    full_candidate_pool = []
    seen_names = set()

    # 先加 BGE (语义保底)
    for node in base_node_datas:
        if node["entity_name"] not in seen_names:
            full_candidate_pool.append(node)
            seen_names.add(node["entity_name"])
            
    # 再加所有 R-GCN (结构增强)
    # 2. 再加 **所有** R-GCN 候选 (不仅仅是 Top-K)
    for name in rgcn_candidate_names:
        if name not in seen_names:
            full_candidate_pool.append(name_to_node_map[name])
            seen_names.add(name)
    
    overall_node_datas = full_candidate_pool[:query_param.top_k*2]
    # ==========================================================================
    retrieval_entities_list = [n["entity_name"] for n in overall_node_datas]
    
    # 这里使用 logger (或者 print) 输出这个数组
    logger.info(f"\n[DEBUG] 用于检索社区与文本块的实体池 (共 {len(retrieval_entities_list)} 个):")
    logger.info(f"   -> {retrieval_entities_list}") 
    use_communities = await _find_most_related_community_from_entities(
        overall_node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        overall_node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    def find_path_with_required_nodes(graph, source, target, required_nodes):
        final_path = []
        current_node = source
        for next_node in required_nodes:
            try:
                sub_path = nx.shortest_path(graph, source=current_node, target=next_node)
            except nx.NetworkXNoPath:
                final_path.extend([next_node])
                current_node = next_node
                continue
            if final_path:
                final_path.extend(sub_path[1:])
            else:
                final_path.extend(sub_path)
            current_node = next_node
        try:
            sub_path = nx.shortest_path(graph, source=current_node, target=target)
            final_path.extend(sub_path[1:])
        except nx.NetworkXNoPath:
            final_path.extend([target])
        return final_path
    key_entities = []
    max_entity_num = query_param.top_m
    if use_communities:
        for c in use_communities:
            cur = []
            community_entities = c["nodes"]
            cur.extend([e for e in full_candidate_pool if e["entity_name"] in community_entities][:max_entity_num])
            key_entities.append(cur)
    else:
        key_entities = [full_candidate_pool[:max_entity_num]]
    key_entities = [[e["entity_name"] for e in k] for k in key_entities]
    key_entities = list(set([k for kk in key_entities for k in kk]))
    try:
        path = find_path_with_required_nodes(knowledge_graph_inst._graph, key_entities[0], key_entities[-1], key_entities[1:-1])
        path_datas = await asyncio.gather(
            *[knowledge_graph_inst.get_node(r) for r in path]
        )
        path_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r) for r in path]
        )
        path_datas = [
            {**n, "entity_name": k, "rank": d}
            for k, n, d in zip(path, path_datas, path_degrees)
            if n is not None
        ]
        use_reasoning_path = await _find_most_related_edges_from_paths(
            path_datas, path, query_param, knowledge_graph_inst
        )
    except ValueError:
        use_reasoning_path = []
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    display_entities = overall_node_datas[:query_param.top_k]
    for i, n in enumerate(display_entities):
        entites_section_list.append([
            i,
            n["entity_name"],
            n.get("entity_type", "UNKNOWN"),
            n.get("description", "UNKNOWN"),
            n["rank"],
        ])
    entities_context = list_of_list_to_csv(entites_section_list)
    reasoning_path_section_list = [["id", "source", "target", "relation_type", "description", "weight", "rank"]]
    for i, e in enumerate(use_reasoning_path):
        src, tgt = e.get("src_tgt", (None, None))
        relation_type = e.get("relation_type", "UNKNOWN")
        description = e.get("description", "UNKNOWN")
        weight = e.get("weight", 1.0)
        rank = e.get("rank", 0)
        reasoning_path_section_list.append([i, src, tgt, relation_type, description, weight, rank])
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)
    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"].replace("\n", " ")])
    communities_context = list_of_list_to_csv(communities_section_list)
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    # 日志记录
    entities_log = [f'"{n["entity_name"]}"' for n in overall_node_datas]
    communities_log = [(c.get("level"), c.get("title", f"Cluster {c.get('cluster')}")) for c in use_communities]
    chunks_log = [(t.get("full_doc_id") or t.get("doc_id"), t.get("chunk_order_index") or t.get("index", 0)) for t in use_text_units]
    
    logger.info(f"====== References (RGCN Diversity Enhanced) ======\n"
                f"Entities ({len(entities_log)}): {entities_log}\n"
                f"Communities ({len(communities_log)}): {communities_log}\n"
                f"Chunks ({len(chunks_log)}): {chunks_log}\n")
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
# -----Synonyms-----
# ```csv
# {synonyms_context}
# ```

import json
from openai import AsyncOpenAI

async def llm_listwise_rerank(query, candidate_nodes, top_n, client, model):
    """
    Listwise Reranking: 让 LLM 对实体列表进行整体打分排序。
    """
    if not candidate_nodes:
        return []

    # 1. 准备输入数据
    # 为了节省 Token，我们只发：ID, Name, Description
    # 限制输入数量，防止 LLM 晕头转向（建议最多 30-50 个候选）
    candidates_input = candidate_nodes[:50] 
    
    # 建立 ID 映射，方便后续还原
    id_map = {i: node for i, node in enumerate(candidates_input)}
    
    # 构造列表字符串
    list_str = ""
    for i, node in enumerate(candidates_input):
        name = node['entity_name']
        desc = node.get('description', 'No description')[:150].replace("\n", " ") # 截断描述
        list_str += f"[{i}] {name}: {desc}\n"

    # 2. 构造 Prompt
    sys_prompt = "You are a Ranker for a Retrieval-Augmented Generation system."
    
    user_prompt = f"""
    I need to retrieve the most relevant entities to answer the following query.
    
    Query: "{query}"
    
    Below is a list of candidate entities retrieved by different algorithms (Semantic Search + Graph Search).
    Some are crucial, some are marginal, and some are noise.
    
    **Task:**
    Select the Top-{top_n} entities that are most helpful for answering the query.
    Sort them by relevance (most relevant first).
    
    **Candidates:**
    {list_str}
    
    **Output Format:**
    Return ONLY a JSON object containing the list of selected IDs.
    {{
        "selected_ids": [id_1, id_2, id_3, ...]
    }}
    """

    # 3. 调用 LLM
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0 # 必须为 0，保证稳定
        )
        
        # 4. 解析结果
        content = response.choices[0].message.content
        if "```json" in content:
            content = content.replace("```json", "").replace("```", "").strip()
            
        result = json.loads(content)
        selected_ids = result.get("selected_ids", [])
        
        # 5. 根据 ID 还原节点列表
        ranked_nodes = []
        seen_ids = set()
        
        for mid in selected_ids:
            if mid in id_map and mid not in seen_ids:
                ranked_nodes.append(id_map[mid])
                seen_ids.add(mid)
        
        # 6. 兜底：如果 LLM 选的不够 top_n，用剩下的补齐
        if len(ranked_nodes) < top_n:
            for i, node in enumerate(candidates_input):
                if i not in seen_ids:
                    ranked_nodes.append(node)
                    if len(ranked_nodes) >= top_n: break
                    
        return ranked_nodes

    except Exception as e:
        print(f"⚠️ Rerank Failed: {e}, returning original order.")
        return candidate_nodes[:top_n]
import aiohttp
import json
import logging

async def call_rerank_api(query, candidates, api_key, base_url, model_name="BAAI/bge-reranker-v2-m3"):
    """
    调用 Rerank API 对实体进行重排序。
    
    Args:
        query: 用户问题
        candidates: 实体列表 [{"entity_name":..., "description":...}, ...]
        api_key: API Key
        base_url: API Base URL (e.g., https://api.siliconflow.cn/v1)
    Returns:
        排序后的实体列表
    """
    if not candidates:
        return []
    if not api_key:
        raise ValueError("Rerank API key is not configured. Set openai.api_key, RERANK_API_KEY, or OPENAI_API_KEY.")
        
    # 1. 构造文档列表 (Name + Description)
    # 描述对于 Reranker 判断相关性至关重要！
    documents = []
    for n in candidates:
        name = n.get("entity_name", "")
        desc = n.get("description", "") or ""
        # 拼接成一段话，截断防止超长
        doc_text = f"{name}: {desc[:500]}" 
        documents.append(doc_text)
   
    url = f"{base_url}/rerank"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "query": query,
        "documents": documents,
        "top_n": len(documents), # 返回所有结果，我们自己截断
        "return_documents": False
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logging.error(f"Rerank API Error {resp.status}: {text}")
                    return candidates # 失败降级：返回原顺序

                result = await resp.json()
                # 结果通常是: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
                ranked_results = result.get("results", [])
                
                # 2. 根据返回的 index 重组列表
                sorted_candidates = []
                for item in ranked_results:
                    original_idx = item["index"]
                    score = item["relevance_score"]
                    
                    # 把 Rerank 分数存进去，方便日志查看
                    node = candidates[original_idx].copy()
                    node["_rerank_score"] = score
                    
                    # 过滤掉分数极低的结果 (负分通常表示不相关)
                    # BGE-Reranker 的分数范围通常是 -10 到 10
                    if score > -2.0: 
                        sorted_candidates.append(node)
                        
                logging.info(f"Rerank Success: {len(candidates)} -> {len(sorted_candidates)} kept.")
                return sorted_candidates

    except Exception as e:
        logging.error(f"Rerank Exception: {e}")
        return candidates # 失败降级


def _resolve_rerank_api_config(global_config: dict | None):
    global_config = global_config or {}
    openai_cfg = global_config.get("openai", {}) or {}
    rerank_cfg = global_config.get("rerank", {}) or {}
    api_key = (
        global_config.get("rerank_api_key")
        or rerank_cfg.get("api_key")
        or openai_cfg.get("api_key")
        or os.getenv("RERANK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    base_url = (
        global_config.get("rerank_base_url")
        or rerank_cfg.get("base_url")
        or openai_cfg.get("base_url")
        or os.getenv("RERANK_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.siliconflow.cn/v1"
    )
    model_name = (
        global_config.get("rerank_model_name")
        or global_config.get("rerank_model")
        or rerank_cfg.get("model")
        or os.getenv("RERANK_MODEL")
        or "BAAI/bge-reranker-v2-m3"
    )
    return api_key, base_url, model_name

# ---------------------------hi-----------------------------
async def _build_hierarchical_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k * 10)          # find the top-k(20) related entities

    if not len(results):    # results just with entity name
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [                          # add rank, which is the degree
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    overall_node_datas = node_datas
    node_datas = node_datas[:query_param.top_k*2]

    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    # use_relations = await _find_most_related_edges_from_entities(
    #     node_datas, query_param, knowledge_graph_inst
    # )

    def find_path_with_required_nodes(graph, source, target, required_nodes):
        # inital final path
        final_path = []
        # 起点设置为当前节点
        current_node = source

        # 遍历必经节点
        for next_node in required_nodes:
            # 找到从当前节点到下一个必经节点的最短路径
            try:
                sub_path = nx.shortest_path(graph, source=current_node, target=next_node)
            except nx.NetworkXNoPath:
                # raise ValueError(f"No path between {current_node} and {next_node}.")
                final_path.extend([next_node])
                current_node = next_node
                continue
            
            # 合并路径（避免重复添加当前节点）
            if final_path:
                final_path.extend(sub_path[1:])  # 从第二个节点开始添加，避免重复
            else:
                final_path.extend(sub_path)
            
            # 更新当前节点为下一个必经节点
            current_node = next_node

        # 最后，从最后一个必经节点到目标节点的路径
        try:
            sub_path = nx.shortest_path(graph, source=current_node, target=target)
            final_path.extend(sub_path[1:])  # 从第二个节点开始添加，避免重复
        except nx.NetworkXNoPath:
            # raise ValueError(f"No path between {current_node} and {target}.")
            final_path.extend([target])

        return final_path

    # find some top-k entities in each communities in use_communities
    key_entities = []
    max_entity_num = query_param.top_m
    if use_communities:
        for c in use_communities:
            cur_community_key_entities = []
            community_entities = c['nodes']
            # find the top-k entities in this community
            cur_community_key_entities.extend(
                [e for e in overall_node_datas if e['entity_name'] in community_entities][:max_entity_num]
            )
            key_entities.append(cur_community_key_entities)
    else:
        key_entities = [overall_node_datas[:max_entity_num]]
    # unique key entities
    key_entities = [[e['entity_name'] for e in k] for k in key_entities]
    # key_entities = list(set([k for kk in key_entities for k in kk]))
    key_entities = sorted(list(set([k for kk in key_entities for k in kk])))
    # find the shortest path between the key entities
    try:
        path = find_path_with_required_nodes(knowledge_graph_inst._graph, key_entities[0], key_entities[-1], key_entities[1:-1])
        # path = list(set(path))
        path_datas = await asyncio.gather(      # get full information of retrieved entities
            *[knowledge_graph_inst.get_node(r) for r in path]
        )
        path_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r) for r in path]
        )
        path_datas = [                          # add rank, which is the degree
            {**n, "entity_name": k, "rank": d}
            for k, n, d in zip(path, path_datas, path_degrees)
            if n is not None
        ]
        # use_reasoning_path = await _find_most_related_edges_from_entities(
        #                     path_datas, query_param, knowledge_graph_inst
        #                 )
        use_reasoning_path = await _find_most_related_edges_from_paths(
                                path_datas, path, query_param, knowledge_graph_inst
                            )
    except ValueError as e:
        print(e)
    
    # # fetch the relations of the reasoning paths
    # reasoning_path = []
    # for i in range(len(path) - 1):
    #     src = path[i]
    #     tgt = path[i + 1]
    #     cur_relation = (await knowledge_graph_inst.get_edge(src, tgt))['description']
    #     reasoning_path.append(cur_relation)
    # reasoning_path = list(set(reasoning_path))

    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_reasoning_path)} reasoning path items, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    
    reasoning_path_section_list = [
    ["id", "source", "target", "relation_type", "description", "weight", "rank"]
]

    for i, e in enumerate(use_reasoning_path):
        # src_tgt 是 (source, target)
        src, tgt = e.get("src_tgt", (None, None))
        relation_type = e.get("relation_type", "UNKNOWN")
        description = e.get("description", "UNKNOWN")
        weight = e.get("weight", 1.0)
        rank = e.get("rank", 0)

        reasoning_path_section_list.append(
            [i, src, tgt, relation_type, description, weight, rank]
        )

    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)
    
    # reasoning_path_context = list_of_list_to_csv([["id", "content"]] + [[i, p] for i, p in enumerate(reasoning_path)])
    
    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"].replace("\n", " ")])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    # display reference info
    entities = [n["entity_name"] for n in node_datas]
    communities = [(c["level"], c["title"]) for c in use_communities]
    chunks = [(t["full_doc_id"], t["chunk_order_index"]) for t in use_text_units]

    references_context = (
        f"Entities ({len(entities)}): {entities}\n\n"
        f"Communities (level, cluster_id) ({len(communities)}): {communities}\n\n"
        f"Chunks (doc_id, chunk_index) ({len(chunks)}): {chunks}\n"
    )

    logging.info(f"====== References ======:\n{references_context}")
    # # ==========================================================================
    # # [修改点] 调用埋点函数
    # # ==========================================================================
    # data = extract_data_snapshot(
    #     query=query,
    #     final_entities=node_datas, 
    #     use_communities=use_communities, 
    #     # use_text_units=use_text_units, 
    #     bridging_paths=use_reasoning_path,
    #     # physical_paths  
    # )

    # # 保存
    # with open("logs/hi.jsonl", "a", encoding="utf-8") as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4)
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
# ---------------------------hi+rgcn+重排序-----------------------------
async def _build_rgcn_path_rank_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    # ==========================================================================
    # 1. BGE 语义检索 (Semantic Anchoring)
    # ==========================================================================
    results = await entities_vdb.query(query, top_k=query_param.top_k*10)

    if not len(results):
        return None
        
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    
    base_node_datas = node_datas[:query_param.top_k]
    
    # ==========================================================================
    # 2. R-GCN 结构扩展 (Structural Expansion)
    # ==========================================================================
    overall_node_datas, rgcn_sims = await _rgcn_expand_entities(
        base_node_datas, knowledge_graph_inst, query_param, global_config
    )
 
    
    bge_entity_names = set(n["entity_name"] for n in base_node_datas)

    # ==========================================================================
    # 3. [Rerank API 版] 实体混合重排序
    # 策略：把 BGE 和 R-GCN 的结果混合，交给 Rerank 模型统一打分
    # ==========================================================================
    
    # A. 准备候选池 (Candidate Pool)
    pool_nodes = []
    seen_in_pool = set()
    
    # # 1. 提取 BGE Top-3 (锁定组)
    bge_top3_nodes = []
    for n in base_node_datas[:3]:
        bge_top3_nodes.append(n)
        seen_in_pool.add(n["entity_name"])

    # 2. 提取剩下的 BGE 实体 (待重排组)
    for n in base_node_datas[3:]:
        if n["entity_name"] not in seen_in_pool:
            pool_nodes.append(n)
            seen_in_pool.add(n["entity_name"])
        
    # 3. 提取 R-GCN 高票实体 (待重排组)
    rgcn_counts = Counter([x[1] for x in rgcn_sims])
    name_to_node = {n["entity_name"]: n for n in overall_node_datas}
    
    # 按票数排序，取前 50 个进入复赛
    top_rgcn_names = sorted(rgcn_counts.keys(), key=lambda k: rgcn_counts[k], reverse=True)
    
    for name in top_rgcn_names[:50]:
        if name not in seen_in_pool and name in name_to_node:
            pool_nodes.append(name_to_node[name])
            seen_in_pool.add(name)
            
    # B. 调用 Rerank API (只对“待重排组”进行打分)
    api_key, base_url, rerank_model_name = _resolve_rerank_api_config(global_config)

    # 执行重排
    # 注意：这里只传入 pool_nodes，不传 bge_top3_nodes
    reranked_pool = await call_rerank_api(
        query=query,
        candidates=pool_nodes,
        api_key=api_key,
        base_url=base_url,
        model_name=rerank_model_name
    )
    
    # C. 组装最终名单 (拼接：锁定组 + 重排后的优胜组)
    final_query_nodes = []
    
    # 1. 先放 BGE Top-3
    final_query_nodes.extend(bge_top3_nodes)          #目前不需要
    
    # 2. 再放 Rerank 后的高分实体
    # 注意去重，虽然逻辑上应该没重复，但为了保险
    seen_final = set(n["entity_name"] for n in bge_top3_nodes)
    
    for node in reranked_pool:
        if node["entity_name"] not in seen_final:
            final_query_nodes.append(node)
            seen_final.add(node["entity_name"])

    # D. 最终截断
    # 保证总数不超过 top_k * 2 (通常为了 Rerank 效果好，可以稍微放宽到 top_k)
    results = final_query_nodes
    final_query_nodes = final_query_nodes[:query_param.top_k*2]
    final_selected_names = set(n["entity_name"] for n in final_query_nodes)
    
    # [Log] 看看谁是 Top 5 (前3个应该是 BGE 原生的)
    debug_top = [f"{n['entity_name']}({n.get('_rerank_score', 'LOCK')})" for n in final_query_nodes[:5]]
    logger.info(f"[Hybrid Rerank] Top 5: {debug_top}")

    # ==========================================================================
    # 4. 溯源路径落地 (Traceback Paths)
    # ==========================================================================
    graph = knowledge_graph_inst._graph
    structural_path_edges = [] 
    added_edges_keys = set()

    # 筛选父子对: Seed (BGE) -> Target (RGCN New)
    parent_child_pairs = []
    seen_pairs = set()
   
    for seed, target, score, _ in rgcn_sims:
        if target in final_selected_names and seed != target:
            # 仅为不在 BGE 列表里的增量实体找路径
            if seed in bge_entity_names and target not in bge_entity_names:
                pair_key = (seed, target)
                if pair_key not in seen_pairs:
                    parent_child_pairs.append((seed, target, score))
                    seen_pairs.add(pair_key)
 # =======================================================
    # [修改 1] 初始化一个列表，专门存原始路径
    # =======================================================
    rgcn_raw_paths_for_stats = []
    # 寻找物理路径
    for src, tgt, score in parent_child_pairs:
        try:
            path_nodes = nx.shortest_path(graph, source=src, target=tgt)
            if len(path_nodes) > 4: continue
             # =======================================================
            # [修改 2] 只要路径有效，就把它存下来！
            # ===================================================
            rgcn_raw_paths_for_stats.append(path_nodes)
            path_sub = graph.subgraph(path_nodes)
            for i in range(len(path_nodes)-1):
                u, v = path_nodes[i], path_nodes[i+1]
                if u == v: continue # 过滤自环
                
                edata = path_sub.get_edge_data(u, v)
                if not edata: continue
                
                edges_list = list(edata.values()) if graph.is_multigraph() else [edata]
                if len(edges_list) > 3: edges_list = edges_list[:3] # 限制平行边

                for attr in edges_list:
                    raw_rel = attr.get("relation_type", attr.get("label", "RELATED"))
                    raw_desc = attr.get("description", "")
                    k = (u, v, raw_rel, raw_desc[:20])
                    
                    if k in added_edges_keys: continue
                    
                    final_desc = raw_desc if raw_desc.strip() else f"{u} is structurally related to {v}."
                    
                    structural_path_edges.append({
                        "src": u, "tgt": v, "relation": raw_rel, "desc": final_desc,
                        "weight": score, "rank": 0 # 占位
                    })
                    added_edges_keys.add(k)
        except: continue

    # 批量获取真实 Rank (度数)
    if structural_path_edges:
        real_ranks = await asyncio.gather(
            *[knowledge_graph_inst.edge_degree(e["src"], e["tgt"]) for e in structural_path_edges]
        )
        for edge, rank in zip(structural_path_edges, real_ranks):
            edge["rank"] = rank
    # ==========================================================================
    # 5. 检索社区与文本块 (Community & Text Units)
    # ⭐️ 关键点：因为 overall_node_datas 变了，这里召回的 chunks 也会变！
    # ==========================================================================
    use_communities = await _find_most_related_community_from_entities(
        final_query_nodes, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        final_query_nodes, query_param, text_chunks_db, knowledge_graph_inst
    )
    # ==========================================================================
    # 6. 互联路径 (Interconnection)
    # ==========================================================================
    def find_path_with_required_nodes(graph, source, target, required_nodes):
            final_path = []
            current_node = source
            for next_node in required_nodes:
                try:
                    sub_path = nx.shortest_path(graph, source=current_node, target=next_node)
                except nx.NetworkXNoPath:
                    final_path.extend([next_node])
                    current_node = next_node
                    continue
                if final_path:
                    final_path.extend(sub_path[1:])
                else:
                    final_path.extend(sub_path)
                current_node = next_node
            try:
                sub_path = nx.shortest_path(graph, source=current_node, target=target)
                final_path.extend(sub_path[1:])
            except nx.NetworkXNoPath:
                final_path.extend([target])
            return final_path
    key_entities = []
    if use_communities:
        for c in use_communities:
            cur = [e["entity_name"] for e in results if e["entity_name"] in c["nodes"]][:query_param.top_m]
            key_entities.extend(cur)
    else:
        key_entities = [e["entity_name"] for e in results[:query_param.top_m]]
    
    key_entities = list(set(key_entities))
    existing_path_edges = []
    if len(key_entities) > 1:
        try:
            p = find_path_with_required_nodes(graph, key_entities[0], key_entities[-1], key_entities[1:-1])
            path_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(r) for r in p])
            path_datas = [d for d in path_datas if d is not None]
            if path_datas:
                existing_path_edges = await _find_most_related_edges_from_paths(
                    path_datas, p, query_param, knowledge_graph_inst
                )
        except Exception:
            pass

    # ==========================================================================
    # 7. 最终合并、去重与排序 (Global Merge & Sort)
    # ==========================================================================
    
    reasoning_path_section_list = [["id", "source", "target", "relation_type", "description", "weight", "rank"]]
    path_id = 0
    seen_edges_fingerprint = set()

    # --- 第一梯队：原有社区互联路径 (Core Logic) ---
    # 这些是 HiRAG 认为最重要的路径，必须排在前面，且保持原顺序
    if existing_path_edges:
        # 原有路径通常已经按重要性排好序了，直接加入
        for e in existing_path_edges:
            src, tgt = e.get("src_tgt", (None, None))
            rel = e.get("relation_type", "UNKNOWN")
            
            # 指纹: (Source, Target, Relation)
            k = tuple(sorted((src, tgt))) + (rel,)
            
            if src and tgt and k not in seen_edges_fingerprint:
                reasoning_path_section_list.append([
                    path_id,
                    src, tgt,
                    rel,
                    e.get("description", ""),
                    e.get("weight", 1.0),
                    e.get("rank", 10) # 保留原始 Rank
                ])
                path_id += 1
                seen_edges_fingerprint.add(k)


    # --- 第二梯队：R-GCN 物理溯源路径 (Supplementary) ---
    # 先对 R-GCN 路径内部做个小排序（按相似度降序），保证追加的是质量相对高的
    structural_path_edges.sort(key=lambda x: x["weight"], reverse=True)

    for edge in structural_path_edges:
        # 制作同样的指纹
        k = tuple(sorted((edge["src"], edge["tgt"]))) + (edge["relation"],)
        
        # 只有当这条边【没有】在第一梯队出现过，才加进去
        if k not in seen_edges_fingerprint:
            reasoning_path_section_list.append([
                path_id,
                edge["src"],
                edge["tgt"],
                edge["relation"],
                edge["desc"],
                f"{float(edge['weight']):.2f}", # R-GCN 相似度
                edge["rank"]             # 真实度数
            ])
            path_id += 1
            seen_edges_fingerprint.add(k)
    # 生成 CSV
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)
    
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(final_query_nodes):
        entites_section_list.append([
            i, n["entity_name"], n.get("entity_type", "UNKNOWN"),
            n.get("description", "UNKNOWN"), n["rank"],
        ])
    entities_context = list_of_list_to_csv(entites_section_list)

    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"].replace("\n", " ")])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    # 日志记录
    entities_log = [f'"{n["entity_name"]}"' for n in final_query_nodes]
    communities_log = [(c.get("level"), c.get("title", f"Cluster {c.get('cluster')}")) for c in use_communities]
    chunks_log = [(t.get("full_doc_id") or t.get("doc_id"), t.get("chunk_order_index") or t.get("index", 0)) for t in use_text_units]
    
    logger.info(f"====== References (RGCN Diversity Enhanced) ======\n"
                f"Entities ({len(entities_log)}): {entities_log}\n"
                f"Communities ({len(communities_log)}): {communities_log}\n"
                f"Chunks ({len(chunks_log)}): {chunks_log}\n")
    # # ==========================================================================
    # # [修改点] 调用埋点函数
    # # ==========================================================================
    # # 1. 准备数据源变量 (防止变量未定义报错)
    # target_bridging = existing_path_edges if 'existing_path_edges' in locals() else []
    # target_physical = structural_path_edges if 'structural_path_edges' in locals() else []

    # # 2. 准备来源集合 (用于打标签)
    # # HI 来源: 你的代码里已经有 bge_entity_names
    # hi_source_set = bge_entity_names if 'bge_entity_names' in locals() else set()
    
    # # RGCN 来源: 你的代码里有 top_rgcn_names (这是 RGCN 推荐出的候选人)
    # # 注意：top_rgcn_names 是个列表，转成 set 查找更快
    # rgcn_source_set = set(top_rgcn_names) if 'top_rgcn_names' in locals() else set()

    # # 3. 提取数据
    # data = extract_data_snapshot(
    #     query=query,
    #     final_entities=final_query_nodes, 
    #     use_communities=use_communities,
        
    #     # [传入] 两个来源集合
    #     hi_source_names=hi_source_set,
    #     rgcn_source_names=rgcn_source_set,
        
    #     bridging_paths=target_bridging,
    #     physical_paths=target_physical  
    # )
    # # 保存
    # with open("logs/hi_rerank_rgcn.jsonl", "a", encoding="utf-8") as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4)
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
#TODO gcn的修改
# ---------------------------hi+gcn+重排序-----------------------------
async def _build_gcn_path_rank_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    # ==========================================================================
    # 1. BGE 语义检索 (Semantic Anchoring)
    # ==========================================================================
    results = await entities_vdb.query(query, top_k=query_param.top_k*10)

    if not len(results):
        return None
        
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    
    base_node_datas = node_datas[:query_param.top_k]
    
    # ==========================================================================
    # 2. R-GCN 结构扩展 (Structural Expansion)
    # ==========================================================================
    overall_node_datas, rgcn_sims = await _rgcn_expand_entities(
        base_node_datas, knowledge_graph_inst, query_param, global_config
    )
 
    
    bge_entity_names = set(n["entity_name"] for n in base_node_datas)

    # ==========================================================================
    # 3. [Rerank API 版] 实体混合重排序
    # 策略：把 BGE 和 R-GCN 的结果混合，交给 Rerank 模型统一打分
    # ==========================================================================
    
    # A. 准备候选池 (Candidate Pool)
    pool_nodes = []
    seen_in_pool = set()
    
    # # 1. 提取 BGE Top-3 (锁定组)
    bge_top3_nodes = []
    for n in base_node_datas[:3]:
        bge_top3_nodes.append(n)
        seen_in_pool.add(n["entity_name"])

    # 2. 提取剩下的 BGE 实体 (待重排组)
    for n in base_node_datas[3:]:
        if n["entity_name"] not in seen_in_pool:
            pool_nodes.append(n)
            seen_in_pool.add(n["entity_name"])
        
    # 3. 提取 R-GCN 高票实体 (待重排组)
    rgcn_counts = Counter([x[1] for x in rgcn_sims])
    name_to_node = {n["entity_name"]: n for n in overall_node_datas}
    
    # 按票数排序，取前 50 个进入复赛
    top_rgcn_names = sorted(rgcn_counts.keys(), key=lambda k: rgcn_counts[k], reverse=True)
    
    for name in top_rgcn_names[:50]:
        if name not in seen_in_pool and name in name_to_node:
            pool_nodes.append(name_to_node[name])
            seen_in_pool.add(name)
            
    # B. 调用 Rerank API (只对“待重排组”进行打分)
    api_key, base_url, rerank_model_name = _resolve_rerank_api_config(global_config)

    # 执行重排
    # 注意：这里只传入 pool_nodes，不传 bge_top3_nodes
    reranked_pool = await call_rerank_api(
        query=query,
        candidates=pool_nodes,
        api_key=api_key,
        base_url=base_url,
        model_name=rerank_model_name
    )
    
    # C. 组装最终名单 (拼接：锁定组 + 重排后的优胜组)
    final_query_nodes = []
    
    # 1. 先放 BGE Top-3
    final_query_nodes.extend(bge_top3_nodes)          #目前不需要
    
    # 2. 再放 Rerank 后的高分实体
    # 注意去重，虽然逻辑上应该没重复，但为了保险
    seen_final = set(n["entity_name"] for n in bge_top3_nodes)
    
    for node in reranked_pool:
        if node["entity_name"] not in seen_final:
            final_query_nodes.append(node)
            seen_final.add(node["entity_name"])

    # D. 最终截断
    # 保证总数不超过 top_k * 2 (通常为了 Rerank 效果好，可以稍微放宽到 top_k)
    results = final_query_nodes
    final_query_nodes = final_query_nodes[:query_param.top_k*2]
    final_selected_names = set(n["entity_name"] for n in final_query_nodes)
    
    # [Log] 看看谁是 Top 5 (前3个应该是 BGE 原生的)
    debug_top = [f"{n['entity_name']}({n.get('_rerank_score', 'LOCK')})" for n in final_query_nodes[:5]]
    logger.info(f"[GCNHybrid Rerank] Top 5: {debug_top}")

    # ==========================================================================
    # 4. 溯源路径落地 (Traceback Paths)
    # ==========================================================================
    graph = knowledge_graph_inst._graph
    structural_path_edges = [] 
    added_edges_keys = set()

    # 筛选父子对: Seed (BGE) -> Target (RGCN New)
    parent_child_pairs = []
    seen_pairs = set()
   
    for seed, target, score, _ in rgcn_sims:
        if target in final_selected_names and seed != target:
            # 仅为不在 BGE 列表里的增量实体找路径
            if seed in bge_entity_names and target not in bge_entity_names:
                pair_key = (seed, target)
                if pair_key not in seen_pairs:
                    parent_child_pairs.append((seed, target, score))
                    seen_pairs.add(pair_key)
 # =======================================================
    # [修改 1] 初始化一个列表，专门存原始路径
    # =======================================================
    rgcn_raw_paths_for_stats = []
    # 寻找物理路径
    for src, tgt, score in parent_child_pairs:
        try:
            path_nodes = nx.shortest_path(graph, source=src, target=tgt)
            if len(path_nodes) > 4: continue
             # =======================================================
            # [修改 2] 只要路径有效，就把它存下来！
            # ===================================================
            rgcn_raw_paths_for_stats.append(path_nodes)
            path_sub = graph.subgraph(path_nodes)
            for i in range(len(path_nodes)-1):
                u, v = path_nodes[i], path_nodes[i+1]
                if u == v: continue # 过滤自环
                
                edata = path_sub.get_edge_data(u, v)
                if not edata: continue
                
                edges_list = list(edata.values()) if graph.is_multigraph() else [edata]
                if len(edges_list) > 3: edges_list = edges_list[:3] # 限制平行边

                for attr in edges_list:
                    raw_rel = attr.get("relation_type", attr.get("label", "RELATED"))
                    raw_desc = attr.get("description", "")
                    k = (u, v, raw_rel, raw_desc[:20])
                    
                    if k in added_edges_keys: continue
                    
                    final_desc = raw_desc if raw_desc.strip() else f"{u} is structurally related to {v}."
                    
                    structural_path_edges.append({
                        "src": u, "tgt": v, "relation": raw_rel, "desc": final_desc,
                        "weight": score, "rank": 0 # 占位
                    })
                    added_edges_keys.add(k)
        except: continue

    # 批量获取真实 Rank (度数)
    if structural_path_edges:
        real_ranks = await asyncio.gather(
            *[knowledge_graph_inst.edge_degree(e["src"], e["tgt"]) for e in structural_path_edges]
        )
        for edge, rank in zip(structural_path_edges, real_ranks):
            edge["rank"] = rank
    # ==========================================================================
    # 5. 检索社区与文本块 (Community & Text Units)
    # ⭐️ 关键点：因为 overall_node_datas 变了，这里召回的 chunks 也会变！
    # ==========================================================================
    use_communities = await _find_most_related_community_from_entities(
        final_query_nodes, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        final_query_nodes, query_param, text_chunks_db, knowledge_graph_inst
    )
    # ==========================================================================
    # 6. 互联路径 (Interconnection)
    # ==========================================================================
    def find_path_with_required_nodes(graph, source, target, required_nodes):
            final_path = []
            current_node = source
            for next_node in required_nodes:
                try:
                    sub_path = nx.shortest_path(graph, source=current_node, target=next_node)
                except nx.NetworkXNoPath:
                    final_path.extend([next_node])
                    current_node = next_node
                    continue
                if final_path:
                    final_path.extend(sub_path[1:])
                else:
                    final_path.extend(sub_path)
                current_node = next_node
            try:
                sub_path = nx.shortest_path(graph, source=current_node, target=target)
                final_path.extend(sub_path[1:])
            except nx.NetworkXNoPath:
                final_path.extend([target])
            return final_path
    key_entities = []
    if use_communities:
        for c in use_communities:
            cur = [e["entity_name"] for e in results if e["entity_name"] in c["nodes"]][:query_param.top_m]
            key_entities.extend(cur)
    else:
        key_entities = [e["entity_name"] for e in results[:query_param.top_m]]
    
    key_entities = list(set(key_entities))
    existing_path_edges = []
    if len(key_entities) > 1:
        try:
            p = find_path_with_required_nodes(graph, key_entities[0], key_entities[-1], key_entities[1:-1])
            path_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(r) for r in p])
            path_datas = [d for d in path_datas if d is not None]
            if path_datas:
                existing_path_edges = await _find_most_related_edges_from_paths(
                    path_datas, p, query_param, knowledge_graph_inst
                )
        except Exception:
            pass
    
   # ==========================================================================
    # 7. 最终合并、去重与排序 (GCN 专用版 - 强制同质化)
    # ==========================================================================
    
    reasoning_path_section_list = [["id", "source", "target", "relation_type", "description", "weight", "rank"]]
    path_id = 0
    seen_edges_fingerprint = set()

    # --- 第一梯队：原有社区互联路径 (Core Logic) ---
    # 这些是 HiRAG 认为最重要的路径，必须排在前面，且保持原顺序
    if existing_path_edges:
        # 原有路径通常已经按重要性排好序了，直接加入
        for e in existing_path_edges:
            src, tgt = e.get("src_tgt", (None, None))
            rel = e.get("relation_type", "UNKNOWN")
            
            # 指纹: (Source, Target, Relation)
            k = tuple(sorted((src, tgt))) + (rel,)
            
            if src and tgt and k not in seen_edges_fingerprint:
                reasoning_path_section_list.append([
                    path_id,
                    src, tgt,
                    rel,
                    e.get("description", ""),
                    e.get("weight", 1.0),
                    e.get("rank", 10) # 保留原始 Rank
                ])
                path_id += 1
                seen_edges_fingerprint.add(k)


    # # --- 第二梯队：R-GCN 物理溯源路径 (Supplementary) ---
    # # 先对 R-GCN 路径内部做个小排序（按相似度降序），保证追加的是质量相对高的
    structural_path_edges.sort(key=lambda x: x["weight"], reverse=True)

    for edge in structural_path_edges:
        # 制作同样的指纹
        k = tuple(sorted((edge["src"], edge["tgt"]))) + (edge["relation"],)
        
        # 只有当这条边【没有】在第一梯队出现过，才加进去
        if k not in seen_edges_fingerprint:
            reasoning_path_section_list.append([
                path_id,
                edge["src"],
                edge["tgt"],
                edge["relation"],
                edge["desc"],
                f"{float(edge['weight']):.2f}", # R-GCN 相似度
                edge["rank"]             # 真实度数
            ])
            path_id += 1
            seen_edges_fingerprint.add(k)
    # 生成 CSV
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)
    # ... (后面的 entities_context 等代码保持不变) ...
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(final_query_nodes):
        entites_section_list.append([
            i, n["entity_name"], n.get("entity_type", "UNKNOWN"),
            n.get("description", "UNKNOWN"), n["rank"],
        ])
    entities_context = list_of_list_to_csv(entites_section_list)

    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"].replace("\n", " ")])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    # 日志记录
    entities_log = [f'"{n["entity_name"]}"' for n in final_query_nodes]
    communities_log = [(c.get("level"), c.get("title", f"Cluster {c.get('cluster')}")) for c in use_communities]
    chunks_log = [(t.get("full_doc_id") or t.get("doc_id"), t.get("chunk_order_index") or t.get("index", 0)) for t in use_text_units]
    
    logger.info(f"====== References (RGCN Diversity Enhanced) ======\n"
                f"Entities ({len(entities_log)}): {entities_log}\n"
                f"Communities ({len(communities_log)}): {communities_log}\n"
                f"Chunks ({len(chunks_log)}): {chunks_log}\n")
    # # ==========================================================================
    # # [修改点] 调用埋点函数
    # # ==========================================================================
    # # 1. 准备数据源变量 (防止变量未定义报错)
    # target_bridging = existing_path_edges if 'existing_path_edges' in locals() else []
    # target_physical = structural_path_edges if 'structural_path_edges' in locals() else []

    # # 2. 准备来源集合 (用于打标签)
    # # HI 来源: 你的代码里已经有 bge_entity_names
    # hi_source_set = bge_entity_names if 'bge_entity_names' in locals() else set()
    
    # # RGCN 来源: 你的代码里有 top_rgcn_names (这是 RGCN 推荐出的候选人)
    # # 注意：top_rgcn_names 是个列表，转成 set 查找更快
    # rgcn_source_set = set(top_rgcn_names) if 'top_rgcn_names' in locals() else set()

    # # 3. 提取数据
    # data = extract_data_snapshot(
    #     query=query,
    #     final_entities=final_query_nodes, 
    #     use_communities=use_communities,
        
    #     # [传入] 两个来源集合
    #     hi_source_names=hi_source_set,
    #     rgcn_source_names=rgcn_source_set,
        
    #     bridging_paths=target_bridging,
    #     physical_paths=target_physical  
    # )
    # # 保存
    # with open("logs/hi_rerank_rgcn.jsonl", "a", encoding="utf-8") as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4)
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
# -------------------hi+rgcn------------------------
async def _build_rgcn_path1_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    # ==========================================================================
    # 1. BGE 语义检索 (Semantic Anchoring)
    # ==========================================================================
    results = await entities_vdb.query(query, top_k=query_param.top_k*10)

    if not len(results):
        return None
        
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    
    base_node_datas = node_datas[:query_param.top_k]
    
    # ==========================================================================
    # 2. R-GCN 结构扩展 (Structural Expansion)
    # ==========================================================================
    overall_node_datas, rgcn_sims = await _rgcn_expand_entities(
        base_node_datas, knowledge_graph_inst, query_param, global_config
    )
 
# ==========================================================================
    # 3. [Top-3 锁定 + 混合重排] 策略
    # 逻辑：
    # 1. BGE Top-3 强制置顶 (Index 0-2)
    # 2. 剩余 BGE + RGCN 通过语义门控后，按 (票数+相似度) 混合排序
    # ==========================================================================
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # A. 准备基础数据
    rgcn_counts = Counter([x[1] for x in rgcn_sims])
    
    # 识别 BGE 的不同梯队
    # bge_top3: 绝对核心，不参与竞争，直接保送
    # bge_rest: 需要和 RGCN 竞争
    bge_top3_names = set(n["entity_name"] for n in base_node_datas[:3])
    bge_all_names = set(n["entity_name"] for n in base_node_datas)
    
    # 获取所有候选实体的名称列表
    all_candidate_names = [n["entity_name"] for n in overall_node_datas]

    # -------------------------------------------------------
    # B. 计算向量相似度 (Batch Processing)
    # -------------------------------------------------------
    # 1. 获取 Query 向量
    query_embedding = await entities_vdb.embedding_func([query]) 
    
    # 2. 获取 RGCN 增量实体的向量 (带分批处理)
    # 我们只计算不是 BGE 的实体，节省开销
    rgcn_only_names = [n for n in all_candidate_names if n not in bge_all_names]
    name_to_sim = {} 

    if rgcn_only_names:
        BATCH_SIZE = 50 
        embeddings_list = []
        
        for i in range(0, len(rgcn_only_names), BATCH_SIZE):
            batch_names = rgcn_only_names[i : i + BATCH_SIZE]
            try:
                batch_embs = await entities_vdb.embedding_func(batch_names)
                embeddings_list.append(batch_embs)
            except Exception as e:
                # logger.warning(f"Embedding fetch failed: {e}")
                pass

        if embeddings_list:
            if isinstance(embeddings_list[0], np.ndarray):
                candidate_embeddings = np.concatenate(embeddings_list, axis=0)
            else:
                candidate_embeddings = np.array([item for batch in embeddings_list for item in batch])
            
            if len(candidate_embeddings) > 0:
                sims = cosine_similarity(query_embedding, candidate_embeddings)[0]
                name_to_sim = dict(zip(rgcn_only_names, sims))

    # -------------------------------------------------------
    # C. 综合打分 (Scoring)
    # -------------------------------------------------------
    candidate_scores = {}
    
    # 阈值设定 (建议 0.2 - 0.35)
    # 0.2 比较宽松，允许一定的语义漂移以换取结构发现
    SEMANTIC_THRESHOLD = 0.20 

    for n in overall_node_datas:
        name = n["entity_name"]
        
        # --- 情况 1: BGE Top-3 (直接跳过打分，因为它们肯定在最前面) ---
        if name in bge_top3_names:
            continue 

        # --- 情况 2: BGE 剩余实体 (给个基准分) ---
        if name in bge_all_names:
            # 3.0 分是一个很好的基准
            # 意味着 RGCN 实体只要有 2 票且相似度还可以，就能超过它
            candidate_scores[name] = 3.0
            continue
            
        # --- 情况 3: RGCN 新实体 (需过安检 + 算分) ---
        sim = name_to_sim.get(name, 0.0)
        
        # 🛑 [安检门]
        if sim < SEMANTIC_THRESHOLD:
            candidate_scores[name] = -1.0 # 淘汰
            continue
            
        # ✅ [算分]
        votes = rgcn_counts.get(name, 0)
        
        # 公式：票数(结构) + 相似度的一半(语义微调)
        # 举例：Votes=3, Sim=0.6 -> Score = 3.3 (超过 BGE 剩余实体，插队成功)
        # 举例：Votes=1, Sim=0.8 -> Score = 1.4 (排在 BGE 后面，作为补充)
        final_score = votes + (sim * 0.5) 
        
        candidate_scores[name] = final_score

    # -------------------------------------------------------
    # D. 分段组装与排序 (Assembly & Sorting)
    # -------------------------------------------------------
    
    final_query_nodes = []
    seen_names = set()
    
    # 1. 【锁定区】强制放入 BGE Top-3 (保持原始顺序)
    for node in base_node_datas[:3]:
        if node["entity_name"] not in seen_names:
            final_query_nodes.append(node)
            seen_names.add(node["entity_name"])
            
    # 2. 【竞争区】对剩余实体进行排序
    # 筛选出 BGE 剩余实体 和 通过安检的 RGCN 实体
    competition_pool = []
    for n in overall_node_datas:
        name = n["entity_name"]
        if name not in seen_names: # 排除已加入的 Top-3
            score = candidate_scores.get(name, -999)
            if score > 0:
                competition_pool.append(n)
                
    # 按分数降序排列 (这里决定了谁排第 4, 5, 6 名)
    competition_pool.sort(
        key=lambda x: candidate_scores.get(x["entity_name"], 0), 
        reverse=True
    )
    
    # 3. 追加到最终列表
    final_query_nodes.extend(competition_pool)
    
    # E. 最终截断
    results =  final_query_nodes
    final_query_nodes = final_query_nodes[:query_param.top_k * 2]
    final_selected_names = set(n["entity_name"] for n in final_query_nodes)

    # [Log] 验证顺序 (前3个应该是 BGE，后面有分数的混合排)
    debug_top = []
    for n in final_query_nodes[:6]:
        name = n['entity_name']
        if name in bge_top3_names:
            tag = "LOCK"
        else:
            tag = f"{candidate_scores.get(name):.1f}"
        debug_top.append(f"{name}({tag})")
        
    logger.info(f"[Hybrid Rank] Top 6: {debug_top}")
    # ==========================================================================
   # ==========================================================================
    # Phase 4: 溯源路径构建 (Traceback Path Construction)
    # [论文创新点]: 显式推理路径，解释“为什么引入这个新实体”
    # ==========================================================================
    graph = knowledge_graph_inst._graph
    structural_path_edges = []
    added_edges_keys = set()
    
    # 筛选父子对: Seed (BGE) -> Target (RGCN New)
    parent_child = []
    bge_names = set(n["entity_name"] for n in base_node_datas)
    
    for seed, tgt, score, _ in rgcn_sims:
        if tgt in final_selected_names and seed != tgt:
            # 仅为 RGCN 带来的“增量实体”找路径
            if seed in bge_names and tgt not in bge_names:
                parent_child.append((seed, tgt, score))

    # 物理路径查找
    for src, tgt, score in parent_child:
        try:
            path_nodes = nx.shortest_path(graph, source=src, target=tgt)
            if len(path_nodes) > 4: continue # 限制跳数
            
            # 提取路径上的边
            path_sub = graph.subgraph(path_nodes)
            for i in range(len(path_nodes)-1):
                u, v = path_nodes[i], path_nodes[i+1]
                if u == v: continue # 过滤自环

                edata = path_sub.get_edge_data(u, v)
                if not edata: continue
                
                # 兼容 MultiGraph
                edges_list = list(edata.values()) if graph.is_multigraph() else [edata]
                if len(edges_list) > 3: edges_list = edges_list[:3] # 限制平行边数量

                for attr in edges_list:
                    raw_rel = attr.get("relation_type", attr.get("label", "RELATED"))
                    raw_desc = attr.get("description", "")
                    k = (u, v, raw_rel, raw_desc[:20])
                    
                    if k in added_edges_keys: continue
                    
                    # 动态描述生成
                    final_desc = raw_desc if raw_desc.strip() else f"{u} is structurally related to {v}."
                    
                    structural_path_edges.append({
                        "src": u, "tgt": v, "relation": raw_rel, "desc": final_desc,
                        "weight": score, "rank": 0 # 占位，稍后查真实Rank
                    })
                    added_edges_keys.add(k)
        except: continue

    # 批量获取真实 Rank (度数)
    if structural_path_edges:
        real_ranks = await asyncio.gather(
            *[knowledge_graph_inst.edge_degree(e["src"], e["tgt"]) for e in structural_path_edges]
        )
        for edge, rank in zip(structural_path_edges, real_ranks):
            edge["rank"] = rank

  
    # ==========================================================================
    # 5. 检索社区与文本块 (Community & Text Units)
    # ⭐️ 关键点：因为 overall_node_datas 变了，这里召回的 chunks 也会变！
    # ==========================================================================
    use_communities = await _find_most_related_community_from_entities(
        final_query_nodes, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        final_query_nodes, query_param, text_chunks_db, knowledge_graph_inst
    )

    # ==========================================================================
    # 6. 互联路径 (Interconnection)
    # ==========================================================================
    def find_path_with_required_nodes(graph, source, target, required_nodes):
        final_path = []
        current_node = source
        for next_node in required_nodes:
            try:
                sub_path = nx.shortest_path(graph, source=current_node, target=next_node)
            except nx.NetworkXNoPath:
                final_path.extend([next_node])
                current_node = next_node
                continue
            if final_path:
                final_path.extend(sub_path[1:])
            else:
                final_path.extend(sub_path)
            current_node = next_node
        try:
            sub_path = nx.shortest_path(graph, source=current_node, target=target)
            final_path.extend(sub_path[1:])
        except nx.NetworkXNoPath:
            final_path.extend([target])
        return final_path

    key_entities = []
    if use_communities:
        for c in use_communities:
            cur = [e["entity_name"] for e in results if e["entity_name"] in c["nodes"]][:query_param.top_m]
            key_entities.extend(cur)
    else:
         key_entities = [e["entity_name"] for e in results[:query_param.top_m]]
    
    key_entities = list(set(key_entities))
    existing_path_edges = []
    if len(key_entities) > 1:
        try:
            p = find_path_with_required_nodes(graph, key_entities[0], key_entities[-1], key_entities[1:-1])
            path_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(r) for r in p])
            path_datas = [d for d in path_datas if d is not None]
            if path_datas:
                existing_path_edges = await _find_most_related_edges_from_paths(
                    path_datas, p, query_param, knowledge_graph_inst
                )
        except Exception:
            pass

    # ==========================================================================
    # 7. 最终合并、去重与排序 (Global Merge & Sort)
    # ==========================================================================
    
    reasoning_path_section_list = [["id", "source", "target", "relation_type", "description", "weight", "rank"]]
    path_id = 0
    seen_edges_fingerprint = set()

    # --- 第一梯队：原有社区互联路径 (Core Logic) ---
    # 这些是 HiRAG 认为最重要的路径，必须排在前面，且保持原顺序
    if existing_path_edges:
        # 原有路径通常已经按重要性排好序了，直接加入
        for e in existing_path_edges:
            src, tgt = e.get("src_tgt", (None, None))
            rel = e.get("relation_type", "UNKNOWN")
            
            # 指纹: (Source, Target, Relation)
            k = tuple(sorted((src, tgt))) + (rel,)
            
            if src and tgt and k not in seen_edges_fingerprint:
                reasoning_path_section_list.append([
                    path_id,
                    src, tgt,
                    rel,
                    e.get("description", ""),
                    e.get("weight", 1.0),
                    e.get("rank", 10) # 保留原始 Rank
                ])
                path_id += 1
                seen_edges_fingerprint.add(k)

    # --- 第二梯队：R-GCN 物理溯源路径 (Supplementary) ---
    # 追加在后面，作为补充证据。如果不重要，截断时会被优先切掉。
    
    # # 先对 R-GCN 路径内部做个小排序（按相似度降序），保证追加的是质量相对高的
    # structural_path_edges.sort(key=lambda x: x["weight"], reverse=True)

    # for edge in structural_path_edges:
    #     # 制作同样的指纹
    #     k = tuple(sorted((edge["src"], edge["tgt"]))) + (edge["relation"],)
        
    #     # 只有当这条边【没有】在第一梯队出现过，才加进去
    #     if k not in seen_edges_fingerprint:
    #         reasoning_path_section_list.append([
    #             path_id,
    #             edge["src"],
    #             edge["tgt"],
    #             edge["relation"],
    #             edge["desc"],
    #             f"{float(edge['weight']):.2f}", # R-GCN 相似度
    #             edge["rank"]             # 真实度数
    #         ])
    #         path_id += 1
    #         seen_edges_fingerprint.add(k)

    # 生成 CSV
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(final_query_nodes):
        entites_section_list.append([
            i, n["entity_name"], n.get("entity_type", "UNKNOWN"),
            n.get("description", "UNKNOWN"), n["rank"],
        ])
    entities_context = list_of_list_to_csv(entites_section_list)

    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"].replace("\n", " ")])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    # 日志记录
    entities_log = [f'"{n["entity_name"]}"' for n in final_query_nodes]
    communities_log = [(c.get("level"), c.get("title", f"Cluster {c.get('cluster')}")) for c in use_communities]
    chunks_log = [(t.get("full_doc_id") or t.get("doc_id"), t.get("chunk_order_index") or t.get("index", 0)) for t in use_text_units]
    
    logger.info(f"====== References (RGCN Diversity Enhanced) ======\n"
                f"Entities ({len(entities_log)}): {entities_log}\n"
                f"Communities ({len(communities_log)}): {communities_log}\n"
                f"Chunks ({len(chunks_log)}): {chunks_log}\n")

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
#---------------------------hi + rank -------------------------
async def _build_hierarchical_rank_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict = None,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k * 10)          # find the top-k(20) related entities

    if not len(results):    # results just with entity name
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [                          # add rank, which is the degree
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
  # ========================== 2. [修正版] BGE Top-50 + 锁定前三 ==========================
    results = node_datas
    # A. 准备候选池
    pool_nodes = []
    seen_names = set()
    
    # 1. 【核心修正】锁定 BGE 前 3 名 (保住下限)
    # 这些实体直接保送进最终列表，不参与 Rerank 被淘汰
    locked_nodes = []
    for n in node_datas[:3]:
        n["_source"] = "BGE_LOCK"
        # 即使锁定，也最好检查下描述，为了下游生成
        if not n.get("description"): n["description"] = "Key entity from initial retrieval."
        locked_nodes.append(n)
        seen_names.add(n["entity_name"])

    # 2. 选取剩下的候选者 (Rank 4 - 50)
    CANDIDATE_LIMIT = 50
    candidates = node_datas[3:CANDIDATE_LIMIT]
    
    for n in candidates:
        if n["entity_name"] not in seen_names:
            n["_source"] = "BGE_RERANK"
            # 描述补全
            desc = n.get("description")
            if not desc or len(str(desc).strip()) == 0:
                 n["description"] = f"Entity related to {query[:20]}..."
            
            pool_nodes.append(n)
            seen_names.add(n["entity_name"])

    # B. 调用 Rerank API (只排后面的人)
    api_key, base_url, rerank_model_name = _resolve_rerank_api_config(global_config)

    if pool_nodes:
        reranked_pool = await call_rerank_api(
            query=query,
            candidates=pool_nodes,
            api_key=api_key,
            base_url=base_url,
            model_name=rerank_model_name
        )
    else:
        reranked_pool = []

    final_query_nodes = []
    
    # 1. 先放入“保送”的 Top 3
    final_query_nodes.extend(locked_nodes)
    
    # 【关键修正】创建一个新的去重集合，初始只包含 Top 3
    # 此时千万不能用 seen_names，因为那个里面包含所有候选者
    added_to_final = set(n["entity_name"] for n in locked_nodes)
    
    # [Log] 调试信息：看看 Rerank 返回了多少个
    if reranked_pool:
        logger.info(f"📊 [Merge Start] Locked: {len(locked_nodes)}, Reranked Candidates: {len(reranked_pool)}")
    
    # 2. 放入 Rerank 后的胜出者
    count_added = 0
    for node in reranked_pool:
        name = node["entity_name"]
        
        # 检查是否已经在最终列表里（即是否撞上了 Top 3）
        if name not in added_to_final: 
             
             # 分数阈值：设低一点 (-10.0)，防止把本来就少的相关实体误杀
             score = node.get('_rerank_score', -99.0)
             
             if score > -10.0: 
                 final_query_nodes.append(node)
                 added_to_final.add(name) # 标记为已加入
                 count_added += 1
    
    logger.info(f"📊 [Merge End] Successfully added {count_added} entities from Rerank pool.")

    # ========================== Step D. 最终截断 (Truncate) ==========================
    
    # 取 Top K * 2 (例如 20 * 2 = 40 个) 给下游
    node_datas = final_query_nodes[:query_param.top_k * 2]
    
    # [Log] 验证 Top 6 现在的组成 (应该混合了 LOCK 和 分数)
    debug_top = []
    for n in node_datas[:6]:
        src = n.get('_source', '?')
        score = n.get('_rerank_score', 'LOCK')
        # 格式化一下分数，好看点
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        debug_top.append(f"{n['entity_name']}({src}:{score_str})")
        
    logger.info(f"[Baseline: Lock3 + Rerank] Final List Top 6: {debug_top}")
    # overall_node_datas = node_datas
    # node_datas = node_datas[:query_param.top_k*2]
    # node_datas = final_query_nodes[:query_param.top_k * 2]

    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    # use_relations = await _find_most_related_edges_from_entities(
    #     node_datas, query_param, knowledge_graph_inst
    # )

    def find_path_with_required_nodes(graph, source, target, required_nodes):
        # inital final path
        final_path = []
        # 起点设置为当前节点
        current_node = source

        # 遍历必经节点
        for next_node in required_nodes:
            # 找到从当前节点到下一个必经节点的最短路径
            try:
                sub_path = nx.shortest_path(graph, source=current_node, target=next_node)
            except nx.NetworkXNoPath:
                # raise ValueError(f"No path between {current_node} and {next_node}.")
                final_path.extend([next_node])
                current_node = next_node
                continue
            
            # 合并路径（避免重复添加当前节点）
            if final_path:
                final_path.extend(sub_path[1:])  # 从第二个节点开始添加，避免重复
            else:
                final_path.extend(sub_path)
            
            # 更新当前节点为下一个必经节点
            current_node = next_node

        # 最后，从最后一个必经节点到目标节点的路径
        try:
            sub_path = nx.shortest_path(graph, source=current_node, target=target)
            final_path.extend(sub_path[1:])  # 从第二个节点开始添加，避免重复
        except nx.NetworkXNoPath:
            # raise ValueError(f"No path between {current_node} and {target}.")
            final_path.extend([target])

        return final_path

    # find some top-k entities in each communities in use_communities
    key_entities = []
    max_entity_num = query_param.top_m
    if use_communities:
        for c in use_communities:
            cur_community_key_entities = []
            community_entities = c['nodes']
            # find the top-k entities in this community
            cur_community_key_entities.extend(
                [e for e in results if e['entity_name'] in community_entities][:max_entity_num]
            )
            key_entities.append(cur_community_key_entities)
    else:
        key_entities = [results[:max_entity_num]]
    # unique key entities
    key_entities = [[e['entity_name'] for e in k] for k in key_entities]
    key_entities = list(set([k for kk in key_entities for k in kk]))
    # find the shortest path between the key entities
    try:
        path = find_path_with_required_nodes(knowledge_graph_inst._graph, key_entities[0], key_entities[-1], key_entities[1:-1])
        # path = list(set(path))
        path_datas = await asyncio.gather(      # get full information of retrieved entities
            *[knowledge_graph_inst.get_node(r) for r in path]
        )
        path_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r) for r in path]
        )
        path_datas = [                          # add rank, which is the degree
            {**n, "entity_name": k, "rank": d}
            for k, n, d in zip(path, path_datas, path_degrees)
            if n is not None
        ]
        # use_reasoning_path = await _find_most_related_edges_from_entities(
        #                     path_datas, query_param, knowledge_graph_inst
        #                 )
        use_reasoning_path = await _find_most_related_edges_from_paths(
                                path_datas, path, query_param, knowledge_graph_inst
                            )
    except ValueError as e:
        print(e)
    
    # # fetch the relations of the reasoning paths
    # reasoning_path = []
    # for i in range(len(path) - 1):
    #     src = path[i]
    #     tgt = path[i + 1]
    #     cur_relation = (await knowledge_graph_inst.get_edge(src, tgt))['description']
    #     reasoning_path.append(cur_relation)
    # reasoning_path = list(set(reasoning_path))

    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_reasoning_path)} reasoning path items, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    
    reasoning_path_section_list = [
    ["id", "source", "target", "relation_type", "description", "weight", "rank"]
]

    for i, e in enumerate(use_reasoning_path):
        # src_tgt 是 (source, target)
        src, tgt = e.get("src_tgt", (None, None))
        relation_type = e.get("relation_type", "UNKNOWN")
        description = e.get("description", "UNKNOWN")
        weight = e.get("weight", 1.0)
        rank = e.get("rank", 0)

        reasoning_path_section_list.append(
            [i, src, tgt, relation_type, description, weight, rank]
        )

    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)
    
    # reasoning_path_context = list_of_list_to_csv([["id", "content"]] + [[i, p] for i, p in enumerate(reasoning_path)])
    
    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"].replace("\n", " ")])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    # display reference info
    entities = [n["entity_name"] for n in node_datas]
    communities = [(c["level"], c["title"]) for c in use_communities]
    chunks = [(t["full_doc_id"], t["chunk_order_index"]) for t in use_text_units]

    references_context = (
        f"Entities ({len(entities)}): {entities}\n\n"
        f"Communities (level, cluster_id) ({len(communities)}): {communities}\n\n"
        f"Chunks (doc_id, chunk_index) ({len(chunks)}): {chunks}\n"
    )

    logging.info(f"====== References ======:\n{references_context}")
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
async def _build_rgcn_path_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    # ==========================================================================
    # 1. BGE 语义检索 (Semantic Anchoring)
    # ==========================================================================
    results = await entities_vdb.query(query, top_k=query_param.top_k*10)

    if not len(results):
        return None
        
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    
    base_node_datas = node_datas[:query_param.top_k]
    
    # ==========================================================================
    # 2. R-GCN 结构扩展 (Structural Expansion)
    # ==========================================================================
    overall_node_datas, rgcn_sims = await _rgcn_expand_entities(
        base_node_datas, knowledge_graph_inst, query_param, global_config
    )
 
    # 排序按照次数
    rgcn_counts = Counter()
    for _, target_name, _, _ in rgcn_sims:
        rgcn_counts[target_name] += 1
    name_to_node_map = {n["entity_name"]: n for n in overall_node_datas}
    rgcn_candidate_names = list(rgcn_counts.keys())
    rgcn_candidate_names.sort(
        key=lambda name: (rgcn_counts[name], name_to_node_map.get(name, {}).get("rank", 0)),
        reverse=True,
    )
    # A. 准备数据
    # 统计 R-GCN 票数
    rgcn_votes = Counter([x[1] for x in rgcn_sims])
    
    # 建立 BGE 查找表 (用于快速判断是否是语义核心)
    bge_names_set = set(n["entity_name"] for n in base_node_datas)
    
    # 计算最大度数 (用于归一化)
    max_degree = max([n.get("rank", 0) for n in overall_node_datas], default=1)
    
    # B. 为所有实体打分
    candidate_scores = {}
    
    for node in overall_node_datas:
        name = node["entity_name"]
        score = 0.0
        
        # 1. R-GCN 结构分 (1票 = 1分)
        # 票数越高，说明结构关联越强
        score += rgcn_votes.get(name, 0)
        
        # 2. BGE 语义分 (3分保底)
        # 只要是 BGE 找回来的，自带 3 分，保证不会轻易被低频 R-GCN 噪音挤掉
        if name in bge_names_set:
            score += 3.0
            
        # 3. 图谱度数分 (0.5分微调)
        # 防止分数相同时随机排序，优先选大节点
        degree_bonus = (node.get("rank", 0) / max_degree) * 0.5
        score += degree_bonus
        
        candidate_scores[name] = score

    # C. 全局排序
    # 这里的顺序非常关键！它决定了下游文本块检索的截断顺序
    sorted_all_nodes = sorted(
        overall_node_datas,
        key=lambda x: candidate_scores.get(x["entity_name"], 0),
        reverse=True
    )
    
    # D. 截断与更新
    # 这一步生成的 final_query_nodes 将同时用于【文本检索】和【路径查找】
    final_query_nodes = sorted_all_nodes[:query_param.top_k * 2]
    
    # 更新 set 用于快速查找
    final_selected_names = set(n["entity_name"] for n in final_query_nodes)
    # 3. 截断
    # overall_node_datas = full_candidate_pool[:query_param.top_k*2]
    # final_selected_names = set(n["entity_name"] for n in overall_node_datas)
    # ==========================================================================
    # 🚀 4. 核心修改：溯源路径落地 (获取真实图谱 Rank)
    # ==========================================================================
   # ==========================================================================
    # Phase 4: 溯源路径构建 (Traceback Path Construction)
    # [论文创新点]: 显式推理路径，解释“为什么引入这个新实体”
    # ==========================================================================
    graph = knowledge_graph_inst._graph
    structural_path_edges = []
    added_edges_keys = set()
    
    # 筛选父子对: Seed (BGE) -> Target (RGCN New)
    parent_child = []
    bge_names = set(n["entity_name"] for n in base_node_datas)
    
    for seed, tgt, score, _ in rgcn_sims:
        if tgt in final_selected_names and seed != tgt:
            # 仅为 RGCN 带来的“增量实体”找路径
            if seed in bge_names and tgt not in bge_names:
                parent_child.append((seed, tgt, score))

    # 物理路径查找
    for src, tgt, score in parent_child:
        try:
            path_nodes = nx.shortest_path(graph, source=src, target=tgt)
            if len(path_nodes) > 4: continue # 限制跳数
            
            # 提取路径上的边
            path_sub = graph.subgraph(path_nodes)
            for i in range(len(path_nodes)-1):
                u, v = path_nodes[i], path_nodes[i+1]
                if u == v: continue # 过滤自环

                edata = path_sub.get_edge_data(u, v)
                if not edata: continue
                
                # 兼容 MultiGraph
                edges_list = list(edata.values()) if graph.is_multigraph() else [edata]
                if len(edges_list) > 3: edges_list = edges_list[:3] # 限制平行边数量

                for attr in edges_list:
                    raw_rel = attr.get("relation_type", attr.get("label", "RELATED"))
                    raw_desc = attr.get("description", "")
                    k = (u, v, raw_rel, raw_desc[:20])
                    
                    if k in added_edges_keys: continue
                    
                    # 动态描述生成
                    final_desc = raw_desc if raw_desc.strip() else f"{u} is structurally related to {v}."
                    
                    structural_path_edges.append({
                        "src": u, "tgt": v, "relation": raw_rel, "desc": final_desc,
                        "weight": score, "rank": 0 # 占位，稍后查真实Rank
                    })
                    added_edges_keys.add(k)
        except: continue

    # 批量获取真实 Rank (度数)
    if structural_path_edges:
        real_ranks = await asyncio.gather(
            *[knowledge_graph_inst.edge_degree(e["src"], e["tgt"]) for e in structural_path_edges]
        )
        for edge, rank in zip(structural_path_edges, real_ranks):
            edge["rank"] = rank

  
    # ==========================================================================
    # 5. 检索社区与文本块 (Community & Text Units)
    # ⭐️ 关键点：因为 overall_node_datas 变了，这里召回的 chunks 也会变！
    # ==========================================================================
    use_communities = await _find_most_related_community_from_entities(
        final_query_nodes, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        final_query_nodes, query_param, text_chunks_db, knowledge_graph_inst
    )

    # ==========================================================================
    # 6. 互联路径 (Interconnection)
    # ==========================================================================
    def find_path_with_required_nodes(graph, source, target, required_nodes):
        final_path = []
        current_node = source
        for next_node in required_nodes:
            try:
                sub_path = nx.shortest_path(graph, source=current_node, target=next_node)
            except nx.NetworkXNoPath:
                final_path.extend([next_node])
                current_node = next_node
                continue
            if final_path:
                final_path.extend(sub_path[1:])
            else:
                final_path.extend(sub_path)
            current_node = next_node
        try:
            sub_path = nx.shortest_path(graph, source=current_node, target=target)
            final_path.extend(sub_path[1:])
        except nx.NetworkXNoPath:
            final_path.extend([target])
        return final_path

    key_entities = []
    if use_communities:
        for c in use_communities:
            cur = [e["entity_name"] for e in results if e["entity_name"] in c["nodes"]][:query_param.top_m]
            key_entities.extend(cur)
    else:
         key_entities = [e["entity_name"] for e in results[:query_param.top_m]]
    
    key_entities = list(set(key_entities))
    existing_path_edges = []
    if len(key_entities) > 1:
        try:
            p = find_path_with_required_nodes(graph, key_entities[0], key_entities[-1], key_entities[1:-1])
            path_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(r) for r in p])
            path_datas = [d for d in path_datas if d is not None]
            if path_datas:
                existing_path_edges = await _find_most_related_edges_from_paths(
                    path_datas, p, query_param, knowledge_graph_inst
                )
        except Exception:
            pass

    # ==========================================================================
    # 7. 最终合并、去重与排序 (Global Merge & Sort)
    # ==========================================================================
    
    reasoning_path_section_list = [["id", "source", "target", "relation_type", "description", "weight", "rank"]]
    path_id = 0
    seen_edges_fingerprint = set()

    # --- 第一梯队：原有社区互联路径 (Core Logic) ---
    # 这些是 HiRAG 认为最重要的路径，必须排在前面，且保持原顺序
    if existing_path_edges:
        # 原有路径通常已经按重要性排好序了，直接加入
        for e in existing_path_edges:
            src, tgt = e.get("src_tgt", (None, None))
            rel = e.get("relation_type", "UNKNOWN")
            
            # 指纹: (Source, Target, Relation)
            k = tuple(sorted((src, tgt))) + (rel,)
            
            if src and tgt and k not in seen_edges_fingerprint:
                reasoning_path_section_list.append([
                    path_id,
                    src, tgt,
                    rel,
                    e.get("description", ""),
                    e.get("weight", 1.0),
                    e.get("rank", 10) # 保留原始 Rank
                ])
                path_id += 1
                seen_edges_fingerprint.add(k)

    # --- 第二梯队：R-GCN 物理溯源路径 (Supplementary) ---
    # 追加在后面，作为补充证据。如果不重要，截断时会被优先切掉。
    
    # 先对 R-GCN 路径内部做个小排序（按相似度降序），保证追加的是质量相对高的
    structural_path_edges.sort(key=lambda x: x["weight"], reverse=True)

    for edge in structural_path_edges:
        # 制作同样的指纹
        k = tuple(sorted((edge["src"], edge["tgt"]))) + (edge["relation"],)
        
        # 只有当这条边【没有】在第一梯队出现过，才加进去
        if k not in seen_edges_fingerprint:
            reasoning_path_section_list.append([
                path_id,
                edge["src"],
                edge["tgt"],
                edge["relation"],
                edge["desc"],
                f"{float(edge['weight']):.2f}", # R-GCN 相似度
                edge["rank"]             # 真实度数
            ])
            path_id += 1
            seen_edges_fingerprint.add(k)

    # 生成 CSV
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(overall_node_datas):
        entites_section_list.append([
            i, n["entity_name"], n.get("entity_type", "UNKNOWN"),
            n.get("description", "UNKNOWN"), n["rank"],
        ])
    entities_context = list_of_list_to_csv(entites_section_list)

    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"].replace("\n", " ")])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    # 日志记录
    entities_log = [f'"{n["entity_name"]}"' for n in final_query_nodes]
    communities_log = [(c.get("level"), c.get("title", f"Cluster {c.get('cluster')}")) for c in use_communities]
    chunks_log = [(t.get("full_doc_id") or t.get("doc_id"), t.get("chunk_order_index") or t.get("index", 0)) for t in use_text_units]
    
    logger.info(f"====== References (RGCN Diversity Enhanced) ======\n"
                f"Entities ({len(entities_log)}): {entities_log}\n"
                f"Communities ({len(communities_log)}): {communities_log}\n"
                f"Chunks ({len(chunks_log)}): {chunks_log}\n")

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
async def _build_hilocal_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)          # find the top-k(20) related entities

    if not len(results):    # results just with entity name
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [                          # add rank, which is the degree
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    node_datas = node_datas[:query_param.top_k]
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )

    logger.info(
        f"Using {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    
    relation_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relation_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relation_context = list_of_list_to_csv(relation_section_list)
    
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relations-----
```csv
{relation_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
async def _build_rgcn_no_path1_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    # ==========================================================================
    # 1. BGE 语义检索 (Semantic Anchoring)
    # ==========================================================================
    results = await entities_vdb.query(query, top_k=query_param.top_k*10)

    if not len(results):
        return None
        
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    
    base_node_datas = node_datas[:query_param.top_k]
    
    # ==========================================================================
    # 2. R-GCN 结构扩展 (Structural Expansion)
    # ==========================================================================
    overall_node_datas, rgcn_sims = await _rgcn_expand_entities(
        base_node_datas, knowledge_graph_inst, query_param, global_config
    )
 
# ==========================================================================
    # 3. [Top-3 锁定 + 混合重排] 策略
    # 逻辑：
    # 1. BGE Top-3 强制置顶 (Index 0-2)
    # 2. 剩余 BGE + RGCN 通过语义门控后，按 (票数+相似度) 混合排序
    # ==========================================================================
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # A. 准备基础数据
    rgcn_counts = Counter([x[1] for x in rgcn_sims])
    
    # 识别 BGE 的不同梯队
    # bge_top3: 绝对核心，不参与竞争，直接保送
    # bge_rest: 需要和 RGCN 竞争
    bge_top3_names = set(n["entity_name"] for n in base_node_datas[:3])
    bge_all_names = set(n["entity_name"] for n in base_node_datas)
    
    # 获取所有候选实体的名称列表
    all_candidate_names = [n["entity_name"] for n in overall_node_datas]

    # -------------------------------------------------------
    # B. 计算向量相似度 (Batch Processing)
    # -------------------------------------------------------
    # 1. 获取 Query 向量
    query_embedding = await entities_vdb.embedding_func([query]) 
    
    # 2. 获取 RGCN 增量实体的向量 (带分批处理)
    # 我们只计算不是 BGE 的实体，节省开销
    rgcn_only_names = [n for n in all_candidate_names if n not in bge_all_names]
    name_to_sim = {} 

    if rgcn_only_names:
        BATCH_SIZE = 50 
        embeddings_list = []
        
        for i in range(0, len(rgcn_only_names), BATCH_SIZE):
            batch_names = rgcn_only_names[i : i + BATCH_SIZE]
            try:
                batch_embs = await entities_vdb.embedding_func(batch_names)
                embeddings_list.append(batch_embs)
            except Exception as e:
                # logger.warning(f"Embedding fetch failed: {e}")
                pass

        if embeddings_list:
            if isinstance(embeddings_list[0], np.ndarray):
                candidate_embeddings = np.concatenate(embeddings_list, axis=0)
            else:
                candidate_embeddings = np.array([item for batch in embeddings_list for item in batch])
            
            if len(candidate_embeddings) > 0:
                sims = cosine_similarity(query_embedding, candidate_embeddings)[0]
                name_to_sim = dict(zip(rgcn_only_names, sims))

    # -------------------------------------------------------
    # C. 综合打分 (Scoring)
    # -------------------------------------------------------
    candidate_scores = {}
    
    # 阈值设定 (建议 0.2 - 0.35)
    # 0.2 比较宽松，允许一定的语义漂移以换取结构发现
    SEMANTIC_THRESHOLD = 0.20 

    for n in overall_node_datas:
        name = n["entity_name"]
        
        # --- 情况 1: BGE Top-3 (直接跳过打分，因为它们肯定在最前面) ---
        if name in bge_top3_names:
            continue 

        # --- 情况 2: BGE 剩余实体 (给个基准分) ---
        if name in bge_all_names:
            # 3.0 分是一个很好的基准
            # 意味着 RGCN 实体只要有 2 票且相似度还可以，就能超过它
            candidate_scores[name] = 3.0
            continue
            
        # --- 情况 3: RGCN 新实体 (需过安检 + 算分) ---
        sim = name_to_sim.get(name, 0.0)
        
        # 🛑 [安检门]
        if sim < SEMANTIC_THRESHOLD:
            candidate_scores[name] = -1.0 # 淘汰
            continue
            
        # ✅ [算分]
        votes = rgcn_counts.get(name, 0)
        
        # 公式：票数(结构) + 相似度的一半(语义微调)
        # 举例：Votes=3, Sim=0.6 -> Score = 3.3 (超过 BGE 剩余实体，插队成功)
        # 举例：Votes=1, Sim=0.8 -> Score = 1.4 (排在 BGE 后面，作为补充)
        final_score = votes + (sim * 0.5) 
        
        candidate_scores[name] = final_score

    # -------------------------------------------------------
    # D. 分段组装与排序 (Assembly & Sorting)
    # -------------------------------------------------------
    
    final_query_nodes = []
    seen_names = set()
    
    # 1. 【锁定区】强制放入 BGE Top-3 (保持原始顺序)
    for node in base_node_datas[:3]:
        if node["entity_name"] not in seen_names:
            final_query_nodes.append(node)
            seen_names.add(node["entity_name"])
            
    # 2. 【竞争区】对剩余实体进行排序
    # 筛选出 BGE 剩余实体 和 通过安检的 RGCN 实体
    competition_pool = []
    for n in overall_node_datas:
        name = n["entity_name"]
        if name not in seen_names: # 排除已加入的 Top-3
            score = candidate_scores.get(name, -999)
            if score > 0:
                competition_pool.append(n)
                
    # 按分数降序排列 (这里决定了谁排第 4, 5, 6 名)
    competition_pool.sort(
        key=lambda x: candidate_scores.get(x["entity_name"], 0), 
        reverse=True
    )
    
    # 3. 追加到最终列表
    final_query_nodes.extend(competition_pool)
    
    # E. 最终截断
    results =  final_query_nodes
    final_query_nodes = final_query_nodes[:query_param.top_k * 2]
    final_selected_names = set(n["entity_name"] for n in final_query_nodes)

    # [Log] 验证顺序 (前3个应该是 BGE，后面有分数的混合排)
    debug_top = []
    for n in final_query_nodes[:6]:
        name = n['entity_name']
        if name in bge_top3_names:
            tag = "LOCK"
        else:
            tag = f"{candidate_scores.get(name):.1f}"
        debug_top.append(f"{name}({tag})")
        
    logger.info(f"[Hybrid Rank] Top 6: {debug_top}")
    # ==========================================================================
   # ==========================================================================
    # Phase 4: 溯源路径构建 (Traceback Path Construction)
    # [论文创新点]: 显式推理路径，解释“为什么引入这个新实体”
    # ==========================================================================
    graph = knowledge_graph_inst._graph
    structural_path_edges = []
    added_edges_keys = set()
    
    # 筛选父子对: Seed (BGE) -> Target (RGCN New)
    parent_child = []
    bge_names = set(n["entity_name"] for n in base_node_datas)
    
    for seed, tgt, score, _ in rgcn_sims:
        if tgt in final_selected_names and seed != tgt:
            # 仅为 RGCN 带来的“增量实体”找路径
            if seed in bge_names and tgt not in bge_names:
                parent_child.append((seed, tgt, score))

    # 物理路径查找
    for src, tgt, score in parent_child:
        try:
            path_nodes = nx.shortest_path(graph, source=src, target=tgt)
            if len(path_nodes) > 4: continue # 限制跳数
            
            # 提取路径上的边
            path_sub = graph.subgraph(path_nodes)
            for i in range(len(path_nodes)-1):
                u, v = path_nodes[i], path_nodes[i+1]
                if u == v: continue # 过滤自环

                edata = path_sub.get_edge_data(u, v)
                if not edata: continue
                
                # 兼容 MultiGraph
                edges_list = list(edata.values()) if graph.is_multigraph() else [edata]
                if len(edges_list) > 3: edges_list = edges_list[:3] # 限制平行边数量

                for attr in edges_list:
                    raw_rel = attr.get("relation_type", attr.get("label", "RELATED"))
                    raw_desc = attr.get("description", "")
                    k = (u, v, raw_rel, raw_desc[:20])
                    
                    if k in added_edges_keys: continue
                    
                    # 动态描述生成
                    final_desc = raw_desc if raw_desc.strip() else f"{u} is structurally related to {v}."
                    
                    structural_path_edges.append({
                        "src": u, "tgt": v, "relation": raw_rel, "desc": final_desc,
                        "weight": score, "rank": 0 # 占位，稍后查真实Rank
                    })
                    added_edges_keys.add(k)
        except: continue

    # 批量获取真实 Rank (度数)
    if structural_path_edges:
        real_ranks = await asyncio.gather(
            *[knowledge_graph_inst.edge_degree(e["src"], e["tgt"]) for e in structural_path_edges]
        )
        for edge, rank in zip(structural_path_edges, real_ranks):
            edge["rank"] = rank

  
    # ==========================================================================
    # 5. 检索社区与文本块 (Community & Text Units)
    # ⭐️ 关键点：因为 overall_node_datas 变了，这里召回的 chunks 也会变！
    # ==========================================================================
    use_communities = await _find_most_related_community_from_entities(
        final_query_nodes, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        final_query_nodes, query_param, text_chunks_db, knowledge_graph_inst
    )

    # ==========================================================================
    # 6. 互联路径 (Interconnection)
    # ==========================================================================
    def find_path_with_required_nodes(graph, source, target, required_nodes):
        final_path = []
        current_node = source
        for next_node in required_nodes:
            try:
                sub_path = nx.shortest_path(graph, source=current_node, target=next_node)
            except nx.NetworkXNoPath:
                final_path.extend([next_node])
                current_node = next_node
                continue
            if final_path:
                final_path.extend(sub_path[1:])
            else:
                final_path.extend(sub_path)
            current_node = next_node
        try:
            sub_path = nx.shortest_path(graph, source=current_node, target=target)
            final_path.extend(sub_path[1:])
        except nx.NetworkXNoPath:
            final_path.extend([target])
        return final_path

    key_entities = []
    if use_communities:
        for c in use_communities:
            cur = [e["entity_name"] for e in results if e["entity_name"] in c["nodes"]][:query_param.top_m]
            key_entities.extend(cur)
    else:
         key_entities = [e["entity_name"] for e in results[:query_param.top_m]]
    
    key_entities = list(set(key_entities))
    existing_path_edges = []
    if len(key_entities) > 1:
        try:
            p = find_path_with_required_nodes(graph, key_entities[0], key_entities[-1], key_entities[1:-1])
            path_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(r) for r in p])
            path_datas = [d for d in path_datas if d is not None]
            if path_datas:
                existing_path_edges = await _find_most_related_edges_from_paths(
                    path_datas, p, query_param, knowledge_graph_inst
                )
        except Exception:
            pass

    # ==========================================================================
    # 7. 最终合并、去重与排序 (Global Merge & Sort)
    # ==========================================================================
    
    reasoning_path_section_list = [["id", "source", "target", "relation_type", "description", "weight", "rank"]]
    path_id = 0
    seen_edges_fingerprint = set()

    # --- 第一梯队：原有社区互联路径 (Core Logic) ---
    # 这些是 HiRAG 认为最重要的路径，必须排在前面，且保持原顺序
    if existing_path_edges:
        # 原有路径通常已经按重要性排好序了，直接加入
        for e in existing_path_edges:
            src, tgt = e.get("src_tgt", (None, None))
            rel = e.get("relation_type", "UNKNOWN")
            
            # 指纹: (Source, Target, Relation)
            k = tuple(sorted((src, tgt))) + (rel,)
            
            if src and tgt and k not in seen_edges_fingerprint:
                reasoning_path_section_list.append([
                    path_id,
                    src, tgt,
                    rel,
                    e.get("description", ""),
                    e.get("weight", 1.0),
                    e.get("rank", 10) # 保留原始 Rank
                ])
                path_id += 1
                seen_edges_fingerprint.add(k)

    # --- 第二梯队：R-GCN 物理溯源路径 (Supplementary) ---
    # 追加在后面，作为补充证据。如果不重要，截断时会被优先切掉。
    
    # 先对 R-GCN 路径内部做个小排序（按相似度降序），保证追加的是质量相对高的
    structural_path_edges.sort(key=lambda x: x["weight"], reverse=True)

    for edge in structural_path_edges:
        # 制作同样的指纹
        k = tuple(sorted((edge["src"], edge["tgt"]))) + (edge["relation"],)
        
        # 只有当这条边【没有】在第一梯队出现过，才加进去
        if k not in seen_edges_fingerprint:
            reasoning_path_section_list.append([
                path_id,
                edge["src"],
                edge["tgt"],
                edge["relation"],
                edge["desc"],
                f"{float(edge['weight']):.2f}", # R-GCN 相似度
                edge["rank"]             # 真实度数
            ])
            path_id += 1
            seen_edges_fingerprint.add(k)

    # 生成 CSV
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(final_query_nodes):
        entites_section_list.append([
            i, n["entity_name"], n.get("entity_type", "UNKNOWN"),
            n.get("description", "UNKNOWN"), n["rank"],
        ])
    entities_context = list_of_list_to_csv(entites_section_list)

    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"].replace("\n", " ")])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    # 日志记录
    entities_log = [f'"{n["entity_name"]}"' for n in final_query_nodes]
    communities_log = [(c.get("level"), c.get("title", f"Cluster {c.get('cluster')}")) for c in use_communities]
    chunks_log = [(t.get("full_doc_id") or t.get("doc_id"), t.get("chunk_order_index") or t.get("index", 0)) for t in use_text_units]
    
    logger.info(f"====== References (RGCN Diversity Enhanced) ======\n"
                f"Entities ({len(entities_log)}): {entities_log}\n"
                f"Communities ({len(communities_log)}): {communities_log}\n"
                f"Chunks ({len(chunks_log)}): {chunks_log}\n")

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
# query functions
async def hierarchical_query(
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
        context = await _build_hierarchical_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
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

async def hierarchical_bridge_query(
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
        context = await _build_hibridge_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
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

async def hierarchical_local_query(
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
        context = await _build_hilocal_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
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

async def hierarchical_global_query(
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
        context = await _build_higlobal_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
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

async def hierarchical_nobridge_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """
    retrieve with only related entities
    """
    use_model_func = global_config["best_model_func"]
    with timer():
        context = await _build_local_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
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

async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_model_func = global_config["best_model_func"]
    with timer():
        results = await chunks_vdb.query(query, top_k=query_param.top_k)
        if not len(results):
            return PROMPTS["fail_response"]
        chunks_ids = [r["id"] for r in results]
        chunks = await text_chunks_db.get_by_ids(chunks_ids)

        maybe_trun_chunks = truncate_list_by_token_size(
            chunks,
            key=lambda x: x["content"],
            max_token_size=query_param.naive_max_token_for_text_unit,
        )
        logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
        section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
        if query_param.only_need_context:
            return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response

async def _plugrag_vote_expansion_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    """Vote-only ablation context: semantic anchors plus graph-encoder vote expansion."""
    use_model_func = global_config["best_model_func"]
    with timer():
        context = await _build_rgcn_query_context(
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


async def plugrag_full_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    """Full PlugRag retrieval: graph expansion, semantic rerank, and trace/path context.

    The graph encoder is selected by the loaded embedding/mapping files. R-GCN,
    Simple GCN, and Multi-edge GCN all use this same query pipeline.
    """
    use_model_func = global_config["best_model_func"]
    with timer():
        context = await _build_rgcn_path_rank_query_context(
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


async def plugrag_vote_only_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    """Vote-only ablation: graph-encoder expansion ranked by structural vote counts."""
    return await _plugrag_vote_expansion_query(
        query,
        knowledge_graph_inst,
        entities_vdb,
        community_reports,
        text_chunks_db,
        query_param,
        global_config,
    )


async def plugrag_semantic_rank_only_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    """Rank-only ablation: semantic retrieval plus rerank, without R-GCN expansion."""
    # 1. 获取模型生成函数
    use_model_func = global_config["best_model_func"]
    
    # 2. 构建上下文 (调用纯 Rerank 的 Context Builder)
    with timer():
        context = await _build_hierarchical_rank_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
            global_config, # 记得传入 global_config 以获取 API Key
        )
        
    # 3. 如果只需要上下文 (调试用)
    if query_param.only_need_context:
        return context
        
    # 4. 如果没检索到内容，返回失败提示
    if context is None:
        return PROMPTS["fail_response"]
        
    # 5. 组装系统提示词
    sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    
    # 6. 调用大模型生成回答
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    
    return response


async def _deprecated_plugrag_simple_gcn_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    """Simple GCN ablation using the same PlugRag retrieval pipeline."""
    # 1. 获取模型生成函数
    use_model_func = global_config["best_model_func"]
    
    # 2. 构建上下文 (调用纯 Rerank 的 Context Builder)
    with timer():
        context = await _build_gcn_path_rank_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
            global_config, # 记得传入 global_config 以获取 API Key
        )
        
    # 3. 如果只需要上下文 (调试用)
    if query_param.only_need_context:
        return context
        
    # 4. 如果没检索到内容，返回失败提示
    if context is None:
        return PROMPTS["fail_response"]
        
    # 5. 组装系统提示词
    sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    
    # 6. 调用大模型生成回答
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    
    return response


# Backward-compatible aliases for historical experiment scripts.
# The public release exposes one full PlugRag query path. R-GCN, Simple GCN,
# and Multi-edge GCN are selected by the configured embedding/mapping files.
plugrag_rgcn_query = plugrag_full_query
plugrag_rgcn_path_query = plugrag_full_query
plugrag_rgcn_rank_path_query = plugrag_full_query
plugrag_simple_gcn_query = plugrag_full_query

rgcn_query = plugrag_full_query
rgcn_path_query = plugrag_full_query
hierarchical_rank_query = plugrag_semantic_rank_only_query
rgcn_path_rank_query = plugrag_full_query
hi_gcn_query = plugrag_full_query
rgcn_vote_query = plugrag_vote_only_query

# Older release aliases kept for compatibility with early cleanup drafts.
plugrag_rank_only_query = plugrag_semantic_rank_only_query
plugrag_hybrid_rank_path_query = plugrag_full_query

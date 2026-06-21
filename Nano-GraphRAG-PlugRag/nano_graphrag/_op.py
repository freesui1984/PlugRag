import re
import os
import json
import asyncio
import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
import aiohttp
from typing import Union
from collections import Counter, defaultdict
from ._splitter import SeparatorSplitter
from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,

    TokenizerWrapper
)
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


def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tokenizer_wrapper: TokenizerWrapper, 
    overlap_token_size=128,
    max_token_size=1024,
):
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))


        chunk_texts = tokenizer_wrapper.decode_batch(chunk_token)

        for i, chunk in enumerate(chunk_texts):
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
    tokenizer_wrapper: TokenizerWrapper,
    overlap_token_size=128,
    max_token_size=1024,
):
    from .prompt import PROMPTS
    # *** 修改 ***: 直接使用 wrapper 编码，而不是获取底层 tokenizer
    separators = [tokenizer_wrapper.encode(s) for s in PROMPTS["default_text_separator"]]
    splitter = SeparatorSplitter(
        separators=separators,
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_tokens = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_tokens]

        decoded_chunks = tokenizer_wrapper.decode_batch(chunk_tokens)
        for i, chunk in enumerate(decoded_chunks):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )
    return results


def get_chunks(new_docs, chunk_func=chunking_by_token_size, tokenizer_wrapper: TokenizerWrapper = None, **chunk_func_params):
    inserting_chunks = {}
    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]

    tokens = [tokenizer_wrapper.encode(doc) for doc in docs]
    chunks = chunk_func(
        tokens, doc_keys=doc_keys, tokenizer_wrapper=tokenizer_wrapper, overlap_token_size=chunk_func_params.get("overlap_token_size", 128), max_token_size=chunk_func_params.get("max_token_size", 1024)
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
    tokenizer_wrapper: TokenizerWrapper,
) -> str:
    use_llm_func: callable = global_config["cheap_model_func"]
    llm_max_tokens = global_config["cheap_model_max_token_size"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]


    tokens = tokenizer_wrapper.encode(description)
    if len(tokens) < summary_max_tokens:
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]

    use_description = tokenizer_wrapper.decode(tokens[:llm_max_tokens])
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
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    relation_type = clean_str(record_attributes[3].upper())
    edge_description = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        relation_type=relation_type,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    tokenizer_wrapper,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:
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
        entity_name, description, global_config, tokenizer_wrapper
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


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    relation_type: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    tokenizer_wrapper,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id, key=relation_type):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id, key=relation_type)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config, tokenizer_wrapper
    )
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        key=relation_type,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order, relation_type=relation_type
        ),
    )


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    tokenizer_wrapper,
    global_config: dict,
    using_amazon_bedrock: bool=False,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)
        if isinstance(final_result, list):
            final_result = final_result[0]["text"]

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result, using_amazon_bedrock)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result, using_amazon_bedrock)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
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
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"], if_relation["relation_type"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
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
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # k is (src, tgt, type)
            # For undirected consistency, we sort src and tgt but keep type
            src, tgt, rel_type = k
            if src > tgt:
                src, tgt = tgt, src
            maybe_edges[(src, tgt, rel_type)].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config, tokenizer_wrapper)
            for k, v in maybe_nodes.items()
        ]
    )
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], k[2], v, knwoledge_graph_inst, global_config, tokenizer_wrapper)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
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
    tokenizer_wrapper: TokenizerWrapper,
) -> tuple[str, int, set, set]: 
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
        tokenizer_wrapper=tokenizer_wrapper,
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
        len(tokenizer_wrapper.encode(sub_communities_describe)),
        set(already_nodes),
        set(already_edges),
    )


async def _pack_single_community_describe(
    knwoledge_graph_inst: BaseGraphStorage,
    community: SingleCommunitySchema,
    tokenizer_wrapper: "TokenizerWrapper",
    max_token_size: int = 12000,
    already_reports: dict[str, CommunitySchema] = {},
    global_config: dict = {},
) -> str:

    

    # 1. 准备原始数据
    nodes_in_order = sorted(community["nodes"])
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    nodes_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_node(n) for n in nodes_in_order]
    )
    edges_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
    )


    # 2. 定义模板和固定开销
    final_template = """-----Reports-----
```csv
{reports}
```
-----Entities-----
```csv
{entities}
```
-----Relationships-----
```csv
{relationships}
```"""
    base_template_tokens = len(tokenizer_wrapper.encode(
        final_template.format(reports="", entities="", relationships="")
    ))
    remaining_budget = max_token_size - base_template_tokens

    # 3. 处理子社区报告
    report_describe = ""
    contain_nodes = set()
    contain_edges = set()
    
    # 启发式截断检测
    truncated = len(nodes_in_order) > 100 or len(edges_in_order) > 100
    
    need_to_use_sub_communities = (
        truncated and 
        community["sub_communities"] and 
        already_reports
    )
    force_to_use_sub_communities = global_config["addon_params"].get(
        "force_to_use_sub_communities", False
    )
    
    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(f"Community {community['title']} using sub-communities")
        # 获取子社区报告及包含的节点/边
        result = _pack_single_community_by_sub_communities(
            community, remaining_budget, already_reports, tokenizer_wrapper
        )
        report_describe, report_size, contain_nodes, contain_edges = result
        remaining_budget = max(0, remaining_budget - report_size)

    # 4. 准备节点和边数据（过滤子社区已包含的）
    def format_row(row: list) -> str:
        return ','.join('"{}"'.format(str(item).replace('"', '""')) for item in row)

    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]

    # 获取度数并创建数据结构
    node_degrees = await knwoledge_graph_inst.node_degrees_batch(nodes_in_order)
    edge_degrees = await knwoledge_graph_inst.edge_degrees_batch(edges_in_order)
    
    # 过滤已存在于子社区的节点/边
    nodes_list_data = [
        [i, name, data.get("entity_type", "UNKNOWN"), 
         data.get("description", "UNKNOWN"), node_degrees[i]]
        for i, (name, data) in enumerate(zip(nodes_in_order, nodes_data))
        if name not in contain_nodes  # 关键过滤
    ]
    
    edges_list_data = [
        [i, edge[0], edge[1], data.get("description", "UNKNOWN"), edge_degrees[i]]
        for i, (edge, data) in enumerate(zip(edges_in_order, edges_data))
        if (edge[0], edge[1]) not in contain_edges  # 关键过滤
    ]
    
    # 按重要性排序
    nodes_list_data.sort(key=lambda x: x[-1], reverse=True)
    edges_list_data.sort(key=lambda x: x[-1], reverse=True)

    # 5. 动态分配预算
    # 计算表头开销
    header_tokens = len(tokenizer_wrapper.encode(
        list_of_list_to_csv([node_fields]) + "\n" + list_of_list_to_csv([edge_fields])
    ))



    data_budget = max(0, remaining_budget - header_tokens)
    total_items = len(nodes_list_data) + len(edges_list_data)
    node_ratio = len(nodes_list_data) / max(1, total_items)
    edge_ratio = 1 - node_ratio




    # 执行截断
    nodes_final = truncate_list_by_token_size(
        nodes_list_data, key=format_row, 
        max_token_size=int(data_budget * node_ratio), 
        tokenizer_wrapper=tokenizer_wrapper
    )
    edges_final = truncate_list_by_token_size(
        edges_list_data, key=format_row,
        max_token_size= int(data_budget * edge_ratio),
        tokenizer_wrapper=tokenizer_wrapper
    )

    # 6. 组装最终输出
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_final)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_final)



    final_output = final_template.format(
        reports=report_describe,
        entities=nodes_describe,
        relationships=edges_describe
    )

    return final_output


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
    tokenizer_wrapper: TokenizerWrapper,
    global_config: dict,
):
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func: callable = global_config["best_model_func"]
    use_string_json_convert_func: callable = global_config["convert_response_to_json_func"]

    communities_schema = await knwoledge_graph_inst.community_schema()
    community_keys, community_values = list(communities_schema.keys()), list(communities_schema.values())
    already_processed = 0

    prompt_template = PROMPTS["community_report"]

    prompt_overhead = len(tokenizer_wrapper.encode(prompt_template.format(input_text="")))

    async def _form_single_community_report(
        community: SingleCommunitySchema, already_reports: dict[str, CommunitySchema]
    ):
        nonlocal already_processed
        describe = await _pack_single_community_describe(
            knwoledge_graph_inst,
            community,
            tokenizer_wrapper=tokenizer_wrapper, 
            max_token_size=global_config["best_model_max_token_size"] - prompt_overhead -200, # extra token for chat template and prompt template
            already_reports=already_reports,
            global_config=global_config,
        )
        prompt = prompt_template.format(input_text=describe)


        response = await use_llm_func(prompt, **llm_extra_kwargs)
        data = use_string_json_convert_func(response)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(f"{now_ticks} Processed {already_processed} communities\r", end="", flush=True)
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
    tokenizer_wrapper,
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
    _related_community_datas = await asyncio.gather(
        *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
    )
    related_community_datas = {
        k: v
        for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
        if v is not None
    }
    related_community_keys = sorted(
        related_community_keys_counts.keys(),
        key=lambda k: (
            related_community_keys_counts[k],
            related_community_datas[k]["report_json"].get("rating", -1),
        ),
        reverse=True,
    )
    sorted_community_datas = [
        related_community_datas[k] for k in related_community_keys
    ]

    use_community_reports = truncate_list_by_token_size(
        sorted_community_datas,
        key=lambda x: x["report_string"],
        max_token_size=query_param.local_max_token_for_community_report,
        tokenizer_wrapper=tokenizer_wrapper, 
    )
    if query_param.local_community_single_one:
        use_community_reports = use_community_reports[:1]
    return use_community_reports


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    tokenizer_wrapper,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    # edges = await knowledge_graph_inst.get_nodes_edges_batch([dp["entity_name"] for dp in node_datas])
    edges = await asyncio.gather(   # get relations related to the retrieved entities
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    ) 
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    # all_one_hop_nodes_data = await knowledge_graph_inst.get_nodes_batch(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(  # get node information from storage
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
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
                "relation_counts": relation_counts,
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.local_max_token_for_text_unit,
        tokenizer_wrapper=tokenizer_wrapper, # 传入 wrapper
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]
    return all_text_units

async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    tokenizer_wrapper,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        if not this_edges:
            continue
        for edge in this_edges:
            # edge is (u, v, key) or (u, v)
            if len(edge) >= 2:
                # 为了去重，我们将 (u, v) 排序，但保留 key
                u, v = sorted([edge[0], edge[1]])
                key = edge[2] if len(edge) == 3 else "RELATED"
                all_edges.add((u, v, key))
    
    all_edges = list(all_edges)
    
    # 获取边数据
    all_edges_pack = []
    all_edges_degree = []
    
    for u, v, key in all_edges:
        edge_data = await knowledge_graph_inst.get_edge(u, v, key=key)
        degree = await knowledge_graph_inst.edge_degree(u, v)
        all_edges_pack.append(edge_data)
        all_edges_degree.append(degree)
    
    all_edges_data = []
    for (u, v, key), v_data, d in zip(all_edges, all_edges_pack, all_edges_degree):
        if v_data is None:
            continue
        
        # 确保基本字段存在
        desc = v_data.get("description", "")
        weight = v_data.get("weight", 1.0)
        rel_type = v_data.get("relation_type", key)
        
        all_edges_data.append({
            "src": u,
            "tgt": v,
            "rank": d,
            "weight": weight,
            "description": desc,
            "relation_type": rel_type
        })
    
    if all_edges_data:
        empty_desc = sum(1 for e in all_edges_data if not e["description"].strip())
        logger.info(f"Retrieved {len(all_edges_data)} edges, {empty_desc} have empty description.")
    
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.local_max_token_for_local_context,
        tokenizer_wrapper=tokenizer_wrapper,
    )
    return all_edges_data

    # node_names = {dp["entity_name"] for dp in node_datas}
    # all_edges = []
    # seen = set()
    
    # for this_edges in all_related_edges:
    #     if not this_edges: continue
    #     for e in this_edges:
    #         # e is (src, tgt, key)
    #         src, tgt, rel_type = e
    #         # Normalize edge for deduplication: (min, max, key)
    #         edge_id = tuple(sorted([src, tgt])) + (rel_type,)
            
    #         if edge_id not in seen:
    #             seen.add(edge_id)
    #             all_edges.append(e) 
                
    # all_edges_pack = await knowledge_graph_inst.get_edges_batch(all_edges)
    # all_edges_degree = await knowledge_graph_inst.edge_degrees_batch([(e[0], e[1]) for e in all_edges])
    
    # all_edges_data = []
    # for e, v, d in zip(all_edges, all_edges_pack, all_edges_degree):
    #     if v is None: continue
    #     src, tgt, rel_type = e
    #     # Boost rank for edges that connect two retrieved nodes (structural paths)
    #     rank = d
    #     if src in node_names and tgt in node_names:
    #         rank += 1000 # Give high priority to "internal" edges
            
    #     all_edges_data.append({
    #         "src_tgt": (src, tgt),
    #         "relation_type": rel_type,
    #         "rank": rank,
    #         **v
    #     })
        
    # all_edges_data = sorted(
    #     all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    # )
    # all_edges_data = truncate_list_by_token_size(
    #     all_edges_data,
    #     key=lambda x: f"{x['src_tgt'][0]} -[{x['relation_type']}]-> {x['src_tgt'][1]}: {x['description']}",
    #     max_token_size=query_param.local_max_token_for_local_context,
    #     tokenizer_wrapper=tokenizer_wrapper, 
    # )
    # return all_edges_data

_RGCN_STATE = {"names": None, "emb": None}

def _load_rgcn_state(global_config: dict):
    if not global_config.get("rgcn_enabled"):
        return None
    ef = global_config.get("rgcn_embeddings_file")
    mf = global_config.get("rgcn_mapping_file")
    wd = global_config.get("working_dir", "")
    
    if not ef or not mf:
        return None
        
    # 处理路径，如果是相对路径则与 working_dir 拼接
    if not os.path.isabs(ef) and wd:
        ef = os.path.join(wd, ef)
    if not os.path.isabs(mf) and wd:
        mf = os.path.join(wd, mf)

    if _RGCN_STATE["emb"] is None:
        try:
            if not os.path.exists(ef) or not os.path.exists(mf):
                logger.warning(f"RGCN files not found: {ef} or {mf}")
                return None
                
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
        except Exception as e:
            logger.error(f"Error loading RGCN state: {e}")
            _RGCN_STATE["names"] = None
            _RGCN_STATE["emb"] = None
    return _RGCN_STATE if _RGCN_STATE["emb"] is not None else None
async def _rgcn_expand_entities(
    node_datas: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    query_param: QueryParam,
    global_config: dict,
    rgcn_embeddings=None,
    rgcn_mapping=None,
    rgcn_expand_k=20,
    rgcn_cluster_limit=200
) -> tuple[list[dict], list[tuple[str,str,float,str]]]:
    
    # [检查点 1]：确认函数是否被调用
    print("\n=================== [DEBUG START] ===================", flush=True)
    print(f"[DEBUG 1] 进入 _rgcn_expand_entities 函数", flush=True)

    if rgcn_embeddings is not None and rgcn_mapping is not None:
        emb = rgcn_embeddings
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        
        idx_to_node = rgcn_mapping.get("idx_to_node") or {}
        names = []
        for i in range(len(emb)):
            v = idx_to_node.get(str(i), idx_to_node.get(i))
            names.append(v if isinstance(v, str) else None)
        state = {"names": names, "emb": emb}
    else:
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
            
        idxs = np.argsort(-s)[:rgcn_expand_k]
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
    
    if len(added_names) > rgcn_cluster_limit:
        added_names = added_names[:rgcn_cluster_limit]
        
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
## --------------------rgcn + nano + rank-------------------------------------
async def _build_local_rgcn_rank_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
    **kwargs,
):
    rgcn_embeddings = kwargs.get("rgcn_embeddings")
    rgcn_mapping = kwargs.get("rgcn_mapping")
    rgcn_rerank_alpha = kwargs.get("rgcn_rerank_alpha", global_config.get("rgcn_rerank_alpha", 0.5))
    rgcn_expand_k = kwargs.get("rgcn_expand_k", global_config.get("rgcn_expand_k", 20))
    rgcn_cluster_limit = kwargs.get("rgcn_cluster_limit", global_config.get("rgcn_cluster_limit", 200))
    rerank_api_key = kwargs.get("rerank_api_key")
    rerank_base_url = kwargs.get("rerank_base_url")
    rerank_model = kwargs.get("rerank_model")

    # 如果 rgcn 参数为 None，尝试从 global_config 中读取配置
    if rgcn_embeddings is None or rgcn_mapping is None:
        state = _load_rgcn_state(global_config)
        if state:
            rgcn_embeddings = state["emb"]
            # 构造与传入格式一致的 mapping
            rgcn_mapping = {"idx_to_node": {str(i): name for i, name in enumerate(state["names"])}}
            logger.info("RGCN embeddings and mapping loaded from config files.")

    results = await entities_vdb.query(query, top_k=query_param.top_k*10)
    if not len(results):
        return None
    
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
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
        base_node_datas,
        knowledge_graph_inst,
        query_param,
        global_config,
        rgcn_embeddings=rgcn_embeddings,
        rgcn_mapping=rgcn_mapping,
        rgcn_expand_k=rgcn_expand_k,
        rgcn_cluster_limit=rgcn_cluster_limit
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
    # 读取配置
    openai_cfg = global_config.get('openai', {})
    
    api_key = rerank_api_key or openai_cfg.get('api_key') or os.getenv("RERANK_API_KEY") or os.getenv("OPENAI_API_KEY")
    rerank_model_name = rerank_model or "BAAI/bge-reranker-v2-m3" 
    base_url = rerank_base_url or openai_cfg.get('base_url') or os.getenv("RERANK_BASE_URL") or "https://api.siliconflow.cn/v1" 

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
    # ✨✨✨ 修改这里：只提取实体名称，不提取分数 ✨✨✨
    # ==========================================================================
    final_entities_debug = [n["entity_name"] for n in final_query_nodes]
    logger.info(f"[GCNHybrid Rerank] Final selected entities list ({len(final_entities_debug)} total): {final_entities_debug}")

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
        # 【修复 1】强制将源节点和目标节点转换为字符串，对齐底层的 upsert 逻辑
        src_str, tgt_str = str(src), str(tgt)
        
        try:
            path_nodes = nx.shortest_path(graph, source=src_str, target=tgt_str)
            if len(path_nodes) > 4: 
                continue
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

        # 【修复 2】把报错分类打印出来，不要用裸的 except
        except nx.NetworkXNoPath:
            # 节点之间本来就没有路径（图不连通），这是正常的，可以 pass
            pass
        except nx.NodeNotFound as e:
            # 找不到节点，通常是类型不匹配或图中确实没这个节点
            print(f"警告: 节点未找到 - {e}")
        except Exception as e:
            # 捕捉其他意料之外的错误
            print(f"警告: 处理 {src_str} -> {tgt_str} 时发生未知错误: {e}")
    print("parent_child_pairs =", len(parent_child_pairs))
    # 批量获取真实 Rank (度数)
    if structural_path_edges:
        real_ranks = await asyncio.gather(
            *[knowledge_graph_inst.edge_degree(e["src"], e["tgt"]) for e in structural_path_edges]
        )
        for edge, rank in zip(structural_path_edges, real_ranks):
            edge["rank"] = rank
    related_edges_from_entities = final_query_nodes[:query_param.top_k]
    use_communities = await _find_most_related_community_from_entities(
        final_query_nodes, query_param, community_reports, tokenizer_wrapper
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        final_query_nodes, query_param, text_chunks_db, knowledge_graph_inst, tokenizer_wrapper
    )
    use_relations = await _find_most_related_edges_from_entities(
        related_edges_from_entities, query_param, knowledge_graph_inst,tokenizer_wrapper
    )
    logger.info(
        f"Using {len(final_query_nodes)} entites, {len(use_communities)} communities, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(final_query_nodes):
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

    reasoning_path_section_list = [["id", "source", "target", "relation_type", "description", "weight", "rank"]]
    
    path_id = 0
    seen_edges_fingerprint = set()

    # # --- 第一梯队：原有社区互联路径 (Core Logic) ---
    if use_relations:
        # 原有路径通常已经按重要性排好序了，直接加入
        for e in use_relations:
            src = e.get("src")
            tgt = e.get("tgt")
            rel = e.get("relation_type", "RELATED")
            
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
    # # 生成 CSV
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)

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
{reasoning_path_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""

## --------------------rgcn + nano + vote-------------------------------------
async def _build_local_rgcn_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
    **kwargs,
):
    rgcn_embeddings = kwargs.get("rgcn_embeddings")
    rgcn_mapping = kwargs.get("rgcn_mapping")
    rgcn_rerank_alpha = kwargs.get("rgcn_rerank_alpha", global_config.get("rgcn_rerank_alpha", 0.5))
    rgcn_expand_k = kwargs.get("rgcn_expand_k", global_config.get("rgcn_expand_k", 20))
    rgcn_cluster_limit = kwargs.get("rgcn_cluster_limit", global_config.get("rgcn_cluster_limit", 200))
    rerank_api_key = kwargs.get("rerank_api_key")
    rerank_base_url = kwargs.get("rerank_base_url")
    rerank_model = kwargs.get("rerank_model")

    # 如果 rgcn 参数为 None，尝试从 global_config 中读取配置
    if rgcn_embeddings is None or rgcn_mapping is None:
        state = _load_rgcn_state(global_config)
        if state:
            rgcn_embeddings = state["emb"]
            # 构造与传入格式一致的 mapping
            rgcn_mapping = {"idx_to_node": {str(i): name for i, name in enumerate(state["names"])}}
            logger.info("RGCN embeddings and mapping loaded from config files.")

    results = await entities_vdb.query(query, top_k=query_param.top_k*10)
    if not len(results):
        return None
    
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
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
        base_node_datas,
        knowledge_graph_inst,
        query_param,
        global_config,
        rgcn_embeddings=rgcn_embeddings,
        rgcn_mapping=rgcn_mapping,
        rgcn_expand_k=rgcn_expand_k,
        rgcn_cluster_limit=rgcn_cluster_limit
    )

    # ==========================================================================
    # ⭐ Candidate Pruning（关键优化：限制进入排序规模）
    # ==========================================================================

    from collections import Counter
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    MAX_CANDIDATES = 50

    rgcn_counts = Counter([x[1] for x in rgcn_sims])

    bge_top3_names = set(n["entity_name"] for n in base_node_datas[:3])
    bge_all_names = set(n["entity_name"] for n in base_node_datas)

    # -----------------------------
    # 🔥 Candidate pruning（保留Top3）
    # -----------------------------
    sorted_candidates = sorted(
        overall_node_datas,
        key=lambda n: rgcn_counts.get(n["entity_name"], 0),
        reverse=True
    )

    overall_node_datas = (
        base_node_datas[:3] +
        [n for n in sorted_candidates if n["entity_name"] not in bge_top3_names]
    )[:MAX_CANDIDATES]


    # ==========================================================================
    # 3. Hybrid Rerank v2 (Unified Scoring Version)
    # ==========================================================================

    query_embedding = await entities_vdb.embedding_func([query])

    candidate_scores = {}
    name_to_emb = {}

    SEMANTIC_THRESHOLD = 0.20

    # ==========================================================================
    # ⭐ BATCH EMBEDDING（优化API调用）
    # ==========================================================================

    names = [n["entity_name"] for n in overall_node_datas]

    BATCH_SIZE = 50

    for i in range(0, len(names), BATCH_SIZE):
        batch = names[i:i + BATCH_SIZE]
        try:
            embs = await entities_vdb.embedding_func(batch)
            for name, emb in zip(batch, embs):
                name_to_emb[name] = emb
        except Exception as e:
            logger.warning(f"Embedding batch failed: {e}")


    # ==========================================================================
    # ⭐ Scoring（注意：Top3不参与评分）
    # ==========================================================================

    for n in overall_node_datas:
        name = n["entity_name"]

        # 🔒 Top-3 不参与排序
        if name in bge_top3_names:
            continue

        # BGE rest
        if name in bge_all_names:
            candidate_scores[name] = 3.0
            continue

        # RGCN candidates
        votes = rgcn_counts.get(name, 0)

        emb = name_to_emb.get(name)
        if emb is None:
            continue

        sim = cosine_similarity(query_embedding, [emb])[0][0]

        if sim < SEMANTIC_THRESHOLD:
            continue

        candidate_scores[name] = votes + sim * 0.5


    # ==========================================================================
    # ⭐ rerank API（仅辅助）
    # ==========================================================================

    openai_cfg = global_config.get('openai', {})
    api_key = rerank_api_key or openai_cfg.get('api_key') or os.getenv("RERANK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = rerank_base_url or openai_cfg.get('base_url') or os.getenv("RERANK_BASE_URL") or "https://api.siliconflow.cn/v1"
    try:
        _ = await call_rerank_api(
            query=query,
            candidates=overall_node_datas,
            api_key=api_key,
            base_url=base_url,
            model_name=rerank_model or "BAAI/bge-reranker-v2-m3"
        )
    except Exception as e:
        logger.warning(f"[Rerank API failed but ignored]: {e}")


    # ==========================================================================
    # ⭐ FINAL RANKING（正确结构：Top3锁定 + 竞争排序）
    # ==========================================================================

    final_query_nodes = []
    seen = set()

    # -----------------------------
    # 🔒 1. Lock Top-3
    # -----------------------------
    for node in base_node_datas[:3]:
        name = node["entity_name"]
        if name not in seen:
            final_query_nodes.append(node)
            seen.add(name)

    # -----------------------------
    # 🧠 2. Competition pool
    # -----------------------------
    competition_pool = []

    for n in overall_node_datas:
        name = n["entity_name"]

        if name in seen:
            continue

        score = candidate_scores.get(name, -1)
        if score > 0:
            competition_pool.append(n)

    # -----------------------------
    # 🧠 3. Sort competition
    # -----------------------------
    competition_pool.sort(
        key=lambda x: candidate_scores.get(x["entity_name"], 0),
        reverse=True
    )

    # -----------------------------
    # 🔗 4. merge
    # -----------------------------
    final_query_nodes.extend(competition_pool)

    # -----------------------------
    # ✂️ 5. truncate
    # -----------------------------
    final_query_nodes = final_query_nodes[:query_param.top_k * 2]


    # ==========================================================================
    # ⭐ Debug Log
    # ==========================================================================

    debug_top = []
    for n in final_query_nodes[:6]:
        name = n["entity_name"]
        score = candidate_scores.get(name, -1)

        if name in bge_top3_names:
            tag = "LOCK"
        else:
            tag = f"{score:.2f}"

        debug_top.append(f"{name}({tag})")

    logger.info(f"[HybridRank-v2 FIXED] Top6: {debug_top}")


    # ==========================================================================
    # 返回
    # ==========================================================================
    results = final_query_nodes
    final_query_nodes = final_query_nodes[:query_param.top_k*2]
    final_selected_names = set(n["entity_name"] for n in final_query_nodes)
    
    # # [Log] 看看谁是 Top 5 (前3个应该是 BGE 原生的)
    # debug_top = [f"{n['entity_name']}({n.get('_rerank_score', 'LOCK')})" for n in final_query_nodes[:5]]
    # logger.info(f"[GCNHybrid Rerank] Top 5: {debug_top}")

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
            if seed in bge_all_names and target not in bge_all_names:
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
        # 【修复 1】强制将源节点和目标节点转换为字符串，对齐底层的 upsert 逻辑
        src_str, tgt_str = str(src), str(tgt)
        
        try:
            path_nodes = nx.shortest_path(graph, source=src_str, target=tgt_str)
            if len(path_nodes) > 4: 
                continue
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

        # 【修复 2】把报错分类打印出来，不要用裸的 except
        except nx.NetworkXNoPath:
            # 节点之间本来就没有路径（图不连通），这是正常的，可以 pass
            pass
        except nx.NodeNotFound as e:
            # 找不到节点，通常是类型不匹配或图中确实没这个节点
            print(f"警告: 节点未找到 - {e}")
        except Exception as e:
            # 捕捉其他意料之外的错误
            print(f"警告: 处理 {src_str} -> {tgt_str} 时发生未知错误: {e}")
    print("parent_child_pairs =", len(parent_child_pairs))
    # 批量获取真实 Rank (度数)
    if structural_path_edges:
        real_ranks = await asyncio.gather(
            *[knowledge_graph_inst.edge_degree(e["src"], e["tgt"]) for e in structural_path_edges]
        )
        for edge, rank in zip(structural_path_edges, real_ranks):
            edge["rank"] = rank
    related_edges_from_entities = final_query_nodes[:query_param.top_k]
    use_communities = await _find_most_related_community_from_entities(
        final_query_nodes, query_param, community_reports, tokenizer_wrapper
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        final_query_nodes, query_param, text_chunks_db, knowledge_graph_inst, tokenizer_wrapper
    )
    use_relations = await _find_most_related_edges_from_entities(
        related_edges_from_entities, query_param, knowledge_graph_inst,tokenizer_wrapper
    )
    logger.info(
        f"Using {len(final_query_nodes)} entites, {len(use_communities)} communities, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(final_query_nodes):
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

    reasoning_path_section_list = [["id", "source", "target", "relation_type", "description", "weight", "rank"]]
    
    path_id = 0
    seen_edges_fingerprint = set()

    # # --- 第一梯队：原有社区互联路径 (Core Logic) ---
    if use_relations:
        # 原有路径通常已经按重要性排好序了，直接加入
        for e in use_relations:
            src = e.get("src")
            tgt = e.get("tgt")
            rel = e.get("relation_type", "RELATED")
            
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
    # # 生成 CSV
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)

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
{reasoning_path_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""

## --------------------nano + rank-------------------------------------
async def _build_local_rank_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
    **kwargs,
):
    rgcn_embeddings = kwargs.get("rgcn_embeddings")
    rgcn_mapping = kwargs.get("rgcn_mapping")
    rgcn_rerank_alpha = kwargs.get("rgcn_rerank_alpha", global_config.get("rgcn_rerank_alpha", 0.5))
    rgcn_expand_k = kwargs.get("rgcn_expand_k", global_config.get("rgcn_expand_k", 20))
    rgcn_cluster_limit = kwargs.get("rgcn_cluster_limit", global_config.get("rgcn_cluster_limit", 200))
    rerank_api_key = kwargs.get("rerank_api_key")
    rerank_base_url = kwargs.get("rerank_base_url")
    rerank_model = kwargs.get("rerank_model")

    # 如果 rgcn 参数为 None，尝试从 global_config 中读取配置
    if rgcn_embeddings is None or rgcn_mapping is None:
        state = _load_rgcn_state(global_config)
        if state:
            rgcn_embeddings = state["emb"]
            # 构造与传入格式一致的 mapping
            rgcn_mapping = {"idx_to_node": {str(i): name for i, name in enumerate(state["names"])}}
            logger.info("RGCN embeddings and mapping loaded from config files.")

    results = await entities_vdb.query(query, top_k=query_param.top_k*10)
    if not len(results):
        return None
    
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
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
    # 3. [Rerank API 版] 实体重排序（无 R-GCN 版本）
    # ==========================================================================

    # A. 准备候选池 (Candidate Pool)
    pool_nodes = []
    seen_in_pool = set()

    # 1. 锁定 BGE Top-3
    bge_top3_nodes = []
    for n in base_node_datas[:3]:
        bge_top3_nodes.append(n)
        seen_in_pool.add(n["entity_name"])

    # 2. 构建候选池（BGE Top50）
    # ⚠️ 注意这里直接从原始 results 取，而不是 base_node_datas
    for r in results[:50]:
        name = r["entity_name"]
        if name not in seen_in_pool:
            node = next((n for n in node_datas if n["entity_name"] == name), None)
            if node:
                pool_nodes.append(node)
                seen_in_pool.add(name)
            
    # B. 调用 Rerank API (只对“待重排组”进行打分)
    # 读取配置
    openai_cfg = global_config.get('openai', {})
    
    api_key = rerank_api_key or openai_cfg.get('api_key') or os.getenv("RERANK_API_KEY") or os.getenv("OPENAI_API_KEY")
    rerank_model_name = rerank_model or "BAAI/bge-reranker-v2-m3" 
    base_url = rerank_base_url or openai_cfg.get('base_url') or os.getenv("RERANK_BASE_URL") or "https://api.siliconflow.cn/v1" 

    # 执行重排
    # 注意：这里只传入 pool_nodes，不传 bge_top3_nodes
    reranked_pool = await call_rerank_api(
        query=query,
        candidates=pool_nodes,
        api_key=api_key,
        base_url=base_url,
        model_name=rerank_model_name
    )

    # C. 组装最终结果
    final_query_nodes = []

    # 1. 先放 Top-3（锁定）
    final_query_nodes.extend(bge_top3_nodes)

    # 2. 再拼 rerank 后的
    seen_final = set(n["entity_name"] for n in bge_top3_nodes)

    for node in reranked_pool:
        if node["entity_name"] not in seen_final:
            final_query_nodes.append(node)
            seen_final.add(node["entity_name"])

    # D. 截断
    final_query_nodes = final_query_nodes[:query_param.top_k * 2]
    results = final_query_nodes
    final_selected_names = set(n["entity_name"] for n in final_query_nodes)
    
    # [Log] 看看谁是 Top 5 (前3个应该是 BGE 原生的)
    debug_top = [f"{n['entity_name']}({n.get('_rerank_score', 'LOCK')})" for n in final_query_nodes[:5]]
    logger.info(f"[GCNHybrid Rerank] Top 5: {debug_top}")

    related_edges_from_entities = final_query_nodes[:query_param.top_k]
    use_communities = await _find_most_related_community_from_entities(
        final_query_nodes, query_param, community_reports, tokenizer_wrapper
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        final_query_nodes, query_param, text_chunks_db, knowledge_graph_inst, tokenizer_wrapper
    )
    use_relations = await _find_most_related_edges_from_entities(
        related_edges_from_entities, query_param, knowledge_graph_inst,tokenizer_wrapper
    )
    logger.info(
        f"Using {len(final_query_nodes)} entites, {len(use_communities)} communities, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(final_query_nodes):
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

    reasoning_path_section_list = [["id", "source", "target", "relation_type", "description", "weight", "rank"]]
    
    path_id = 0
    seen_edges_fingerprint = set()

    # # --- 第一梯队：原有社区互联路径 (Core Logic) ---
    if use_relations:
        # 原有路径通常已经按重要性排好序了，直接加入
        for e in use_relations:
            src = e.get("src")
            tgt = e.get("tgt")
            rel = e.get("relation_type", "RELATED")
            
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

    # # 生成 CSV
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)

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
{reasoning_path_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k*2)
    if not len(results):
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
            {**n, "entity_name": k["entity_name"], "rank": d}
            for k, n, d in zip(results, node_datas, node_degrees)
            if n is not None
        ]
    # ==========================================================================
    # ✨✨✨ 在这里添加打印最终检索到的实体列表 ✨✨✨
    # ==========================================================================
    retrieved_entities = [n["entity_name"] for n in node_datas]
    logger.info(f"[Local Query] Retrieved entities list ({len(retrieved_entities)} total): {retrieved_entities}")
    # ==========================================================================
    related_edges_from_entities = node_datas[:query_param.top_k]
    
    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports, tokenizer_wrapper
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst, tokenizer_wrapper
    )
    use_relations = await _find_most_related_edges_from_entities(
        related_edges_from_entities, query_param, knowledge_graph_inst, tokenizer_wrapper
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
    ["id", "source", "target", "relation_type", "description", "weight", "rank"]
    ]

    for i, e in enumerate(use_relations):

        src = e.get("src")
        tgt = e.get("tgt")

        if not src or not tgt:
            continue

        relations_section_list.append(
            [
                i,
                src,
                tgt,
                e.get("relation_type", "RELATED"),
                e.get("description", ""),
                e.get("weight", 1.0),
                e.get("rank", 0),
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


async def local_rgcn_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
    rgcn_embeddings: torch.Tensor = None,
    rgcn_mapping: dict = None,
    rerank_api_key: str = None,
    rerank_base_url: str = None,
    rerank_model: str = None,
) -> str:
    use_model_func = global_config["best_model_func"]
    context = await _build_local_rgcn_rank_query_context(
        query,
        knowledge_graph_inst,
        entities_vdb,
        community_reports,
        text_chunks_db,
        query_param,
        tokenizer_wrapper,
        global_config,
        rgcn_embeddings=rgcn_embeddings,
        rgcn_mapping=rgcn_mapping,
        rgcn_rerank_alpha=global_config.get("rgcn_rerank_alpha", 0.5),
        rgcn_expand_k=global_config.get("rgcn_expand_k", 10),
        rgcn_cluster_limit=global_config.get("rgcn_cluster_limit", 50),
        rerank_api_key=rerank_api_key,
        rerank_base_url=rerank_base_url,
        rerank_model=rerank_model,
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

async def local_rgcn_vote_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
    rgcn_embeddings: torch.Tensor = None,
    rgcn_mapping: dict = None,
    rerank_api_key: str = None,
    rerank_base_url: str = None,
    rerank_model: str = None,
) -> str:
    use_model_func = global_config["best_model_func"]
    context = await _build_local_rgcn_query_context(
        query,
        knowledge_graph_inst,
        entities_vdb,
        community_reports,
        text_chunks_db,
        query_param,
        tokenizer_wrapper,
        global_config,
        rgcn_embeddings=rgcn_embeddings,
        rgcn_mapping=rgcn_mapping,
        rgcn_rerank_alpha=global_config.get("rgcn_rerank_alpha", 0.5),
        rgcn_expand_k=global_config.get("rgcn_expand_k", 10),
        rgcn_cluster_limit=global_config.get("rgcn_cluster_limit", 50),
        rerank_api_key=rerank_api_key,
        rerank_base_url=rerank_base_url,
        rerank_model=rerank_model,
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
async def local_rank_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
    rgcn_embeddings: torch.Tensor = None,
    rgcn_mapping: dict = None,
    rerank_api_key: str = None,
    rerank_base_url: str = None,
    rerank_model: str = None,
) -> str:
    use_model_func = global_config["best_model_func"]
    context = await _build_local_rank_query_context(
        query,
        knowledge_graph_inst,
        entities_vdb,
        community_reports,
        text_chunks_db,
        query_param,
        tokenizer_wrapper,
        global_config,
        rgcn_embeddings=rgcn_embeddings,
        rgcn_mapping=rgcn_mapping,
        rgcn_rerank_alpha=global_config.get("rgcn_rerank_alpha", 0.5),
        rgcn_expand_k=global_config.get("rgcn_expand_k", 10),
        rgcn_cluster_limit=global_config.get("rgcn_cluster_limit", 50),
        rerank_api_key=rerank_api_key,
        rerank_base_url=rerank_base_url,
        rerank_model=rerank_model,
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

async def local_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
) -> str:
    use_model_func = global_config["best_model_func"]
    context = await _build_local_query_context(
        query,
        knowledge_graph_inst,
        entities_vdb,
        community_reports,
        text_chunks_db,
        query_param,
        tokenizer_wrapper,
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

async def _map_global_communities(
    query: str,
    communities_data: list[CommunitySchema],
    query_param: QueryParam,
    global_config: dict,
    tokenizer_wrapper,
):
    use_string_json_convert_func = global_config["convert_response_to_json_func"]
    use_model_func = global_config["best_model_func"]
    community_groups = []
    while len(communities_data):
        this_group = truncate_list_by_token_size(
            communities_data,
            key=lambda x: x["report_string"],
            max_token_size=query_param.global_max_token_for_community_report,
            tokenizer_wrapper=tokenizer_wrapper, # 传入 wrapper
        )
        community_groups.append(this_group)
        communities_data = communities_data[len(this_group) :]

    async def _process(community_truncated_datas: list[CommunitySchema]) -> dict:
        communities_section_list = [["id", "content", "rating", "importance"]]
        for i, c in enumerate(community_truncated_datas):
            communities_section_list.append(
                [
                    i,
                    c["report_string"],
                    c["report_json"].get("rating", 0),
                    c["occurrence"],
                ]
            )
        community_context = list_of_list_to_csv(communities_section_list)
        sys_prompt_temp = PROMPTS["global_map_rag_points"]
        sys_prompt = sys_prompt_temp.format(context_data=community_context)
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
            **query_param.global_special_community_map_llm_kwargs,
        )
        data = use_string_json_convert_func(response)
        return data.get("points", [])

    logger.info(f"Grouping to {len(community_groups)} groups for global search")
    responses = await asyncio.gather(*[_process(c) for c in community_groups])
    return responses


async def global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
) -> str:
    community_schema = await knowledge_graph_inst.community_schema()
    community_schema = {
        k: v for k, v in community_schema.items() if v["level"] <= query_param.level
    }
    if not len(community_schema):
        return PROMPTS["fail_response"]
    use_model_func = global_config["best_model_func"]

    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )
    sorted_community_schemas = sorted_community_schemas[
        : query_param.global_max_consider_community
    ]
    community_datas = await community_reports.get_by_ids(
        [k[0] for k in sorted_community_schemas]
    )
    community_datas = [c for c in community_datas if c is not None]
    community_datas = [
        c
        for c in community_datas
        if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating
    ]
    community_datas = sorted(
        community_datas,
        key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
        reverse=True,
    )
    logger.info(f"Revtrieved {len(community_datas)} communities")

    map_communities_points = await _map_global_communities(
        query, community_datas, query_param, global_config, tokenizer_wrapper
    )
    final_support_points = []
    for i, mc in enumerate(map_communities_points):
        for point in mc:
            if "description" not in point:
                continue
            final_support_points.append(
                {
                    "analyst": i,
                    "answer": point["description"],
                    "score": point.get("score", 1),
                }
            )
    final_support_points = [p for p in final_support_points if p["score"] > 0]
    if not len(final_support_points):
        return PROMPTS["fail_response"]
    final_support_points = sorted(
        final_support_points, key=lambda x: x["score"], reverse=True
    )
    final_support_points = truncate_list_by_token_size(
        final_support_points,
        key=lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report,
        tokenizer_wrapper=tokenizer_wrapper, # 传入 wrapper
    )
    points_context = []
    for dp in final_support_points:
        points_context.append(
            f"""----Analyst {dp['analyst']}----
Importance Score: {dp['score']}
{dp['answer']}
"""
        )
    points_context = "\n".join(points_context)
    if query_param.only_need_context:
        return points_context
    sys_prompt_temp = PROMPTS["global_reduce_rag_response"]
    response = await use_model_func(
        query,
        sys_prompt_temp.format(
            report_data=points_context, response_type=query_param.response_type
        ),
    )
    return response


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
):
    use_model_func = global_config["best_model_func"]
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.naive_max_token_for_text_unit,
        tokenizer_wrapper=tokenizer_wrapper, # 传入 wrapper
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

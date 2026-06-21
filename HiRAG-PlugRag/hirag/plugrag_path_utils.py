import networkx as nx
import json
import os
import re
def find_path_with_required_nodes_detailed(graph, source, target, required_nodes):
    """
    [工具函数] 寻找经过必须点的路径，并记录每一段的详细信息
    返回: (最终拼接路径, 详细分段日志)
    """
    final_path = []
    detailed_segments = [] 
    
    current_node = source
    
    # 构建访问序列: source -> req1 -> req2 ... -> target
    visit_sequence = list(required_nodes) + [target]
    
    for next_node in visit_sequence:
        segment_record = {
            "from": current_node,
            "to": next_node,
            "path": [],
            "status": "success"
        }
        
        try:
            # 计算 A -> B 的最短路
            sub_path = nx.shortest_path(graph, source=current_node, target=next_node)
            segment_record["path"] = sub_path
            
            # 拼接逻辑
            if not final_path:
                final_path.extend(sub_path)
            else:
                final_path.extend(sub_path[1:]) # 去重连接点

        except nx.NetworkXNoPath:
            # 记录失败
            segment_record["path"] = [current_node, next_node]
            segment_record["status"] = "no_path"
            
            if not final_path:
                final_path.extend([next_node])
            else:
                final_path.extend([next_node])
            
            current_node = next_node
            detailed_segments.append(segment_record)
            continue
            
        current_node = next_node
        detailed_segments.append(segment_record)

    return final_path, detailed_segments

def save_hirag_verification_log(query, key_entities, path_nodes, edges_data, detailed_segments, file_path="hirag_formulas_verify.jsonl"):
    """
    [工具函数] 保存 HiRAG 论文公式验证数据 (支持详细路书)
    """
    try:
        # 1. 格式化 A->B 的路径段
        formatted_paths = []
        if detailed_segments:
            for seg in detailed_segments:
                formatted_paths.append({
                    "pair": f"{seg['from']} to {seg['to']}",
                    "status": seg["status"],
                    "hops": len(seg['path']) - 1 if seg['path'] else 0,
                    "full_path": seg["path"] # 这里就是你要的完整路径列表
                })

        # 2. 处理 R_hat (提取边属性)
        r_hat_serializable = []
        for edge in edges_data:
            src, tgt = edge.get("src_tgt", (None, None))
            if src is None: 
                src = edge.get("src")
                tgt = edge.get("tgt")
            
            r_hat_serializable.append({
                "source": src,
                "target": tgt,
                "relation": edge.get("relation_type", edge.get("relation", "RELATED")),
                "description": edge.get("description", edge.get("desc", "")),
                "weight": edge.get("weight", 0),
                "rank": edge.get("rank", 0)
            })

        # 3. 组装最终 JSON
        log_data = {
            "query": query,
            
            # Formula 14: V_hat
            "formula_14_V_hat": {
                "description": "Key Entities (Top-M)",
                "count": len(key_entities),
                "entities": key_entities
            },
            
            # Formula 15: R (这里现在有了 segments_detail)
            "formula_15_R": {
                "description": "Reasoning Paths (Detailed Segments)",
                "total_segments": len(formatted_paths),
                "segments_detail": formatted_paths, # ✅ 你的 A->B 细节在这里
                "flattened_sequence": path_nodes    # 原始长串
            },
            
            # Formula 16: R_hat
            "formula_16_R_hat": {
                "description": "Refined Subgraph Triples",
                "count": len(r_hat_serializable),
                "triples": r_hat_serializable
            }
        }

        # 4. 追加写入
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
            
    except Exception as e:
        print(f"⚠️ [HiRAG Utils Error] Failed to save log: {e}")
def save_rgcn_metrics_log(
    query, 
    final_query_nodes, 
    use_communities, 
    structural_path_edges, 
    rgcn_raw_paths,
    final_reasoning_triples=None,  # 接收 rgcn_added_only_list (增量路径)
    all_reasoning_triples=None,    # 接收 reasoning_path_section_list (全量路径)
    log_file_path=None
):
    """
    数据埋点函数：记录 RGCN 路径分析数据。
    包含：RGCN 增量路径、最终全量路径、社区覆盖情况等。
    输出格式：带缩进的 JSON，方便肉眼观察。
    """
    try:
        # =========================================================
        # 第一步：计算中间变量 (必须在定义字典前完成计算！)
        # =========================================================
        
        # 1. 计算 final_entity_set (最终选中的实体集合)
        final_entity_set = {n["entity_name"] for n in final_query_nodes}
        
        # 2. 计算 selected_comm_ids 和 entity_to_comm_map
        selected_comm_ids = set()
        entity_to_comm_map = {} 
        
        for comm in use_communities:
            c_id = None
            # --- ID 提取逻辑 (兼容字典和对象) ---
            if isinstance(comm, dict):
                c_id = comm.get("id") or comm.get("report_id") or comm.get("cluster")
                # 修复 "Cluster XX" 这种标题的解析问题
                if c_id is None and "title" in comm:
                    match = re.search(r"Cluster\s+(\d+)", str(comm["title"]))
                    if match:
                        c_id = match.group(1)
            elif hasattr(comm, "id"):
                c_id = comm.id
                
            # --- 建立映射 ---
            if c_id is not None:
                c_id_str = str(c_id)
                selected_comm_ids.add(c_id_str)
                
                # 提取该社区下的所有节点
                nodes_list = []
                if isinstance(comm, dict):
                    nodes_list = comm.get("nodes", [])
                elif hasattr(comm, "nodes"):
                    nodes_list = comm.nodes 
                
                for node_name in nodes_list:
                    entity_to_comm_map[node_name] = c_id_str

        # 3. 计算 rgcn_path_entities_set (RGCN 推荐的实体集合)
        rgcn_path_entities_set = set()
        for edge in structural_path_edges:
            src = edge.get("src")
            tgt = edge.get("tgt")
            if src: rgcn_path_entities_set.add(src)
            if tgt: rgcn_path_entities_set.add(tgt)
            
        # 4. 定义辅助函数：统一格式化三元组列表
        def format_triples(triples_list):
            formatted = []
            if triples_list:
                for row in triples_list:
                    # row 格式预期: [id, source, target, relation, description, weight, rank]
                    if isinstance(row, list) and len(row) >= 4: 
                        formatted.append({
                            "id": row[0],
                            "source": row[1],
                            "target": row[2],
                            "relation": row[3],
                            "weight": row[5] if len(row) > 5 else "N/A",
                            "rank": row[6] if len(row) > 6 else "N/A"
                        })
            return formatted

        # =========================================================
        # 第二步：组装日志字典 (所有变量此刻已就绪)
        # =========================================================
        analysis_record = {
            "query": query,
            
            # 基础 RGCN 信息
            "rgcn_path_entities": list(rgcn_path_entities_set),
            "rgcn_raw_paths": rgcn_raw_paths,
            
            # 覆盖率分析数据
            "final_selected_entities": list(final_entity_set),
            "selected_comm_ids": list(selected_comm_ids),
            # "entity_to_comm_map": entity_to_comm_map,
            
            # 🔥 核心路径数据
            # 1. 只有 RGCN 新增的路径 (增量)
            "rgcn_added_paths": format_triples(final_reasoning_triples),
            
            # 2. 最终喂给大模型的全部路径 (全量 = 社区 + RGCN)
            "all_context_paths": format_triples(all_reasoning_triples) 
        }

        # =========================================================
        # 第三步：写入文件 (带缩进格式化)
        # =========================================================
        if log_file_path is None:
            log_file_path = os.environ.get("PLUGRAG_ANALYSIS_LOG", "rgcn_coverage_analysis.jsonl")

        # 获取目录路径并确保存在
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        with open(log_file_path, "a", encoding="utf-8") as f:
            # 🔥 这里加了 indent=4 让它自动换行和缩进，方便肉眼观察
            # separators=(',', ': ') 去掉多余空格，保持紧凑美观
            f.write(json.dumps(analysis_record, ensure_ascii=False, indent=4, separators=(',', ': ')) + "\n\n")

    except Exception as e:
        # 错误处理：优先使用 logger，否则 print
        error_msg = f"Failed to save RGCN analysis log: {str(e)}"
        if 'logger' in globals():
            logger.error(error_msg)
        else:
            print(f"⚠️ {error_msg}")
# import re
# import json
# import os

# def extract_data_snapshot(
#     query,                # <--- [新增] 接收问题文本
#     final_entities,       
#     use_communities,      
#     bridging_paths=None,       
#     physical_paths=None   
# ):
#     """
#     增强版提取器：包含问题文本 (Query) 以及路径的详细描述 (Description)。
#     """

#     # --- 1. 内部辅助：提取纯数字社区 ID ---
#     def get_clean_comm_id(comm):
#         c_id = None
#         if isinstance(comm, dict):
#             c_id = comm.get("id") or comm.get("report_id") or comm.get("cluster")
#             if c_id is None and "title" in comm:
#                 match = re.search(r"Cluster\s+(\d+)", str(comm["title"]))
#                 if match:
#                     c_id = match.group(1)
#         elif hasattr(comm, "id"):
#             c_id = comm.id
#         return str(c_id) if c_id is not None else "UNKNOWN"

#     # --- 2. 内部辅助：通用路径格式化 (带描述) ---
#     def format_paths(paths, path_type="bridging"):
#         if not paths: return []
#         formatted = []
#         for p in paths:
#             try:
#                 src, tgt, rel, desc = "Unknown", "Unknown", "link", ""
                
#                 # 策略 A: 处理 existing_path_edges (原有路径)
#                 # 结构: {'src_tgt': ('A', 'B'), 'relation_type': 'REL', 'description': '...'}
#                 if path_type == "bridging" and isinstance(p, dict) and "src_tgt" in p:
#                     src_val, tgt_val = p["src_tgt"]
#                     src, tgt = str(src_val), str(tgt_val)
#                     rel = str(p.get("relation_type") or p.get("relation") or "link")
#                     desc = str(p.get("description", ""))

#                 # 策略 B: 处理 structural_path_edges (RGCN 路径)
#                 # 结构: {'src': 'A', 'tgt': 'B', 'desc': '...'}
#                 elif path_type == "physical" and isinstance(p, dict):
#                     src = str(p.get("src") or p.get("source") or "Unknown")
#                     tgt = str(p.get("tgt") or p.get("target") or "Unknown")
#                     rel = str(p.get("relation") or p.get("relation_type") or "link")
#                     # RGCN 有时候字段叫 desc，有时候叫 description
#                     desc = str(p.get("desc") or p.get("description") or "")
                
#                 # 策略 C: 兜底处理 CSV 列表结构 [id, src, tgt, rel, desc, ...]
#                 elif isinstance(p, list) and len(p) >= 4:
#                     if p[0] == "id": continue 
#                     src, tgt, rel = str(p[1]), str(p[2]), str(p[3])
#                     # 如果列表有第5项，通常是描述
#                     if len(p) > 4:
#                         desc = str(p[4])

#                 # 清洗数据
#                 s_clean = src.strip('"')
#                 t_clean = tgt.strip('"')
#                 r_clean = rel.strip('"')
#                 d_clean = desc.strip().strip('"')
                
#                 # 格式化输出：如果描述存在，就加上
#                 path_str = f"{s_clean} -> {t_clean} ({r_clean})"
#                 if d_clean:
#                     path_str += f" | Desc: {d_clean}"
                
#                 formatted.append(path_str)
            
#             except Exception:
#                 continue
#         return formatted

#     # --- 3. 组装数据 ---
#     bridging_paths = bridging_paths or []
#     physical_paths = physical_paths or []
#     final_entities = final_entities or []

#     snapshot = {
#         # [新增] 问题字段
#         "query": query,

#         # 1. 实体
#         "entities": [e["entity_name"].strip('"') for e in final_entities if "entity_name" in e],
        
#         # 2. 社区 ID
#         "community_ids": list(set(get_clean_comm_id(c) for c in use_communities)) if use_communities else [],
        
#         # 3. 原有互联路径 (带描述)
#         "bridging_paths": format_paths(bridging_paths, path_type="bridging"),
        
#         # 4. RGCN 新增路径 (带描述)
#         "rgcn_physical_paths": format_paths(physical_paths, path_type="physical")
#     }
    
#     return snapshot
import re
import json
import os

def extract_data_snapshot(
    query,
    final_entities,
    use_communities,
    hi_source_names=None,   
    rgcn_source_names=None,
    bridging_paths=None,
    physical_paths=None
):
    """
    分类提取器：
    将实体分为三个独立列表：[HI独有], [RGCN独有], [HI+RGCN交集]。
    """
    
    # --- 1. 内部辅助：清洗社区 ID ---
    def get_clean_comm_id(comm):
        c_id = None
        if isinstance(comm, dict):
            c_id = comm.get("id") or comm.get("report_id") or comm.get("cluster")
            if c_id is None and "title" in comm:
                match = re.search(r"Cluster\s+(\d+)", str(comm["title"]))
                if match:
                    c_id = match.group(1)
        elif hasattr(comm, "id"):
            c_id = comm.id
        return str(c_id) if c_id is not None else "UNKNOWN"

    # --- 2. 内部辅助：路径格式化 ---
    def format_paths(paths, path_type="bridging"):
        if not paths: return []
        formatted = []
        for p in paths:
            try:
                src, tgt, rel, desc = "Unknown", "Unknown", "link", ""
                
                if path_type == "bridging" and isinstance(p, dict) and "src_tgt" in p:
                    src_val, tgt_val = p["src_tgt"]
                    src, tgt = str(src_val), str(tgt_val)
                    rel = str(p.get("relation_type") or p.get("relation") or "link")
                    desc = str(p.get("description", ""))
                elif path_type == "physical" and isinstance(p, dict):
                    src = str(p.get("src") or p.get("source") or "Unknown")
                    tgt = str(p.get("tgt") or p.get("target") or "Unknown")
                    rel = str(p.get("relation") or p.get("relation_type") or "link")
                    desc = str(p.get("desc") or p.get("description") or "")
                elif isinstance(p, list) and len(p) >= 4:
                    if p[0] == "id": continue 
                    src, tgt, rel = str(p[1]), str(p[2]), str(p[3])
                    if len(p) > 4: desc = str(p[4])

                s_clean = src.strip('"')
                t_clean = tgt.strip('"')
                r_clean = rel.strip('"')
                d_clean = desc.strip().strip('"')
                
                path_str = f"{s_clean} -> {t_clean} ({r_clean})"
                if d_clean: path_str += f" | Desc: {d_clean}"
                
                formatted.append(path_str)
            except: continue
        return formatted

    # --- 3. 核心逻辑：实体分流 ---
    # 先把来源名单洗干净
    clean_hi_set = set(str(n).strip('"').strip() for n in (hi_source_names or []))
    clean_rgcn_set = set(str(n).strip('"').strip() for n in (rgcn_source_names or []))
    
    hi_only_list = []
    rgcn_only_list = []
    intersection_list = []
    
    for e in final_entities:
        raw_name = e.get("entity_name", "")
        name = str(raw_name).strip('"').strip()
        
        in_hi = name in clean_hi_set
        in_rgcn = name in clean_rgcn_set
        
        # 分流逻辑
        if in_hi and in_rgcn:
            intersection_list.append(name) # 两者都有
        elif in_hi:
            hi_only_list.append(name)      # 只有 HI
        elif in_rgcn:
            rgcn_only_list.append(name)    # 只有 RGCN
        # 都不在的通常忽略，或按需记录
            
    # --- 4. 组装数据 ---
    bridging_paths = bridging_paths or []
    physical_paths = physical_paths or []

    snapshot = {
        "query": query,
        
        # 🔥 三个独立的实体列表
        "hi_only_entities": hi_only_list,
        "rgcn_only_entities": rgcn_only_list,
        "hi_rgcn_merged_entities": intersection_list,
        
        "community_ids": list(set(get_clean_comm_id(c) for c in use_communities)) if use_communities else [],
        "bridging_paths": format_paths(bridging_paths, path_type="bridging"),
        "rgcn_physical_paths": format_paths(physical_paths, path_type="physical")
    }
    
    return snapshot

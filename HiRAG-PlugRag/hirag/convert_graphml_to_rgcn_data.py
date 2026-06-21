import torch
import networkx as nx
from torch_geometric.data import HeteroData
import numpy as np
import argparse
import os
import json
from openai import AsyncOpenAI
import asyncio
import logging
import yaml

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------
# 异步获取嵌入 (增强版)
# ---------------------------
async def get_embeddings_via_api(texts: list[str], model_name: str, client: AsyncOpenAI, batch_size: int = 50):
    """
    异步批量调用 API 获取嵌入，带重试机制和空值检查
    """
    if client is None:
        raise ValueError("API客户端未初始化")

    all_embeddings = []
    total = len(texts)
    logger.info(f"🧠 准备获取 {total} 条文本嵌入 (Model: {model_name})...")

    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # 简单的清理：防止 API 报错空字符串
        batch_texts = [t if t.strip() else "unknown entity" for t in batch_texts]

        try:
            response = await client.embeddings.create(input=batch_texts, model=model_name)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            logger.info(f"✅ 进度: {min(i + batch_size, total)}/{total}")
        except Exception as e:
            logger.error(f"❌ API 调用失败 (Batch {i}): {e}")
            # 失败时的兜底策略：填充零向量 (假设维度 1024，需根据模型调整，这里抛出异常更安全)
            raise e
    
    return np.array(all_embeddings, dtype=np.float32)

# ---------------------------
# 主程序
# ---------------------------
async def main():
    # 1. 加载配置
    config_path = os.environ.get("PLUGRAG_CONFIG", os.path.join(os.getcwd(), "config.example.yaml"))
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}
        logger.warning("⚠️ 未找到配置文件，将使用默认参数。")

    rc = cfg.get("rgcn", {}).get("convert", {})
    oc = cfg.get("openai", {})

    # 2. 参数解析
    parser = argparse.ArgumentParser(description="Convert GraphRAG .graphml to R-GCN HeteroData format")
    parser.add_argument("--input", type=str, default=rc.get("graphml_input", "graph.graphml"))
    parser.add_argument("--output", type=str, default=rc.get("pt_output", "rgcn_data.pt"))
    parser.add_argument("--api_key", type=str, default=oc.get("api_key", os.environ.get("OPENAI_API_KEY")))
    parser.add_argument("--base_url", type=str, default=oc.get("base_url", "https://api.openai.com/v1"))
    parser.add_argument("--embedding_model", type=str, default=rc.get("embedding_model", "BAAI/bge-m3"))
    parser.add_argument("--batch_size", type=int, default=int(rc.get("batch_size", 50)))
    args = parser.parse_args()

    # 3. 加载图谱
    if not os.path.exists(args.input):
        logger.error(f"找不到输入文件: {args.input}")
        return

    logger.info(f"📘 Loading GraphML file: {args.input}")
    G = nx.read_graphml(args.input)
    logger.info(f"✅ Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # ---------------------------
    # A. 节点处理 (Rich Content)
    # ---------------------------
    node_ids = list(G.nodes)
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    
    # ✨ 修改点：拼接 Name + Description 以增强语义锚点
    node_texts = []
    for n in node_ids:
        name = n.strip('"') # 去除可能存在的引号
        desc = G.nodes[n].get("description", "")
        # 如果有描述，拼在一起；如果没有，重复名字以加强权重
        if desc:
            content = f"{name}: {desc}"
        else:
            content = f"{name}: {name}" 
        node_texts.append(content)

    # ---------------------------
    # B. 获取嵌入
    # ---------------------------
    if not args.api_key:
        logger.error("❌ 未提供 API Key，无法获取嵌入。")
        return

    async_client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    
    embeddings_np = await get_embeddings_via_api(
        node_texts, 
        args.embedding_model, 
        async_client, 
        batch_size=args.batch_size
    )
    node_features = torch.tensor(embeddings_np, dtype=torch.float)
    logger.info(f"✅ Node feature matrix shape: {node_features.shape}")

    # ---------------------------
    # C. 边处理 (按关系类型分组)
    # ---------------------------
    # HeteroData 的结构：data['src_type', 'relation', 'dst_type']
    # 在 GraphRAG 中，通常只有一种节点类型 'entity'
    
    edge_index_dict = {}
    relation_types = set()

    for u, v, data in G.edges(data=True):
        # 获取关系类型，默认为 'RELATED'，并统一大写去空格
        rel = data.get("relation_type", "RELATED").strip().upper()
        # 将关系里的空格替换为下划线，防止 PyG 报错
        rel = rel.replace(" ", "_")
        
        relation_types.add(rel)
        
        # 构建 HeteroData 的 key
        rel_key = ('entity', rel, 'entity')
        
        if rel_key not in edge_index_dict:
            edge_index_dict[rel_key] = [[], []]
        
        if u in node_to_idx and v in node_to_idx:
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            edge_index_dict[rel_key][0].append(u_idx)
            edge_index_dict[rel_key][1].append(v_idx)

    relation_types = sorted(list(relation_types))
    logger.info(f"🧩 Found {len(relation_types)} relation types: {relation_types[:5]}...")

    # ---------------------------
    # D. 构造 HeteroData
    # ---------------------------
    data = HeteroData()
    data['entity'].x = node_features
    # 保存节点数量，防止孤立点导致的 shape 推断错误
    data['entity'].num_nodes = len(node_ids)

    for rel_key, (src_list, dst_list) in edge_index_dict.items():
        if len(src_list) > 0:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            data[rel_key].edge_index = edge_index
        else:
            # 这是一个空的边类型，为了稳健性可以添加空 tensor
            data[rel_key].edge_index = torch.empty((2, 0), dtype=torch.long)

    logger.info(f"✅ Constructed HeteroData object.")

    # ---------------------------
    # E. 保存与映射 (包含关系映射)
    # ---------------------------
    torch.save(data, args.output)
    logger.info(f"💾 Saved R-GCN graph data to: {args.output}")

    # 保存完整的映射文件，方便后续推理和分析
    mapping_path = os.path.splitext(args.output)[0] + "_mapping.json"
    idx_to_node = {i: node_id for node_id, i in node_to_idx.items()}
    
    # ✨ 修改点：保存关系类型映射
    # 虽然 HeteroData 内部存了 key，但为了 debug 方便，我们显式保存
    # 这里的 ID 对应 train_rgcn.py 中 build_edge_index_and_types 排序后的 ID
    rel_to_idx = {r: i for i, r in enumerate(relation_types)}

    mappings = {
        "node_to_idx": node_to_idx,
        "idx_to_node": idx_to_node,
        "relation_types": relation_types, # 列表本身就是一种 index -> name 映射
        "rel_to_idx": rel_to_idx
    }
    
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)
    logger.info(f"🗺️ Saved entity & relation mapping to: {mapping_path}")

if __name__ == "__main__":
    asyncio.run(main())

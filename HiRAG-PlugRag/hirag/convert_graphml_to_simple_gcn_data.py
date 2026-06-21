import torch
import networkx as nx
from torch_geometric.data import Data  # 【关键】这里导入的是 Data 而不是 HeteroData
import numpy as np
import argparse
import os
import json
from openai import AsyncOpenAI
import asyncio
import logging
import yaml

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# 1. 异步获取嵌入 (保持不变)
# ---------------------------
async def get_embeddings_via_api(texts: list[str], model_name: str, client: AsyncOpenAI, batch_size: int = 50):
    """
    异步批量调用 API 获取嵌入
    """
    if client is None:
        raise ValueError("API客户端未初始化")

    all_embeddings = []
    logger.info(f"🚀 准备获取 {len(texts)} 条文本的嵌入 (Model: {model_name})...")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            response = await client.embeddings.create(input=batch_texts, model=model_name)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            if i % 100 == 0:
                logger.info(f"   ⏳ 已处理 {min(i + batch_size, len(texts))}/{len(texts)} 条...")
        except Exception as e:
            logger.error(f"❌ API 请求失败: {e}")
            raise e
    
    logger.info("✅ 所有文本嵌入获取完成！")
    return np.array(all_embeddings, dtype=np.float32)

# ---------------------------
# 2. 主程序
# ---------------------------
async def main():
    # --- 加载配置文件 ---
    config_path = os.environ.get("PLUGRAG_CONFIG", os.path.join(os.getcwd(), "config.example.yaml"))
    if not os.path.exists(config_path):
        logger.error(f"找不到配置文件: {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    rc = cfg.get("gcn", {}).get("convert", cfg.get("rgcn", {}).get("convert", {}))
    oc = cfg.get("openai", {})

    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="Convert GraphRAG .graphml to GCN (Homogeneous) format")
    parser.add_argument("--input", type=str, default=rc.get("graphml_input", ""))
    parser.add_argument("--output", type=str, default=rc.get("pt_output", "graph_data.pt"))
    parser.add_argument("--embedding_model", type=str, default=rc.get("embedding_model", oc.get("embedding_model", "BAAI/bge-m3")))
    parser.add_argument("--batch_size", type=int, default=int(rc.get("batch_size", 20)))
    args = parser.parse_args()

    logger.info(f"📘 正在加载 GraphML 文件: {args.input}")
    try:
        G = nx.read_graphml(args.input)
    except Exception as e:
        logger.error(f"无法读取 GraphML 文件: {e}")
        return
        
    logger.info(f"✅ 图谱加载成功: {len(G.nodes)} 节点, {len(G.edges)} 边")

    # ---------------------------
    # 3. 节点处理 (Node Processing)
    # ---------------------------
    node_ids = list(G.nodes)
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    
    # 提取节点文本用于Embedding
    # 注意：确保 graphml 里的属性名是 'description'，有些版本可能是 'desc' 或其他
    node_texts = [G.nodes[n].get("description", G.nodes[n].get("name", "")) for n in node_ids]
    
    # 初始化 API 客户端
    api_key = oc.get("api_key") or os.environ.get("OPENAI_API_KEY")
    base_url = oc.get("base_url") or os.environ.get("OPENAI_BASE_URL") or "https://api.siliconflow.cn/v1"
    if not api_key:
        raise ValueError("Embedding API key is not configured. Set openai.api_key or OPENAI_API_KEY.")
    
    async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    # 获取特征
    embeddings_np = await get_embeddings_via_api(node_texts, args.embedding_model, async_client, batch_size=args.batch_size)
    node_features = torch.tensor(embeddings_np, dtype=torch.float)
    logger.info(f"✅ 节点特征矩阵形状: {node_features.shape}")
    
    # ---------------------------
# 4. 边处理 (Edge Processing - 去除多重边版本)
# ---------------------------
    logger.info("🔄 正在构建同质图边索引 (Simple Graph Mode)...")

    edge_set = set()

    for u, v in G.edges():  # 不需要 data=True
        if u not in node_to_idx or v not in node_to_idx:
            continue

        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]

        # 去重
        edge_set.add((u_idx, v_idx))

    logger.info(f"📊 原始边数: {len(G.edges)}")
    logger.info(f"📊 去重后边数: {len(edge_set)}")

    # 转为 tensor
    if len(edge_set) > 0:
        src, dst = zip(*edge_set)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # ---------------------------
    # 5. 构造并保存 Data 对象
    # ---------------------------
    # 使用 Data 而不是 HeteroData，这是 GCN 的标准输入格式
    data = Data(x=node_features, edge_index=edge_index)
    
    # 添加辅助信息（可选，方便调试）
    data.num_nodes = len(node_ids)
    
    logger.info(f"✅ 构建完成: {data}")
    
    # 保存 .pt 文件
    torch.save(data, args.output)
    logger.info(f"💾 GCN 数据集已保存至: {args.output}")

    # 保存 ID 映射 (JSON)
    mapping_path = os.path.splitext(args.output)[0] + "_mapping.json"
    idx_to_node = {i: node_id for i, node_id in enumerate(node_ids)}
    mappings = {"node_to_idx": node_to_idx, "idx_to_node": idx_to_node}
    
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)
    logger.info(f"🗺️ 实体映射表已保存至: {mapping_path}")

if __name__ == "__main__":
    asyncio.run(main())

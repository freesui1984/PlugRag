import argparse
import asyncio
import json
import logging
import os

import networkx as nx
import numpy as np
import torch
import yaml
from openai import AsyncOpenAI
from torch_geometric.data import HeteroData


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


async def get_embeddings_via_api(texts, model_name, client, batch_size=20):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = [text if str(text).strip() else "unknown entity" for text in texts[i:i + batch_size]]
        response = await client.embeddings.create(
            input=batch_texts,
            model=model_name,
            encoding_format="float",
        )
        all_embeddings.extend([item.embedding for item in response.data])
        logger.info("Embedding progress: %s/%s", min(i + batch_size, len(texts)), len(texts))
    return np.array(all_embeddings, dtype=np.float32)


def iter_edges(graph):
    if graph.is_multigraph():
        for u, v, key, attr in graph.edges(keys=True, data=True):
            yield u, v, key, attr or {}
    else:
        for u, v, attr in graph.edges(data=True):
            yield u, v, None, attr or {}


def relation_label(edge_key, edge_attr):
    raw = (
        edge_attr.get("relation_type")
        or edge_attr.get("relation")
        or edge_attr.get("label")
        or edge_key
        or "RELATED"
    )
    return str(raw).strip().upper().replace(" ", "_")


def node_text(graph, node_id):
    attrs = graph.nodes[node_id]
    name = str(attrs.get("name") or node_id).strip('"')
    desc = attrs.get("description") or attrs.get("desc") or ""
    return f"{name}: {desc}" if desc else f"{name}: {name}"


async def main():
    config_path = os.environ.get("PLUGRAG_CONFIG", os.path.join(os.getcwd(), "config.example.yaml"))
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    convert_cfg = cfg.get("multi_edge_rgcn_transe", {}).get(
        "convert", cfg.get("rgcn", {}).get("convert", {})
    )
    openai_cfg = cfg.get("openai", {})

    parser = argparse.ArgumentParser(
        description="Convert GraphRAG .graphml to Multi-edge R-GCN HeteroData format"
    )
    parser.add_argument("--input", default=convert_cfg.get("graphml_input", "graph.graphml"))
    parser.add_argument("--output", default=convert_cfg.get("pt_output", "multi_edge_rgcn_data.pt"))
    parser.add_argument(
        "--api_key",
        default=openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY"),
    )
    parser.add_argument(
        "--base_url",
        default=openai_cfg.get("base_url")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
    )
    parser.add_argument(
        "--embedding_model",
        default=convert_cfg.get(
            "embedding_model", openai_cfg.get("embedding_model", "BAAI/bge-m3")
        ),
    )
    parser.add_argument("--batch_size", type=int, default=int(convert_cfg.get("batch_size", 20)))
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"GraphML input not found: {args.input}")
    if not args.api_key:
        raise ValueError("Embedding API key is not configured.")

    graph = nx.read_graphml(args.input)
    node_ids = list(graph.nodes)
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    texts = [node_text(graph, node_id) for node_id in node_ids]

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url, timeout=600.0)
    node_features = torch.tensor(
        await get_embeddings_via_api(texts, args.embedding_model, client, args.batch_size),
        dtype=torch.float,
    )

    edge_index_dict = {}
    relation_types = set()
    for u, v, key, attr in iter_edges(graph):
        if u not in node_to_idx or v not in node_to_idx:
            continue
        rel = relation_label(key, attr)
        relation_types.add(rel)
        rel_key = ("entity", rel, "entity")
        edge_index_dict.setdefault(rel_key, [[], []])
        edge_index_dict[rel_key][0].append(node_to_idx[u])
        edge_index_dict[rel_key][1].append(node_to_idx[v])

    data = HeteroData()
    data["entity"].x = node_features
    data["entity"].num_nodes = len(node_ids)
    for rel_key, (src, dst) in edge_index_dict.items():
        data[rel_key].edge_index = torch.tensor([src, dst], dtype=torch.long)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(data, args.output)

    relation_types = sorted(relation_types)
    mapping_path = os.path.splitext(args.output)[0] + "_mapping.json"
    mappings = {
        "node_to_idx": node_to_idx,
        "idx_to_node": {i: node_id for node_id, i in node_to_idx.items()},
        "relation_types": relation_types,
        "rel_to_idx": {rel: i for i, rel in enumerate(relation_types)},
    }
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)

    logger.info("Saved multi-edge R-GCN data to: %s", args.output)
    logger.info("Saved mapping to: %s", mapping_path)


if __name__ == "__main__":
    asyncio.run(main())

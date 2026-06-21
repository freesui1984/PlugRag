import argparse
import asyncio
import json
import logging
import os
from typing import Any, Dict, Iterable, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import yaml
from openai import AsyncOpenAI
from torch_geometric.data import Data, HeteroData


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = config_path or os.environ.get("PLUGRAG_CONFIG", "config_example.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _working_path(working_dir: str, file_name: str) -> str:
    return file_name if os.path.isabs(file_name) else os.path.join(working_dir, file_name)


def _mode_default_output(config: Dict[str, Any], mode: str) -> str:
    storage = config.get("storage", {})
    graphml = config.get("graphml_conversion", {})
    working_dir = storage.get("working_dir", "work_dir_data_mix")
    if mode == "rgcn":
        default_file = storage.get("rgcn_data_file", "rgcn/rgcn_data.pt")
        output = graphml.get("rgcn_output_file", default_file)
    elif mode == "simple_gcn":
        default_file = storage.get("gcn_data_file", "gcn/gcn_data.pt")
        output = graphml.get("gcn_output_file", default_file)
    elif mode == "multi_edge":
        default_file = storage.get("multi_edge_data_file", "multi_edge/multi_edge_rgcn_data.pt")
        output = graphml.get("multi_edge_output_file", default_file)
    else:
        raise ValueError(f"Unknown conversion mode: {mode}")
    return _working_path(working_dir, output)


def conversion_defaults(config: Dict[str, Any], mode: str) -> Dict[str, Any]:
    api_config = config.get("api_keys", {})
    model_config = config.get("models", {})
    storage = config.get("storage", {})
    graphml = config.get("graphml_conversion", {})
    working_dir = storage.get("working_dir", "work_dir_data_mix")
    return {
        "input": graphml.get(
            "graphml_input",
            os.path.join(working_dir, "graph_chunk_entity_relation.graphml"),
        ),
        "output": _mode_default_output(config, mode),
        "api_key": api_config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY"),
        "base_url": (
            api_config.get("openai_base_url")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ),
        "embedding_model": graphml.get(
            "embedding_model", model_config.get("embedding_model", "BAAI/bge-m3")
        ),
        "batch_size": int(graphml.get("batch_size", 20)),
    }


async def get_embeddings_via_api(
    texts: list[str],
    model_name: str,
    client: AsyncOpenAI,
    batch_size: int = 20,
) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = [text if text.strip() else "unknown entity" for text in texts[i:i + batch_size]]
        response = await client.embeddings.create(
            input=batch_texts,
            model=model_name,
            encoding_format="float",
        )
        all_embeddings.extend([item.embedding for item in response.data])
        logger.info("Embedding progress: %s/%s", min(i + batch_size, len(texts)), len(texts))
    return np.array(all_embeddings, dtype=np.float32)


def iter_edges(graph: nx.Graph) -> Iterable[Tuple[str, str, Optional[str], Dict[str, Any]]]:
    if graph.is_multigraph():
        for u, v, key, attr in graph.edges(keys=True, data=True):
            yield u, v, key, attr or {}
    else:
        for u, v, attr in graph.edges(data=True):
            yield u, v, None, attr or {}


def relation_label(edge_key: Optional[str], edge_attr: Dict[str, Any]) -> str:
    raw = (
        edge_attr.get("relation_type")
        or edge_attr.get("relation")
        or edge_attr.get("label")
        or edge_key
        or "RELATED"
    )
    return str(raw).strip().upper().replace(" ", "_")


def node_text(graph: nx.Graph, node_id: str) -> str:
    attrs = graph.nodes[node_id]
    name = str(attrs.get("name") or node_id).strip('"')
    desc = attrs.get("description") or attrs.get("desc") or ""
    return f"{name}: {desc}" if desc else f"{name}: {name}"


async def convert_graphml(
    input_path: str,
    output_path: str,
    api_key: str,
    base_url: str,
    embedding_model: str,
    batch_size: int,
    mode: str,
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"GraphML input not found: {input_path}")
    if not api_key:
        raise ValueError("Embedding API key is not configured.")

    logger.info("Loading GraphML file: %s", input_path)
    graph = nx.read_graphml(input_path)
    node_ids = list(graph.nodes)
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    texts = [node_text(graph, node_id) for node_id in node_ids]

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=600.0)
    embeddings_np = await get_embeddings_via_api(
        texts,
        embedding_model,
        client,
        batch_size=batch_size,
    )
    node_features = torch.tensor(embeddings_np, dtype=torch.float)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if mode == "simple_gcn":
        data, mappings = build_simple_gcn_data(graph, node_to_idx, node_features)
    elif mode in {"rgcn", "multi_edge"}:
        data, mappings = build_rgcn_data(graph, node_to_idx, node_features)
    else:
        raise ValueError(f"Unknown conversion mode: {mode}")

    torch.save(data, output_path)
    mapping_path = os.path.splitext(output_path)[0] + "_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)
    logger.info("Saved converted data to: %s", output_path)
    logger.info("Saved mapping to: %s", mapping_path)
    return data, mappings


def build_rgcn_data(graph: nx.Graph, node_to_idx: Dict[str, int], node_features: torch.Tensor):
    edge_index_dict: Dict[Tuple[str, str, str], list[list[int]]] = {}
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
    data["entity"].num_nodes = len(node_to_idx)
    for rel_key, (src, dst) in edge_index_dict.items():
        data[rel_key].edge_index = torch.tensor([src, dst], dtype=torch.long)

    relation_types = sorted(relation_types)
    idx_to_node = {i: node_id for node_id, i in node_to_idx.items()}
    mappings = {
        "node_to_idx": node_to_idx,
        "idx_to_node": idx_to_node,
        "relation_types": relation_types,
        "rel_to_idx": {rel: i for i, rel in enumerate(relation_types)},
    }
    return data, mappings


def build_simple_gcn_data(graph: nx.Graph, node_to_idx: Dict[str, int], node_features: torch.Tensor):
    edge_set = set()
    for u, v, _key, _attr in iter_edges(graph):
        if u in node_to_idx and v in node_to_idx:
            edge_set.add((node_to_idx[u], node_to_idx[v]))

    if edge_set:
        src, dst = zip(*sorted(edge_set))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index)
    data.num_nodes = len(node_to_idx)
    idx_to_node = {i: node_id for node_id, i in node_to_idx.items()}
    mappings = {"node_to_idx": node_to_idx, "idx_to_node": idx_to_node}
    return data, mappings


def parse_args(mode: str):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _unknown = pre_parser.parse_known_args()
    config = load_config(pre_args.config)
    defaults = conversion_defaults(config, mode)

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--input", default=defaults["input"], help="Input GraphML path.")
    parser.add_argument("--output", default=defaults["output"], help="Output .pt path.")
    parser.add_argument("--api_key", default=defaults["api_key"], help="Embedding API key.")
    parser.add_argument("--base_url", default=defaults["base_url"], help="Embedding API base URL.")
    parser.add_argument(
        "--embedding_model",
        default=defaults["embedding_model"],
        help="Embedding model name.",
    )
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    return parser.parse_args()


def run_cli(mode: str):
    args = parse_args(mode)
    asyncio.run(
        convert_graphml(
            input_path=args.input,
            output_path=args.output,
            api_key=args.api_key,
            base_url=args.base_url,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            mode=mode,
        )
    )

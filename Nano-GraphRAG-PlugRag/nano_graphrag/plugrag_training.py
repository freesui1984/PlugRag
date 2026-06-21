import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from openai import AsyncOpenAI

from . import GraphRAG
from ._utils import (
    always_get_an_event_loop,
    compute_args_hash,
    wrap_embedding_func_with_attrs,
)
from .base import BaseKVStorage
from .rgcn_utils import (
    convert_to_gcn_data,
    convert_to_multi_edge_rgcn_data,
    convert_to_rgcn_data,
)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = config_path or os.environ.get("PLUGRAG_CONFIG", "config_example.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _working_path(working_dir: str, file_name: str) -> str:
    return file_name if os.path.isabs(file_name) else os.path.join(working_dir, file_name)


def _write_mapping_copy(working_dir: str, mapping_file: str, mappings: Dict[str, Any]) -> None:
    mapping_path = _working_path(working_dir, mapping_file)
    os.makedirs(os.path.dirname(mapping_path) or working_dir, exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)


def _require_api_key(api_key: Optional[str]) -> None:
    if not api_key:
        raise ValueError(
            "API key is not configured. Set api_keys.openai_api_key or OPENAI_API_KEY."
        )


def build_openai_functions(config: Dict[str, Any]):
    api_config = config.get("api_keys", {})
    llm_config = config.get("llm", {})
    model_config = config.get("models", {})

    api_key = api_config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    base_url = (
        api_config.get("openai_base_url")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    llm_model = llm_config.get("model", "gpt-4o-mini")
    embedding_model = model_config.get("embedding_model", "BAAI/bge-m3")
    embedding_dim = int(model_config.get("embedding_dim", 1024))
    max_token_size = int(model_config.get("max_token_size", 8192))
    temperature = float(llm_config.get("temperature", 0.0))

    @wrap_embedding_func_with_attrs(
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
    )
    async def openai_embedding(texts: List[str]) -> np.ndarray:
        _require_api_key(api_key)
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=600.0)
        response = await client.embeddings.create(
            model=embedding_model,
            input=texts,
            encoding_format="float",
        )
        return np.array([item.embedding for item in response.data])

    async def openai_model_if_cache(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        _require_api_key(api_key)
        hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=600.0)

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        if hashing_kv is not None:
            args_hash = compute_args_hash(llm_model, messages)
            cached = await hashing_kv.get_by_id(args_hash)
            if cached is not None:
                return cached["return"]

        response = await client.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        answer = response.choices[0].message.content

        if hashing_kv is not None:
            await hashing_kv.upsert({args_hash: {"return": answer, "model": llm_model}})
        return answer

    return openai_embedding, openai_model_if_cache


def load_contexts(path: str) -> List[str]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Context file not found: {input_path}")

    if input_path.suffix.lower() == ".jsonl":
        docs: List[str] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, str):
                    docs.append(item)
                elif isinstance(item, dict):
                    docs.append(
                        item.get("content")
                        or item.get("text")
                        or item.get("context")
                        or item.get("article")
                        or json.dumps(item, ensure_ascii=False)
                    )
        return docs

    if input_path.suffix.lower() == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [
                item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
                for item in data
            ]
        if isinstance(data, dict):
            values = data.get("documents") or data.get("contexts")
            if isinstance(values, list):
                return [
                    item
                    if isinstance(item, str)
                    else json.dumps(item, ensure_ascii=False)
                    for item in values
                ]
            return [json.dumps(data, ensure_ascii=False)]

    return [input_path.read_text(encoding="utf-8")]


def build_rag(config: Dict[str, Any]) -> GraphRAG:
    api_config = config.get("api_keys", {})
    model_config = config.get("models", {})
    storage_config = config.get("storage", {})
    retrieval_config = config.get("retrieval", {})
    indexing_config = config.get("indexing", {})
    embedding_func, model_func = build_openai_functions(config)

    return GraphRAG(
        working_dir=storage_config.get("working_dir", "work_dir_data_mix"),
        embedding_func=embedding_func,
        best_model_func=model_func,
        cheap_model_func=model_func,
        enable_naive_rag=_as_bool(retrieval_config.get("enable_naive_rag"), True),
        enable_llm_cache=_as_bool(indexing_config.get("enable_llm_cache"), False),
        rgcn_enabled=_as_bool(retrieval_config.get("rgcn_enabled"), True),
        rgcn_data_file=storage_config.get("rgcn_data_file", "rgcn/rgcn_data.pt"),
        rgcn_embeddings_file=storage_config.get(
            "rgcn_embeddings_file", "rgcn/rgcn_embeddings.pt"
        ),
        rgcn_mapping_file=storage_config.get(
            "rgcn_mapping_file", "rgcn/rgcn_embeddings_mapping.json"
        ),
        gcn_data_file=storage_config.get("gcn_data_file", "gcn/gcn_data.pt"),
        gcn_embeddings_file=storage_config.get(
            "gcn_embeddings_file", "gcn/gcn_embeddings.pt"
        ),
        gcn_mapping_file=storage_config.get(
            "gcn_mapping_file", "gcn/gcn_embeddings_mapping.json"
        ),
        multi_edge_data_file=storage_config.get(
            "multi_edge_data_file", "multi_edge/multi_edge_rgcn_data.pt"
        ),
        multi_edge_embeddings_file=storage_config.get(
            "multi_edge_embeddings_file", "multi_edge/rgcn_transe_embeddings.pt"
        ),
        multi_edge_mapping_file=storage_config.get(
            "multi_edge_mapping_file", "multi_edge/rgcn_transe_mapping.json"
        ),
        rgcn_expand_k=int(retrieval_config.get("rgcn_expand_k", 10)),
        rgcn_cluster_limit=int(retrieval_config.get("rgcn_cluster_limit", 50)),
        rgcn_rerank_alpha=float(retrieval_config.get("rgcn_rerank_alpha", 0.5)),
        rerank_api_key=api_config.get("openai_api_key")
        or os.environ.get("RERANK_API_KEY"),
        rerank_base_url=api_config.get("openai_base_url")
        or os.environ.get("RERANK_BASE_URL"),
        rerank_model_name=model_config.get("rerank_model", "BAAI/bge-reranker-v2-m3"),
    )


def maybe_insert_contexts(rag: GraphRAG, config: Dict[str, Any], label: str) -> None:
    indexing_config = config.get("indexing", {})
    context_file = indexing_config.get("context_file", "")
    if context_file and _as_bool(indexing_config.get("insert_before_training"), False):
        docs = load_contexts(context_file)
        print(f"Inserting {len(docs)} document(s) before {label} training...")
        rag.insert(docs)


async def aconvert_rgcn_data_from_config(config: Dict[str, Any]):
    storage_config = config.get("storage", {})
    train_config = config.get("rgcn_training", {})
    rag = build_rag(config)
    data_file = train_config.get(
        "data_file", storage_config.get("rgcn_data_file", "rgcn/rgcn_data.pt")
    )
    data_path = _working_path(rag.working_dir, data_file)
    os.makedirs(os.path.dirname(data_path) or rag.working_dir, exist_ok=True)
    data, mappings = await convert_to_rgcn_data(
        rag.chunk_entity_relation_graph,
        rag.entities_vdb,
        data_path,
    )
    mapping_file = train_config.get(
        "mapping_file",
        storage_config.get("rgcn_mapping_file", "rgcn/rgcn_embeddings_mapping.json"),
    )
    _write_mapping_copy(rag.working_dir, mapping_file, mappings)
    return data, mappings


async def aconvert_gcn_data_from_config(config: Dict[str, Any]):
    storage_config = config.get("storage", {})
    train_config = config.get("gcn_training", {})
    rag = build_rag(config)
    data_file = train_config.get(
        "data_file", storage_config.get("gcn_data_file", "gcn/gcn_data.pt")
    )
    data_path = _working_path(rag.working_dir, data_file)
    os.makedirs(os.path.dirname(data_path) or rag.working_dir, exist_ok=True)
    data, mappings = await convert_to_gcn_data(
        rag.chunk_entity_relation_graph,
        rag.entities_vdb,
        data_path,
    )
    mapping_file = train_config.get(
        "mapping_file",
        storage_config.get("gcn_mapping_file", "gcn/gcn_embeddings_mapping.json"),
    )
    _write_mapping_copy(rag.working_dir, mapping_file, mappings)
    return data, mappings


async def aconvert_multi_edge_data_from_config(config: Dict[str, Any]):
    storage_config = config.get("storage", {})
    train_config = config.get("multi_edge_rgcn_transe_training", {})
    rag = build_rag(config)
    data_file = train_config.get(
        "data_file",
        storage_config.get("multi_edge_data_file", "multi_edge/multi_edge_rgcn_data.pt"),
    )
    data_path = _working_path(rag.working_dir, data_file)
    os.makedirs(os.path.dirname(data_path) or rag.working_dir, exist_ok=True)
    data, mappings = await convert_to_multi_edge_rgcn_data(
        rag.chunk_entity_relation_graph,
        rag.entities_vdb,
        data_path,
    )
    mapping_file = train_config.get(
        "mapping_file",
        storage_config.get("multi_edge_mapping_file", "multi_edge/rgcn_transe_mapping.json"),
    )
    _write_mapping_copy(rag.working_dir, mapping_file, mappings)
    return data, mappings


def convert_rgcn_data_from_config(config: Dict[str, Any]):
    loop = always_get_an_event_loop()
    return loop.run_until_complete(aconvert_rgcn_data_from_config(config))


def convert_gcn_data_from_config(config: Dict[str, Any]):
    loop = always_get_an_event_loop()
    return loop.run_until_complete(aconvert_gcn_data_from_config(config))


def convert_multi_edge_data_from_config(config: Dict[str, Any]):
    loop = always_get_an_event_loop()
    return loop.run_until_complete(aconvert_multi_edge_data_from_config(config))


def convert_training_data_from_config(config: Dict[str, Any], encoder: str = "both"):
    encoder = encoder.lower()
    if encoder not in {"rgcn", "gcn", "multi_edge", "both", "all"}:
        raise ValueError("encoder must be one of: rgcn, gcn, multi_edge, both, all")

    outputs = {}
    if encoder in {"rgcn", "both", "all"}:
        outputs["rgcn"] = convert_rgcn_data_from_config(config)
    if encoder in {"gcn", "both", "all"}:
        outputs["gcn"] = convert_gcn_data_from_config(config)
    if encoder in {"multi_edge", "all"}:
        outputs["multi_edge"] = convert_multi_edge_data_from_config(config)
    return outputs


def _rgcn_training_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    storage_config = config.get("storage", {})
    train_config = config.get("rgcn_training", {})
    return {
        "convert_data": _as_bool(train_config.get("convert_data"), True),
        "data_file": train_config.get(
            "data_file", storage_config.get("rgcn_data_file", "rgcn/rgcn_data.pt")
        ),
        "output_file": train_config.get(
            "output_file",
            storage_config.get("rgcn_embeddings_file", "rgcn/rgcn_embeddings.pt"),
        ),
        "mapping_file": train_config.get(
            "mapping_file",
            storage_config.get(
                "rgcn_mapping_file", "rgcn/rgcn_embeddings_mapping.json"
            ),
        ),
        "hidden_dim": int(train_config.get("hidden_dim", 512)),
        "lr": float(train_config.get("learning_rate", 0.001)),
        "epochs": int(train_config.get("epochs", 100)),
        "num_bases": int(train_config.get("num_bases", 10)),
        "num_layers": int(train_config.get("num_layers", 3)),
        "alpha": float(train_config.get("alpha", 0.8)),
        "beta": float(train_config.get("beta", 0.2)),
        "mse_scale": float(train_config.get("mse_scale", 500.0)),
        "early_stopping": int(train_config.get("early_stopping", 20)),
        "seed": int(train_config.get("seed", 42)),
    }


def _gcn_training_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    storage_config = config.get("storage", {})
    train_config = config.get("gcn_training", {})
    return {
        "convert_data": _as_bool(train_config.get("convert_data"), True),
        "data_file": train_config.get(
            "data_file", storage_config.get("gcn_data_file", "gcn/gcn_data.pt")
        ),
        "output_file": train_config.get(
            "output_file",
            storage_config.get("gcn_embeddings_file", "gcn/gcn_embeddings.pt"),
        ),
        "mapping_file": train_config.get(
            "mapping_file",
            storage_config.get("gcn_mapping_file", "gcn/gcn_embeddings_mapping.json"),
        ),
        "hidden_dim": int(train_config.get("hidden_dim", 512)),
        "lr": float(train_config.get("learning_rate", 0.001)),
        "epochs": int(train_config.get("epochs", 100)),
        "num_layers": int(train_config.get("num_layers", 3)),
        "alpha": float(train_config.get("alpha", 0.8)),
        "beta": float(train_config.get("beta", 0.2)),
        "mse_scale": float(train_config.get("mse_scale", 50.0)),
        "early_stopping": int(train_config.get("early_stopping", 20)),
        "seed": int(train_config.get("seed", 42)),
    }


def _multi_edge_training_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    storage_config = config.get("storage", {})
    train_config = config.get("multi_edge_rgcn_transe_training", {})
    return {
        "convert_data": _as_bool(train_config.get("convert_data"), True),
        "data_file": train_config.get(
            "data_file",
            storage_config.get("multi_edge_data_file", "multi_edge/multi_edge_rgcn_data.pt"),
        ),
        "output_file": train_config.get(
            "output_file",
            storage_config.get(
                "multi_edge_embeddings_file", "multi_edge/rgcn_transe_embeddings.pt"
            ),
        ),
        "mapping_file": train_config.get(
            "mapping_file",
            storage_config.get(
                "multi_edge_mapping_file", "multi_edge/rgcn_transe_mapping.json"
            ),
        ),
        "hidden_dim": int(train_config.get("hidden_dim", 512)),
        "lr": float(train_config.get("learning_rate", 0.001)),
        "epochs": int(train_config.get("epochs", 100)),
        "num_bases": int(train_config.get("num_bases", 10)),
        "num_layers": int(train_config.get("num_layers", 3)),
        "alpha": float(train_config.get("alpha", 0.8)),
        "beta": float(train_config.get("beta", 0.2)),
        "mse_scale": float(train_config.get("mse_scale", 50.0)),
        "margin": float(train_config.get("margin", 1.0)),
        "early_stopping": int(train_config.get("early_stopping", 30)),
        "seed": int(train_config.get("seed", 42)),
    }


def train_rgcn_from_config(config: Dict[str, Any]):
    rag = build_rag(config)
    maybe_insert_contexts(rag, config, "R-GCN")
    print("Training Nano-GraphRAG PlugRag R-GCN encoder...")
    return rag.train_rgcn(**_rgcn_training_kwargs(config))


def train_gcn_from_config(config: Dict[str, Any]):
    rag = build_rag(config)
    maybe_insert_contexts(rag, config, "Simple GCN")
    print("Training Nano-GraphRAG PlugRag Simple GCN ablation encoder...")
    return rag.train_gcn(**_gcn_training_kwargs(config))


def train_multi_edge_from_config(config: Dict[str, Any]):
    rag = build_rag(config)
    maybe_insert_contexts(rag, config, "Multi-edge R-GCN TransE")
    print("Training Nano-GraphRAG PlugRag Multi-edge R-GCN TransE ablation encoder...")
    return rag.train_multi_edge_rgcn_transe(**_multi_edge_training_kwargs(config))


def _parse_config_arg() -> Optional[str]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config. Defaults to PLUGRAG_CONFIG or config_example.yaml.",
    )
    args = parser.parse_args()
    return args.config


def _parse_convert_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config. Defaults to PLUGRAG_CONFIG or config_example.yaml.",
    )
    parser.add_argument(
        "--encoder",
        default="both",
        choices=["rgcn", "gcn", "multi_edge", "both", "all"],
        help="Which converted training data to prepare.",
    )
    return parser.parse_args()


def main_train_rgcn() -> None:
    config = load_config(_parse_config_arg())
    train_rgcn_from_config(config)
    print("Done.")


def main_train_gcn() -> None:
    config = load_config(_parse_config_arg())
    train_gcn_from_config(config)
    print("Done.")


def main_train_multi_edge() -> None:
    config = load_config(_parse_config_arg())
    train_multi_edge_from_config(config)
    print("Done.")


def main_convert_training_data() -> None:
    args = _parse_convert_args()
    config = load_config(args.config)
    convert_training_data_from_config(config, args.encoder)
    print("Done.")

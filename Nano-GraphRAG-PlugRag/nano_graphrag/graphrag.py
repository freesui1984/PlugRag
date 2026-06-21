import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast
import json
import torch


from ._llm import (
    amazon_bedrock_embedding,
    create_amazon_bedrock_complete_function,
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete,
)
from ._op import (
    chunking_by_token_size,
    extract_entities,
    generate_community_report,
    get_chunks,
    local_query,
    global_query,
    naive_query,
    local_rgcn_query,
    local_rank_query,
    local_rgcn_vote_query,
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from .rgcn_utils import (
    convert_to_gcn_data,
    convert_to_multi_edge_rgcn_data,
    convert_to_rgcn_data,
    train_multi_edge_rgcn_transe,
    train_hybrid,
    train_hybrid_gcn,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
    TokenizerWrapper,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)


@dataclass
class GraphRAG:
    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = False

    # text chunking
    tokenizer_type: str = "tiktoken"  # or 'huggingface'
    tiktoken_model_name: str = "gpt-4o"
    huggingface_model_name: str = "bert-base-uncased"  # default HF model
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            TokenizerWrapper,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # graph clustering
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "num_walks": 10,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # community reports
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    # text embedding
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    query_better_than_threshold: float = 0.2

    # LLM
    using_azure_openai: bool = False
    using_amazon_bedrock: bool = False
    best_model_id: str = "us.anthropic.claude-3-sonnet-20240229-v1:0"
    cheap_model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0"
    best_model_func: callable = gpt_4o_complete
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 16
    cheap_model_func: callable = gpt_4o_mini_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16

    # entity extraction
    entity_extraction_func: callable = extract_entities

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # rgcn
    rgcn_enabled: bool = False
    rgcn_data_file: str = "rgcn_data.pt"
    rgcn_embeddings_file: str = "rgcn_embeddings.pt"
    rgcn_mapping_file: str = "rgcn_embeddings_mapping.json"
    gcn_data_file: str = "gcn_data.pt"
    gcn_embeddings_file: str = "gcn_embeddings.pt"
    gcn_mapping_file: str = "gcn_embeddings_mapping.json"
    multi_edge_data_file: str = "multi_edge/multi_edge_rgcn_data.pt"
    multi_edge_embeddings_file: str = "multi_edge/rgcn_transe_embeddings.pt"
    multi_edge_mapping_file: str = "multi_edge/rgcn_transe_mapping.json"
    rgcn_rerank_alpha: float = 0.5
    rgcn_expand_k: int = 20
    rgcn_cluster_limit: int = 200

    # rerank
    rerank_api_key: str = None
    rerank_base_url: str = None
    rerank_model_name: str = None

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

        self.tokenizer_wrapper = TokenizerWrapper(
            tokenizer_type=self.tokenizer_type,
            model_name=self.tiktoken_model_name if self.tokenizer_type == "tiktoken" else self.huggingface_model_name
        )

        if self.using_azure_openai:
            # If there's no OpenAI API key, use Azure OpenAI
            if self.best_model_func == gpt_4o_complete:
                self.best_model_func = azure_gpt_4o_complete
            if self.cheap_model_func == gpt_4o_mini_complete:
                self.cheap_model_func = azure_gpt_4o_mini_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info(
                "Switched the default openai funcs to Azure OpenAI if you didn't set any of it"
            )

        if self.using_amazon_bedrock:
            self.best_model_func = create_amazon_bedrock_complete_function(self.best_model_id)
            self.cheap_model_func = create_amazon_bedrock_complete_function(self.cheap_model_id)
            self.embedding_func = amazon_bedrock_embedding
            logger.info(
                "Switched the default openai funcs to Amazon Bedrock"
            )

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )



    def train_rgcn(self, **kwargs):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.atrain_rgcn(**kwargs))

    def train_gcn(self, **kwargs):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.atrain_gcn(**kwargs))

    def train_multi_edge_rgcn_transe(self, **kwargs):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.atrain_multi_edge_rgcn_transe(**kwargs))

    async def atrain_rgcn(self, **kwargs):
        """
        Convert current graph to RGCN format and perform training.
        """
        if not self.rgcn_enabled:
            logger.warning("rgcn_enabled is False, but triggering train_rgcn. Proceeding anyway.")
        
        output_pt_path = os.path.join(self.working_dir, kwargs.get("output_file", self.rgcn_embeddings_file))
        data_pt_path = os.path.join(self.working_dir, kwargs.get("data_file", self.rgcn_data_file))
        mapping_out_path = os.path.join(self.working_dir, kwargs.get("mapping_file", self.rgcn_mapping_file))
        os.makedirs(os.path.dirname(output_pt_path) or self.working_dir, exist_ok=True)
        os.makedirs(os.path.dirname(data_pt_path) or self.working_dir, exist_ok=True)
        os.makedirs(os.path.dirname(mapping_out_path) or self.working_dir, exist_ok=True)
        
        # 1. Convert or load
        if kwargs.get("convert_data", True):
            data, mappings = await convert_to_rgcn_data(
                self.chunk_entity_relation_graph, 
                self.entities_vdb, 
                data_pt_path
            )
        else:
            data = torch.load(data_pt_path, map_location="cpu", weights_only=False)
            mapping_path = mapping_out_path
            if not os.path.exists(mapping_path):
                mapping_path = os.path.splitext(data_pt_path)[0] + "_mapping.json"
            with open(mapping_path, "r", encoding="utf-8") as f:
                mappings = json.load(f)
        
        # 2. Train
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_relations = len(mappings["relation_types"])
        
        logger.info(f"Starting RGCN training on {device}...")
        embeddings = train_hybrid(
            data,
            hidden=kwargs.get("hidden_dim", 512),
            lr=kwargs.get("lr", 1e-4),
            epochs=kwargs.get("epochs", 100),
            device=device,
            output_path=output_pt_path,
            num_bases=kwargs.get("num_bases", min(20, num_relations)),
            num_layers=kwargs.get("num_layers", 3),
            alpha=kwargs.get("alpha", 0.8),
            beta=kwargs.get("beta", 0.2),
            mse_scale=kwargs.get("mse_scale", 500.0),
            early_stopping_patience=kwargs.get("early_stopping", 20),
            seed=kwargs.get("seed", 42)
        )
        
        if embeddings is not None:
            logger.info(f"RGCN training completed. Embeddings saved to {output_pt_path}")
            mapping_src = os.path.splitext(data_pt_path)[0] + "_mapping.json"
            if os.path.exists(mapping_src):
                with open(mapping_src, 'r', encoding='utf-8') as f:
                    mapping_data = json.load(f)
                with open(mapping_out_path, 'w', encoding='utf-8') as f:
                    json.dump(mapping_data, f, ensure_ascii=False, indent=4)
        else:
            logger.error("RGCN training failed or returned no embeddings.")
        
        return embeddings

    async def atrain_multi_edge_rgcn_transe(self, **kwargs):
        """
        Convert the current multi-edge graph to relation-aware R-GCN data and
        train the Multi-edge R-GCN + TransE ablation encoder.
        """
        output_pt_path = os.path.join(
            self.working_dir, kwargs.get("output_file", self.multi_edge_embeddings_file)
        )
        data_pt_path = os.path.join(
            self.working_dir, kwargs.get("data_file", self.multi_edge_data_file)
        )
        mapping_out_path = os.path.join(
            self.working_dir, kwargs.get("mapping_file", self.multi_edge_mapping_file)
        )
        os.makedirs(os.path.dirname(output_pt_path) or self.working_dir, exist_ok=True)
        os.makedirs(os.path.dirname(data_pt_path) or self.working_dir, exist_ok=True)
        os.makedirs(os.path.dirname(mapping_out_path) or self.working_dir, exist_ok=True)

        if kwargs.get("convert_data", True):
            data, mappings = await convert_to_multi_edge_rgcn_data(
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                data_pt_path,
            )
        else:
            data = torch.load(data_pt_path, map_location="cpu", weights_only=False)
            mapping_path = mapping_out_path
            if not os.path.exists(mapping_path):
                mapping_path = os.path.splitext(data_pt_path)[0] + "_mapping.json"
            with open(mapping_path, "r", encoding="utf-8") as f:
                mappings = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_relations = len(mappings.get("relation_types", []))
        logger.info(f"Starting Multi-edge RGCN+TransE training on {device}...")
        embeddings = train_multi_edge_rgcn_transe(
            data,
            hidden=kwargs.get("hidden_dim", 512),
            lr=kwargs.get("lr", 1e-3),
            epochs=kwargs.get("epochs", 100),
            device=device,
            output_path=output_pt_path,
            num_bases=kwargs.get("num_bases", min(20, num_relations) if num_relations else None),
            num_layers=kwargs.get("num_layers", 3),
            alpha=kwargs.get("alpha", 0.8),
            beta=kwargs.get("beta", 0.2),
            margin=kwargs.get("margin", 1.0),
            mse_scale=kwargs.get("mse_scale", 50.0),
            early_stopping_patience=kwargs.get("early_stopping", 30),
            seed=kwargs.get("seed", 42),
        )

        if embeddings is not None:
            logger.info(f"Multi-edge RGCN+TransE training completed. Embeddings saved to {output_pt_path}")
            mapping_src = os.path.splitext(data_pt_path)[0] + "_mapping.json"
            if os.path.exists(mapping_src):
                with open(mapping_src, "r", encoding="utf-8") as f:
                    mapping_data = json.load(f)
                with open(mapping_out_path, "w", encoding="utf-8") as f:
                    json.dump(mapping_data, f, ensure_ascii=False, indent=4)
        else:
            logger.error("Multi-edge RGCN+TransE training failed or returned no embeddings.")

        return embeddings

    async def atrain_gcn(self, **kwargs):
        """
        Convert current graph to homogeneous GCN format and train the Simple GCN ablation.
        """
        output_pt_path = os.path.join(self.working_dir, kwargs.get("output_file", self.gcn_embeddings_file))
        data_pt_path = os.path.join(self.working_dir, kwargs.get("data_file", self.gcn_data_file))
        mapping_out_path = os.path.join(self.working_dir, kwargs.get("mapping_file", self.gcn_mapping_file))
        os.makedirs(os.path.dirname(output_pt_path) or self.working_dir, exist_ok=True)
        os.makedirs(os.path.dirname(data_pt_path) or self.working_dir, exist_ok=True)
        os.makedirs(os.path.dirname(mapping_out_path) or self.working_dir, exist_ok=True)

        if kwargs.get("convert_data", True):
            data, _mappings = await convert_to_gcn_data(
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                data_pt_path,
            )
        else:
            data = torch.load(data_pt_path, map_location="cpu", weights_only=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Starting Simple GCN ablation training on {device}...")
        embeddings = train_hybrid_gcn(
            data,
            hidden=kwargs.get("hidden_dim", 512),
            lr=kwargs.get("lr", 1e-3),
            epochs=kwargs.get("epochs", 100),
            device=device,
            output_path=output_pt_path,
            num_layers=kwargs.get("num_layers", 3),
            alpha=kwargs.get("alpha", 0.8),
            beta=kwargs.get("beta", 0.2),
            mse_scale=kwargs.get("mse_scale", 50.0),
            early_stopping_patience=kwargs.get("early_stopping", 20),
            seed=kwargs.get("seed", 42),
        )

        if embeddings is not None:
            logger.info(f"Simple GCN training completed. Embeddings saved to {output_pt_path}")
            mapping_src = os.path.splitext(data_pt_path)[0] + "_mapping.json"
            if not os.path.exists(mapping_src) and os.path.exists(mapping_out_path):
                mapping_src = mapping_out_path
            if os.path.exists(mapping_src):
                with open(mapping_src, "r", encoding="utf-8") as f:
                    mapping_data = json.load(f)
                with open(mapping_out_path, "w", encoding="utf-8") as f:
                    json.dump(mapping_data, f, ensure_ascii=False, indent=4)
        else:
            logger.error("Simple GCN training failed or returned no embeddings.")

        return embeddings

    # def bootstrap_from_existing_graph(self, rebuild_reports: bool = True):
    #     loop = always_get_an_event_loop()
    #     return loop.run_until_complete(self.abootstrap_from_existing_graph(rebuild_reports))

    # async def abootstrap_from_existing_graph(self, rebuild_reports: bool = True):
    #     await self._insert_start()
    #     try:
    #         return await self._abootstrap_from_existing_graph(rebuild_reports)
    #     finally:
    #         await self._insert_done()

    # async def _abootstrap_from_existing_graph(self, rebuild_reports: bool = True):
    #     all_nodes = list(self.chunk_entity_relation_graph._graph.nodes(data=True))
    #     if len(all_nodes) == 0:
    #         logger.error(
    #             f"No nodes found in graph {self.chunk_entity_relation_graph._graphml_xml_file}. "
    #             f"Please ensure the GraphML file exists in the correct directory."
    #         )
    #         return False
        
    #     # Check if entities are already indexed
    #     vdb_nodes_count = await self.entities_vdb.get_counts() if self.entities_vdb is not None else 0
        
    #     # If the VDB already has substantial data, DO NOT re-index everything
    #     if vdb_nodes_count >= (len(all_nodes) * 0.9): 
    #         logger.info(f"Entities VDB already has {vdb_nodes_count} nodes. Skipping re-indexing.")
    #     else:
    #         logger.info(f"VDB nodes ({vdb_nodes_count}) < Graph nodes ({len(all_nodes)}). Indexing missing nodes...")
    #         data_for_vdb = {}
    #         for node_id, node_data in all_nodes:
    #             content = f"{node_id}{node_data.get('description', '')}"
    #             data_for_vdb[compute_mdhash_id(node_id, prefix="ent-")] = {
    #                 "content": content,
    #                 "entity_name": node_id,
    #             }
    #         if self.entities_vdb is not None and len(data_for_vdb):
    #             await self.entities_vdb.upsert(data_for_vdb)

    #     if rebuild_reports:
    #         # FORCE RE-CLUSTERING IF REPORTS ARE EMPTY
    #         # report_counts = await self.community_reports.get_counts()
    #         if report_counts > 0:
    #             logger.info(f"Community reports already exist ({report_counts}). Skipping report generation.")
    #         else:
    #             logger.warning("Community reports missing or corrupted. Re-generating from graph...")
    #             # Important: we must perform clustering before generating reports
    #             await self.chunk_entity_relation_graph.clustering(self.graph_cluster_algorithm)
    #             await generate_community_report(
    #                 self.community_reports, self.chunk_entity_relation_graph, self.tokenizer_wrapper, asdict(self)
    #             )
    #     return True

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        mode_aliases = {
            "plugrag_full": "rgcn",
            "plugrag_vote_only": "vote_rgcn",
            "plugrag_rank_only": "rank",
        }
        mode = mode_aliases.get(param.mode, param.mode)

        if mode == "local" and not self.enable_local:
            raise ValueError("enable_local is False, cannot query in local mode")
        if mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")

        # Load RGCN if enabled
        rgcn_embeddings = None
        rgcn_mapping = None
        if param.rgcn_mode and self.rgcn_enabled:
            emb_path = os.path.join(self.working_dir, self.rgcn_embeddings_file)
            map_path = os.path.join(self.working_dir, self.rgcn_mapping_file)
            if os.path.exists(emb_path) and os.path.exists(map_path):
                import torch
                logger.info(f"Loading RGCN embeddings from {emb_path}")
                rgcn_data = torch.load(emb_path, map_location="cpu")
                rgcn_embeddings = rgcn_data["embeddings"]
                with open(map_path, "r", encoding="utf-8") as f:
                    rgcn_mapping = json.load(f)
            else:
                logger.warning("RGCN mode enabled but embeddings/mapping not found. Run train_rgcn() first.")

        if mode == "rgcn":
            response = await local_rgcn_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                asdict(self),
                rgcn_embeddings=rgcn_embeddings,
                rgcn_mapping=rgcn_mapping,
                rerank_api_key=self.rerank_api_key,
                rerank_base_url=self.rerank_base_url,
                rerank_model=self.rerank_model_name,
            )
        elif mode == "vote_rgcn":
            response = await local_rgcn_vote_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                asdict(self),
                rgcn_embeddings=rgcn_embeddings,
                rgcn_mapping=rgcn_mapping,
                rerank_api_key=self.rerank_api_key,
                rerank_base_url=self.rerank_base_url,
                rerank_model=self.rerank_model_name,
            )
        elif mode == "rank":
            response = await local_rank_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                asdict(self),
                rgcn_embeddings=rgcn_embeddings,
                rgcn_mapping=rgcn_mapping,
                rerank_api_key=self.rerank_api_key,
                rerank_base_url=self.rerank_base_url,
                rerank_model=self.rerank_model_name,
            )
        elif mode == "local":
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                asdict(self),
            )
        elif mode == "global":
            response = await global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                asdict(self),
            )
        elif mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def ainsert(self, string_or_strings):
        await self._insert_start()
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            # ---------- new docs
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning(f"All docs are already in the storage")
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # ---------- chunking

            inserting_chunks = get_chunks(
                new_docs=new_docs,
                chunk_func=self.chunk_func,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size,
                tokenizer_wrapper=self.tokenizer_wrapper,
            )

            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # TODO: don't support incremental update for communities now, so we have to drop all
            await self.community_reports.drop()

            # ---------- extract/summary entity and upsert to graph
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await self.entity_extraction_func(
                inserting_chunks,
                knwoledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                tokenizer_wrapper=self.tokenizer_wrapper,
                global_config=asdict(self),
                using_amazon_bedrock=self.using_amazon_bedrock,
            )
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg
            # ---------- update clusterings of graph
            logger.info("[Community Report]...")
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm
            )
            await generate_community_report(
                self.community_reports, self.chunk_entity_relation_graph, self.tokenizer_wrapper, asdict(self)
            )

            # ---------- commit upsertings and indexing
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

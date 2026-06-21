import html
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union, cast
import networkx as nx
import numpy as np

from .._utils import logger
from ..base import (
    BaseGraphStorage,
    SingleCommunitySchema,
)
from ..prompt import GRAPH_FIELD_SEP


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        # self._graph = preloaded_graph or nx.Graph()
        self._graph = preloaded_graph or nx.MultiGraph()
        self._clustering_algorithms = {
            "leiden": self._leiden_clustering,
        }
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    # async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
    #     return self._graph.has_edge(source_node_id, target_node_id)
    async def has_edge(self, source_node_id: str, target_node_id: str, key: str | None = None) -> bool:
        """检查是否存在特定的边（由key区分）。"""
        return self._graph.has_edge(source_node_id, target_node_id, key=key)
    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        # [numberchiffre]: node_id not part of graph returns `DegreeView({})` instead of 0
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0) + (
            self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        )

    # async def get_edge(
    #     self, source_node_id: str, target_node_id: str
    # ) -> Union[dict, None]:
    #     return self._graph.edges.get((source_node_id, target_node_id))
    async def get_edge(
            self, source_node_id: str, target_node_id: str, key: str | None = None
        ) -> dict | None:
            """
            获取特定边的数据。
            注意: networkx的 get_edge_data 才是官方推荐的用法。
            这个方法主要是为了兼容性。
            """
            # MultiGraph.get_edge_data 支持 key 参数
            return self._graph.get_edge_data(source_node_id, target_node_id, key=key)
    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None
 # VVVVVV PASTE THE NEW METHOD HERE VVVVVV
    # async def get_edge_data(self, source_node_id: str, target_node_id: str) -> dict | None:
    #     """
    #     Gets data for edges between two nodes.
    #     For a MultiGraph, this returns a dictionary of all parallel edges.
    #     """
    #     return self._graph.get_edge_data(source_node_id, target_node_id)
    async def get_edge_data(self, source_node_id: str, target_node_id: str, key: str | None = None) -> dict | None:
        """
        获取边的属性数据，支持通过key查找特定平行边。
        这是解决您问题的核心修复。
        """
        return self._graph.get_edge_data(source_node_id, target_node_id, key=key)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    # async def upsert_node(self, node_id: str, node_data: dict[str, str]):
    #     self._graph.add_node(node_id, **node_data)

    # async def upsert_edge(
    #     self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    # ):
    #     self._graph.add_edge(source_node_id, target_node_id, **edge_data)
    async def upsert_node(self, node_id: str, node_data: dict):
        self._graph.add_node(
            str(node_id),  # 确保 node_id 是字符串
            **{k: self._make_graphml_serializable(v) for k, v in node_data.items()}
        )
    @staticmethod
    def _make_graphml_serializable(val: any) -> any:
        """
        将值转换为 GraphML (networkx) 可接受的格式:
        - 保持 str/bool/int/float
        - 将 numpy 标量转换为原生类型
        - 将 dict/list/tuple/set/ndarray 转换为 JSON 字符串
        - 其他情况回退到 str(val)
        """
        # 基础 Python 类型
        if isinstance(val, (str, bool, int, float)):
            return val
        
        # NumPy 标量类型
        if hasattr(val, 'item') and callable(val.item):
             # 这是一个更通用的检查 numpy 类型的方法
             return val.item()

        # 集合类型 -> JSON 字符串
        if isinstance(val, (dict, list, tuple, set)):
            try:
                # 对于字典，递归地清理其值
                if isinstance(val, dict):
                    cleaned_val = {
                        # 确保这里调用的是 NetworkXStorage._make_graphml_serializable
                        k: NetworkXStorage._make_graphml_serializable(v) for k, v in val.items()
                    }
                    return json.dumps(cleaned_val, ensure_ascii=False)
                # 对于列表等，递归地清理其元素
                else:
                    cleaned_val = [NetworkXStorage._make_graphml_serializable(v) for v in val]
                    return json.dumps(cleaned_val, ensure_ascii=False)
            except Exception:
                # 如果JSON序列化失败，回退到字符串
                return str(val)

        # 针对 ndarray 的特殊处理
        if "ndarray" in str(type(val)):
             return json.dumps(val.tolist(), ensure_ascii=False)

        # 回退到字符串
        return str(val)
    # async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict, key: str = None):
    #     """
    #     Upserts an edge to the graph. For MultiGraph, uses a key to distinguish parallel edges.
    #     """
    #     # Create a copy to avoid modifying the original dict, and remove the key from the attributes
    #     attributes_to_save = edge_data.copy()
    #     attributes_to_save.pop("key", None)
        
    #     self._graph.add_edge(
    #         str(source_node_id), 
    #         str(target_node_id), 
    #         key=key,  # <-- This key is essential for creating/updating the correct parallel edge
    #         **{k: self._make_graphml_serializable(v) for k, v in attributes_to_save.items()}
    #     )
    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict, key: str | None = None):
        """
        插入或更新一条边。
        在MultiGraph中，(u, v, key) 共同定义了一条唯一的边。
        如果 (u, v, key) 已存在，add_edge会自动更新其属性，实现upsert功能。
        """
        # 确保ID是字符串，并对属性进行序列化
        self._graph.add_edge(
            str(source_node_id), 
            str(target_node_id), 
            key=key, 
            **{k: self._make_graphml_serializable(v) for k, v in edge_data.items()}
        )

    async def remove_edge(self, source_node_id: str, target_node_id: str, key: str | None = None):
        """删除指定的边，支持MultiGraph的key参数"""
        try:
            if key is not None:
                # For MultiGraph with specific key
                self._graph.remove_edge(str(source_node_id), str(target_node_id), key=key)
            else:
                # For simple edge or all edges between nodes
                self._graph.remove_edge(str(source_node_id), str(target_node_id))
        except nx.NetworkXError:
            # Edge doesn't exist, ignore
            pass

    async def clustering(self, algorithm: str):
        if algorithm not in self._clustering_algorithms:
            raise ValueError(f"Clustering algorithm {algorithm} not supported")
        await self._clustering_algorithms[algorithm]()

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                occurrence=0.0,
                sub_communities=[],
            )
        )
        max_num_ids = 0
        levels = defaultdict(set)
        for node_id, node_data in self._graph.nodes(data=True):
            if "clusters" not in node_data:
                continue
            clusters = json.loads(node_data["clusters"])
            this_node_edges = self._graph.edges(node_id)

            for cluster in clusters:
                level = cluster["level"]
                cluster_key = str(cluster["cluster"])
                levels[level].add(cluster_key)
                results[cluster_key]["level"] = level
                results[cluster_key]["title"] = f"Cluster {cluster_key}"
                results[cluster_key]["nodes"].add(node_id)
                results[cluster_key]["edges"].update(
                    [tuple(sorted(e)) for e in this_node_edges]
                )
                source_id_str = node_data.get("source_id", "")
                results[cluster_key]["chunk_ids"].update(
                    source_id_str.split(GRAPH_FIELD_SEP)
                )   
                max_num_ids = max(max_num_ids, len(results[cluster_key]["chunk_ids"]))

        ordered_levels = sorted(levels.keys())
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]
            # compute the sub-communities by nodes intersection
            for comm in this_level_comms:
                results[comm]["sub_communities"] = [
                    c
                    for c in next_level_comms
                    if results[c]["nodes"].issubset(results[comm]["nodes"])
                ]

        for k, v in results.items():
            v["edges"] = list(v["edges"])
            v["edges"] = [list(e) for e in v["edges"]]
            v["nodes"] = list(v["nodes"])
            v["chunk_ids"] = list(v["chunk_ids"])
            v["occurrence"] = len(v["chunk_ids"]) / max_num_ids
        return dict(results)

    def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):
        for node_id, clusters in cluster_data.items():
            self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)

    async def _leiden_clustering(self):
        from graspologic.partition import hierarchical_leiden
        """
        It uses the hierarchical_leiden function from the graspologic library
        The Leiden algorithm is used in the HiRAG.ainsert method
        """
        graph = NetworkXStorage.stable_largest_connected_component(self._graph)
        community_mapping = hierarchical_leiden(
            graph,
            max_cluster_size=self.global_config["max_graph_cluster_size"],
            random_seed=self.global_config["graph_cluster_seed"],
        )

        node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
        __levels = defaultdict(set)
        for partition in community_mapping:
            level_key = partition.level
            cluster_id = partition.cluster
            node_communities[partition.node].append(
                {"level": level_key, "cluster": cluster_id}
            )
            __levels[level_key].add(cluster_id)
        node_communities = dict(node_communities)
        __levels = {k: len(v) for k, v in __levels.items()}
        logger.info(f"Each level has communities: {dict(__levels)}")
        self._cluster_data_to_subgraphs(node_communities)

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids

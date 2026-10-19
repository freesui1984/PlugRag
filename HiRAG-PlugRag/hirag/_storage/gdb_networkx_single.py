"""Relation-agnostic NetworkX storage for the isolated HiRAG Single line.

This storage intentionally uses ``nx.Graph``.  It rejects MultiGraph edge keys
and relation-type attributes so a relation-aware extraction path cannot be
silently written into a Single workspace.
"""

import os
from dataclasses import dataclass

import networkx as nx

from .._utils import logger
from .gdb_networkx import NetworkXStorage


@dataclass
class NetworkXSingleStorage(NetworkXStorage):
    """A simple undirected graph storage for relation-agnostic HiRAG runs."""

    @staticmethod
    def load_nx_graph(file_name: str) -> nx.Graph | None:
        if not os.path.exists(file_name):
            return None

        graph = nx.read_graphml(file_name)
        if graph.is_multigraph():
            raise ValueError(
                f"Single graph storage cannot load a MultiGraph: {file_name}. "
                "Use the explicit Multi-to-Single preparation script first."
            )
        return nx.Graph(graph)

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = self.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                "Loaded single graph from %s with %s nodes, %s edges",
                self._graphml_xml_file,
                preloaded_graph.number_of_nodes(),
                preloaded_graph.number_of_edges(),
            )
        self._graph = preloaded_graph or nx.Graph()
        self._clustering_algorithms = {"leiden": self._leiden_clustering}
        self._node_embed_algorithms = {"node2vec": self._node2vec_embed}

    async def has_edge(
        self, source_node_id: str, target_node_id: str, key: str | None = None
    ) -> bool:
        return key is None and self._graph.has_edge(source_node_id, target_node_id)

    async def get_edge(
        self, source_node_id: str, target_node_id: str, key: str | None = None
    ) -> dict | None:
        if key is not None:
            return None
        return self._graph.get_edge_data(source_node_id, target_node_id)

    async def get_edge_data(
        self, source_node_id: str, target_node_id: str, key: str | None = None
    ) -> dict | None:
        return await self.get_edge(source_node_id, target_node_id, key)

    async def upsert_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_data: dict,
        key: str | None = None,
    ):
        if key is not None:
            raise ValueError("Single graph storage does not accept MultiGraph edge keys.")
        if edge_data.get("relation_type") not in (None, ""):
            raise ValueError(
                "Single graph storage does not accept relation_type. "
                "Use the relation-agnostic edge merge before writing."
            )

        self._graph.add_edge(
            str(source_node_id),
            str(target_node_id),
            **{
                attribute: self._make_graphml_serializable(value)
                for attribute, value in edge_data.items()
            },
        )

    async def remove_edge(
        self, source_node_id: str, target_node_id: str, key: str | None = None
    ):
        if key is not None:
            raise ValueError("Single graph storage does not accept MultiGraph edge keys.")
        try:
            self._graph.remove_edge(str(source_node_id), str(target_node_id))
        except nx.NetworkXError:
            pass

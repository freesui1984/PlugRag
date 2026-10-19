import asyncio
from unittest.mock import AsyncMock

import networkx as nx

from hirag import pathrag_evidence_aligned, ppr_top3
from hirag.base import QueryParam


class GraphStore:
    def __init__(self):
        self._graph = nx.Graph()
        self._graph.add_edge("seed_a", "bridge", weight=2, description="a to bridge")
        self._graph.add_edge("bridge", "seed_b", weight=1, description="bridge to b")
        self._graph.add_edge("seed_b", "seed_c", weight=1, description="b to c")
        self.global_config = {}

    async def get_node(self, name):
        return {"entity_type": "TEST", "description": name, "source_id": name}

    async def node_degree(self, name):
        return self._graph.degree(name)


class VectorStore:
    async def query(self, _query, top_k):
        return [
            {"entity_name": "seed_a", "score": 0.9},
            {"entity_name": "seed_b", "score": 0.8},
            {"entity_name": "seed_c", "score": 0.7},
        ][:top_k]


def test_ppr_top3_builds_context_without_full_retriever(monkeypatch):
    monkeypatch.setattr(ppr_top3, "_find_most_related_community_from_entities", AsyncMock(return_value=[]))
    monkeypatch.setattr(ppr_top3, "_find_most_related_text_unit_from_entities", AsyncMock(return_value=[]))
    monkeypatch.setattr(ppr_top3, "_find_ppr_related_edges_from_entities", AsyncMock(return_value=[]))

    context = asyncio.run(
        ppr_top3._build_ppr_top3_query_context(
            "question", GraphStore(), VectorStore(), None, None, QueryParam(top_k=4), {}
        )
    )

    assert context is not None
    assert "seed_a" in context and "seed_b" in context and "seed_c" in context
    assert "bridge" in context
    assert "ppr_propagation" in context


def test_aligned_pathrag_builds_context_without_full_retriever(monkeypatch):
    monkeypatch.setattr(pathrag_evidence_aligned, "_communities_from_entities", AsyncMock(return_value=[]))
    monkeypatch.setattr(pathrag_evidence_aligned, "_sources_from_entities", AsyncMock(return_value=[]))

    context = asyncio.run(
        pathrag_evidence_aligned.build_pathrag_evidence_aligned_context(
            "question", GraphStore(), VectorStore(), None, None, QueryParam(), {}
        )
    )

    assert context is not None
    assert "-----Reasoning Path-----" in context
    assert "seed_a" in context and "seed_b" in context


def test_aligned_pathrag_accepts_parallel_relations(monkeypatch):
    monkeypatch.setattr(pathrag_evidence_aligned, "_communities_from_entities", AsyncMock(return_value=[]))
    monkeypatch.setattr(pathrag_evidence_aligned, "_sources_from_entities", AsyncMock(return_value=[]))
    store = GraphStore()
    store._graph = nx.MultiGraph(store._graph)
    store._graph.add_edge("seed_a", "bridge", weight=3, description="second relation")

    context = asyncio.run(
        pathrag_evidence_aligned.build_pathrag_evidence_aligned_context(
            "question", store, VectorStore(), None, None, QueryParam(), {}
        )
    )

    assert context is not None
    assert "seed_a" in context and "bridge" in context

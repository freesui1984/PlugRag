import asyncio
import json

import networkx as nx
import pytest

from hirag._op import (
    _build_hierarchical_query_context,
    build_single_graph_from_multi_graph,
)
from hirag._storage import NetworkXSingleStorage
from hirag.base import QueryParam
from hirag.prompt import GRAPH_FIELD_SEP


def test_multi_graph_projection_merges_parallel_edges_and_strips_relation_type():
    graph = nx.MultiGraph()
    graph.add_node("ALICE", description="Alice")
    graph.add_node("BOB", description="Bob")
    graph.add_edge(
        "ALICE",
        "BOB",
        key="KNOWS",
        weight=2.0,
        description="Alice knows Bob",
        source_id=f"chunk-a{GRAPH_FIELD_SEP}chunk-b",
        order=2,
        relation_type="KNOWS",
    )
    graph.add_edge(
        "ALICE",
        "BOB",
        key="WORKS_WITH",
        weight="3",
        description="Alice works with Bob",
        source_id="chunk-b",
        order=1,
        relation_type="WORKS_WITH",
    )

    single_graph, audit_rows = build_single_graph_from_multi_graph(graph)

    assert isinstance(single_graph, nx.Graph)
    assert not single_graph.is_multigraph()
    assert single_graph.number_of_edges() == 1
    attributes = single_graph["ALICE"]["BOB"]
    assert attributes["weight"] == 5.0
    assert attributes["order"] == 1
    assert "relation_type" not in attributes
    assert set(attributes["source_id"].split(GRAPH_FIELD_SEP)) == {"chunk-a", "chunk-b"}
    assert audit_rows == [
        {
            "source": "ALICE",
            "target": "BOB",
            "multi_edge_count": 2,
            "multi_edge_keys": ["KNOWS", "WORKS_WITH"],
            "relation_types_removed": ["KNOWS", "WORKS_WITH"],
            "multi_total_weight": 5.0,
            "single_weight": 5.0,
            "source_chunk_count": 2,
        }
    ]


def test_single_storage_rejects_relation_aware_writes(tmp_path):
    storage = NetworkXSingleStorage(
        namespace="chunk_entity_relation",
        global_config={"working_dir": str(tmp_path)},
    )

    with pytest.raises(ValueError, match="relation_type"):
        asyncio.run(
            storage.upsert_edge(
                "ALICE", "BOB", {"weight": 1.0, "relation_type": "KNOWS"}
            )
        )

    asyncio.run(storage.upsert_edge("ALICE", "BOB", {"weight": 1.0}))
    assert asyncio.run(storage.has_edge("ALICE", "BOB"))
    assert not asyncio.run(storage.has_edge("ALICE", "BOB", key="KNOWS"))


class _FakeVectorStore:
    async def query(self, _query, top_k):
        return [{"entity_name": "ALICE"}, {"entity_name": "BOB"}][:top_k]


class _FakeKVStore:
    def __init__(self, records):
        self.records = records

    async def get_by_id(self, key):
        return self.records.get(key)


def test_native_hi_context_omits_relation_type_for_single_graph(tmp_path):
    storage = NetworkXSingleStorage(
        namespace="chunk_entity_relation", global_config={"working_dir": str(tmp_path)}
    )
    clusters = json.dumps([{"level": 0, "cluster": 1}])
    node_data = {
        "entity_type": "PERSON",
        "description": "A test entity.",
        "source_id": "chunk-1",
        "clusters": clusters,
    }
    asyncio.run(storage.upsert_node("ALICE", node_data))
    asyncio.run(storage.upsert_node("BOB", node_data))
    asyncio.run(
        storage.upsert_edge(
            "ALICE",
            "BOB",
            {
                "description": "Alice knows Bob.",
                "source_id": "chunk-1",
                "weight": 1.0,
                "order": 1,
            },
        )
    )
    reports = _FakeKVStore(
        {
            "1": {
                "report_string": "Alice and Bob.",
                "report_json": {"rating": 1},
                "level": 0,
                "title": "Cluster 1",
                "nodes": ["ALICE", "BOB"],
            }
        }
    )
    chunks = _FakeKVStore(
        {
            "chunk-1": {
                "content": "Alice knows Bob.",
                "full_doc_id": "doc-1",
                "chunk_order_index": 0,
            }
        }
    )

    context = asyncio.run(
        _build_hierarchical_query_context(
            "Who knows Bob?",
            storage,
            _FakeVectorStore(),
            reports,
            chunks,
            QueryParam(top_k=2),
        )
    )

    assert "relation_type" not in context
    assert '"id",\t"source",\t"target",\t"description",\t"weight",\t"rank"' in context

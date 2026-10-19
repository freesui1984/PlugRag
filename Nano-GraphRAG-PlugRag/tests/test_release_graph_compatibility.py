import asyncio

import networkx as nx
import pytest

from nano_graphrag._storage.gdb_networkx import NetworkXStorage


@pytest.mark.parametrize('graph_type',[nx.Graph,nx.MultiGraph])
def test_storage_reads_and_writes_both_graph_types(graph_type):
    store = NetworkXStorage.__new__(NetworkXStorage)
    store._graph = graph_type()
    key = 'RELATED' if store._graph.is_multigraph() else None
    asyncio.run(store.upsert_edge('a','b',{'weight':2,'description':'link'},key=key))
    assert asyncio.run(store.has_edge('a','b',key=key))
    assert asyncio.run(store.get_edge('a','b',key=key))['weight'] == 2
    assert asyncio.run(store.get_edge_data('a','b',key=key))['description'] == 'link'
    edges = asyncio.run(store.get_node_edges('a'))
    assert edges == [('a','b',key)] if key is not None else edges == [('a','b')]

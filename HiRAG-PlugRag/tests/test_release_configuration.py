import asyncio
import json
from dataclasses import fields
from unittest.mock import AsyncMock, patch

from hirag import HiRAG
from hirag.base import QueryParam
from hirag._op import _find_most_related_community_from_entities


def test_paper_configuration_fields_are_exposed():
    names = {f.name for f in fields(HiRAG)}
    assert {'ppr_graph_retrieval','pathrag_evidence_aligned'} <= names
    assert QueryParam().max_community_reports is None
    assert callable(HiRAG.single_graph)


def test_optional_community_limit_is_applied_after_ranking():
    nodes = [{'clusters':json.dumps([{'cluster':1,'level':0},{'cluster':2,'level':0}])}]
    store = type('Store',(),{'get_by_id':AsyncMock(side_effect=[
        {'report_json':{'rating':1},'report_string':'low'},
        {'report_json':{'rating':9},'report_string':'high'}])})()
    with patch('hirag._op.truncate_list_by_token_size',side_effect=lambda data,**kwargs:data):
        reports = asyncio.run(_find_most_related_community_from_entities(nodes,QueryParam(max_community_reports=1),store))
    assert [r['report_string'] for r in reports] == ['high']

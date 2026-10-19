# Controlled retrieval baselines: review candidate

This first-stage release adds two retrieval-strategy adaptations inside HiRAG.
It is not a complete reproduction of HippoRAG or PathRAG. Historical answers and
evaluation artifacts will be reviewed and released separately; they are not part
of this commit. This code has not been certified to regenerate historical runs.

## Entry points

| QueryParam.mode | Implementation | Query function |
| --- | --- | --- |
| hi_ppr_top3 | hirag/ppr_top3.py | ppr_graph_top3_query |
| pathrag_evidence_aligned | hirag/pathrag_evidence_aligned.py | pathrag_evidence_aligned_query |

HiRAG.aquery in hirag/hirag.py dispatches these modes directly to separate
modules, rather than placing the new query functions in hirag/_op.py.
pathrag_evidence.py supplies helpers used by the aligned PathRAG implementation.
Both modes reuse the existing graph/index and HiRAG answer-generation prompt.
Neither uses PlugRag R-GCN expansion or its cross-encoder reranker.

PPR preserves three valid semantic anchors, uses them as personalized PageRank
seeds, and fills remaining entity positions by propagation score. The aligned
PathRAG strategy propagates resources with decay and threshold pruning, retains
parent traces, and assembles entity, path, community and source evidence.

## Configuration and checks

Use the existing environment installation instructions. Copy config.example.yaml
to an ignored local configuration. Set batch_query.mode to one of the modes above,
batch_query.graph_encoder to none, and hirag.working_dir to an existing compatible
graph/index. Put credentials in environment variables or an ignored local file.

From HiRAG-PlugRag:

```powershell
$env:PLUGRAG_CONFIG = 'config.local.yaml'
python hi_batch_query.py
python -m pytest -q tests
```

Generation needs model endpoints and may incur charges. Release verification
uses mocked/offline tests and does not call those endpoints.

## Known review items

- The inherited vector-hit scorer interprets distance as smaller-is-better,
  while the NanoVectorDB adapter supplies cosine similarity in that field.
  This can reverse seed weights in both baselines. It is preserved for review;
  a correction would change retrieval outputs and require separate validation.
- PPR undirected multigraph relation deduplication can retain reverse duplicates.
- Aligned PathRAG does not implement the optional max_community_reports count
  cap; the default None leaves token-budget-only behavior in place.
- The optional prepare_single_graph_from_multi.py utility projects the graph
  but intentionally omits community reports. Its output is not query-ready until
  matching clustering/report assets have been regenerated. Missing reports can
  cause the shared community lookup to fail.

The passing focused tests do not establish the absence of these inherited
implementation problems. No algorithm correction or historical data rewrite is
included in this review candidate.

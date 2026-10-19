# PathRAG V3 Experiment Manifest

This manifest collects the final answer and bidirectional evaluation artifacts for the
PathRAG comparison experiments. Each dataset directory contains a
`pathrag_baseline_v3/answers/` subdirectory and a `pathrag_baseline_v3/evaluation/`
subdirectory.

| Dataset | Retrieval variant | Answer rows | Evaluation rows per comparison | Overall vs Full | Overall vs HiRAG |
| --- | --- | ---: | ---: | ---: | ---: |
| Mix | `pathrag_evidence_aligned_v3` | 130 | 260 | 42.69% vs 57.31% | 46.92% vs 53.08% |
| CS | `pathrag_evidence_aligned_v3` | 100 | 200 | 44.00% vs 56.00% | 45.00% vs 55.00% |
| Legal | `pathrag_evidence_aligned_v3` | 100 | 200 | 33.50% vs 66.50% | 38.50% vs 61.50% |
| Agriculture | `pathrag_evidence_aligned_v3` | 100 | 200 | 39.00% vs 61.00% | 37.50% vs 62.50% |

All percentages are Overall Winner rates after bidirectional Answer 1/Answer 2
position correction. The original run directories remain intact for reproducibility.

All four datasets use the `pathrag_evidence_aligned_v3` naming and the same
bidirectional position-corrected reporting convention.

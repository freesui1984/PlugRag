# Paper-aligned Overall win rates

Rates are percentages for the method to the left of the comparison arrow. Macro and pooled use different denominators.

| Experiment | Mix | CS | Legal | Agriculture | Macro | Pooled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HiRAG/R-GCN/vs-native | 56.15 | 55.00 | 55.50 | 54.00 | 55.16 | 55.23 |
| HiRAG/Simple GCN/vs-native | 46.92 | 49.50 | 51.50 | 49.50 | 49.36 | 49.19 |
| HiRAG/Multi-edge GCN/vs-native | 45.38 | 51.00 | 53.00 | 55.00 | 51.10 | 50.70 |
| Nano-GraphRAG/R-GCN/vs-native | 55.77 | 55.50 | 53.50 | 57.00 | 55.44 | 55.47 |
| Nano-GraphRAG/Simple GCN/vs-native | 53.85 | 51.00 | 57.00 | 51.50 | 53.34 | 53.37 |
| Nano-GraphRAG/Multi-edge GCN/vs-native | 50.38 | 52.00 | 49.00 | 49.00 | 50.10 | 50.12 |
| controlled/0/Nano-GraphRAG → HiRAG | 44.23 | 48.50 | 48.50 | 48.00 | 47.31 | 47.09 |
| controlled/1/Nano-GraphRAG+PlugRag → HiRAG | 52.69 | 52.00 | 51.00 | 57.50 | 53.30 | 53.26 |
| controlled/2/HiRAG+PlugRag → PPR-based | 55.77 | 52.00 | 59.00 | 56.00 | 55.69 | 55.70 |
| controlled/3/HiRAG+PlugRag → PathRAG-style | 57.31 | 56.00 | 66.50 | 61.00 | 60.20 | 60.00 |
| HiRAG/ablation/R-GCN → Multi-edge GCN | 52.31 | 50.50 | 54.00 | 50.50 | 51.83 | 51.86 |
| HiRAG/ablation/R-GCN → Simple GCN | 54.62 | 49.50 | 49.00 | 48.00 | 50.28 | 50.58 |
| Nano-GraphRAG/ablation/R-GCN → Multi-edge GCN | 52.69 | 50.50 | 52.50 | 54.50 | 52.55 | 52.56 |
| Nano-GraphRAG/ablation/R-GCN → Simple GCN | 50.77 | 55.00 | 46.00 | 53.50 | 51.32 | 51.28 |
| HiRAG/ablation/Full→ w/o Trace | 53.08 | 51.00 | 56.00 | 51.50 | 52.89 | 52.91 |
| HiRAG/ablation/w/o Trace → Original | 52.31 | 53.50 | 50.00 | 52.00 | 51.95 | 51.98 |
| Nano-GraphRAG/ablation/Full→ w/o Trace | 48.46 | 52.00 | 51.50 | 55.00 | 51.74 | 51.51 |
| Nano-GraphRAG/ablation/w/o Trace → Original | 52.69 | 56.00 | 53.00 | 51.00 | 53.17 | 53.14 |
| HiRAG/ablation/Full→ Rank | 51.15 | 56.00 | 52.00 | 53.00 | 53.04 | 52.91 |
| HiRAG/ablation/Full→ Vote | 50.38 | 53.50 | 55.00 | 51.00 | 52.47 | 52.33 |
| Nano-GraphRAG/ablation/Full→ Rank | 50.00 | 50.00 | 53.50 | 53.50 | 51.75 | 51.63 |
| Nano-GraphRAG/ablation/Full→ Vote | 48.85 | 58.00 | 55.00 | 53.50 | 53.84 | 53.49 |

The 188 mapped experimental percentages (dataset values, dimension values and reported aggregates) passed the offline audit. Graph-structure statistics were not audited. Some displayed additional aggregates are computed for transparency and are not paper claims.
See `experiments.json` for each evaluation file and comparator direction. See `RELEASE_NOTES.md` for unresolved run/model/input provenance.

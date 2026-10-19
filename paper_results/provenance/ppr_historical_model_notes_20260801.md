# PPR Top3 Experimental Summary

Date: 2026-08-01

## Scope

This document summarizes the current PPR top3 retrieval experiments against PlugRag full and HiRAG.

The main comparable setting is:

- PPR answer generation: DeepSeek-V3 style runs where available.
- Evaluation: existing V3/SenseNova DeepSeek evaluator files.
- Parsing rule: only the explicit `Winner` field is used. Explanations are ignored to avoid false matches of `Answer 1` or `Answer 2`.
- Invalid partial files are excluded from the main table.

## Main Results

| Dataset | Comparison | Rows | Method A | Method B | Overall A | Overall B | Unknown | Result |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| Mix | PPR top3 V3 vs PlugRag full | 260 | PPR top3 | PlugRag full | 44.23% | 55.77% | 0.00% | Full wins after rerun full baseline |
| Mix | PPR top3 V3 vs HiRAG | 260 | PPR top3 | HiRAG | 53.08% | 46.92% | 0.00% | PPR slightly wins |
| CS | PPR top3 V3 vs PlugRag full | 200 | PPR top3 | PlugRag full | 48.00% | 52.00% | 0.00% | Full slightly wins |
| CS | PPR top3 V3 vs HiRAG | 200 | PPR top3 | HiRAG | 46.50% | 53.50% | 0.00% | HiRAG wins |
| Legal | PPR top3 V3 vs PlugRag full | 200 | PPR top3 | PlugRag full | 41.00% | 59.00% | 0.00% | Full clearly wins |
| Legal | PPR top3 V3 vs HiRAG | 200 | PPR top3 | HiRAG | 39.00% | 61.00% | 0.00% | HiRAG clearly wins |
| Agriculture | PPR top3 V3 vs PlugRag full | 200 | PPR top3 | PlugRag full | 44.00% | 56.00% | 0.00% | Full wins |
| Agriculture | PPR top3 V3 vs HiRAG | 200 | PPR top3 | HiRAG | 45.50% | 54.50% | 0.00% | HiRAG wins |

## Metric Details

### Mix: PPR top3 V3 vs PlugRag full

| Metric | PPR top3 | PlugRag full |
| --- | ---: | ---: |
| Comprehensiveness | 46.15% | 53.85% |
| Empowerment | 45.00% | 55.00% |
| Diversity | 45.38% | 54.62% |
| Overall Winner | 44.23% | 55.77% |

### Mix: PPR top3 V3 vs HiRAG

| Metric | PPR top3 | HiRAG |
| --- | ---: | ---: |
| Comprehensiveness | 52.69% | 47.31% |
| Empowerment | 52.31% | 47.69% |
| Diversity | 57.69% | 42.31% |
| Overall Winner | 53.08% | 46.92% |

### CS: PPR top3 V3 vs PlugRag full

| Metric | PPR top3 | PlugRag full |
| --- | ---: | ---: |
| Comprehensiveness | 47.50% | 52.50% |
| Empowerment | 52.00% | 48.00% |
| Diversity | 46.50% | 53.50% |
| Overall Winner | 48.00% | 52.00% |

### CS: PPR top3 V3 vs HiRAG

| Metric | PPR top3 | HiRAG |
| --- | ---: | ---: |
| Comprehensiveness | 47.00% | 53.00% |
| Empowerment | 48.00% | 52.00% |
| Diversity | 50.00% | 50.00% |
| Overall Winner | 46.50% | 53.50% |

### Legal: PPR top3 V3 vs PlugRag full

| Metric | PPR top3 | PlugRag full |
| --- | ---: | ---: |
| Comprehensiveness | 42.00% | 58.00% |
| Empowerment | 41.50% | 58.50% |
| Diversity | 36.00% | 64.00% |
| Overall Winner | 41.00% | 59.00% |

### Legal: PPR top3 V3 vs HiRAG

| Metric | PPR top3 | HiRAG |
| --- | ---: | ---: |
| Comprehensiveness | 41.50% | 58.50% |
| Empowerment | 42.00% | 58.00% |
| Diversity | 36.00% | 64.00% |
| Overall Winner | 39.00% | 61.00% |

### Agriculture: PPR top3 V3 vs PlugRag full

| Metric | PPR top3 | PlugRag full |
| --- | ---: | ---: |
| Comprehensiveness | 45.00% | 55.00% |
| Empowerment | 44.50% | 55.50% |
| Diversity | 44.00% | 56.00% |
| Overall Winner | 44.00% | 56.00% |

### Agriculture: PPR top3 V3 vs HiRAG

| Metric | PPR top3 | HiRAG |
| --- | ---: | ---: |
| Comprehensiveness | 45.00% | 55.00% |
| Empowerment | 47.50% | 52.50% |
| Diversity | 44.50% | 55.50% |
| Overall Winner | 45.50% | 54.50% |

## Extra Check: SenseNova-Generated Legal PPR

A newer Legal PPR answer file was generated with SenseNova DeepSeek and then evaluated with SenseNova DeepSeek.

| Dataset | Comparison | Rows | Method A | Method B | Overall A | Overall B | Unknown | Note |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| Legal | PPR top3 SenseNova-gen vs HiRAG | 200 | PPR top3 | HiRAG | 17.50% | 82.50% | 0.00% | This is not directly comparable to the V3-generation setting. |

Metric details:

| Metric | PPR top3 | HiRAG |
| --- | ---: | ---: |
| Comprehensiveness | 18.00% | 82.00% |
| Empowerment | 20.00% | 80.00% |
| Diversity | 14.50% | 85.50% |
| Overall Winner | 17.50% | 82.50% |

Important correction: an earlier quick parser mistakenly scanned the full evaluation object, including the `Explanation` text, for `Answer 1` and `Answer 2`. The corrected parser reads only the nested `Winner` field. Therefore, the correct result for this SenseNova-generated Legal PPR run is 17.50% vs 82.50%, not the earlier temporary 49.50% vs 50.50%.

## Current Interpretation

PPR top3 is not a stable improvement over PlugRag full. After using the rerun Mix full baseline, PPR is below PlugRag full on Mix, CS, Legal, and Agriculture.

PPR top3 is also not a stable improvement over HiRAG. It slightly wins on Mix, but loses on CS, Legal, and Agriculture.

The strongest conclusion for a paper is therefore not "PPR improves the system". The more defensible use is as a graph-propagation baseline: PPR represents a pure graph-propagation retrieval strategy, while PlugRag full represents the proposed semantic-structural plug-in strategy. Under this framing, PlugRag full is generally stronger than PPR on most datasets, especially Legal and Agriculture.

## Files Used

| Dataset | Comparison | Evaluation file |
| --- | --- | --- |
| Mix | PPR vs full | `evel/datasets/mix/results/mix_full_v3_rerun_vs_ppr_v3_eval_v3_resume_result_openai.jsonl` |
| Mix | PPR vs HiRAG | `evel/datasets/mix/results/mix_ppr_top3_v3_vs_hirag_eval_flash_result_deepseek.jsonl` |
| CS | PPR vs full | `evel/datasets/cs/results/cs_ppr_top3_v3_100_vs_full_eval_sensenova_deepseek_20260731_result_openai.jsonl` |
| CS | PPR vs HiRAG | `evel/datasets/cs/results/cs_ppr_top3_v3_100_vs_hirag_eval_sensenova_deepseek_20260801_result_openai.jsonl` |
| Legal | PPR vs full | `evel/datasets/legal/results/legal_ppr_top3_v3_100_vs_full_eval_v3_20260731_result_openai.jsonl` |
| Legal | PPR vs HiRAG | `evel/datasets/legal/results/legal_ppr_top3_v3_100_vs_hirag_eval_sensenova_deepseek_20260801_result_openai.jsonl` |
| Agriculture | PPR vs full | `evel/datasets/agriculture/results/agriculture_ppr_top3_v3_100_vs_full_eval_official_v3_20260730_result_openai.jsonl` |
| Agriculture | PPR vs HiRAG | `evel/datasets/agriculture/results/agriculture_ppr_top3_v3_100_vs_hirag_eval_sensenova_deepseek_20260801_result_openai.jsonl` |
| Legal | SenseNova-gen PPR vs HiRAG | `evel/datasets/legal/results/legal_ppr_top3_sensenova_answer_vs_hirag_eval_sensenova_deepseek_20260801_result_openai.jsonl` |

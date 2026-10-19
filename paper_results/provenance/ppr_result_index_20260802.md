# PPR 实验结果与文件归档清单

日期：2026-08-02

说明：项目实际使用的评测数据目录名称是 `evel`，因此本清单放在 `HiRAG-PlugRag/evel` 下。本文档是当前 PPR 补充实验的持久化索引，不替代原始 JSONL 文件。

## 1. 当前保留原则

每个数据集保留与最终统计表对应的 PPR 实验结果：

1. `PPR top3 vs PlugRag full`
2. `PPR top3 vs HiRAG`

答案目录保留：

- HiRAG 原始答案；
- PlugRag full 答案；
- PPR top3 答案；
- 原有 PlugRag 消融答案，如 `rank_only`、`simple_gcn`、`multi_edge_gcn`、`rgcn_no_trace`、`vote_only`。

GCN、消融和其他非 PPR 结果没有移动，仍位于各数据集原来的 `results` 目录中。

## 2. 最终 PPR 评测文件

### Mix

结果目录：`evel/datasets/mix/results/`

- `mix_full_v3_rerun_vs_ppr_v3_eval_v3_resume_result_openai.jsonl`
- `mix_ppr_top3_v3_vs_hirag_eval_flash_result_deepseek.jsonl`

答案目录：`evel/datasets/mix/answers/`

- `mix_hirag_ppr_top3_siliconflow_v3.jsonl`
- `mix_hirag_plugrag_full_siliconflow_v3_rerun_20260730.jsonl`
- `mix_hirag_baseline.jsonl`
- 原有 PlugRag 消融答案继续保留。

### CS

结果目录：`evel/datasets/cs/results/`

- `cs_ppr_top3_v3_100_vs_full_eval_sensenova_deepseek_20260731_result_openai.jsonl`
- `cs_ppr_top3_v3_100_vs_hirag_eval_sensenova_deepseek_20260801_result_openai.jsonl`

答案目录：`evel/datasets/cs/answers/`

- `cs_hirag_ppr_top3_siliconflow_v3_100_20260731.jsonl`
- `cs_hirag_plugrag_rgcn_path.jsonl`
- `cs_hirag_baseline.jsonl`
- 原有 PlugRag 消融答案继续保留。

### Legal

结果目录：`evel/datasets/legal/results/`

- `legal_ppr_top3_v3_100_vs_full_eval_v3_20260731_result_openai.jsonl`
- `legal_ppr_top3_v3_100_vs_hirag_eval_sensenova_deepseek_20260801_result_openai.jsonl`

答案目录：`evel/datasets/legal/answers/`

- `legal_hirag_ppr_top3_siliconflow_v3_100_20260731.jsonl`
- `legal_hirag_plugrag_rgcn_path.jsonl`
- `legal_hirag_baseline.jsonl`
- `legal_hirag_plugrag_multi_edge_gcn.jsonl`
- `legal_hirag_plugrag_rank_only.jsonl`
- `legal_hirag_plugrag_rgcn_no_trace.jsonl`
- `legal_hirag_plugrag_simple_gcn.jsonl`
- `legal_hirag_plugrag_vote_only.jsonl`

查询目录保留：`evel/datasets/legal/queries/legal_query.jsonl`

### Agriculture

结果目录：`evel/datasets/agriculture/results/`

- `agriculture_ppr_top3_v3_100_vs_full_eval_official_v3_20260730_result_openai.jsonl`
- `agriculture_ppr_top3_v3_100_vs_hirag_eval_sensenova_deepseek_20260801_result_openai.jsonl`

答案目录：`evel/datasets/agriculture/answers/`

- `agriculture_hirag_ppr_top3_siliconflow_v3_100_20260730.jsonl`
- `agriculture_hirag_plugrag_rgcn_path.jsonl`
- `agriculture_hirag_baseline.jsonl`
- 原有 PlugRag 消融答案继续保留。

## 3. PPR 旧文件位置

旧版、重复版、修补版 PPR 评测结果及 PPR 相关日志统一放在：

`evel/datasets/_trash_ppr_results_20260802/`

目录下按数据集分为：

- `_trash_ppr_results_20260802/mix/`
- `_trash_ppr_results_20260802/cs/`
- `_trash_ppr_results_20260802/legal/`
- `_trash_ppr_results_20260802/agriculture/`

其中包括旧模型评测、`before_schema_repair`、fresh rerun、旧 Full 对比、旧 HiRAG 对比和中断日志。它们没有删除，只是不再作为最终统计文件。

Legal 额外的旧答案和缺失查询文件位于：

`evel/datasets/_trash_legal_answers_queries_20260802/`

- `answers/`：Legal 的 DeepSeek、SenseNova、V3 重跑、`repaired`、`missing_patch` 等旧答案；
- `queries/`：Legal 的 missing query 片段。

## 4. 最终统计结果

评测文件均使用正反答案顺序进行评测，因此：

- Mix 的 260 行代表 130 个问题；
- CS、Legal、Agriculture 的 200 行分别代表 100 个问题；
- 主结果重新统计时只读取嵌套的 `Winner` 字段；
- 主表中的 `Unknown` 均为 0%。

| Dataset | Comparison | Method A | Method B | Overall A | Overall B | Unknown |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Mix | PPR vs PlugRag full | PPR top3 | PlugRag full | 44.23% | 55.77% | 0.00% |
| Mix | PPR vs HiRAG | PPR top3 | HiRAG | 53.08% | 46.92% | 0.00% |
| CS | PPR vs PlugRag full | PPR top3 | PlugRag full | 48.00% | 52.00% | 0.00% |
| CS | PPR vs HiRAG | PPR top3 | HiRAG | 46.50% | 53.50% | 0.00% |
| Legal | PPR vs PlugRag full | PPR top3 | PlugRag full | 41.00% | 59.00% | 0.00% |
| Legal | PPR vs HiRAG | PPR top3 | HiRAG | 39.00% | 61.00% | 0.00% |
| Agriculture | PPR vs PlugRag full | PPR top3 | PlugRag full | 44.00% | 56.00% | 0.00% |
| Agriculture | PPR vs HiRAG | PPR top3 | HiRAG | 45.50% | 54.50% | 0.00% |

## 5. 结果解释

当前数据支持以下结论：

- PlugRag full 在四个数据集上均高于 PPR top3；
- PPR 只有在 Mix 上相对 HiRAG 有小幅优势；
- PPR 在 CS、Legal、Agriculture 上没有稳定超过 HiRAG；
- PPR 应作为图传播检索基线，而不是 PlugRag 的新改进方法；
- PlugRag 的优势应解释为语义实体检索与结构扩展的组合效果，而不是单纯图传播效果。

由于历史结果使用过不同生成/评测模型版本，正式论文中应在统一模型、统一提示词和统一上下文预算下重新确认最终表格。当前清单中的主表仅用于明确文件对应关系和现有结果口径。

## 6. 相关说明文档

- `docs/ppr_v3_results_summary_20260801.md`
- `docs/ppr_answer_analysis_requirement_audit_20260802.md`
- 本文件：`evel/ppr/ppr_results_and_archive_manifest_20260802.md`

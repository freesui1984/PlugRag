# 论文实验与代码发布核对说明

本目录用于核对最新版中文论文的实验数据，尚未宣称历史实验已经能够完整重新生成。
原始答案、评价记录均直接复制，未修改；没有重新调用收费模型接口。
这是供作者审查的本地候选版本；数值通过核对，不代表已确认原有检索实现不存在问题。

## 已完成的核对

- 185 个 JSONL 文件均有来源路径、记录数、文件大小和 SHA-256 校验值；其中新增的 `mix_hirag_plugrag_full_v3_candidate_answers.jsonl` 为 Mix 数据集的 HiRAG + PlugRag Full 补充候选答案（V3 生成），原始文件名为 `1.jsonl`。
- 88 组“数据集—比较实验”与论文对应，188 处百分比（含宏平均、合并值）复算一致。
- 四个数据集的问题顺序一致，答案非空，问题没有重复。
- PPR、PathRAG 对比与更新后的 HiRAG、Nano 主结果分别归档，不与旧批次混用。

从仓库根目录运行：

```powershell
python paper_results/audit.py
python -m unittest discover -s paper_results/tests -v
```

`SUMMARY.md` 是结果总表；`experiments.json` 指明每个论文数值对应的评价文件及答案顺序；
`files.json` 记录原始文件来源和校验值。`raw` 下的文件保留原始相对路径。
仓库原有 `evel` 中仍可能有旧结果，核对新版论文时请以本目录的明确映射为准。

## 上传前仍需注意

1. 按作者于 2026-10-19 的最新更正：答案生成使用 V3；非跨框架评价（主实验、PPR／PathRAG 受控比较及消融）使用 V3；跨框架评价使用 Pro。
   此前整理中的 PPR“Pro 评测”和常规评价“R1”标注已撤回。原始文件名和历史说明保留，不作为当前模型判断依据。
2. Mix 已补入 Full 重跑答案的高度吻合候选 `answers/mix_hirag_plugrag_full_v3_candidate_answers.jsonl`（原始文件名为 `1.jsonl`）：130 题顺序与生成日志一致，最后写入时间相差约 0.128 秒。作者不记得是否改名，因此仍标记为候选，不声称已确定原评价所用的完整答案批次。证据见 `provenance/mix_full_candidate.json`。
3. 判断记录本身没有保存模型名和输入答案哈希，因此不能仅凭文件归档位置证明完整运行溯源。
4. 上游配置中的早停轮数、基矩阵数与论文不同，当前示例配置不作为原实验参数的证据。
5. 原始图谱、索引和训练权重未打包。现阶段适合称为“论文结果核对归档及补充基线代码”，
   不宜称为“全部实验已完成端到端复现”。完整说明见 `RELEASE_NOTES.md`。

运行新增基线时，将 HiRAG 示例配置复制为本地配置，设置 `batch_query.mode` 为
`hi_ppr_top3` 或 `pathrag_evidence_aligned`，并设置 `batch_query.graph_encoder: none`。
模型密钥不应写入公开文件。先阅读英文 README 的环境和运行步骤。

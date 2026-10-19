# Nano vs HiRAG 汇总

本目录集中保存 Mix、CS、Legal 和 Agriculture 四个数据集上的三组对比实验。答案与评测结果分开存放，原始文件仅复制归档，没有移动或删除。

## 目录结构

```text
nano_vs_hirag/
├── answers/
│   ├── mix/
│   ├── cs/
│   ├── legal/
│   └── agriculture/
└── evaluation/
    ├── mix/
    ├── cs/
    ├── legal/
    └── agriculture/
```

按实验作者确认，四个 `Nano Full vs Nano` 结果使用 DeepSeek-R1；此前汇总文件名及原运行目录中的 `v4pro` 标记有误，现统一命名为 `nanofull_vs_nano_r1_result.jsonl`。另外两组与 HiRAG 的比较仍按原记录使用 SenseNova `deepseek-v4-pro`。原运行目录保留原样以便追溯；评测 JSONL 本身不记录模型名。Mix 包含 130 个问题和 260 条双向评测记录；其余数据集各包含 100 个问题和 200 条双向评测记录。表中胜率均已按答案位置交换进行校正。

## Mix

| 对比实验 | Comprehensiveness | Empowerment | Diversity | 前者 Overall | 后者 Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nano Full vs Nano | 56.15% | 52.31% | 56.92% | **55.77%** | 44.23% |
| Nano vs HiRAG | 46.92% | 42.69% | 51.54% | 44.23% | **55.77%** |
| Nano Full vs HiRAG | 52.69% | 50.38% | 53.85% | **52.69%** | 47.31% |

## CS

| 对比实验 | Comprehensiveness | Empowerment | Diversity | 前者 Overall | 后者 Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nano vs HiRAG | 49.00% | 47.00% | 52.50% | 48.50% | **51.50%** |
| Nano Full vs HiRAG | 53.00% | 53.00% | 53.50% | **52.00%** | 48.00% |
| Nano Full vs Nano | 55.00% | 52.00% | 58.50% | **55.50%** | 44.50% |

## Legal

| 对比实验 | Comprehensiveness | Empowerment | Diversity | 前者 Overall | 后者 Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nano vs HiRAG | 49.50% | 48.00% | 49.50% | 48.50% | **51.50%** |
| Nano Full vs HiRAG | 53.50% | 50.50% | 48.50% | **51.00%** | 49.00% |
| Nano Full vs Nano | 55.00% | 52.00% | 51.50% | **53.50%** | 46.50% |

## Agriculture

| 对比实验 | Comprehensiveness | Empowerment | Diversity | 前者 Overall | 后者 Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nano vs HiRAG | 48.50% | 46.50% | 43.00% | 48.00% | **52.00%** |
| Nano Full vs HiRAG | 60.50% | 55.00% | 53.00% | **57.50%** | 42.50% |
| Nano Full vs Nano | 55.50% | 56.50% | 59.00% | **57.00%** | 43.00% |

## 总体结论

原生 Nano 在四个数据集上均未超过原生 HiRAG，Overall 分别为 44.23%、48.50%、48.50% 和 48.00%。接入 PlugRag Full 后，Nano 在四个数据集上均超过原生 Nano，同时也超过原生 HiRAG。该结果支持两点：PlugRag 的增益能够迁移到 Nano-GraphRAG 主干；这种增益并非仅来自原生 Nano 本身，而来自 PlugRag 的结构扩展、混合重排与路径证据组织。

需要注意，单个数据集上的接近胜率应表述为趋势，不宜仅凭百分比宣称统计显著；正式论文中应结合置信区间与显著性检验报告。
`Nano Full vs Nano` 与另外两组比较使用不同评测模型，其胜率不能直接相减。

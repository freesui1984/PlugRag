# PlugRag

**PlugRag: A Plug-and-Play Graph Retrieval Enhancement Method Integrating Semantic and Structural Representations**

[English README](README.md)

PlugRag 是一个面向 KG-enhanced RAG 的插件式实体级图检索增强方法。它不改变底层 RAG 框架原有的索引、图构建和答案生成流程，而是在实体检索与上下文组织阶段加入语义-结构联合表示、结构化实体召回、混合排序和结构溯源路径，从而提升结构证据组织能力。

<p align="center">
  <img src="assets/figure2-plugrag-overview.png" alt="PlugRag overview" width="900">
</p>

## 工作简介

现有 KG-enhanced RAG 方法通常依赖语义相似度、邻居扩展或路径搜索进行实体召回。这类方法能够利用图结构补充证据，但也容易引入弱相关邻居、冗余路径，或者遗漏关键中间实体。PlugRag 的目标不是替换 HiRAG 或 Nano-GraphRAG，而是作为一个可插拔增强层接入这些框架。

PlugRag 的核心思想包括：

- **语义-结构联合表示**：使用实体名称和描述的文本 embedding 作为初始表示，并通过 GCN / R-GCN 注入图拓扑与关系类型信息。
- **结构化实体召回**：以高置信语义实体作为种子，在图编码器表示空间中扩展结构相关实体。
- **混合排序**：结合语义过滤、结构投票频率和 Cross-Encoder reranking，抑制结构扩展带来的噪声。
- **结构溯源路径**：为结构扩展实体生成从语义种子到目标实体的关系路径，使上下文来源更可解释。

## 目录结构

```text
github_code_release_20260619/
├── HiRAG-PlugRag/
│   ├── hirag/
│   ├── evel/
│   ├── config.example.yaml
│   ├── environment.yml
├── Nano-GraphRAG-PlugRag/
│   ├── nano_graphrag/
│   ├── evel/
│   ├── config_example.yaml
│   ├── environment.yml
├── paper_results/
│   ├── raw/
│   ├── experiments.json
│   ├── files.json
│   └── audit.py
└── assets/
    └── figure2-plugrag-overview.png
```

HiRAG 和 Nano-GraphRAG 两个子目录可以分别运行，安装、配置、训练、问答和评测方法统一见下文。

## 无侵入式接入

PlugRag 的接入方式是无侵入式的：

- 保留原始框架的文档切分、实体关系抽取、知识图谱构建和 LLM 生成流程。
- 不要求重写 HiRAG / Nano-GraphRAG 的全局检索逻辑。
- 只在实体级 local retrieval 与 context construction 阶段加入增强模块。
- 图编码器训练结果通过 YAML 配置和 checkpoint 路径选择，不需要把 R-GCN / GCN / Multi-edge GCN 拆成多个复杂查询入口。

PlugRag 提供三个查询模式：

- `plugrag_full`：完整 PlugRag，包含结构扩展、混合排序和 trace/path 上下文。
- `plugrag_rank_only`：语义排序消融，不使用图编码器结构扩展。
- `plugrag_vote_only`：结构投票消融，不使用语义融合重排。

HiRAG 还提供 `hi_ppr_top3` 和 `pathrag_evidence_aligned`，用于 PPR 和 PathRAG 风格的受控检索比较。这两个模式共用 HiRAG 图谱和答案生成流程，并非完整复现 HippoRAG 或 PathRAG 系统。使用时将 `batch_query.graph_encoder` 设为 `none`；实现分别位于 `HiRAG-PlugRag/hirag/ppr_top3.py` 和 `HiRAG-PlugRag/hirag/pathrag_evidence_aligned.py`。

## 安装

推荐 Python 3.10。

HiRAG 版本：

```bash
cd HiRAG-PlugRag
conda env create -f environment.yml
conda activate plugrag-hirag
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

Nano-GraphRAG 版本：

```bash
cd Nano-GraphRAG-PlugRag
conda env create -f environment.yml
conda activate plugrag-nano
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

如果需要 CUDA 版本 PyTorch，请先按本机 CUDA 版本安装 PyTorch，再安装 `requirements.txt`。Nano 的 HNSW、Ollama、Qdrant、Milvus、DSPy 等扩展依赖放在 `requirements-optional.txt` 中。

## 配置

运行前请复制配置文件并填写 API 信息：

```bash
cp HiRAG-PlugRag/config.example.yaml HiRAG-PlugRag/config.local.yaml
cp Nano-GraphRAG-PlugRag/config_example.yaml Nano-GraphRAG-PlugRag/config_local.yaml
```

Windows PowerShell：

```powershell
copy HiRAG-PlugRag/config.example.yaml HiRAG-PlugRag/config.local.yaml
copy Nano-GraphRAG-PlugRag/config_example.yaml Nano-GraphRAG-PlugRag/config_local.yaml
$env:PLUGRAG_CONFIG="config.local.yaml" # HiRAG；Nano 使用 config_local.yaml
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_BASE_URL="https://api.siliconflow.cn/v1"
```

论文实验中使用 BGE-M3 作为初始实体语义表示，使用 R-GCN / Simple GCN / Multi-edge R-GCN+TransE 作为图编码器变体，使用 BGE-Reranker-v2-M3 进行 reranking。

Linux / macOS 使用 `export PLUGRAG_CONFIG=config.local.yaml`（Nano 使用 `config_local.yaml`）选择配置，并用 `export OPENAI_API_KEY=...` 等环境变量设置接口信息。

训练和问答前需准备基础框架的图谱与索引。HiRAG 通过 `hirag.working_dir` 指定，Nano 通过 `storage.working_dir` 指定；Nano 也可通过 `indexing.context_file` 和 `indexing.insert_before_training` 从语料建索引。在本地 YAML 中设置模型接口、实体嵌入与映射文件、问题和输出路径。下列命令以已有兼容图谱与索引为前提。

## 训练

HiRAG 从 GraphML 转换训练数据：

```bash
cd HiRAG-PlugRag
python -m hirag.convert_graphml_to_rgcn_data --input work_dir_data_mix/graph_chunk_entity_relation_02.graphml --output work_dir_data_mix/rgcn/rgcn_data.pt
python -m hirag.convert_graphml_to_simple_gcn_data --input work_dir_data_mix/graph_chunk_entity_relation_02.graphml --output work_dir_data_mix/gcn/gcn_data.pt
python -m hirag.convert_graphml_to_multi_edge_rgcn_data --input work_dir_data_mix/graph_chunk_entity_relation_02.graphml --output work_dir_data_mix/multi_edge/multi_edge_rgcn_data.pt
```

HiRAG 训练图编码器：

```bash
python -m hirag.train_plugrag_rgcn_encoder
python -m hirag.train_simple_gcn_ablation
python -m hirag.train_multi_edge_rgcn_transe_ablation
```

Nano-GraphRAG 转换训练数据：

```bash
cd Nano-GraphRAG-PlugRag
python prepare_plugrag_training_data.py --config config_example.yaml --encoder all
```

Nano-GraphRAG 训练图编码器：

```bash
python train_plugrag_rgcn_encoder.py --config config_example.yaml
python train_simple_gcn_ablation.py --config config_example.yaml
python train_multi_edge_rgcn_transe_ablation.py --config config_example.yaml
```

## 批量问答

批量问答由 YAML 中的 `batch_query` 控制：

```yaml
batch_query:
  input_file: "queries.jsonl"
  output_file: "answers.jsonl"
  concurrency: 5
  mode: "plugrag_full"
  graph_encoder: "rgcn"
  top_k: 20
```

HiRAG：

```bash
cd HiRAG-PlugRag
python hi_batch_query.py
```

Nano-GraphRAG：

```bash
cd Nano-GraphRAG-PlugRag
python hi_batch_query.py
```

消融实验只需要切换 `mode` 或图编码器路径。例如，Simple GCN 消融仍然使用 `plugrag_full`。HiRAG 可通过 `batch_query.graph_encoder: "simple_gcn"` 切换；Nano 可将 `storage.rgcn_embeddings_file` 和 `storage.rgcn_mapping_file` 指向 Simple GCN 文件。

## 评测

实验使用 [UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain) 的四个子集：

- Mix
- CS
- Legal
- Agriculture

更新后的论文答案和评测记录分别放在：

```text
paper_results/raw/HiRAG-PlugRag/evel/datasets/
paper_results/raw/Nano-GraphRAG-PlugRag/evel/datasets/
```

`paper_results/experiments.json` 给出比较实验、评测文件和答案顺序的对应关系，`paper_results/files.json` 记录来源路径与校验值。新版结果请按这些映射读取；原有 `evel/` 中也有历史批次。补充的 Mix Full 答案文件名保留 `candidate`，表示其历史批次身份尚未完全确认。

从仓库根目录运行以下命令，可离线核对归档文件和论文百分比，不调用模型 API：

```bash
python paper_results/audit.py
python -m unittest discover -s paper_results/tests -q
```

需要重新进行成对评测时，在对应基础框架目录下运行：

```bash
cp evel/evaluation_code/eval_config.example.yaml evel/evaluation_code/eval_config.yaml
python evel/evaluation_code/batch_eval.py --help
```

也可通过 `PLUGRAG_EVAL_CONFIG` 指定评测配置文件，避免将 API 密钥写入仓库。

评测维度包括 Comprehensiveness、Empowerment、Diversity 和 Overall Winner。论文实验采用双向位置交换的 pairwise evaluation，以缓解 Answer 1 / Answer 2 的位置偏差。

经作者确认，答案生成使用 DeepSeek-V3；跨框架比较由 Pro 评判，框架内比较、PPR/PathRAG 对比及消融评测由 DeepSeek-R1 评判。答案生成与评测是不同阶段。历史文件名予以保留，不作为模型身份的依据；报告使用的评测模型见 `paper_results/experiments.json`。

## 运行说明

- 图谱、索引及训练产物不随代码打包，需准备本地数据和相应路径。
- 示例配置用于新实验，不代表每个历史批次的实际配置；离线核对不等于重新生成答案或确认历史模型身份。
- 问答及模型评测需要有效 API 凭据，调用可能产生费用。请勿提交密钥。
- 代码测试可在 `HiRAG-PlugRag` 中运行 `python -m pytest -q tests`。

## 主要发现

在 UltraDomain 的四个子集上，PlugRag 在多数设置中带来正向但相对温和的提升。R-GCN 变体在 HiRAG 和 Nano-GraphRAG 两个 backbone 上表现相对稳定；Simple GCN 和 Multi-edge GCN 的效果更依赖具体数据集结构、关系标签质量和图连通性。消融实验显示，语义-结构表示、混合排序和结构溯源路径对降低结构噪声、改善证据组织有帮助。

## 致谢

本仓库基于并感谢以下开源项目：

- [hhy-huang/HiRAG](https://github.com/hhy-huang/HiRAG)
- [gusye1234/nano-graphrag](https://github.com/gusye1234/nano-graphrag)

同时感谢 [UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain) 数据集的贡献者。

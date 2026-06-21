# Nano-GraphRAG-PlugRag

本目录是论文中基于 Nano-GraphRAG 的 PlugRag 代码发布版。相比原始 Nano-GraphRAG，本版本加入了图编码器训练、结构邻居扩展、语义与结构融合排序、trace/path 上下文，以及 rank-only / vote-only 消融。

## 1. 推荐环境

推荐使用 Python 3.10。Python 3.11 通常也可以，Python 3.12 暂不建议作为复现实验环境。

```bash
conda env create -f environment.yml
conda activate plugrag-nano
python -m pip install --upgrade pip setuptools wheel
```

先安装与你机器匹配的 PyTorch。CPU 环境：

```bash
pip install "torch>=2.1,<2.6" --index-url https://download.pytorch.org/whl/cpu
```

CUDA 12.1 环境：

```bash
pip install "torch>=2.1,<2.6" --index-url https://download.pytorch.org/whl/cu121
```

然后安装项目依赖和本地包：

```bash
pip install -r requirements.txt
pip install -e .
```

可选后端和 examples 依赖单独安装：

```bash
pip install -r requirements-optional.txt
```

`requirements-optional.txt` 包含 HNSW、Amazon Bedrock、DSPy、Ollama、Qdrant、Milvus、SentenceTransformers、HuggingFace tokenizer 等扩展。论文核心 Nano-GraphRAG+PlugRag 流程不强制依赖这些包。

安装完成后可以做一次最小检查：

```bash
python -c "from nano_graphrag import GraphRAG, QueryParam; print('Nano-GraphRAG-PlugRag import OK')"
```

## 2. API 与配置

默认配置文件是 `config_example.yaml`。建议复制一份再修改：

```bash
copy config_example.yaml config_local.yaml
```

Windows PowerShell：

```powershell
$env:PLUGRAG_CONFIG="config_local.yaml"
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_BASE_URL="https://api.siliconflow.cn/v1"
```

Linux / macOS：

```bash
export PLUGRAG_CONFIG=config_local.yaml
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=https://api.siliconflow.cn/v1
```

`config_example.yaml` 主要包含这些区域：

- `api_keys` / `models`：LLM、embedding、rerank 模型和 API 地址。
- `storage`：Nano working directory，以及三类图编码器训练结果路径。
- `graphml_conversion`：GraphML 转训练数据的输入输出。
- `indexing`：是否从原始语料重新插入并构建 Nano 图。
- `retrieval`：PlugRag 检索、扩展、rerank 参数。
- `batch_query`：批量问答输入输出和查询模式。
- `rgcn_training`：论文主 R-GCN 编码器训练参数。
- `gcn_training`：Simple GCN 消融训练参数。
- `multi_edge_rgcn_transe_training`：Multi-edge R-GCN+TransE 消融训练参数。

## 3. 准备 Nano 工作目录

如果已经有 Nano-GraphRAG 的索引目录，在配置中指定：

```yaml
storage:
  working_dir: "work_dir_data_mix"
```

如果希望训练前从原始语料重新插入，可以设置：

```yaml
indexing:
  context_file: "datasets/mix_contexts.jsonl"
  insert_before_training: true
```

`context_file` 支持 `.txt`、`.json`、`.jsonl`。对于 `.jsonl`，每行可以包含 `content`、`text`、`context` 或 `article` 字段。

## 4. 转换训练数据

从当前 Nano working directory 转换三类训练数据：

```bash
python prepare_plugrag_training_data.py --config config_example.yaml --encoder all
```

只转换某一类：

```bash
python prepare_plugrag_training_data.py --config config_example.yaml --encoder rgcn
python prepare_plugrag_training_data.py --config config_example.yaml --encoder gcn
python prepare_plugrag_training_data.py --config config_example.yaml --encoder multi_edge
```

如果已有 GraphML，也可以直接转换：

```bash
python -m nano_graphrag.convert_graphml_to_rgcn_data --config config_example.yaml
python -m nano_graphrag.convert_graphml_to_simple_gcn_data --config config_example.yaml
python -m nano_graphrag.convert_graphml_to_multi_edge_rgcn_data --config config_example.yaml
```

## 5. 训练图编码器

```bash
python train_plugrag_rgcn_encoder.py --config config_example.yaml
python train_simple_gcn_ablation.py --config config_example.yaml
python train_multi_edge_rgcn_transe_ablation.py --config config_example.yaml
```

三类编码器的含义：

- `rgcn`：论文主模型，保留关系类型的 R-GCN。
- `simple_gcn`：去掉关系类型后的同质图 GCN 消融。
- `multi_edge_rgcn_transe`：保留多关系边，并加入 TransE 风格结构损失的消融。

如果已经转换好了 `.pt` 训练数据，可以在对应训练区域设置：

```yaml
convert_data: false
```

## 6. 批量问答

在 YAML 中设置：

```yaml
batch_query:
  input_file: "queries.jsonl"
  output_file: "answers.jsonl"
  concurrency: 1
  mode: "plugrag_full"
  top_k: 20
```

运行：

```bash
python hi_batch_query.py
```

常用查询模式：

- `plugrag_full`：论文主模式，结构扩展 + 语义重排 + trace/path 上下文。
- `plugrag_rank_only`：只保留语义检索/重排，不使用图编码器结构扩展。
- `plugrag_vote_only`：只保留结构投票排序，不使用语义融合重排。

旧实验名仍然兼容：`rgcn` 会映射到 full，`rank` 会映射到 rank-only，`vote_rgcn` 会映射到 vote-only。

## 7. 评测数据与脚本

开源评测数据放在 `evel/` 下，按照 `mix`、`cs`、`legal`、`agriculture` 四个数据集分开整理。评测脚本保留论文实验使用的原始 `batch_eval.py`：

```bash
python evel/evaluation_code/batch_eval.py --help
```

`evel/evaluation_code/collect_eval_results.py` 只用于汇总已有 OpenAI 评测 JSONL，不替代论文原始评测脚本。

## 8. 核心文件

- `nano_graphrag/graphrag.py`：Nano-GraphRAG 主类、PlugRag 查询模式、训练入口。
- `nano_graphrag/_op.py`：结构扩展、融合排序、rank-only、vote-only、full 查询逻辑。
- `nano_graphrag/rgcn_utils.py`：R-GCN、Simple GCN、Multi-edge R-GCN+TransE 的训练核心。
- `nano_graphrag/graphml_conversion.py`：GraphML 转三类训练数据。
- `nano_graphrag/plugrag_training.py`：YAML 驱动的数据转换和训练流程。
- `prepare_plugrag_training_data.py`：命令行转换入口。
- `train_plugrag_rgcn_encoder.py`：R-GCN 主模型训练入口。
- `train_simple_gcn_ablation.py`：Simple GCN 消融训练入口。
- `train_multi_edge_rgcn_transe_ablation.py`：Multi-edge R-GCN+TransE 消融训练入口。

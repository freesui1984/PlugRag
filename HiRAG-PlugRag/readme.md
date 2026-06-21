# HiRAG-PlugRag

本目录是论文中基于 HiRAG 的 PlugRag 代码发布版。核心改动是在原始 HiRAG 分层检索框架上加入图编码器训练、结构邻居扩展、trace/path 上下文、语义与结构融合排序，以及 rank-only / vote-only 消融。

## 1. 推荐环境

推荐使用 Python 3.10。Python 3.11 通常也可以，Python 3.12 暂不建议作为复现实验环境，因为 PyTorch Geometric、graspologic 等包在部分机器上更容易遇到兼容问题。

```bash
conda env create -f environment.yml
conda activate plugrag-hirag
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

如果 `torch-geometric` 在你的环境里安装失败，请先确认 PyTorch 版本，再按 PyG 官方选择器安装匹配 wheel。安装完成后可以做一次最小检查：

```bash
python -c "from hirag import HiRAG, QueryParam; print('HiRAG-PlugRag import OK')"
```

## 2. API 与配置

默认配置文件是 `config.example.yaml`。建议复制一份再修改：

```bash
copy config.example.yaml config.local.yaml
```

Windows PowerShell：

```powershell
$env:PLUGRAG_CONFIG="config.local.yaml"
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_BASE_URL="https://api.siliconflow.cn/v1"
```

Linux / macOS：

```bash
export PLUGRAG_CONFIG=config.local.yaml
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=https://api.siliconflow.cn/v1
```

`config.example.yaml` 主要包含这些区域：

- `openai`：LLM、embedding 模型、API key、base URL。
- `hirag`：HiRAG working directory、缓存、分层检索、embedding 并发。
- `batch_query`：批量问答输入输出、查询模式、图编码器选择。
- `rgcn`：论文主模型 R-GCN 的 GraphML 转换和训练参数。
- `gcn`：Simple GCN 消融的转换和训练参数。
- `multi_edge_rgcn_transe`：Multi-edge R-GCN+TransE 消融的转换和训练参数。

## 3. 准备 HiRAG 工作目录

如果已经有 HiRAG 的索引目录，在配置中指定：

```yaml
hirag:
  working_dir: "work_dir_data_mix"
```

该目录通常包含 GraphML、entity vector storage、text chunks、community reports 和 LLM cache。若要从原始语料重新构建，可以使用 HiRAG 的插入流程：

```python
from hirag import HiRAG

graph_func = HiRAG(
    working_dir="work_dir_data_mix",
    enable_hierachical_mode=True,
    enable_naive_rag=True,
)

with open("contexts.txt", "r", encoding="utf-8") as f:
    graph_func.insert(f.read())
```

## 4. 转换训练数据

从 GraphML 转 R-GCN 训练数据：

```bash
python -m hirag.convert_graphml_to_rgcn_data --input work_dir_data_mix/graph_chunk_entity_relation_02.graphml --output work_dir_data_mix/rgcn/rgcn_data.pt
```

从 GraphML 转 Simple GCN 训练数据：

```bash
python -m hirag.convert_graphml_to_simple_gcn_data --input work_dir_data_mix/graph_chunk_entity_relation_02.graphml --output work_dir_data_mix/gcn/gcn_data.pt
```

从 GraphML 转 Multi-edge R-GCN+TransE 训练数据：

```bash
python -m hirag.convert_graphml_to_multi_edge_rgcn_data --input work_dir_data_mix/graph_chunk_entity_relation_02.graphml --output work_dir_data_mix/multi_edge/multi_edge_rgcn_data.pt
```

转换脚本会同时生成 mapping 文件，例如 `rgcn_data_mapping.json`、`gcn_data_mapping.json`、`multi_edge_rgcn_data_mapping.json`。

## 5. 训练图编码器

训练参数从 `PLUGRAG_CONFIG` 指向的 YAML 中读取。

```bash
python -m hirag.train_plugrag_rgcn_encoder
python -m hirag.train_simple_gcn_ablation
python -m hirag.train_multi_edge_rgcn_transe_ablation
```

三类编码器的含义：

- `rgcn`：论文主模型，保留关系类型的 R-GCN。
- `simple_gcn`：去掉关系类型后的同质图 GCN 消融。
- `multi_edge_rgcn_transe`：保留多关系边，并加入 TransE 风格结构损失的消融。

## 6. 批量问答

在 YAML 中设置：

```yaml
batch_query:
  input_file: "queries.jsonl"
  output_file: "answers.jsonl"
  concurrency: 5
  mode: "plugrag_full"
  graph_encoder: "rgcn"
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

GCN / R-GCN / Multi-edge 的差异不再写成不同 query mode，而是由 `batch_query.graph_encoder` 选择加载哪一份训练结果：

```yaml
batch_query:
  mode: "plugrag_full"
  graph_encoder: "simple_gcn"
```

## 7. 评测数据与脚本

开源评测数据放在 `evel/` 下，按照 `mix`、`cs`、`legal`、`agriculture` 四个数据集分开整理。评测脚本保留论文实验使用的原始 `batch_eval.py`：

```bash
python evel/evaluation_code/batch_eval.py --help
```

`evel/evaluation_code/collect_eval_results.py` 只用于汇总已有 OpenAI 评测 JSONL，不替代论文原始评测脚本。

## 8. 核心文件

- `hirag/hirag.py`：HiRAG 主类、查询分发、PlugRag 模式别名。
- `hirag/_op.py`：分层检索、结构扩展、rank-only、vote-only、full 查询逻辑。
- `hirag/convert_graphml_to_rgcn_data.py`：GraphML 转 R-GCN 训练数据。
- `hirag/convert_graphml_to_simple_gcn_data.py`：GraphML 转 Simple GCN 训练数据。
- `hirag/convert_graphml_to_multi_edge_rgcn_data.py`：GraphML 转 Multi-edge R-GCN 数据。
- `hirag/train_plugrag_rgcn_encoder.py`：R-GCN 主模型训练。
- `hirag/train_simple_gcn_ablation.py`：Simple GCN 消融训练。
- `hirag/train_multi_edge_rgcn_transe_ablation.py`：Multi-edge R-GCN+TransE 消融训练。
- `hirag/plugrag_path_utils.py`：trace/path、覆盖率统计和解释性日志工具。

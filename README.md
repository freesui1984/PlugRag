# PlugRag

**PlugRag: A Plug-and-Play Graph Retrieval Enhancement Method Integrating Semantic and Structural Representations**

[Chinese README](README_zh.md)

PlugRag is a plug-and-play entity-level graph retrieval enhancement method for KG-enhanced retrieval-augmented generation. It does not replace the indexing, graph construction, or generation workflow of the host RAG framework. Instead, it adds semantic-structural entity representations, structured entity recall, hybrid ranking, and structural trace paths during entity retrieval and context construction.

<p align="center">
  <img src="assets/figure2-plugrag-overview.png" alt="PlugRag overview" width="900">
</p>

## Overview

Existing KG-enhanced RAG systems often retrieve entities through semantic similarity, neighborhood expansion, or path search. These mechanisms can improve evidence coverage, but they may also introduce weakly related neighbors, redundant paths, or miss important intermediate entities. PlugRag addresses this issue as a non-intrusive enhancement layer that can be attached to graph-based RAG frameworks such as HiRAG and Nano-GraphRAG.

PlugRag consists of four main components:

- **Semantic-Structural Representation**: initialize entities with text embeddings from entity names and descriptions, then inject graph topology and relation-type information with GCN / R-GCN encoders.
- **Structured Entity Recall**: use high-confidence semantic entities as seeds and expand structurally related entities in the graph-encoder representation space.
- **Hybrid Ranking**: combine semantic filtering, structural voting frequency, and Cross-Encoder reranking to reduce structural noise.
- **Structural Trace Path**: generate relation paths from semantic seeds to expanded entities, making expanded evidence more interpretable.

## Repository Structure

```text
github_code_release_20260619/
|-- HiRAG-PlugRag/
|   |-- hirag/
|   |-- evel/
|   |-- config.example.yaml
|   |-- environment.yml
|   `-- readme.md
|-- Nano-GraphRAG-PlugRag/
|   |-- nano_graphrag/
|   |-- evel/
|   |-- config_example.yaml
|   |-- environment.yml
|   `-- readme.md
|-- assets/
|   `-- figure2-plugrag-overview.png
`-- README_zh.md
```

The two backbone implementations can be used independently. See the README inside each subdirectory for detailed environment, configuration, training, and inference commands.

## Non-Intrusive Integration

PlugRag is designed as a non-intrusive module:

- It keeps the original document chunking, entity-relation extraction, knowledge graph construction, and LLM generation workflow.
- It does not rewrite the global retrieval logic of HiRAG or Nano-GraphRAG.
- It only enhances local entity retrieval and context construction.
- Graph encoder variants are selected through configuration and checkpoint paths instead of separate query modes.

The release uses three official query modes:

- `plugrag_full`: full PlugRag with structural expansion, hybrid ranking, and trace/path context.
- `plugrag_rank_only`: semantic ranking ablation without graph-encoder structural expansion.
- `plugrag_vote_only`: structural voting ablation without semantic reranking fusion.

## Installation

Python 3.10 is recommended.

HiRAG version:

```bash
cd HiRAG-PlugRag
conda env create -f environment.yml
conda activate plugrag-hirag
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

Nano-GraphRAG version:

```bash
cd Nano-GraphRAG-PlugRag
conda env create -f environment.yml
conda activate plugrag-nano
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

If you need a CUDA build of PyTorch, install the matching PyTorch package first and then install `requirements.txt`. Optional Nano backends such as HNSW, Ollama, Qdrant, Milvus, and DSPy are listed in `Nano-GraphRAG-PlugRag/requirements-optional.txt`.

## Configuration

Copy the example configuration before running experiments:

```bash
cp HiRAG-PlugRag/config.example.yaml HiRAG-PlugRag/config.local.yaml
cp Nano-GraphRAG-PlugRag/config_example.yaml Nano-GraphRAG-PlugRag/config_local.yaml
```

Windows PowerShell:

```powershell
copy HiRAG-PlugRag/config.example.yaml HiRAG-PlugRag/config.local.yaml
copy Nano-GraphRAG-PlugRag/config_example.yaml Nano-GraphRAG-PlugRag/config_local.yaml
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_BASE_URL="https://api.siliconflow.cn/v1"
```

Linux / macOS:

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=https://api.siliconflow.cn/v1
```

In the paper experiments, BGE-M3 is used for initial entity semantic embeddings, R-GCN / Simple GCN / Multi-edge R-GCN+TransE are used as graph encoder variants, and BGE-Reranker-v2-M3 is used for reranking.

## Training

### HiRAG

Convert GraphML to graph encoder training data:

```bash
cd HiRAG-PlugRag
python -m hirag.convert_graphml_to_rgcn_data --input work_dir_data_mix/graph_chunk_entity_relation_02.graphml --output work_dir_data_mix/rgcn/rgcn_data.pt
python -m hirag.convert_graphml_to_simple_gcn_data --input work_dir_data_mix/graph_chunk_entity_relation_02.graphml --output work_dir_data_mix/gcn/gcn_data.pt
python -m hirag.convert_graphml_to_multi_edge_rgcn_data --input work_dir_data_mix/graph_chunk_entity_relation_02.graphml --output work_dir_data_mix/multi_edge/multi_edge_rgcn_data.pt
```

Train graph encoders:

```bash
python -m hirag.train_plugrag_rgcn_encoder
python -m hirag.train_simple_gcn_ablation
python -m hirag.train_multi_edge_rgcn_transe_ablation
```

### Nano-GraphRAG

Convert training data from an existing Nano working directory or GraphML:

```bash
cd Nano-GraphRAG-PlugRag
python prepare_plugrag_training_data.py --config config_example.yaml --encoder all
```

Train graph encoders:

```bash
python train_plugrag_rgcn_encoder.py --config config_example.yaml
python train_simple_gcn_ablation.py --config config_example.yaml
python train_multi_edge_rgcn_transe_ablation.py --config config_example.yaml
```

## Question Answering

Batch QA is controlled by the `batch_query` section in YAML:

```yaml
batch_query:
  input_file: "queries.jsonl"
  output_file: "answers.jsonl"
  concurrency: 5
  mode: "plugrag_full"
  graph_encoder: "rgcn"
  top_k: 20
```

HiRAG:

```bash
cd HiRAG-PlugRag
python hi_batch_query.py
```

Nano-GraphRAG:

```bash
cd Nano-GraphRAG-PlugRag
python hi_batch_query.py
```

For ablations, keep the query mode simple and switch the encoder checkpoint paths. For example, Simple GCN still runs through `plugrag_full`; HiRAG can use `batch_query.graph_encoder: "simple_gcn"`, while Nano can point `storage.rgcn_embeddings_file` and `storage.rgcn_mapping_file` to the Simple GCN files.

## Evaluation

Experiments are conducted on four subsets of [UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain):

- Mix
- CS
- Legal
- Agriculture

Released evaluation files are organized under:

```text
HiRAG-PlugRag/evel/
Nano-GraphRAG-PlugRag/evel/
```

Each dataset folder contains JSONL release files: queries, answers, and OpenAI/LLM-as-a-judge result files. The evaluation code keeps the original `batch_eval.py` used for the paper experiments.

```bash
cp evel/evaluation_code/eval_config.example.yaml evel/evaluation_code/eval_config.yaml
python evel/evaluation_code/batch_eval.py --help
```

The evaluation script also accepts `PLUGRAG_EVAL_CONFIG` if you prefer to keep API credentials outside the repository.

The evaluation dimensions are:

- Comprehensiveness
- Empowerment
- Diversity
- Overall Winner

The paper uses bidirectional position-swapped pairwise evaluation to reduce answer-order bias.

## Reproducibility Notes

- Public configuration files do not contain private API keys.
- Full answer and evaluation result JSONL files are included under each `evel/datasets/<dataset>` folder.
- Generated training checkpoints and local working directories are intentionally excluded by `.gitignore`; users can regenerate them from the provided conversion and training scripts.
- API-dependent QA and LLM-as-a-judge evaluation require valid API credentials and may produce small variations if the provider changes model behavior.

## Main Findings

Across the four UltraDomain subsets, PlugRag generally provides positive but modest gains. The R-GCN variant is relatively stable across both HiRAG and Nano-GraphRAG. Simple GCN and Multi-edge GCN are more sensitive to dataset structure, relation-label quality, and graph connectivity. Ablation studies indicate that semantic-structural representation, hybrid ranking, and structural trace paths help reduce structural noise and improve evidence organization.

## Acknowledgements

This repository builds on and sincerely thanks the following open-source projects:

- [hhy-huang/HiRAG](https://github.com/hhy-huang/HiRAG)
- [gusye1234/nano-graphrag](https://github.com/gusye1234/nano-graphrag)

We also thank the creators of [UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain) for providing the benchmark used in the experiments.

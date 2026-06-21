# Evaluation Code

This folder keeps the evaluation script used for the paper results.

## Batch Evaluation

- `batch_eval.py`: preserves the original LLM-as-a-judge prompt, Answer 1 / Answer 2 swap logic, Batch API request creation, and result statistics code used by the experiments.
- The public release adds portable config loading. Copy `eval_config.example.yaml` to `eval_config.yaml`, or set `PLUGRAG_EVAL_CONFIG` to your local config path.

```bash
cp evaluation_code/eval_config.example.yaml evaluation_code/eval_config.yaml
python evaluation_code/batch_eval.py --help
```

## Summarize Judge Results

```bash
python evaluation_code/collect_eval_results.py \
  --root datasets/mix/results \
  --out mix_eval_summary.csv \
  --md mix_eval_summary.md
```

The summary script handles the swapped Answer 1 / Answer 2 evaluation format used in the paper.

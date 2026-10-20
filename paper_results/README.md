# Paper results and controlled retrieval release

This is a paper-aligned numerical archive, not a new run of the experiments.
This release candidate is pending author review; numerical verification is not
certification that the inherited retrieval implementation is free of defects.
The source manuscript is identified by filename and SHA-256 in `experiments.json`;
the manuscript itself is not published here. Existing upstream `evel/` results may
be from older batches. Use this directory's explicit mapping for the revised paper.

## Model reporting

Following the author's latest correction on 2026-10-19, answer generation uses
V3; non-cross-framework evaluation (main, controlled PPR/PathRAG comparisons and
ablations) uses V3; cross-framework evaluation uses Pro. Earlier release labels
of R1 for ordinary evaluation and Pro for PPR evaluation are superseded. Original
filenames and historical documents are retained, not relabelled or used to infer
current model identities. These labels are author-reported, not embedded API IDs.

## Recompute the reported numbers

From the repository root, using Python 3.10 or newer (standard library only):

```powershell
python paper_results/audit.py
python -m unittest discover -s paper_results/tests -v
```

The audit checks 185 immutable JSONL files (including one supplemental candidate
answer file), 88 dataset/comparison combinations,
and 188 reported percentages, including aggregate values. It rejects malformed
or missing winner fields, wrong counts, changed checksums, unexpected raw files,
duplicate questions, and inconsistent question order among archived input files.
Mix contains 130 questions; CS, Legal and Agriculture each contain 100. Each
comparison has two judgments per question (260 or 200 rows), not twice as many
independent questions. Comparisons overlap in their questions.

The first N judgments use the original answer order, followed by N judgments
with swapped answer positions. `target_first_pass_answer` records whether the
reported method was Answer 1 or Answer 2 in the first pass. Only nested `Winner`
fields are counted; explanations are not parsed for labels. Macro averages use
unrounded dataset win rates; pooled rates use summed wins and judgment counts.
These two averages must not be interchanged.

## Files

- `experiments.json`: experiment identity, manuscript expectations, comparator
  orientation, evaluation file, and historical provenance qualifications.
- `files.json`: original relative paths, roles, sizes, row counts, and SHA-256.
- `raw/`: byte-for-byte copies, retaining source relative paths to distinguish
  updated main results from older ablation/comparison answer batches.
- `verification.json`: generated report; regenerate with
  `python paper_results/audit.py --output paper_results/verification.json`.
- `SUMMARY.md`: readable cross-dataset Overall table.
- `RELEASE_NOTES.md`: unresolved provenance issues and source-code boundaries.
- `provenance/`: selected existing result-index documents, preserved as evidence,
  not new certifications that their historical settings were uniform.
  `mix_full_candidate.json` records the content/order and log-timing evidence
  for the supplemental Mix `answers/mix_hirag_plugrag_full_v3_candidate_answers.jsonl` (original source name: `1.jsonl`). Its exact historical input-batch
  identity remains unverified; no judgments were regenerated from this candidate.

## Additional retrieval code

The HiRAG modes are `hi_ppr_top3` and `pathrag_evidence_aligned`.
They adapt PPR and PathRAG-style retrieval to the existing HiRAG graph and answer
generation pipeline. They are not full reproductions of HippoRAG or PathRAG.
The PPR function bodies and PathRAG module function/class bodies were checked
against the local experiment implementation before transplantation.

Use the existing per-backbone environment setup instructions. In
`HiRAG-PlugRag/config.local.yaml`, copied from `config.example.yaml`, set
`batch_query.mode` to the required mode, `batch_query.graph_encoder` to `none`
for these two controlled baselines, and point `hirag.working_dir` to your existing
graph/index. Run from `HiRAG-PlugRag/`:

```powershell
$env:PLUGRAG_CONFIG = 'config.local.yaml'
python hi_batch_query.py
python -m pytest -q tests
```

Generation requires paid or locally hosted model endpoints, graph stores and
embeddings; none are bundled, and no model requests were made during this release
audit. Keep credentials in environment variables or ignored local configuration.
For a prepared relation-agnostic native graph, set `hirag.graph_storage` to
`single`; never point it at a multigraph. The optional preparation utility is
`HiRAG-PlugRag/scripts/prepare_single_graph_from_multi.py --help`. Its projection
output deliberately omits community reports and is not query-ready until
matching clustering/report assets have been regenerated. Do not reuse stale
multigraph reports as if they belonged to the projected native graph.

Example YAML values configure the released implementation. They do not establish
the exact parameters or model endpoints of every historical answer batch. Do not
infer a judgment model from its filename or rename a batch to imply a different
model. See the release notes before describing this as a fully reproducible run.

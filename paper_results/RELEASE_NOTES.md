# Release audit boundaries

## Verified

The revised manuscript's main, controlled, cross-framework and ablation numeric
results can be recomputed from the explicitly mapped historical judgments.
This covers the experimental comparison tables and their dimension breakdown;
graph-size/connectivity statistics and qualitative case-study figures were not
recomputed because the original graph stores are outside this release bundle.
No judgment, answer, or source manuscript was rewritten to obtain agreement.
All archived answers are nonempty and have unique questions, with consistent
question order within each dataset. Numeric agreement alone is not a test of
experimental fairness, statistical significance, or full execution provenance.

## Needs author verification before public claims of full reproducibility

1. The author's latest correction on 2026-10-19 is V3 answer generation, V3
   non-cross-framework evaluation, and Pro cross-framework evaluation. This
   supersedes earlier release annotations of R1 for ordinary evaluation and Pro
   for PPR evaluation. The exact Pro service/version is not inferred. These are
   author-reported labels, not model identifiers embedded in judgment rows.
   The source Word file used for numeric extraction is unchanged and may retain
   earlier model descriptions; this correction only updates the release metadata.
2. Some updated Nano files contain `r1` in their names and historical directories
   contain `v4pro`. Those filenames and archived descriptions are retained as
   historical evidence; they do not override the latest reporting rule above.
3. The PPR index lists a Mix Full rerun answer file named
   `mix_hirag_plugrag_full_siliconflow_v3_rerun_20260730.jsonl`. That exact
   file was not located. Its comparison judgments and the PPR answers are
   preserved, but another Full batch has not been substituted as the exact input.
4. Historical judgment rows lack answer hashes, query IDs, timestamps and model
   identifiers. Candidate answers copied from a corresponding archive have not
   thereby been proven to be the exact API inputs. `candidate_answer_pairs`
   checks question order only, not that linkage.
5. Trained weights, per-run graph/index stores and original endpoint snapshots
   are not included. Regeneration may change with model services and code versions.
6. The upstream example training configuration uses early stopping 30 and
   R-GCN bases 10, whereas the manuscript reports 20 and 20 respectively.
   The example remains labelled as implementation defaults, not a recovered run
   configuration. Do not silently treat the example as proof of manuscript values.

## Code integration scope

The clean clone starts from upstream commit
`958be2303504dff2a3a413f498892beb22fa9914`.
PPR and PathRAG-style retrieval are isolated into dedicated modules and query
modes. Their function/class bodies match the local experiment code; the PPR
scoring helper import was relocated to the isolated PPR module.

The local native HiRAG context formatter and single-graph projection/storage are
included, along with Nano simple/multigraph storage compatibility. Configuration
plumbing and focused offline tests are added for the released modes. The batch
launcher now exposes the context limits and an explicit 8000-token generation
cap. This is release plumbing, not evidence that an old run used that cap.
The optional community count limit defaults to `None`, preserving token-only
behavior. Unrelated local theme-fusion, corrective-generation, claim-routing and
budget-search variants are excluded.

Two local exploratory test files were investigated but not imported into the
release test suite: `test_rgcn_expansion_order.py` expects a voting-before-cap
change not present in the active upstream expansion function;
`test_hybrid_candidate_selection.py` references helper functions absent from the
active Nano package. Their original files are unchanged. No algorithm change was
invented to satisfy these tests. This is another reason not to equate arbitrary
local tests, historical results and the current upstream code snapshot.

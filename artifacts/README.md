# Artifact Bundle

Saved traces and rescoring outputs for all paper results. Every number in the paper can be reproduced from these files without rerunning retrieval.

## Layout

```
runs/
  locomo/{lexical,allminilm,bge_m3,mxbai}/    8 native runs (4 retrievers x 2 benchmarks)
  longmemeval/{lexical,allminilm,bge_m3}/
  locomo_density/{lexical,allminilm}_{f1,f5,f8}/   density sweep (parser_max_facts 1/5/8)
  transfer/{locomo_mem0,locomo_memoryos}/       Mem0 and MemoryOS transfer runs
  beam/{1m_lexical,100k_lexical,100k_mxbai}/    BEAM stress tests

semantic_audit/
  judged_cases.csv              1,902-case LLM audit labels (supports/partial/does_not_support)
  *_audit_cases.csv             per-run disaggrement cases fed to the auditor

validation/
  semantic_validation_sample_115.csv    115-case human-annotated validation subset
  external_semantic_review_*.csv        36-case inter-annotator overlap and adjudication
  *_shared_subset_stats.json            shared-subset bootstrap statistics
  within_provider_bootstrap_stats.json  per-provider bootstrap CIs

paper_summaries/
  query_target_alignment.*       cross-run instability summaries
  target_qa_alignment.*          answer-level alignment analysis
  target_qa_manifest.*           target coverage manifest
```

## Per-Run Files

Each run directory contains:

- `trace.jsonl.gz` -- gzipped ranked output (query, memory IDs, ranks, scores). Decompress with `gzip -d`.
- `target_rescore.json` -- rescoring results under Raw, Source, and Canonical targets.
- `target_rescore.summary.txt` -- human-readable summary of per-target metrics.
- `result.summary.txt` -- original run summary.

Shared-subset variants (e.g., `trace.shared_canonical_*.jsonl.gz`) restrict to queries where all compared runs have canonical coverage.

## Reproducing

Install the package and run the scorer on any trace:

```
pip install -e .
gzip -dk artifacts/runs/locomo/lexical/trace.jsonl.gz
mtel-mem score \
  --targets examples/minimal_target_mappings.jsonl \
  --trace artifacts/runs/locomo/lexical/trace.jsonl \
  --out-json report.json
```

Run pinned validation checks against these artifacts:

```
python -c "
import json, os
os.chdir('.')
with open('validation_expectations.json') as f:
    checks = json.load(f)
# ... (see validation_expectations.json for the full check list)
"
```

Or run the test suite:

```
pip install -e .
pytest tests/ -v
```

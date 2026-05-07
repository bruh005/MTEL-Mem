# MTEL-Mem

Multi-target evaluation layer for transformed conversational memory stores. Rescores saved retrieval traces under Raw, Source, and Canonical target sets without rerunning retrieval.

## What This Repo Contains

**Scoring engine** (`src/mtel_mem/`): loads a ranked trace and a target mapping, computes nDCG/MRR/Recall under each target, and reports query-level instability.

**Benchmark adapters**: LoCoMo (built in) and LongMemEval-S (optional download).

**Paper artifacts** (`artifacts/`): gzipped traces, rescoring outputs, semantic audit labels, and validation data for every result in the paper. See [`artifacts/README.md`](artifacts/README.md) for the full layout.

## Quick Start

```bash
pip install -e .
mtel-mem init
```

To also prepare LongMemEval-S:

```bash
mtel-mem init --with-longmemeval-s
```

## Usage

Score the bundled example:

```bash
mtel-mem score-example
```

Score your own trace:

```bash
mtel-mem score \
  --targets examples/minimal_target_mappings.jsonl \
  --trace examples/minimal_ranked_trace.jsonl \
  --out-json reports/score.json
```

Validate manifests:

```bash
mtel-mem validate-manifest --manifest manifests/locomo.json
```

Run the validation suite:

```bash
mtel-mem validate-suite --out-json reports/validation_report.json
```

## Reproducing Paper Results

Every number in the paper can be verified from the bundled artifacts. Traces are stored as gzipped JSONL; decompress before scoring:

```bash
gzip -dk artifacts/runs/locomo/lexical/trace.jsonl.gz
mtel-mem score \
  --targets examples/minimal_target_mappings.jsonl \
  --trace artifacts/runs/locomo/lexical/trace.jsonl \
  --out-json report.json
```

41 pinned scalar and density-winner checks are defined in `validation_expectations.json` and verified against the bundled rescoring outputs.

## Tests

```bash
pip install -e .
pytest tests/ -v
```

## Layout

```
src/mtel_mem/
  adapters/       benchmark manifest helpers (LoCoMo, LongMemEval-S)
  core/           rescore, metrics, instability, validation
  cli.py          command-line interface
  schemas.py      shared data contracts
artifacts/        paper traces, rescores, semantic audit labels (see artifacts/README.md)
data/             LoCoMo benchmark data
manifests/        benchmark manifest definitions
examples/         minimal trace and target mapping for smoke tests
tests/            unit tests
```

# MTEL-Mem

`MTEL-Mem` is the reusable multi-target evaluation layer for the upgraded paper.

The repository is now self-contained for the built-in LoCoMo benchmark:

- `locomo10.json` ships in `data/`
- fixture/eval/stats JSON are generated locally with `mtel-mem init`
- `eval/results` are not required for normal usage

`LongMemEval-S` remains supported, but it is optional and downloaded on demand.

## Scope

Current benchmark adapters:

- `LoCoMo` (built in)
- `LongMemEval-S` (optional download)

Current layer features:

- built-in and optional benchmark manifests
- local fixture/eval generation for repository portability
- a small core scoring engine for `raw`, `source`, and `canonical` target sets
- query-level instability summaries
- schema/invariant validation, null controls, positive controls, and optional paper-regression checks
- a CLI entrypoint for manifest validation, setup, BYO-trace scoring, and a validation scorecard

## Layout

- `src/mtel_mem/`
  - `bootstrap.py`: local data preparation helpers
  - `schemas.py`: shared data contracts
  - `adapters/`: benchmark manifest helpers
  - `core/`: scoring and instability logic
  - `cli.py`: command-line interface
- `data/`
  - `locomo10.json`
  - generated fixture/eval JSON after setup
- `manifests/`
  - `locomo.json`
  - `longmemeval_s.json`
- `examples/`
  - `minimal_target_mappings.jsonl`
  - `minimal_ranked_trace.jsonl`
- `tests/`
  - unit tests for metrics, the example pipeline, manifest wiring, and portable validation behavior

## Quick Start

Install the package in editable mode:

```powershell
pip install -e .
```

Prepare the built-in LoCoMo assets:

```powershell
mtel-mem init
```

If you also want LongMemEval-S:

```powershell
mtel-mem init --with-longmemeval-s
```

Validate the bundled manifests:

```powershell
mtel-mem validate-manifest --manifest manifests/locomo.json
mtel-mem validate-manifest --manifest manifests/longmemeval_s.json
```

Run the example scorer:

```powershell
mtel-mem score-example
```

Run a bring-your-own trace:

```powershell
mtel-mem score --targets examples/minimal_target_mappings.jsonl --trace examples/minimal_ranked_trace.jsonl --out-json reports/example_score.json
```

Run the portable validation suite:

```powershell
mtel-mem validate-suite --out-json reports/validation_report.json --out-md reports/validation_report.md
```

Include the paper-only regression checks only when the external artifacts are available:

```powershell
mtel-mem validate-suite --include-paper-regression
```

Run the unit tests:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

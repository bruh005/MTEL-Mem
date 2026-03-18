# MTEL-Mem

`MTEL-Mem` is the resource layer for the upgraded paper.

It sits on top of the existing store-aware evaluation assets already in this repository and gives them a cleaner identity:

- `TIAP` is the method.
- `MTEL-Mem` is the reusable multi-target evaluation layer.

## Scope

Current benchmark adapters:

- `LoCoMo`
- `LongMemEval-S`

Current layer features:

- benchmark manifests that point to the existing benchmark JSON assets in `../eval/data/`
- a small core scoring engine for `raw`, `source`, and `canonical` target sets
- query-level instability summaries
- schema/invariant validation, null controls, positive controls, and paper-regression checks
- a CLI entrypoint for manifest validation, BYO-trace scoring, and a validation scorecard
- tests and example files for the generic input/output contract

## Layout

- `src/mtel_mem/`
  - `schemas.py`: shared data contracts
  - `adapters/`: benchmark manifest helpers
  - `core/`: scoring and instability logic
  - `cli.py`: command-line interface
- `manifests/`
  - `locomo.json`
  - `longmemeval_s.json`
- `examples/`
  - `minimal_target_mappings.jsonl`
  - `minimal_ranked_trace.jsonl`
- `tests/`
  - unit tests for metrics, the example pipeline, and manifest wiring

## Quick Start

Validate the repository-backed benchmark manifests:

```powershell
python -m mtel_mem validate-manifest --manifest MTEL-Mem/manifests/locomo.json
python -m mtel_mem validate-manifest --manifest MTEL-Mem/manifests/longmemeval_s.json
```

Run the example scorer:

```powershell
$env:PYTHONPATH = "MTEL-Mem/src"
python -m mtel_mem score-example
```

Run a bring-your-own trace with one command:

```powershell
$env:PYTHONPATH = "MTEL-Mem/src"
python -m mtel_mem score --targets MTEL-Mem/examples/minimal_target_mappings.jsonl --trace MTEL-Mem/examples/minimal_ranked_trace.jsonl --out-json MTEL-Mem/reports/example_score.json
```

Run the validation scorecard:

```powershell
$env:PYTHONPATH = "MTEL-Mem/src"
python -m mtel_mem validate-suite --out-json MTEL-Mem/reports/validation_report.json --out-md MTEL-Mem/reports/validation_report.md
```

Run the unit tests:

```powershell
$env:PYTHONPATH = "MTEL-Mem/src"
python -m unittest discover -s MTEL-Mem/tests -v
```

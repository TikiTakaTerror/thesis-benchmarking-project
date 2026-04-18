# Multi-Seed Sweeps

R6 adds deterministic multi-seed orchestration on top of the managed run system.

## What It Does

R6 can:

- execute one managed run per seed
- keep the same dataset, model family, benchmark suite, and supervision setting
- write one aggregate CSV summary
- write one detailed JSON summary with per-seed rows

## Supported Datasets

Current sweep support:

- `mnlogic`
- `synthetic_mnlogic`

## Main Entry Point

- [src/train/sweeps.py](/Users/abdullahsaeed/thesis-benchmarking-project/src/train/sweeps.py)

Main function:

- `execute_seed_sweep(...)`

## Export Files

Each sweep writes:

- `results/summaries/seed_sweep__...csv`
- `results/summaries/seed_sweep__...json`

CSV content:

- one row per aggregated metric
- columns:
  - `metric_name`
  - `count`
  - `mean`
  - `std`
  - `min`
  - `max`

JSON content:

- selection metadata
- ordered seed list
- run IDs
- aggregate metric rows
- per-seed run rows

## Executable Preset

R6 adds a real MNLogic preset:

- [src/configs/runs/r6_mnlogic_pipeline_multiseed.yaml](/Users/abdullahsaeed/thesis-benchmarking-project/src/configs/runs/r6_mnlogic_pipeline_multiseed.yaml)

It runs:

- dataset: `mnlogic`
- model family: `pipeline`
- benchmark suite: `core_eval`
- supervision: `concept_50`
- seeds: `301, 302, 303`

## Verification

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/check_seed_sweeps.py
```

Expected final line:

```text
[OK] Multi-seed orchestration check passed.
```

# rsbench External Integration

## Goal

R7 makes `rsbench` runs depend on the real local `external/rsbench-code/` checkout in a concrete way.

This phase does **not** claim that the project now executes the full official `rsseval` training stack.
What it does implement is:

- inspect the local official `rsbench-code` repository
- capture git commit, dirty state, and component availability
- discover official XOR reference models from `rsseval`
- probe whether the official `rsscount` exact counter is runnable in the current environment
- store that metadata inside every `rsbench` run
- expose numeric benchmark-environment metrics in `metrics.json`

## Implemented Components

- `src/benchmarks/rsbench_external.py`
  rsbench-code repository inspection helpers
- `src/benchmarks/rsbench.py`
  now attaches external benchmark metadata and metrics to `rsbench` runs
- `scripts/check_rsbench_external_integration.py`
  end-to-end verification for real MNLogic + backend API launch

## Stored rsbench Metadata

Each `rsbench` run now stores:

- configured `external/rsbench-code` root path
- git commit
- git dirty state
- remote URL
- presence of:
  - `rssgen`
  - `rsseval`
  - `rsscount`
- discovered official XOR reference models such as:
  - `xorcbm`
  - `xordpl`
  - `xornn`
- mapped official reference model for the selected project family where available
- `rsscount` exact-availability probe result
- benchmark-environment warnings

This metadata is written into:

- `results/runs/<run_id>/config_snapshot.yaml`

Numeric environment flags are written into:

- `results/runs/<run_id>/metrics.json`

## Stored rsbench Environment Metrics

Current metrics:

- `rsbench_external_repo_present`
- `rsbench_external_repo_dirty`
- `rsbench_external_rssgen_present`
- `rsbench_external_rsseval_present`
- `rsbench_external_rsscount_present`
- `rsbench_official_xor_model_count`
- `rsbench_reference_model_available`
- `rsbench_rsscount_exact_available`

## Current Limitation

This phase still does **not** run the full external `rsseval` model-training pipeline.

Also, in the current local macOS Python 3.10 environment, the official `rsscount` exact counter is probed but not fully available because `pyeda.inter` is not importable. The project now surfaces that limitation explicitly instead of hiding it.

## Exact Verification Steps

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/check_rsbench_external_integration.py
python scripts/check_benchmark_adapters.py
python scripts/check_real_mnlogic_run.py
```

## Expected Output

You should see output similar to:

```text
[OK] rsbench external repo detected with rssgen/rsseval/rsscount components
[OK] rsbench git metadata captured: commit=..., dirty=True
[OK] Official XOR reference models discovered: ['xorcbm', 'xordpl', 'xornn']
[OK] rsscount exact availability captured: False (...)
[OK] Real MNLogic rsbench run stored external benchmark metadata and metrics
[OK] Backend API stored external rsbench metadata for real MNLogic launch
[OK] rsbench external integration check passed.
```

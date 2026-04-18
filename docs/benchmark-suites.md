# Benchmark Suites

## Goal

Phase 13 adds the missing benchmark-suite adapter layer and introduces support for more than one suite.

Implemented suites:
- `rsbench`
- `core_eval`

The implementation stays intentionally small:
- `rsbench` provides rsbench-style ID/OOD split handling for the current synthetic flow
- `core_eval` provides a direct internal benchmark suite that uses the shared evaluator without extra external assumptions

## Implemented Components

- `src/benchmarks/base.py`
  Benchmark adapter contract and typed config parsing
- `src/benchmarks/registry.py`
  Config loading and adapter lookup
- `src/benchmarks/rsbench.py`
  rsbench-style adapter with ID and OOD evaluation splits
- `src/benchmarks/core_eval.py`
  Internal direct-evaluation benchmark adapter
- `src/configs/benchmarks/core_eval.yaml`
  New benchmark-suite config
- `scripts/check_benchmark_adapters.py`
  End-to-end verification of both suites through the managed synthetic run flow

## Current Benchmark Behavior

### `core_eval`

- train split + one `test` split
- runs the shared evaluator directly
- stores:
  - `test_*` metrics
  - `benchmark_primary_score`
  - `core_eval_primary_score`

### `rsbench`

- train split + `id` and `ood` evaluation splits
- stores:
  - `id_*` metrics
  - `ood_*` metrics
  - `id_performance`
  - `ood_performance`
  - `benchmark_primary_score`
  - `rsbench_primary_score`

## Exact Verification Steps

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/check_benchmark_adapters.py
python scripts/check_backend_api.py
python scripts/check_minimal_ui.py
```

## Expected Output

You should see output similar to:

```text
[OK] Config-backed options expose both benchmark suites
[OK] Benchmark adapters loaded and reported supported datasets
[OK] core_eval run stored direct-evaluation metrics: primary=...
[OK] rsbench run stored ID/OOD metrics: id=..., ood=...
[OK] Benchmark adapter smoke check passed.
```

## Current Limitation

Phase 13 does not integrate the real external `rsbench-code` repository yet.

What is implemented now:
- the benchmark adapter contract
- multiple suite selection
- suite-specific split planning for the current synthetic managed-run flow
- suite-specific metric computation and storage

What remains later:
- real dataset-backed rsbench execution
- additional external benchmark environments beyond the current adapter structure

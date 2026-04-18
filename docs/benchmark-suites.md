# Benchmark Suites

## Goal

Phase 13 adds the missing benchmark-suite adapter layer and introduces support for more than one suite.

Implemented suites:
- `rsbench`
- `core_eval`

The implementation stays intentionally small:
- `rsbench` provides rsbench-style ID/OOD split handling for both `mnlogic` and `kand_logic` and now inspects the local official `external/rsbench-code/` checkout
- `core_eval` provides a direct internal benchmark suite that uses the shared evaluator without extra external assumptions

## Implemented Components

- `src/benchmarks/base.py`
  Benchmark adapter contract and typed config parsing
- `src/benchmarks/registry.py`
  Config loading and adapter lookup
- `src/benchmarks/rsbench.py`
  rsbench-style adapter with ID and OOD evaluation splits
- `src/benchmarks/rsbench_external.py`
  official local rsbench-code repository inspection and metadata extraction
- `src/benchmarks/core_eval.py`
  Internal direct-evaluation benchmark adapter
- `src/configs/benchmarks/core_eval.yaml`
  New benchmark-suite config
- `scripts/check_benchmark_adapters.py`
  End-to-end verification of both suites through the managed synthetic run flow
- `scripts/check_rsbench_external_integration.py`
  end-to-end verification that rsbench runs capture external benchmark metadata

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
- inspects the local official `external/rsbench-code/` checkout
- stores:
  - `id_*` metrics
  - `ood_*` metrics
  - `id_performance`
  - `ood_performance`
  - `benchmark_primary_score`
  - `rsbench_primary_score`
  - benchmark-environment metrics such as:
    - `rsbench_external_repo_present`
    - `rsbench_official_xor_model_count`
    - `rsbench_reference_model_available`
    - `rsbench_rsscount_exact_available`

## Exact Verification Steps

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/check_benchmark_adapters.py
python scripts/check_rsbench_external_integration.py
python scripts/check_backend_api.py
python scripts/check_minimal_ui.py
```

## Expected Output

You should see output similar to:

```text
[OK] Config-backed options expose both benchmark suites
[OK] Benchmark adapters loaded and reported supported datasets
[OK] core_eval run stored direct-evaluation metrics: primary=...
[OK] rsbench run stored ID/OOD metrics: id=..., ood=..., xor_models=3
[OK] rsbench external integration check passed.
[OK] Benchmark adapter smoke check passed.
```

## Current Limitation

The project now captures the real local `rsbench-code` environment, but it still does **not** execute the full official external `rsseval` model-training stack.

What is implemented now:
- the benchmark adapter contract
- multiple suite selection
- suite-specific split planning for the current synthetic managed-run flow
- suite-specific metric computation and storage
- external benchmark-environment capture for `rsbench`

What remains later:
- full official external benchmark execution through `rsseval` and `rsscount`
- additional external benchmark environments beyond the current adapter structure

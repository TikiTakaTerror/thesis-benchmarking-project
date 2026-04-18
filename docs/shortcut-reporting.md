# Shortcut Reporting and Plots

## Goal

R8 improves the reporting layer in three ways:

- makes `rsbench` store explicit shortcut-gap metrics
- writes thesis-friendly plot assets into `results/plots/`
- shows those plots directly in the comparison and benchmark-summary pages

## Implemented Components

- `src/benchmarks/rsbench.py`
  now stores shortcut-gap and relative OOD-drop metrics
- `src/services/plots.py`
  shared plot generator for:
  - run comparisons
  - benchmark summaries
  - seed sweeps
- `src/ui/routes.py`
  now generates comparison and benchmark plots on page render
- `src/api/app.py`
  now serves `results/plots/` at `/plots`
- `scripts/check_shortcut_reporting.py`
  end-to-end R8 smoke check

## New rsbench Metrics

When both `id` and `ood` metrics are available, `rsbench` now stores:

- `rsbench_shortcut_gap`
  `id_accuracy - ood_accuracy`
- `rsbench_shortcut_relative_drop`
  `(id_accuracy - ood_accuracy) / id_accuracy`
- `rsbench_concept_gap`
  `id_concept_accuracy - ood_concept_accuracy`

These do not magically solve shortcut evaluation, but they make the current robustness signal much easier to compare across runs.

## Plot Outputs

Plots are written under:

- `results/plots/`

Current generated plots include:

- comparison score plots
- comparison robustness plots
- benchmark overview plots
- benchmark shortcut-gap plots
- seed-sweep mean/std plots

## UI Behavior

`/compare`
- still writes CSV/JSON comparison exports
- now also shows generated comparison plot images

`/benchmarks`
- now shows benchmark overview plots
- now shows mean shortcut-gap values in the grouped table

## Exact Verification Steps

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/check_shortcut_reporting.py
python scripts/check_benchmark_views.py
python scripts/check_seed_sweeps.py
```

## Expected Output

You should see output similar to:

```text
[OK] rsbench runs stored shortcut-gap metrics
[OK] Comparison page rendered and served generated plot files
[OK] Benchmark summary page rendered shortcut reporting and plot assets
[OK] Seed sweep wrote aggregate plot assets
[OK] Shortcut reporting and plots check passed.
```

## Current Limitation

The shortcut-gap metrics are only as good as the available `id` and `ood` splits.

So R8 improves visibility and reporting, but it does **not** solve the upstream MNLogic split-quality limitation discovered earlier.

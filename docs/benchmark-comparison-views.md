# Benchmark Comparison Views

## Goal

Phase 11 adds the missing stored-result views that were intentionally deferred in Phase 10:

- a run comparison page
- a benchmark summary page

The frontend remains server-rendered and minimal. These pages sit on top of the existing SQLite run registry and stored comparison exports.

## Implemented Components

- `src/services/reporting.py`
  Shared helpers for comparison tables, grouped benchmark summaries, and stable comparison export names
- `src/ui/routes.py`
  Added `GET /compare` and `GET /benchmarks`
- `src/ui/templates/compare.html`
  Run selection and comparison table
- `src/ui/templates/benchmark_summary.html`
  Grouped benchmark summary view
- `scripts/check_benchmark_views.py`
  End-to-end smoke check for the new pages

## Current UI Routes

- `GET /`
- `POST /ui/launch`
- `GET /runs/{run_id}`
- `GET /compare`
- `GET /benchmarks`
- `GET /static/styles.css`

## Comparison Page

`/compare` accepts repeated `run_id` query parameters.

Example:

```text
/compare?run_id=<run_id_1>&run_id=<run_id_2>
```

Behavior:
- compares selected runs with one shared metric set
- writes CSV and JSON comparison exports into `results/summaries/`
- writes PNG plot assets into `results/plots/`
- keeps the view aligned with the existing backend comparison export flow

## Benchmark Summary Page

`/benchmarks` groups stored runs by:

- benchmark suite
- dataset
- model family
- supervision

Each row shows:
- run counts
- completed and failed counts
- best and mean task metrics
- mean concept accuracy
- mean shortcut gap when available
- mean runtime
- latest run link
- compare shortcut when at least two completed runs exist in the same group

## Exact Verification Steps

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/check_benchmark_views.py
python scripts/check_minimal_ui.py
```

Optional manual server run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Then open:

- `http://127.0.0.1:8000/compare`
- `http://127.0.0.1:8000/benchmarks`

## Expected Output

You should see output similar to:

```text
[OK] Synthetic runs created for comparison and benchmark summary checks
[OK] Empty comparison page rendered
[OK] Comparison page rendered and export files were written
[OK] Benchmark summary page rendered grouped results
[OK] Benchmark comparison views smoke check passed.
```

## Current Limitation

The benchmark summary page reflects only the currently stored runs.

Today that means:
- synthetic launch runs dominate the visible data unless you add real managed runs later
- only `rsbench` is configured as a benchmark suite so far

The grouping logic is still useful now because it is already aligned with the future multi-suite extension path.

# Run Management

## Goal

Phase 8 adds run lifecycle management and result storage.

It provides:
- one SQLite-backed run registry
- one per-run folder under `results/runs/`
- config snapshots, metrics, artifact metadata, and event logs
- comparison exports in `results/summaries/`

It does not provide:
- backend API endpoints
- frontend result pages
- dataset-to-tensor production pipelines for real MNLogic yet

## Storage Design

Phase 8 uses two storage layers together:

- filesystem artifacts
  - `results/runs/<run_id>/config_snapshot.yaml`
  - `results/runs/<run_id>/metadata.json`
  - `results/runs/<run_id>/metrics.json`
  - `results/runs/<run_id>/artifacts.json`
  - `results/runs/<run_id>/events.jsonl`
- SQLite registry
  - `results/experiment_registry.sqlite3`

This combination keeps the implementation simple:
- SQLite is good for listing and filtering runs later
- JSON and YAML are good for transparent inspection and thesis reproducibility

## Implemented Components

- `src/services/config.py`
  Typed loading of the base project config and resolved storage paths
- `src/services/run_manager.py`
  Run creation, status updates, metric persistence, registry listing, and comparison export
- `src/train/runner.py`
  Minimal managed run execution helper

## Current Run Lifecycle

The Phase 8 lifecycle is:

1. create run folder and registry row
2. save config snapshot
3. mark run as `running`
4. execute training and evaluation
5. save checkpoint and metrics
6. mark run as `completed` or `failed`
7. refresh `results/summaries/run_registry.csv` and `run_registry.json`

## Exact Verification Steps

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/check_run_management.py
python scripts/check_environment.py
```

## Expected Output

You should see output similar to:

```text
[OK] Project config loaded: thesis-benchmarking-project
[OK] SQLite registry path: /Users/abdullahsaeed/thesis-benchmarking-project/results/experiment_registry.sqlite3
[OK] Completed managed run: ...
[OK] Completed managed run: ...
[OK] Listed completed runs: 2
[OK] Registry CSV updated: /Users/abdullahsaeed/thesis-benchmarking-project/results/summaries/run_registry.csv
[OK] Comparison CSV written: /Users/abdullahsaeed/thesis-benchmarking-project/results/summaries/phase8_smoke_comparison.csv
[OK] Run management smoke check passed.
```

Exact run IDs differ because they are timestamped.

## Expected Artifacts

After verification, you should have:

- `results/experiment_registry.sqlite3`
- at least two new folders under `results/runs/`
- `results/summaries/run_registry.csv`
- `results/summaries/run_registry.json`
- `results/summaries/phase8_smoke_comparison.csv`
- `results/summaries/phase8_smoke_comparison.json`

# Minimal Frontend

## Goal

Phase 10 added the minimal frontend foundation.

It provides:
- one server-rendered dashboard at `/`
- one stored-run detail page at `/runs/{run_id}`
- one launch form for the synthetic managed-run flow or the prepared real MNLogic dataset
- a recent-runs table and small metric cards

Since Phase 11, the same frontend also exposes:
- a run comparison page at `/compare`
- a benchmark summary page at `/benchmarks`

It still does not provide:
- Kand-Logic launch from the UI yet

## Implemented Components

- `src/ui/routes.py`
  Server-rendered UI routes
- `src/ui/templates/`
  Jinja templates for the dashboard and run detail page
- `src/ui/static/styles.css`
  Minimal styling
- `scripts/check_minimal_ui.py`
  End-to-end UI smoke check with `TestClient`

## Current UI Routes

- `GET /`
- `POST /ui/launch`
- `GET /runs/{run_id}`
- `GET /compare`
- `GET /benchmarks`
- `GET /static/styles.css`

## Exact Verification Steps

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/check_minimal_ui.py
python scripts/check_benchmark_views.py
```

Optional manual server run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Then open:

- `http://127.0.0.1:8000/`

## Expected Output

You should see output similar to:

```text
[OK] Dashboard page rendered
[OK] Static stylesheet served
[OK] Launch form redirected to /runs/...
[OK] Run detail page rendered for ...
[OK] Dashboard listed the launched run
[OK] Minimal frontend smoke check passed.
```

## Current Limitation

The frontend can now launch real MNLogic runs, but the underlying prepared dataset still carries the upstream rsbench XOR split warning exposed in `source_info.json`.

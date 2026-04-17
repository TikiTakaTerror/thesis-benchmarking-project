# Backend API

## Goal

Phase 9 added the backend API.

Since Phase 10, the same FastAPI app also serves the minimal frontend.
This document still focuses only on the API routes.

It provides:
- health and option endpoints
- run listing and run detail endpoints
- stored snapshot access for config, metadata, metrics, and artifacts
- run comparison endpoint
- synchronous synthetic run launch endpoint for backend verification

It does not provide:
- background job execution
- real dataset-to-training launch for MNLogic yet

## Implemented Components

- `src/api/app.py`
  FastAPI application and Phase 9 routes
- `src/api/schemas.py`
  Request and response models
- `src/services/catalog.py`
  Config-backed option discovery
- `src/train/synthetic.py`
  Synthetic managed-run launcher shared by the API and smoke checks

## Current Endpoint Surface

- `GET /api/v1/health`
- `GET /api/v1/options`
- `GET /api/v1/runs`
- `GET /api/v1/runs/{run_id}`
- `GET /api/v1/runs/{run_id}/snapshot/{snapshot_type}`
- `POST /api/v1/runs/compare`
- `POST /api/v1/runs/launch/synthetic`

## Exact Verification Steps

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/check_backend_api.py
```

Optional manual server run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Optional manual endpoint checks:

```bash
curl http://127.0.0.1:8000/api/v1/health
curl http://127.0.0.1:8000/api/v1/options
```

## Expected Output

You should see output similar to:

```text
[OK] Health endpoint responded for phase 9
[OK] Options endpoint returned 3 model families
[OK] Synthetic launch completed: ...
[OK] Synthetic launch completed: ...
[OK] Run listing endpoint returned launched runs
[OK] Run detail endpoint returned ...
[OK] Metrics snapshot endpoint returned stored metrics
[OK] Compare endpoint wrote: /Users/abdullahsaeed/thesis-benchmarking-project/results/summaries/phase9_api_smoke_comparison.csv
[OK] Backend API smoke check passed.
```

## Current Limitation

The launch endpoint is intentionally limited to synthetic managed runs in Phase 9.

Reason:
- the real dataset-to-training execution flow is not wired yet
- this still gives a working backend contract for the later frontend phase without pretending the full experiment engine is already complete

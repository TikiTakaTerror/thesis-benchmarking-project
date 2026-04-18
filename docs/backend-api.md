# Backend API

## Goal

Phase 9 added the backend API.

Since Phase 10, the same FastAPI app also serves the minimal frontend.
Since Phase 11, that same app also serves the comparison and benchmark summary pages.
This document still focuses only on the API routes.

It provides:
- health and option endpoints
- run listing and run detail endpoints
- stored snapshot access for config, metadata, metrics, and artifacts
- run comparison endpoint
- synchronous synthetic run launch endpoint for backend verification
- synchronous real MNLogic run launch endpoint backed by the prepared dataset

It does not provide:
- background job execution
- Kand-Logic launch yet

## Implemented Components

- `src/api/app.py`
  FastAPI application and Phase 9 routes
- `src/api/schemas.py`
  Request and response models
- `src/services/catalog.py`
  Config-backed option discovery
- `src/train/synthetic.py`
  Synthetic managed-run launcher shared by the API and smoke checks
- `src/train/real_data.py`
  Real MNLogic managed-run launcher with dataset-aware runtime config alignment

## Current Endpoint Surface

- `GET /api/v1/health`
- `GET /api/v1/options`
- `GET /api/v1/runs`
- `GET /api/v1/runs/{run_id}`
- `GET /api/v1/runs/{run_id}/snapshot/{snapshot_type}`
- `POST /api/v1/runs/compare`
- `POST /api/v1/runs/launch/synthetic`
- `POST /api/v1/runs/launch/mnlogic`

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
[OK] Health endpoint responded for phase 11
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

The API can now launch real MNLogic runs, but the current prepared dataset still carries the upstream rsbench XOR split warning:

- `val` is single-class
- `test` is single-class
- `ood` is single-class

The API exposes those warnings instead of hiding them.

# Real MNLogic Managed Runs

This document describes the first real dataset-backed execution path added after the original synthetic-only system.

## What R4 Adds

R4 makes `mnlogic` runnable through the managed run stack:

- real prepared MNLogic batches are loaded from `data/processed/mnlogic/`
- the managed runner builds dataset-aware runtime model configs for:
  - `pipeline`
  - `ltn`
  - `deepproblog`
- the backend API can launch a real MNLogic run through:
  - `POST /api/v1/runs/launch/mnlogic`
- the minimal frontend can now launch:
  - `mnlogic`
  - `synthetic_mnlogic`

## Important Limitation

The current real MNLogic dataset comes from the official `rsbench-code/rssgen` XOR generator output converted in R2.

That upstream raw output currently has a data-quality problem:

- `val` is single-class
- `test` is single-class
- `ood` is single-class

This is recorded in:

- [data/processed/mnlogic/metadata/source_info.json](/Users/abdullahsaeed/thesis-benchmarking-project/data/processed/mnlogic/metadata/source_info.json)

R4 does **not** hide that problem. Real runs still work, but the warning is preserved in the stored config snapshot and the API response.

## Runtime Config Alignment

The static model YAML files still contain smoke-test concept names.

R4 does not overwrite those files globally. Instead it builds a runtime config from the real dataset metadata:

- concepts become `a`, `b`, `c`
- labels come from the prepared label schema
- the logic expression is read from `source_info.json`
- family-specific logic sections are rewritten at run time

This keeps the earlier synthetic smoke checks stable while making real MNLogic runs semantically correct.

## Managed Run Sources

Synthetic path:

- `src/train/synthetic.py`

Real MNLogic path:

- `src/train/real_data.py`

## Verification

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/check_real_mnlogic_run.py
```

Expected final line:

```text
[OK] Real MNLogic managed-run check passed.
```

## Current Scope

R4 enables:

- real MNLogic managed training/evaluation
- real MNLogic backend launch
- real MNLogic UI launch

R4 does not yet solve:

- the upstream rsbench XOR split imbalance
- concept-supervision percentage regimes like `0% / 50% / 100%`
- multi-seed sweep orchestration
- Kand-Logic real-data execution

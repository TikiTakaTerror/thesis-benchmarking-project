# Environment Setup

## Goal

Set up a local Python environment for the backend-first experiment system without installing the external neuro-symbolic repositories yet.

Phase 1 covers:
- Python interpreter target
- Python package installation
- local virtual environment creation
- environment smoke-check verification

Phase 1 does not cover:
- dataset downloads
- `rsbench-code` checkout
- later neuro-symbolic model implementation
- model training

## Python Version

Use Python `3.10.18`.

Reason:
- it is available in the current machine environment
- it is a safer compatibility target for planned PyTorch plus later neuro-symbolic libraries than the system `python3` version `3.14.3`

## Files Added In This Phase

- `.python-version`
- `requirements.txt`
- `requirements-dev.txt`
- `scripts/bootstrap_env.sh`
- `scripts/check_environment.py`

## Exact Setup Commands

Run these commands from the project root:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt
python scripts/check_environment.py
```

## One-Command Alternative

If you want the scripted version instead:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
./scripts/bootstrap_env.sh
```

## Expected Smoke-Check Output

The final command should end with output similar to this:

```text
[OK] Python version: 3.10.x
[OK] Imported torch
[OK] Imported torchvision
[OK] Imported deepproblog
[OK] Imported problog
[OK] Imported numpy
[OK] Imported pandas
[OK] Imported sklearn
[OK] Imported yaml
[OK] Imported pydantic
[OK] Imported fastapi
[OK] Imported uvicorn
[OK] Imported jinja2
[OK] Imported matplotlib
[OK] Imported tqdm
[OK] Required path exists: external/rsbench-code
[OK] Required path exists: external/deepproblog
[OK] Required path exists: external/LTNtorch
[OK] Required path exists: src/configs
[OK] Required path exists: results/runs
[OK] Environment check passed.
```

Version numbers may differ slightly within the allowed ranges.

## If `python3.10` Is Missing

Run:

```bash
python3.10 --version
```

If that command fails on your machine, stop before Phase 2 and tell me exactly what this command prints. I will then give you platform-specific installation steps for Python 3.10.

## External Dependency Status

Do not clone or download anything into these folders yet:
- `/Users/abdullahsaeed/thesis-benchmarking-project/external/rsbench-code`
- `/Users/abdullahsaeed/thesis-benchmarking-project/external/deepproblog`
- `/Users/abdullahsaeed/thesis-benchmarking-project/external/LTNtorch`

Those paths remain placeholders in the repository layout. Later phases use pinned Python packages for DeepProbLog and LTNtorch instead of local checkouts.

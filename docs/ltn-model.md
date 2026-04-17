# LTN Model

## Goal

Phase 6 implements Family C only:
- shared neural encoder
- concept prediction head
- label prediction head
- LTNtorch logical satisfaction as differentiable supervision
- final label prediction influenced by logic scores

It does not implement:
- DeepProbLog
- benchmark storage
- frontend integration

## Dependency Choice

This phase uses the official `LTNtorch` PyPI package, pinned to `1.0.2`, instead of cloning the placeholder `external/LTNtorch/` directory.

## Implemented Components

- `src/models/ltn_model/config.py`
  Typed config for concepts, labels, logic formulas, and training defaults
- `src/models/ltn_model/model.py`
  Real `LTNModelAdapter` with:
  - `train(...)`
  - `predict(...)`
  - `predict_concepts(...)`
  - `evaluate(...)`
  - `save_checkpoint(...)`
  - `load_checkpoint(...)`

## Logic Design

The current LTN setup uses:
- learnable concept probabilities from the concept head
- learnable label probabilities from the label head
- LTN predicates grounded on those probabilities
- universal formulas aggregated with `AggregPMeanError`
- overall logical satisfaction aggregated with `SatAgg`

Training uses:
- task label supervision
- concept supervision
- logical satisfaction loss

Final prediction uses:
- neural label probabilities
- logic-derived label scores
- a configurable blend between them

## Exact Verification Steps

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/check_ltn_model.py
python scripts/check_model_foundations.py
```

## Expected Output

You should see output similar to:

```text
[OK] Loaded LTN config with 3 concepts and 2 labels
[OK] Synthetic LTN dataset created: train=96, val=32
[OK] Training completed
[OK] Validation label accuracy: ...
[OK] Validation concept accuracy: ...
[OK] Train logic satisfaction: ...
[OK] Checkpoint reload matched predictions.
[OK] LTN smoke check passed.
```

The exact metric values can differ, but the script should finish successfully and the validation metrics should be high.

## Expected Artifact

The verification script writes:

- `/Users/abdullahsaeed/thesis-benchmarking-project/results/runs/phase6_ltn_smoke/ltn_checkpoint.pt`

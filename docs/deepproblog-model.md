# DeepProbLog Model

## Goal

Phase 7 implements Family B only:
- shared neural encoder
- one neural predicate per concept
- exact probabilistic logic inference with DeepProbLog
- final label prediction produced by the logic program

It does not implement:
- run storage
- backend API
- frontend integration

## Dependency Choice

This phase uses the official `deepproblog` PyPI package, pinned to `2.0.6`, instead of cloning the placeholder `external/deepproblog/` directory.

## Implemented Components

- `src/models/deepproblog_model/config.py`
  Typed config for concepts, labels, logic-program settings, and training defaults
- `src/models/deepproblog_model/model.py`
  Real `DeepProbLogModelAdapter` with:
  - `train(...)`
  - `predict(...)`
  - `predict_concepts(...)`
  - `evaluate(...)`
  - `save_checkpoint(...)`
  - `load_checkpoint(...)`

## Logic Design

The current DeepProbLog setup uses:
- a shared CNN encoder
- one binary neural predicate per concept
- an exact DeepProbLog program that maps concept predicates to the final label
- label training through the logic-program probability of the correct label
- optional concept supervision through direct concept logits

The current Phase 7 implementation supports a binary label setup:
- one positive label defined by a ProbLog rule
- one negative label defined as the logical complement of the positive label

## Exact Verification Steps

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/check_deepproblog_model.py
python scripts/check_model_foundations.py
```

## Expected Output

You should see output similar to:

```text
[OK] Loaded DeepProbLog config with 3 concepts and 2 labels
[OK] Synthetic DeepProbLog dataset created: train=48, val=16
[OK] Training completed
[OK] Validation label accuracy: 1.0000
[OK] Validation concept accuracy: 1.0000
[OK] Validation rule satisfaction rate: 1.0000
[OK] Checkpoint reload matched predictions.
[OK] DeepProbLog smoke check passed.
```

The exact metric values can differ slightly, but the script should finish successfully and the validation metrics should stay high.

## Expected Artifact

The verification script writes:

- `/Users/abdullahsaeed/thesis-benchmarking-project/results/runs/phase7_deepproblog_smoke/deepproblog_checkpoint.pt`

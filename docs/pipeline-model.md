# Custom Pipeline Model

## Goal

Phase 4 implements Family A only:
- shared neural encoder
- concept prediction head
- symbolic rule executor
- final label prediction from concepts and rules

It does not implement:
- the common evaluation engine
- the LTN family
- the DeepProbLog family
- frontend or API wiring

## Implemented Components

- `src/logic/soft_rules.py`
  Differentiable soft-logic executor with `and`, `or`, and `not`
- `src/models/pipeline/config.py`
  Typed config for concepts, labels, rules, and training defaults
- `src/models/pipeline/model.py`
  Real pipeline adapter with:
  - `train(...)`
  - `predict(...)`
  - `predict_concepts(...)`
  - `evaluate(...)`
  - `save_checkpoint(...)`
  - `load_checkpoint(...)`

## Training Input Contract

The pipeline adapter currently trains on batches of tensors with this shape:

```python
{
    "images": torch.Tensor,          # [batch, channels, height, width]
    "label_ids": torch.Tensor,       # [batch]
    "concept_targets": torch.Tensor  # [batch, num_concepts], optional
}
```

This keeps the phase bounded. The real dataset-to-tensor training wiring will be added later.

## Rule Execution Design

Two symbolic paths are used:
- soft path
  Concept probabilities from `sigmoid(concept_logits)` are sent through soft logic rules for differentiable label supervision
- hard path
  Concept probabilities are thresholded and then passed through the same rules for final symbolic predictions

Soft operators:
- `not(x) = 1 - x`
- `and(x1, x2, ..., xn) = product(xi)`
- `or(x1, x2, ..., xn) = 1 - product(1 - xi)`

## Exact Verification Steps

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/check_pipeline_model.py
```

## Expected Output

You should see output similar to:

```text
[OK] Loaded pipeline config with 3 concepts and 2 labels
[OK] Synthetic dataset created: train=96, val=32
[OK] Training completed
[OK] Validation label accuracy: ...
[OK] Validation concept accuracy: ...
[OK] Checkpoint reload matched predictions.
[OK] Pipeline smoke check passed.
```

The exact loss values can differ, but the script should finish successfully and write a checkpoint artifact.

## Expected Artifact

The verification script writes:

- `/Users/abdullahsaeed/thesis-benchmarking-project/results/runs/phase4_pipeline_smoke/pipeline_checkpoint.pt`


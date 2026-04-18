# MNLogic Tensor Loading

## Goal

R3 makes the prepared real MNLogic dataset consumable by the current model-family code.

This phase adds:
- real image loading from `data/processed/mnlogic/`
- resizing into the shared encoder input shape
- conversion into PyTorch tensors
- split-specific `DataLoader` objects that already match the current model batch contract

This phase does **not** yet:
- launch managed real MNLogic runs
- replace the API/UI synthetic launch path
- fix dataset-aware model logic configuration

## What It Loads

The loader reads the prepared-manifest dataset:

```text
data/processed/mnlogic/
  images/
  metadata/
  splits/
```

Each sample becomes a dictionary with:

```python
{
    "images": torch.Tensor,        # [C, H, W]
    "label_ids": torch.Tensor,     # scalar long
    "concept_targets": torch.Tensor,
    "sample_id": str,
    "image_path": str,
    "split_name": str,
}
```

The default `DataLoader` then batches those fields into the same structure already
accepted by the model families.

## Current Important Limitation

The real prepared MNLogic dataset currently exposes concept names:

- `a`
- `b`
- `c`

The current smoke-test model-family configs still use:

- `has_triangle`
- `is_red`
- `count_is_two`

So after R3:
- the real data can be loaded correctly
- but the model configs are still semantically mismatched

That alignment is the next real-run wiring task.

## Exact Verification Steps

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/check_mnlogic_tensor_loading.py
```

## Expected Output

You should see output similar to:

```text
[OK] Built real MNLogic dataloaders for model family: pipeline
[OK] Tensor config: channels=3, input_size=(64, 64), batch_size=16
[OK] Dataset concept names: ['a', 'b', 'c']
[OK] Model concept names: ['has_triangle', 'is_red', 'count_is_two']
[WARN] Dataset and model concept names do not currently align. This is expected after R2 and will be fixed during the real-run wiring phase.
[OK] Train batch image shape: (8, 3, 64, 64)
[OK] Train batch label shape: (8,)
[OK] Train batch concept shape: (8, 3)
[OK] Train batch pixel range: min=..., max=...
[OK] Test batch image shape: (8, 3, 64, 64)
[OK] OOD batch image shape: (8, 3, 64, 64)
[OK] MNLogic tensor loading check passed.
```

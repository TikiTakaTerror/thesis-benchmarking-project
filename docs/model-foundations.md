# Model Foundations

## Goal

Phase 3 adds the shared model foundation only.

It provides:
- a common model adapter interface
- a reusable shared encoder implementation
- reusable prediction heads
- generic checkpoint helpers for shared neural components
- family adapter stubs for the three planned model families

It does not provide:
- training loops
- symbolic rule execution
- LTN constraint optimization
- DeepProbLog inference
- benchmark evaluation

## Important Design Decision

The project-level model interface is implemented as an adapter object, not as the raw `torch.nn.Module` interface.

Reason:
- PyTorch already uses `module.train()` to switch layers like dropout and batch norm into training mode
- this project also needs a higher-level `train(...)` method meaning “run the experiment training procedure”
- keeping that method on an adapter avoids a naming collision and keeps the experiment API clear

## Shared Encoder

Phase 3 adds one baseline shared encoder:
- `small_cnn`

Default config shape:

```yaml
shared_encoder:
  name: small_cnn
  pretrained: false
  input_channels: 3
  input_size: [64, 64]
  conv_channels: [32, 64]
  feature_dim: 128
  dropout: 0.1
```

The encoder outputs one feature vector per image with size `feature_dim`.

## Common Heads

Phase 3 adds one reusable head module:
- `PredictionHead`

It can be used for:
- concept logits
- label logits
- neural predicate logits in later integrations

## Family Adapter Status

Added stubs:
- `PipelineModelAdapter`
- `LTNModelAdapter`
- `DeepProbLogModelAdapter`

These stubs already share the same high-level interface, but their methods intentionally raise `NotImplementedError` until the correct implementation phases.

## Exact Verification Steps

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/check_model_foundations.py
```

## Expected Output

You should see output similar to:

```text
[OK] Loaded model config: pipeline
[OK] Shared encoder built: small_cnn
[OK] Encoder output shape: (2, 128)
[OK] Concept logits shape: (2, 3)
[OK] Label logits shape: (2, 2)
[OK] Checkpoint roundtrip matched.
[OK] Registered adapter: pipeline -> PipelineModelAdapter
[OK] Registered adapter: ltn -> LTNModelAdapter
[OK] Registered adapter: deepproblog -> DeepProbLogModelAdapter
[OK] Model foundations check passed.
```

The trainable parameter count may also be printed and can vary if the config changes later.

## Smoke-Check Artifact

The verification script writes a temporary checkpoint here:

- `/Users/abdullahsaeed/thesis-benchmarking-project/results/runs/phase3_shared_encoder_smoke/shared_components.pt`

That file is expected and is part of the verification flow.


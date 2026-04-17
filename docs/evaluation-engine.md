# Evaluation Engine

## Goal

Phase 5 adds the shared evaluation layer.

It provides:
- common task metrics
- common concept metrics
- semantic consistency metrics
- lightweight control metrics
- split-aware evaluation for ID and OOD-like batches

It does not provide:
- benchmark storage or run registry
- plots or comparison dashboards
- ablation or intervention tooling

## Implemented Metrics

Task view:
- `accuracy`
- `macro_f1`

Concept view:
- `concept_accuracy`
- `concept_macro_f1`
- `exact_concept_vector_match`

Semantic view:
- `rule_satisfaction_rate`
- `violation_rate`
- `concept_label_consistency`

Control view:
- `parameter_count`
- `num_examples`
- `num_batches`
- `evaluation_time_seconds`
- `seed` when provided

Loss metrics when logits and targets are available:
- `loss`
- `label_loss`
- `concept_loss`

## Current Assumption

The common evaluator currently expects tensor batches with:

```python
{
    "images": torch.Tensor,
    "label_ids": torch.Tensor,
    "concept_targets": torch.Tensor,  # optional
}
```

This matches the current pipeline smoke-check setup and is enough for Phase 5.

## Exact Verification Steps

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/check_evaluation_engine.py
```

## Expected Output

You should see output similar to:

```text
[OK] Pipeline model trained for evaluation smoke check
[OK] id accuracy: ...
[OK] id macro_f1: ...
[OK] id concept_accuracy: ...
[OK] id rule_satisfaction_rate: ...
[OK] ood accuracy: ...
[OK] ood macro_f1: ...
[OK] Evaluation engine smoke check passed.
```

Exact metric values can differ, but the script should finish successfully and print both ID and OOD-prefixed metrics.


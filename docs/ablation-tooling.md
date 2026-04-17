# Ablation And Intervention Tooling

## Goal

Phase 12 adds evaluator-driven tooling for:

- symbolic-layer ablation analysis
- concept intervention analysis

The implementation stays backend-first. There is no new frontend surface in this phase.

## Implemented Metrics

When supported by the model family and when concept targets are available, the shared evaluator now adds:

- `symbolic_layer_ablated_accuracy`
- `symbolic_layer_ablated_macro_f1`
- `symbolic_layer_ablation_gain`
- `concept_intervention_accuracy`
- `concept_intervention_macro_f1`
- `concept_intervention_gain`

## Current Phase 12 Coverage

Concept intervention:
- `pipeline`: supported
- `ltn`: supported
- `deepproblog`: supported

Symbolic-layer ablation:
- `pipeline`: not exposed in Phase 12 because the current Family A implementation does not include a separate non-symbolic label head
- `ltn`: supported through the raw neural label head
- `deepproblog`: not exposed in Phase 12 because the current Family B implementation does not include a separate non-symbolic label head

This is explicit by design. The tooling does not fake a non-symbolic baseline when the current family implementation does not actually have one.

## Implementation Notes

- `src/models/base.py`
  Optional adapter methods for symbolic ablation and concept intervention
- `src/eval/analysis.py`
  Shared metric computation for ablation and intervention analysis
- `src/eval/engine.py`
  Integrates the new analysis metrics into the common evaluator
- model-family adapters
  Add family-specific prediction hooks for intervened concepts and, when available, non-symbolic label prediction

## Exact Verification Steps

Run:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/check_ablation_tooling.py
python scripts/check_evaluation_engine.py
python scripts/check_run_management.py
```

## Expected Output

You should see output similar to:

```text
[OK] Pipeline stored concept intervention metrics: gain=...
[OK] LTN stored ablation and intervention metrics: ablation_gain=..., intervention_gain=...
[OK] DeepProbLog stored concept intervention metrics: gain=...
[OK] Optional symbolic ablation metrics stay absent for unsupported families
[OK] Ablation and intervention tooling smoke check passed.
```

The exact numeric values can differ, but the script should finish successfully and the named metrics should be present in stored runs.

## Manual Inspection

After running the script, inspect the newest run metrics files:

```bash
find results/runs -maxdepth 2 -name metrics.json | sort | tail -n 3
```

You should see new metrics such as:
- `test_concept_intervention_gain`
- `test_concept_intervention_accuracy`
- `test_symbolic_layer_ablation_gain` for the LTN run

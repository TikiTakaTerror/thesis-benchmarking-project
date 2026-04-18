# Architecture Overview

## Goal

This project is an experiment-first platform for comparing three neuro-symbolic model families under a shared protocol:

1. Custom concept-first symbolic pipeline
2. DeepProbLog-based model
3. LTNtorch-based model

The core system is the backend experiment engine. The frontend is intentionally minimal and only exists to launch runs and inspect results.

## Design Principles

- Keep the experiment engine primary and the UI secondary.
- Start with MNLogic before adding Kand-Logic.
- Use shared interfaces for datasets, models, benchmarks, and evaluation so families remain comparable.
- Keep the implementation simple, readable, and student-friendly.
- Add new model families and datasets through adapters, not by rewriting the training pipeline.

## Planned Stack

- Backend language: Python
- Deep learning framework: PyTorch
- Backend API: FastAPI
- Frontend: minimal server-rendered UI or small API-driven UI inside the same codebase
- Result storage: per-run artifacts in `results/runs/`, with lightweight structured summaries and optional SQLite indexing later

## High-Level System Flow

1. A run configuration selects dataset, model family, benchmark suite, supervision setting, and seed.
2. A dataset adapter exposes standard train/validation/test/OOD splits and schema access.
3. A model adapter builds a family-specific model around a comparable shared encoder policy.
4. Training and evaluation use common services where possible.
5. The evaluation layer computes task, concept, semantic, and control metrics in a shared format.
6. Results are stored with config snapshots and artifacts so runs can be compared later.

## Planned Interfaces

### Dataset Adapter

- `load_train_split(...)`
- `load_val_split(...)`
- `load_test_split(...)`
- `load_ood_split(...)`
- `get_concept_schema(...)`
- `get_label_schema(...)`

### Model Adapter

- Adapter note: this project-level interface is intentionally not the raw `torch.nn.Module` API, because PyTorch already uses `train()` to toggle training mode. Family adapters will wrap neural modules and expose experiment-oriented methods.
- `train(...)`
- `predict(...)`
- `predict_concepts(...)`
- `evaluate(...)`
- `save_checkpoint(...)`
- `load_checkpoint(...)`

## Shared Encoder Policy

- A family adapter may use different reasoning machinery, but all families should be able to use the same shared encoder specification for fair comparison.
- Phase 3 adds a simple `small_cnn` encoder as the default shared encoder baseline.
- Future family implementations should load encoder settings from the same config shape so encoder capacity stays aligned across experiments.

### Benchmark Suite Adapter

- `list_datasets()`
- `prepare_dataset(...)`
- `run_evaluation(...)`
- `compute_suite_specific_metrics(...)`

## Repository Structure

```text
project/
  external/
    rsbench-code/
    deepproblog/
    LTNtorch/
  src/
    data/
    models/
      pipeline/
      deepproblog_model/
      ltn_model/
    logic/
    train/
    eval/
    benchmarks/
    services/
    api/
    ui/
    utils/
    configs/
  results/
    runs/
    summaries/
    plots/
  scripts/
  docs/
```

## Folder Responsibilities

- `external/`: third-party libraries or benchmark environments placed locally in predictable paths.
- `data/`: local raw and prepared dataset storage used by dataset adapters and validation scripts.
- `src/data/`: dataset adapters and dataset preparation code.
- `src/data/loaders.py`: Phase R3 real image-to-tensor and DataLoader helpers for prepared datasets such as MNLogic.
- `src/models/`: model-family implementations behind a shared interface.
- `src/models/shared_encoder.py`: reusable shared encoder implementation and config parsing.
- `src/models/heads.py`: reusable prediction heads for concept and label logits.
- `src/models/base.py`: common adapter contract used by all model families.
- `src/models/pipeline/`: the custom concept-first symbolic pipeline family.
- `src/models/deepproblog_model/`: the DeepProbLog-backed model family.
- `src/models/ltn_model/`: the LTNtorch-backed model family.
- `src/logic/`: symbolic rules, logic templates, and logic utilities shared across families.
- `src/train/`: training orchestration, loops, checkpoint handling, and run execution helpers.
- `src/train/runner.py`: Phase 8 managed-run execution helper that ties training, evaluation, and artifact persistence together.
- `src/eval/`: metric computation and evaluation flows.
- `src/eval/engine.py`: split-aware common evaluation runner for shared metrics.
- `src/eval/analysis.py`: Phase 12 ablation and intervention analysis over model predictions and intervened concepts.
- `src/benchmarks/`: benchmark suite adapters such as `rsbench` and `core_eval`.
- `src/benchmarks/base.py`: shared benchmark adapter contract and typed config parsing.
- `src/benchmarks/registry.py`: adapter lookup and config loading for benchmark suites.
- `src/services/`: application services that coordinate configs, storage, and run metadata.
- `src/services/config.py`: typed loading of the base project config and resolved storage paths.
- `src/services/run_manager.py`: SQLite-backed run registry plus per-run filesystem storage helpers.
- `src/api/`: backend API layer for run control and results access.
- `src/api/app.py`: FastAPI application exposing run listing, run detail, comparison, synthetic launch, and the mounted server-rendered UI.
- `src/ui/`: minimal user-facing interface, kept intentionally small.
- `src/ui/routes.py`: server-rendered dashboard, run-detail, comparison, and benchmark summary pages mounted into the FastAPI app.
- `src/services/reporting.py`: thin view-focused reporting helpers for comparison tables and grouped benchmark summaries.
- `src/utils/`: shared utility helpers that do not belong to the other modules.
- `src/configs/`: configuration placeholders for datasets, models, supervision modes, benchmark suites, and run presets.
- `results/`: run artifacts, summaries, and plots.
- `scripts/`: utility scripts for setup and maintenance.
- `docs/`: architecture notes, progress tracking, and reproducibility guidance.

## Phase Boundaries

- Phase 0: structure, docs, config placeholders
- Phase 1: environment and dependency setup
- Phase 2: dataset infrastructure for MNLogic first, using a simple prepared-manifest format
- Phase 3: shared encoder and common model interfaces, without family-specific training logic
- Phase 4: custom concept-first symbolic pipeline with soft-logic training and hard symbolic prediction
- Phase 5: evaluation engine and shared metric computation
- Phase 6: LTNtorch integration with logical satisfaction loss and logic-influenced prediction
- Phase 7: DeepProbLog integration with neural predicates and exact probabilistic logic inference
- Phase 8: run management and result storage with SQLite registry, per-run artifacts, and comparison exports
- Phase 9: backend API over the run registry with synthetic launch support
- Phase 10: minimal server-rendered frontend over the existing backend API and run registry
- Phase 11: comparison views and grouped benchmark summaries over stored runs
- Phase 12: ablation and intervention tooling integrated into the shared evaluator
- Phase 13: benchmark-suite adapter support for `rsbench` and an internal `core_eval` suite
- Phase 14: final cleanup, reproducibility scripts, and handoff documentation

## Phase 0 Outcome

Phase 0 intentionally stops before any executable experiment logic. The repository is prepared so later phases can add functionality without reshaping the project layout.

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

- `train(...)`
- `predict(...)`
- `predict_concepts(...)`
- `evaluate(...)`
- `save_checkpoint(...)`
- `load_checkpoint(...)`

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
- `src/models/`: model-family implementations behind a shared interface.
- `src/logic/`: symbolic rules, logic templates, and logic utilities shared across families.
- `src/train/`: training orchestration, loops, checkpoint handling, and run execution helpers.
- `src/eval/`: metric computation and evaluation flows.
- `src/benchmarks/`: benchmark suite adapters such as rsbench.
- `src/services/`: application services that coordinate configs, storage, and run metadata.
- `src/api/`: backend API layer for later run control and results access.
- `src/ui/`: minimal user-facing interface, kept intentionally small.
- `src/utils/`: shared utility helpers that do not belong to the other modules.
- `src/configs/`: configuration placeholders for datasets, models, supervision modes, benchmark suites, and run presets.
- `results/`: run artifacts, summaries, and plots.
- `scripts/`: utility scripts for setup and maintenance.
- `docs/`: architecture notes, progress tracking, and reproducibility guidance.

## Phase Boundaries

- Phase 0: structure, docs, config placeholders
- Phase 1: environment and dependency setup
- Phase 2: dataset infrastructure for MNLogic first, using a simple prepared-manifest format
- Phase 3: shared encoder and common model interfaces
- Phase 4: custom symbolic pipeline
- Phase 5: evaluation engine
- Phase 6: LTNtorch integration
- Phase 7: DeepProbLog integration
- Phase 8: run management and result storage
- Phase 9: backend API
- Phase 10: minimal frontend
- Phase 11+: comparison views, ablations, extra benchmark suites, cleanup

## Phase 0 Outcome

Phase 0 intentionally stops before any executable experiment logic. The repository is prepared so later phases can add functionality without reshaping the project layout.

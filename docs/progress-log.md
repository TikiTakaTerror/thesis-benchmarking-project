# Progress Log

This file is updated phase by phase so the project history stays explicit and reproducible.

## Phase Status

| Phase | Name | Status |
| --- | --- | --- |
| 0 | Planning and repository skeleton | Completed |
| 1 | Environment and dependency setup | Completed |
| 2 | Dataset infrastructure | Completed |
| 3 | Shared encoder and common model interfaces | Completed |
| 4 | Custom concept-first symbolic pipeline | Completed |
| 5 | Evaluation engine and metric computation | Completed |
| 6 | LTNtorch integration | Completed |
| 7 | DeepProbLog integration | Completed |
| 8 | Run management and result storage | Completed |
| 9 | Backend API | Pending |
| 10 | Minimal frontend | Pending |
| 11 | Benchmark comparison views | Pending |
| 12 | Ablation and intervention tooling | Pending |
| 13 | Additional benchmark suites | Pending |
| 14 | Cleanup, documentation, reproducibility | Pending |

## 2026-04-17 - Phase 0

Goal:
- establish the repository skeleton
- define the architecture direction
- add configuration placeholders
- avoid implementation work

Completed:
- created the main folder structure for code, configs, external dependencies, results, scripts, and docs
- added a short architecture document
- added placeholder configs for datasets, model families, supervision modes, benchmark suites, and a sample run preset
- added placeholder README files for artifact and external dependency directories

Notes:
- no dataset, training, evaluation, or UI logic has been implemented yet
- the project is ready for Phase 1 environment setup after user verification

## 2026-04-17 - Phase 1

Goal:
- define a reproducible Python environment target
- add dependency files for the backend-first experiment system
- add a simple bootstrap path and smoke-test verification
- avoid dataset or model implementation work

Completed:
- selected Python 3.10 as the project interpreter target for compatibility with planned neuro-symbolic dependencies
- added `requirements.txt` and `requirements-dev.txt`
- added `scripts/bootstrap_env.sh` to create a local virtual environment and install dependencies
- added `scripts/check_environment.py` to verify imports and required project directories
- added `docs/environment-setup.md` with exact commands and expected outputs

Notes:
- external repositories such as `rsbench-code`, `DeepProbLog`, and `LTNtorch` are still placeholders in this phase
- no training, dataset loading, or evaluation logic has been added yet

## 2026-04-17 - Phase 2

Goal:
- implement the dataset infrastructure with MNLogic first
- define a simple prepared dataset layout that later datasets can reuse
- add validation and demo scripts so the dataset layer is verifiable now
- avoid model or training work

Completed:
- added a common dataset adapter interface and typed dataset records
- added a prepared-manifest dataset adapter that loads schemas and split CSVs
- added an `MNLogicDatasetAdapter` and a registry-based dataset factory
- added local dataset storage folders under `data/raw` and `data/processed`
- updated the MNLogic config to point to an explicit prepared dataset layout
- added `scripts/create_mnlogic_demo_dataset.py` to generate a tiny verification dataset
- added `scripts/check_mnlogic_dataset.py` to validate a prepared MNLogic dataset and print split counts
- added documentation for the expected dataset layout and verification steps

Notes:
- the real MNLogic download and conversion flow is not implemented yet
- the adapter currently targets a simple prepared format based on schema JSON plus split CSV manifests
- this keeps later Kand-Logic support straightforward because it can reuse the same contract

## 2026-04-17 - Phase 3

Goal:
- add a shared encoder foundation for all model families
- define a common model adapter interface
- add reusable prediction heads and generic checkpoint helpers
- avoid implementing family-specific training or reasoning logic

Completed:
- added a common model adapter contract and shared `ModelOutputs` structure
- added a `small_cnn` shared encoder with config parsing and parameter counting helpers
- added reusable prediction heads for concept and label logits
- added generic module-bundle checkpoint helpers for shared components
- added registry helpers to load model configs and instantiate family adapter stubs
- added family adapter stubs for the pipeline, LTN, and DeepProbLog families
- updated model configs with explicit shared encoder defaults
- added `scripts/check_model_foundations.py` to verify config loading, forward shapes, adapter registration, and checkpoint roundtrip
- added documentation for the Phase 3 model foundation layer

Notes:
- the family adapters are still stubs and intentionally raise `NotImplementedError` for training and reasoning methods
- actual pipeline logic is deferred to Phase 4, LTN integration to Phase 6, and DeepProbLog integration to Phase 7

## 2026-04-17 - Phase 4

Goal:
- implement the custom concept-first symbolic pipeline
- connect the shared encoder to a concept head and symbolic rule executor
- support batch-level training, prediction, evaluation, and checkpointing
- avoid touching the LTN and DeepProbLog families

Completed:
- added a reusable soft-logic rule executor with `and`, `or`, and `not`
- added typed pipeline config parsing for concepts, labels, symbolic rules, and training defaults
- replaced the pipeline family stub with a real `PipelineModelAdapter`
- implemented differentiable soft-rule supervision and hard symbolic prediction
- implemented pipeline-specific `train(...)`, `predict(...)`, `predict_concepts(...)`, `evaluate(...)`, `save_checkpoint(...)`, and `load_checkpoint(...)`
- updated the pipeline model config with example concepts, labels, rules, and working smoke-test defaults
- added `scripts/check_pipeline_model.py` to train the pipeline end to end on a synthetic concept dataset and verify checkpoint reload
- added `docs/pipeline-model.md` with the exact Phase 4 verification flow

Notes:
- the pipeline currently trains on tensor batches, not yet directly from the dataset adapter
- the common evaluation engine is still deferred to Phase 5
- the LTN and DeepProbLog families remain untouched in this phase

## 2026-04-17 - Phase 5

Goal:
- add a shared evaluation engine for common metrics
- compute task, concept, semantic, and control metrics in one place
- support split-aware evaluation for ID and OOD-style batches
- keep the implementation family-agnostic where possible

Completed:
- added shared evaluation tensor containers and metric helpers under `src/eval/`
- implemented common task metrics including `accuracy` and `macro_f1`
- implemented concept metrics including `concept_accuracy`, `concept_macro_f1`, and `exact_concept_vector_match`
- implemented semantic metrics including `rule_satisfaction_rate`, `violation_rate`, and `concept_label_consistency`
- implemented control metrics including `parameter_count`, `num_examples`, `num_batches`, and `evaluation_time_seconds`
- added split-aware evaluation helpers for evaluating multiple named splits such as ID and OOD
- updated the pipeline model to use the common evaluation engine
- added `scripts/check_evaluation_engine.py` to verify the evaluator on synthetic ID and OOD-like data
- preserved compatibility with the existing pipeline smoke-check output

Notes:
- run storage, summaries, and comparison tables are still deferred to Phase 8
- ablation and intervention metrics are still deferred to Phase 12

## 2026-04-17 - Phase 6

Goal:
- integrate the LTNtorch model family
- keep the shared encoder policy aligned with the other families
- use logical constraints as differentiable supervision
- make final prediction influenced by logical satisfaction

Completed:
- installed and pinned the official `LTNtorch` dependency
- added typed LTN model-family config parsing for concepts, labels, logic rules, formulas, and training defaults
- replaced the LTN family stub with a real `LTNModelAdapter`
- implemented concept and label neural heads on top of the shared encoder
- implemented LTN predicates, connectives, quantification, and satisfaction aggregation using the real `ltn` package
- implemented LTN-specific training with task loss, concept loss, and logical satisfaction loss
- implemented logic-influenced final prediction by blending neural label probabilities with logic-derived label scores
- implemented checkpoint save/load for the LTN family
- added `scripts/check_ltn_model.py` for end-to-end synthetic verification
- added `docs/ltn-model.md` with exact installation and verification instructions

Notes:
- this phase uses the official PyPI `LTNtorch` package instead of the placeholder `external/LTNtorch/` folder
- DeepProbLog remains deferred to Phase 7

## 2026-04-17 - Phase 7

Goal:
- integrate the DeepProbLog model family
- keep the shared encoder policy aligned with the other families
- use neural predicates and a probabilistic logic program for final inference
- keep the implementation compatible with the shared training and evaluation interfaces

Completed:
- installed and pinned the official `deepproblog` dependency
- added typed DeepProbLog model-family config parsing for concepts, labels, logic-program settings, and training defaults
- replaced the DeepProbLog family stub with a real `DeepProbLogModelAdapter`
- implemented one shared encoder plus one binary neural predicate head per concept
- implemented exact probabilistic logic inference with the real `deepproblog` package
- implemented DeepProbLog-specific training with logic-program label loss and optional concept supervision
- implemented checkpoint save/load for the DeepProbLog family
- added `scripts/check_deepproblog_model.py` for end-to-end synthetic verification
- added `docs/deepproblog-model.md` with exact installation and verification instructions

Notes:
- this phase uses the official PyPI `deepproblog` package instead of the placeholder `external/deepproblog/` folder
- all three planned model families now have working Phase-level implementations

## 2026-04-17 - Phase 8

Goal:
- add run lifecycle management around the existing model families
- persist config snapshots, metrics, artifacts, and status in a structured way
- keep result inspection and comparison simple for later API and UI phases
- avoid building the API or frontend yet

Completed:
- added typed project config loading for resolved storage paths and registry settings
- added a SQLite-backed run registry under `results/experiment_registry.sqlite3`
- added per-run filesystem storage with `config_snapshot.yaml`, `metadata.json`, `metrics.json`, `artifacts.json`, and `events.jsonl`
- added run listing and comparison export helpers
- added a minimal training runner that executes one managed run and persists its outputs
- added `scripts/check_run_management.py` for end-to-end verification of run creation, storage, listing, and comparison export
- added `docs/run-management.md` with exact verification steps
- added an executable Phase 8 smoke run config under `src/configs/runs/phase8_pipeline_storage_smoke.yaml`

Notes:
- Phase 8 intentionally uses synthetic tensor batches for verification because the real dataset-to-training pipeline is not wired yet
- API access and frontend views over these stored results remain deferred to later phases

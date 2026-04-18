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
| 9 | Backend API | Completed |
| 10 | Minimal frontend | Completed |
| 11 | Benchmark comparison views | Completed |
| 12 | Ablation and intervention tooling | Completed |
| 13 | Additional benchmark suites | Completed |
| 14 | Cleanup, documentation, reproducibility | Completed |

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

## 2026-04-17 - Phase 9

Goal:
- add a backend API over the existing run registry and stored artifacts
- expose minimal run-control and result-inspection endpoints
- keep the launch flow honest by using only synthetic managed runs for now
- avoid building the frontend yet

Completed:
- added a FastAPI application with health, option, run list, run detail, snapshot, compare, and synthetic launch endpoints
- added pydantic request and response schemas for the backend API
- added config-backed option discovery for datasets, models, benchmarks, supervision settings, and run presets
- added a reusable synthetic managed-run launcher shared by the API
- added `scripts/check_backend_api.py` for end-to-end backend verification with `TestClient`
- added `docs/backend-api.md` with exact verification and manual server commands
- added `httpx` to the dev dependencies so `TestClient` works in the local environment

Notes:
- the Phase 9 launch endpoint is intentionally limited to synthetic managed runs
- real dataset-backed launch remains deferred until the training/data orchestration is wired more completely

## 2026-04-17 - Phase 10

Goal:
- add a minimal frontend on top of the backend API and run registry
- keep the UI secondary to the experiment system
- expose launch controls, recent runs, and run detail pages
- avoid building comparison pages yet

Completed:
- added a server-rendered dashboard at `/`
- added a run-detail page at `/runs/{run_id}`
- added a launch form for the current synthetic managed-run flow
- added static styling and Jinja templates for the minimal UI
- mounted the UI and static files into the existing FastAPI application
- added `scripts/check_minimal_ui.py` for end-to-end frontend verification
- added `docs/minimal-frontend.md` with exact verification and manual server steps
- added `python-multipart` as a runtime dependency for HTML form submission

## 2026-04-17 - Phase 11

Goal:
- add the missing stored-result comparison views
- keep the frontend server-rendered and minimal
- expose one simple run comparison page and one grouped benchmark summary page
- avoid ablation or intervention tooling in this phase

Completed:
- added a thin reporting helper layer for UI-facing comparison and grouped benchmark summaries
- added a run comparison page at `/compare`
- added a benchmark summary page at `/benchmarks`
- updated the dashboard so runs can be selected and compared directly from the recent-runs table
- updated the top navigation to include comparison and benchmark pages
- added `scripts/check_benchmark_views.py` for end-to-end verification of the new pages and comparison export generation
- added `docs/benchmark-comparison-views.md` with exact verification and manual browser steps

Notes:
- the comparison page still compares only already stored runs and uses a fixed shared metric set
- the benchmark summary page is already grouped by benchmark suite, dataset, model family, and supervision, even though only `rsbench` is configured so far
- real dataset-backed UI launch remains deferred and is unchanged in this phase

## 2026-04-17 - Phase 12

Goal:
- add symbolic-layer ablation tooling where the family exposes a non-symbolic label path
- add concept intervention tooling across the implemented model families
- integrate these metrics into the shared evaluator and managed run flow
- avoid additional benchmark-suite work in this phase

Completed:
- extended the model adapter base with optional symbolic ablation and concept intervention hooks
- added family-specific concept intervention support for the pipeline, LTN, and DeepProbLog families
- added symbolic-layer ablation support for the LTN family through its raw neural label head
- added `src/eval/analysis.py` for shared ablation and intervention metric computation
- integrated the new metrics into the common evaluation engine so managed runs store them automatically
- added `scripts/check_ablation_tooling.py` for end-to-end verification through the managed-run path
- added `docs/ablation-tooling.md` with exact verification and manual inspection steps

Notes:
- Phase 12 keeps symbolic-layer ablation explicit instead of inventing a fake non-symbolic baseline for families that do not currently expose one
- concept intervention metrics are available only when evaluation batches include `concept_targets`
- no new frontend pages were added in this phase

## 2026-04-17 - Phase 13

Goal:
- add a real benchmark-suite adapter layer
- support more than one benchmark suite in the current system
- make benchmark suite selection change evaluation split behavior and stored suite metrics
- avoid final cleanup work in this phase

Completed:
- added a shared benchmark adapter contract and typed benchmark config parsing
- added a benchmark adapter registry under `src/benchmarks/registry.py`
- implemented `rsbench` as an ID/OOD-style benchmark adapter for the current synthetic managed-run flow
- implemented `core_eval` as an internal benchmark suite that uses the shared evaluator directly
- updated synthetic managed runs so benchmark suite selection changes the prepared evaluation splits and stored suite metrics
- added `src/configs/benchmarks/core_eval.yaml`
- added `scripts/check_benchmark_adapters.py` for end-to-end verification of both suites
- added `docs/benchmark-suites.md` with exact verification instructions

Notes:
- Phase 13 does not integrate the real external `rsbench-code` repository yet
- the current implementation uses the benchmark adapter layer to drive the synthetic managed-run path, which keeps the system testable now without pretending the external benchmark environment is already wired

## 2026-04-18 - Phase 14

Goal:
- clean up the repository handoff surface
- add one final full-project verification path
- add one reproducibility snapshot export path
- finish the top-level documentation so the project is rerunnable without phase-by-phase reconstruction

Completed:
- added `scripts/check_project_ready.py` to run the full smoke-check suite from one command
- added `scripts/export_repro_snapshot.py` to export a timestamped reproducibility snapshot into `results/summaries/`
- added `docs/reproducibility.md` with exact final verification and snapshot export instructions
- refreshed the top-level `README.md` into a final handoff-oriented quickstart
- updated architecture and script documentation for the final project state

Notes:
- the repository now has a single-command final verification path and a single-command reproducibility export path
- the remaining limitations are about real-data and real external benchmark execution, not about the synthetic verification stack or the implemented comparison framework

Notes:
- the frontend intentionally launches the synthetic managed-run flow only
- run comparison and benchmark summary pages remain deferred to Phase 11

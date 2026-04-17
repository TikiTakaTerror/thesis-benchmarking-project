# Progress Log

This file is updated phase by phase so the project history stays explicit and reproducible.

## Phase Status

| Phase | Name | Status |
| --- | --- | --- |
| 0 | Planning and repository skeleton | Completed |
| 1 | Environment and dependency setup | Completed |
| 2 | Dataset infrastructure | Completed |
| 3 | Shared encoder and common model interfaces | Pending |
| 4 | Custom concept-first symbolic pipeline | Pending |
| 5 | Evaluation engine and metric computation | Pending |
| 6 | LTNtorch integration | Pending |
| 7 | DeepProbLog integration | Pending |
| 8 | Run management and result storage | Pending |
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


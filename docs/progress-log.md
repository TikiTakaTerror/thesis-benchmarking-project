# Progress Log

This file is updated phase by phase so the project history stays explicit and reproducible.

## Phase Status

| Phase | Name | Status |
| --- | --- | --- |
| 0 | Planning and repository skeleton | Completed |
| 1 | Environment and dependency setup | Completed |
| 2 | Dataset infrastructure | Pending |
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


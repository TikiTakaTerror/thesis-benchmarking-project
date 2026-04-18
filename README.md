# Thesis Benchmarking Project

Experiment-first platform for a Bachelor's thesis comparing selected neuro-symbolic AI architecture families under one evaluation protocol.

## Implemented Scope

- MNLogic-first dataset infrastructure with a prepared-dataset contract
- shared encoder and common model adapter interface
- Family A: custom concept-first symbolic pipeline
- Family B: DeepProbLog-based model
- Family C: LTNtorch-based model
- shared evaluation engine with task, concept, semantic, control, ablation, and intervention metrics
- managed runs with SQLite-backed result storage
- backend API
- minimal server-rendered frontend
- run comparison and benchmark summary pages
- benchmark-suite adapter support for `core_eval` and `rsbench`
- reproducibility snapshot export and final project-wide verification

## Current Limits

- real dataset-backed managed runs are not fully wired yet
- real external benchmark execution through `external/rsbench-code/` is not integrated yet
- final thesis-specific plot/report generation is still lightweight

## Fast Start

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt
python scripts/check_project_ready.py
python scripts/export_repro_snapshot.py
```

## Most Important Docs

- [docs/architecture.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/architecture.md)
- [docs/reproducibility.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/reproducibility.md)
- [docs/environment-setup.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/environment-setup.md)
- [docs/dataset-setup.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/dataset-setup.md)
- [docs/evaluation-engine.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/evaluation-engine.md)
- [docs/run-management.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/run-management.md)
- [docs/backend-api.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/backend-api.md)
- [docs/minimal-frontend.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/minimal-frontend.md)
- [docs/benchmark-comparison-views.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/benchmark-comparison-views.md)
- [docs/ablation-tooling.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/ablation-tooling.md)
- [docs/benchmark-suites.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/benchmark-suites.md)
- [docs/progress-log.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/progress-log.md)

## Final Verification

Run:

```bash
source .venv/bin/activate
python scripts/check_project_ready.py
```

Expected final line:

```text
[OK] Full project verification passed.
```

## Reproducibility Export

Run:

```bash
source .venv/bin/activate
python scripts/export_repro_snapshot.py
```

This writes a timestamped JSON snapshot under `results/summaries/` containing environment, package, git, config, and option metadata for thesis handoff and appendix use.

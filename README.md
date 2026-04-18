# Thesis Benchmarking Project

Experiment-first platform for a Bachelor's thesis comparing selected neuro-symbolic AI architecture families under one evaluation protocol.

## Implemented Scope

- MNLogic-first dataset infrastructure with a prepared-dataset contract
- real MNLogic prepared-dataset conversion and real managed-run support
- real Kand-Logic prepared-dataset conversion and real managed-run support
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
- official local `rsbench-code` environment capture for `rsbench` runs
- config-driven supervision modes for `label_only`, `concept_50`, and `full`
- multi-seed managed-run orchestration with aggregate CSV/JSON seed-sweep summaries
- shortcut-gap reporting and generated plot assets under `results/plots/`
- reproducibility snapshot export and final project-wide verification

## Current Limits

- the real MNLogic dataset is wired into managed runs, but its current upstream rsbench XOR `val/test/ood` splits are degenerate and carry explicit warnings
- the real Kand-Logic dataset is wired into managed runs, but DeepProbLog exact inference on Kand is substantially slower than MNLogic and currently needs smaller smoke-test limits
- the local official `external/rsbench-code/` checkout is now inspected and attached to `rsbench` runs, but the full external `rsseval` training stack is still not executed from this project
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
- [docs/real-mnlogic-runs.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/real-mnlogic-runs.md)
- [docs/kand-logic-integration.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/kand-logic-integration.md)
- [docs/supervision-modes.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/supervision-modes.md)
- [docs/multi-seed-sweeps.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/multi-seed-sweeps.md)
- [docs/rsbench-external-integration.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/rsbench-external-integration.md)
- [docs/shortcut-reporting.md](/Users/abdullahsaeed/thesis-benchmarking-project/docs/shortcut-reporting.md)
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

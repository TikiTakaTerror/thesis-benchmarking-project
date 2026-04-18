# Reproducibility

## Goal

Phase 14 closes the project with:

- one final end-to-end verification command
- one reproducibility snapshot export command
- one clear handoff path for thesis use

This phase does not add new experiment features.

## Final Verification

Run the full project smoke-check suite:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/check_project_ready.py
```

This runs:
- environment verification
- demo MNLogic dataset creation and validation
- shared model foundations
- pipeline, LTN, and DeepProbLog family smoke checks
- evaluation engine
- run management
- backend API
- minimal frontend
- comparison views
- ablation tooling
- benchmark adapter support

Expected final line:

```text
[OK] Full project verification passed.
```

## Reproducibility Snapshot

Export one structured snapshot of the current environment and repository state:

```bash
cd /Users/abdullahsaeed/thesis-benchmarking-project
source .venv/bin/activate
python scripts/export_repro_snapshot.py
```

Expected output:

```text
[OK] Reproducibility snapshot written: /Users/abdullahsaeed/thesis-benchmarking-project/results/summaries/repro_snapshot__...
[OK] Git commit: ...
[OK] Project phase captured: 14
```

The snapshot JSON records:
- project config
- Python version and executable
- platform information
- installed package versions
- available datasets, model families, supervision settings, and benchmark suites
- git commit, branch, and dirty status
- recommended verification commands

## Recommended Thesis Artifacts To Keep

For any thesis run or appendix bundle, keep:
- the exported reproducibility snapshot JSON from `results/summaries/`
- the corresponding per-run `config_snapshot.yaml`
- the corresponding per-run `metrics.json`
- the corresponding per-run `artifacts.json`
- `results/summaries/run_registry.csv`

## Manual Steps For Real Experiments

This repository is now reproducible for the implemented smoke-test and synthetic benchmark flow.

For real thesis experiments, you still need to provide:
- real MNLogic prepared data under:
  - `/Users/abdullahsaeed/thesis-benchmarking-project/data/processed/mnlogic/`
- optional Kand-Logic prepared data under:
  - `/Users/abdullahsaeed/thesis-benchmarking-project/data/processed/kand_logic/`
- any future real external benchmark environment under:
  - `/Users/abdullahsaeed/thesis-benchmarking-project/external/rsbench-code/`

## Current Known Limits

Still not fully implemented:
- real dataset-backed external benchmark execution through `rsbench-code`
- a complete real-data training orchestration path from prepared MNLogic/Kand-Logic into managed runs
- thesis-specific plots and final report asset generation beyond the current run and summary files

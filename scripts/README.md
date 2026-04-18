# Scripts Directory

This directory is reserved for small utility scripts such as:
- environment bootstrap
- dataset preparation helpers
- repeatable run commands

Current scripts:
- `bootstrap_env.sh`: create `.venv`, install dependencies, and run the environment smoke check
- `check_environment.py`: verify Python version, import core dependencies, and confirm required project directories exist
- `create_mnlogic_demo_dataset.py`: generate a tiny prepared MNLogic-style dataset for infrastructure verification
- `check_mnlogic_dataset.py`: validate a prepared MNLogic dataset layout and print split-level summary information
- `prepare_mnlogic_from_rsbench.py`: convert raw official rsbench MNLogic output into the local prepared-manifest dataset format
- `check_model_foundations.py`: verify shared encoder config loading, forward-pass shapes, adapter registration, and checkpoint roundtrip
- `check_pipeline_model.py`: train the custom symbolic pipeline on synthetic data, validate it, and verify checkpoint reload
- `check_evaluation_engine.py`: train the pipeline model, evaluate ID and OOD-like splits, and verify the shared metric engine
- `check_ltn_model.py`: train the LTN model family on synthetic data, validate logic-guided learning, and verify checkpoint reload
- `check_deepproblog_model.py`: train the DeepProbLog model family on synthetic data, validate exact logic-guided prediction, and verify checkpoint reload
- `check_run_management.py`: execute two managed synthetic runs, store them in the SQLite registry and per-run folders, and verify comparison exports
- `check_backend_api.py`: verify the FastAPI backend endpoints, including synthetic launch, run detail, snapshot access, and comparison
- `check_minimal_ui.py`: verify the server-rendered frontend dashboard, launch form, run-detail page, and static stylesheet
- `check_benchmark_views.py`: verify the run comparison page, benchmark summary page, and comparison export generation
- `check_ablation_tooling.py`: verify concept intervention metrics for all families and symbolic-layer ablation metrics where the family exposes a non-symbolic label path
- `check_benchmark_adapters.py`: verify the benchmark adapter registry, the new `core_eval` suite, and rsbench-style ID/OOD metric storage
- `check_project_ready.py`: run the full end-to-end smoke-check suite used for the final Phase 14 handoff
- `export_repro_snapshot.py`: export a timestamped reproducibility snapshot with environment, package, git, config, and option metadata

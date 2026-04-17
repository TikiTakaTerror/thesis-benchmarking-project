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
- `check_model_foundations.py`: verify shared encoder config loading, forward-pass shapes, adapter registration, and checkpoint roundtrip
- `check_pipeline_model.py`: train the custom symbolic pipeline on synthetic data, validate it, and verify checkpoint reload
- `check_evaluation_engine.py`: train the pipeline model, evaluate ID and OOD-like splits, and verify the shared metric engine
- `check_ltn_model.py`: train the LTN model family on synthetic data, validate logic-guided learning, and verify checkpoint reload

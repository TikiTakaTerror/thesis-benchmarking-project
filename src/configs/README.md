# Config Layout

Configuration files are grouped by concern so runs can stay explicit and comparable.

- `base.yaml`: project-level defaults
- `datasets/`: dataset-specific placeholders
- `models/`: model-family placeholders
- `benchmarks/`: benchmark suite placeholders
- `supervision/`: supervision settings used by the managed-run layer
- `runs/`: example composed run presets

These files started as placeholders in Phase 0.

Current status:
- `base.yaml` is active for storage and backend API configuration
- some run presets are executable synthetic smoke presets
- dataset and benchmark configs still remain partially placeholder-driven until later phases wire the full experiment engine
- supervision configs are active after R5:
  - `label_only` = 0% concept supervision
  - `concept_50` = 50% concept supervision
  - `full` = 100% concept supervision

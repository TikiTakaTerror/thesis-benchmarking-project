# Results Directory

Experiment outputs will be stored here.

Current layout:
- `runs/`: per-run artifacts and config snapshots
- `summaries/`: aggregated tables or comparison exports
- `plots/`: generated comparison figures
- `experiment_registry.sqlite3`: SQLite registry for run metadata and stored metrics

Phase 8 wires the run registry and per-run artifact folders.

#!/usr/bin/env python3
"""Verify multi-seed orchestration and aggregate summary exports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services import load_project_config
from src.services.runtime import get_run_manager
from src.train import execute_seed_sweep


RUN_CONFIG_PATH = (
    PROJECT_ROOT / "src" / "configs" / "runs" / "r6_mnlogic_pipeline_multiseed.yaml"
)


def main() -> int:
    project_config = load_project_config()
    run_manager = get_run_manager()
    run_config = _load_yaml(RUN_CONFIG_PATH)

    print(f"[OK] Loaded seed-sweep preset: {RUN_CONFIG_PATH.name}")

    result = execute_seed_sweep(
        run_manager,
        project_config=project_config,
        dataset=str(run_config["selection"]["dataset"]),
        model_family=str(run_config["selection"]["model_family"]),
        benchmark_suite=str(run_config["selection"]["benchmark_suite"]),
        supervision=str(run_config["selection"]["supervision"]),
        seeds=[int(seed) for seed in run_config["seeds"]],
        sweep_name=str(run_config["run"]["name"]),
        training_overrides=dict(run_config["training"]),
        limit_per_split=int(run_config["data"]["limit_per_split"]),
        metric_names=list(run_config["metrics"]),
    )

    if len(result.records) != 3:
        _fail(f"Expected 3 seeded runs, got {len(result.records)}")
    if not Path(result.csv_path).exists() or not Path(result.json_path).exists():
        _fail("Seed-sweep exports were not written")
    if not result.plot_paths:
        _fail("Seed sweep did not report any generated plot paths")
    for plot_path in result.plot_paths:
        if not Path(plot_path).exists():
            _fail(f"Seed-sweep plot path does not exist: {plot_path}")
    print(
        f"[OK] Seed sweep executed 3 managed runs: "
        f"{[record.selection.seed for record in result.records]}"
    )

    summary_payload = _load_json(Path(result.json_path))
    if summary_payload["selection"]["seeds"] != [301, 302, 303]:
        _fail("Seed sweep JSON export stored unexpected seeds")
    aggregate_rows = summary_payload.get("aggregate_rows", [])
    if not aggregate_rows:
        _fail("Seed sweep JSON export is missing aggregate rows")

    aggregate_by_metric = {
        row["metric_name"]: row
        for row in aggregate_rows
        if isinstance(row, dict) and "metric_name" in row
    }
    for metric_name in (
        "benchmark_primary_score",
        "test_accuracy",
        "test_concept_accuracy",
        "run_runtime_seconds",
    ):
        if metric_name not in aggregate_by_metric:
            _fail(f"Seed sweep aggregate rows are missing {metric_name}")

    concept_accuracy_row = aggregate_by_metric["test_concept_accuracy"]
    if int(concept_accuracy_row["count"]) != 3:
        _fail("Seed sweep concept-accuracy aggregate should have count=3")
    print(
        "[OK] Aggregate JSON summary stored mean/std rows for the requested metrics"
    )

    if len(summary_payload.get("run_rows", [])) != 3:
        _fail("Seed sweep JSON export is missing per-seed rows")
    print(
        f"[OK] Seed sweep JSON written: {result.json_path}"
    )
    print(
        f"[OK] Seed sweep CSV written: {result.csv_path}"
    )
    print("[OK] Seed sweep plot assets were written.")
    print("[OK] Multi-seed orchestration check passed.")
    return 0


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        _fail(f"Invalid YAML config: {path}")
    return payload


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        _fail(f"Invalid JSON summary: {path}")
    return payload


def _fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())

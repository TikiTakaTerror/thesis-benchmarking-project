#!/usr/bin/env python3
"""Verify the Phase 11 comparison and benchmark summary pages."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api import create_app
from src.services.reporting import build_comparison_export_basename
from src.services.runtime import get_project_config, get_run_manager
from src.train.synthetic import execute_synthetic_managed_run


def main() -> int:
    project_config = get_project_config()
    run_manager = get_run_manager()

    created_runs = []
    created_runs.append(
        execute_synthetic_managed_run(
            run_manager,
            project_config=project_config,
            model_family="pipeline",
            seed=31,
            supervision="full",
            run_name="phase11_compare_pipeline_full_seed_31",
            total_samples=32,
            train_size=24,
            training_overrides={
                "epochs": 3,
                "learning_rate": 0.003,
                "concept_loss_weight": 6.0,
            },
        ).record
    )
    created_runs.append(
        execute_synthetic_managed_run(
            run_manager,
            project_config=project_config,
            model_family="pipeline",
            seed=37,
            supervision="full",
            run_name="phase11_compare_pipeline_full_seed_37",
            total_samples=32,
            train_size=24,
            training_overrides={
                "epochs": 3,
                "learning_rate": 0.003,
                "concept_loss_weight": 6.0,
            },
        ).record
    )
    created_runs.append(
        execute_synthetic_managed_run(
            run_manager,
            project_config=project_config,
            model_family="pipeline",
            seed=41,
            supervision="label_only",
            run_name="phase11_compare_pipeline_label_only_seed_41",
            total_samples=32,
            train_size=24,
            training_overrides={
                "epochs": 3,
                "learning_rate": 0.003,
                "concept_loss_weight": 6.0,
            },
        ).record
    )
    print("[OK] Synthetic runs created for comparison and benchmark summary checks")

    client = TestClient(create_app())

    empty_compare = client.get("/compare")
    _require_ok(empty_compare, "empty compare page")
    if "Run Comparison" not in empty_compare.text:
        _fail("Compare page is missing the main heading")
    print("[OK] Empty comparison page rendered")

    compare_ids = [created_runs[0].run_id, created_runs[1].run_id]
    compare_response = client.get(
        "/compare",
        params=[("run_id", run_id) for run_id in compare_ids],
    )
    _require_ok(compare_response, "comparison page")
    compare_html = compare_response.text
    if created_runs[0].run_name not in compare_html or created_runs[1].run_name not in compare_html:
        _fail("Comparison page did not render both selected runs")
    if "Test Accuracy" not in compare_html:
        _fail("Comparison page is missing comparison metrics")
    if "Shortcut Gap" not in compare_html:
        _fail("Comparison page is missing the shortcut-gap metric")
    if "/plots/" not in compare_html:
        _fail("Comparison page did not render generated plot images")

    export_stem = build_comparison_export_basename(compare_ids)
    export_csv = PROJECT_ROOT / "results" / "summaries" / f"{export_stem}.csv"
    export_json = PROJECT_ROOT / "results" / "summaries" / f"{export_stem}.json"
    if not export_csv.exists() or not export_json.exists():
        _fail("Comparison export files were not written")
    print("[OK] Comparison page rendered and export files were written")

    benchmark_response = client.get("/benchmarks")
    _require_ok(benchmark_response, "benchmark summary page")
    benchmark_html = benchmark_response.text
    if "Benchmark Summary" not in benchmark_html:
        _fail("Benchmark summary page is missing the main heading")
    if "synthetic_mnlogic" not in benchmark_html:
        _fail("Benchmark summary page did not render the dataset grouping")
    if "label_only" not in benchmark_html:
        _fail("Benchmark summary page did not render the supervision grouping")
    if "Compare Recent" not in benchmark_html:
        _fail("Benchmark summary page did not render the compare shortcut")
    if "Mean Shortcut Gap" not in benchmark_html:
        _fail("Benchmark summary page did not render the shortcut-gap column")
    if "/plots/" not in benchmark_html:
        _fail("Benchmark summary page did not render generated plot images")
    print("[OK] Benchmark summary page rendered grouped results")

    print("[OK] Benchmark comparison views smoke check passed.")
    return 0


def _require_ok(response, name: str) -> None:
    if response.status_code != 200:
        _fail(f"{name} failed with status {response.status_code}: {response.text}")


def _fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())

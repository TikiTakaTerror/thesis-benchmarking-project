#!/usr/bin/env python3
"""Verify R8 shortcut metrics, reporting, and plot generation."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api import create_app
from src.services import build_comparison_export_basename, get_project_config, get_run_manager
from src.train import execute_seed_sweep
from src.train.synthetic import execute_synthetic_managed_run


def main() -> int:
    project_config = get_project_config()
    run_manager = get_run_manager()

    run_a = execute_synthetic_managed_run(
        run_manager,
        project_config=project_config,
        model_family="pipeline",
        benchmark_suite="rsbench",
        supervision="full",
        seed=81,
        run_name="r8_shortcut_reporting_pipeline_81",
        total_samples=32,
        train_size=24,
        training_overrides={
            "epochs": 3,
            "learning_rate": 0.003,
            "concept_loss_weight": 6.0,
        },
    ).record
    run_b = execute_synthetic_managed_run(
        run_manager,
        project_config=project_config,
        model_family="pipeline",
        benchmark_suite="rsbench",
        supervision="label_only",
        seed=83,
        run_name="r8_shortcut_reporting_pipeline_83",
        total_samples=32,
        train_size=24,
        training_overrides={
            "epochs": 3,
            "learning_rate": 0.003,
            "concept_loss_weight": 6.0,
        },
    ).record

    for record in (run_a, run_b):
        for metric_name in (
            "rsbench_shortcut_gap",
            "rsbench_shortcut_relative_drop",
            "rsbench_concept_gap",
        ):
            if metric_name not in record.metrics:
                _fail(f"Missing shortcut metric {metric_name} on {record.run_id}")
    print("[OK] rsbench runs stored shortcut-gap metrics")

    client = TestClient(create_app())
    compare_ids = [run_a.run_id, run_b.run_id]
    compare_response = client.get(
        "/compare",
        params=[("run_id", run_id) for run_id in compare_ids],
    )
    _require_ok(compare_response, "comparison page")
    compare_html = compare_response.text
    if "Shortcut Gap" not in compare_html:
        _fail("Comparison page did not render the Shortcut Gap column")
    if "/plots/" not in compare_html:
        _fail("Comparison page did not render plot image URLs")
    export_stem = build_comparison_export_basename(compare_ids)
    expected_plot = PROJECT_ROOT / "results" / "plots" / f"{export_stem}__comparison_scores.png"
    if not expected_plot.exists():
        _fail(f"Comparison plot was not written: {expected_plot}")
    plot_response = client.get(f"/plots/{expected_plot.name}")
    _require_ok(plot_response, "comparison plot")
    if plot_response.headers.get("content-type") != "image/png":
        _fail("Comparison plot is not served as image/png")
    print("[OK] Comparison page rendered and served generated plot files")

    benchmark_response = client.get("/benchmarks")
    _require_ok(benchmark_response, "benchmark summary page")
    benchmark_html = benchmark_response.text
    if "Mean Shortcut Gap" not in benchmark_html:
        _fail("Benchmark summary page is missing the Mean Shortcut Gap column")
    if "/plots/" not in benchmark_html:
        _fail("Benchmark summary page did not render plot image URLs")
    benchmark_plot = PROJECT_ROOT / "results" / "plots" / "benchmark_summary__latest__benchmark_overview.png"
    if not benchmark_plot.exists():
        _fail(f"Benchmark overview plot was not written: {benchmark_plot}")
    print("[OK] Benchmark summary page rendered shortcut reporting and plot assets")

    sweep_result = execute_seed_sweep(
        run_manager,
        project_config=project_config,
        dataset="synthetic_mnlogic",
        model_family="pipeline",
        benchmark_suite="rsbench",
        supervision="full",
        seeds=[401, 402],
        sweep_name="r8_shortcut_reporting_sweep",
        training_overrides={
            "epochs": 3,
            "learning_rate": 0.003,
            "concept_loss_weight": 6.0,
        },
        metric_names=[
            "benchmark_primary_score",
            "id_accuracy",
            "ood_accuracy",
            "rsbench_shortcut_gap",
        ],
    )
    if not sweep_result.plot_paths:
        _fail("Seed sweep did not return any generated plot paths")
    for plot_path in sweep_result.plot_paths:
        if not Path(plot_path).exists():
            _fail(f"Seed-sweep plot path does not exist: {plot_path}")
    print("[OK] Seed sweep wrote aggregate plot assets")

    print("[OK] Shortcut reporting and plots check passed.")
    return 0


def _require_ok(response, name: str) -> None:
    if response.status_code != 200:
        _fail(f"{name} failed with status {response.status_code}: {response.text}")


def _fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())

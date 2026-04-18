#!/usr/bin/env python3
"""Verify Phase 13 benchmark-suite adapter support."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks import create_benchmark_adapter
from src.services.catalog import list_available_options
from src.services.runtime import get_project_config, get_run_manager
from src.train.synthetic import execute_synthetic_managed_run


def main() -> int:
    options = list_available_options()
    benchmark_suites = options["benchmark_suites"]
    if "core_eval" not in benchmark_suites or "rsbench" not in benchmark_suites:
        _fail(f"Expected benchmark suites core_eval and rsbench, got {benchmark_suites}")
    print("[OK] Config-backed options expose both benchmark suites")

    core_adapter = create_benchmark_adapter("core_eval")
    rsbench_adapter = create_benchmark_adapter("rsbench")
    if "mnlogic" not in core_adapter.list_datasets():
        _fail("core_eval adapter does not list mnlogic")
    if "mnlogic" not in rsbench_adapter.list_datasets():
        _fail("rsbench adapter does not list mnlogic")
    print("[OK] Benchmark adapters loaded and reported supported datasets")

    project_config = get_project_config()
    run_manager = get_run_manager()

    core_result = execute_synthetic_managed_run(
        run_manager,
        project_config=project_config,
        model_family="pipeline",
        benchmark_suite="core_eval",
        seed=61,
        run_name="phase13_core_eval_smoke_seed_61",
        total_samples=32,
        train_size=24,
        training_overrides={
            "epochs": 4,
            "learning_rate": 0.003,
            "concept_loss_weight": 6.0,
        },
    )
    core_metrics = core_result.record.metrics
    _require_metric(core_metrics, "test_accuracy", "core_eval test_accuracy")
    _require_metric(core_metrics, "benchmark_primary_score", "core_eval benchmark_primary_score")
    _require_metric(core_metrics, "core_eval_primary_score", "core_eval_primary_score")
    if "ood_accuracy" in core_metrics:
        _fail("core_eval should not store OOD metrics for the synthetic smoke run")
    print(
        "[OK] core_eval run stored direct-evaluation metrics: "
        f"primary={core_metrics['benchmark_primary_score']:.4f}"
    )

    rsbench_result = execute_synthetic_managed_run(
        run_manager,
        project_config=project_config,
        model_family="pipeline",
        benchmark_suite="rsbench",
        seed=67,
        run_name="phase13_rsbench_smoke_seed_67",
        total_samples=32,
        train_size=24,
        training_overrides={
            "epochs": 4,
            "learning_rate": 0.003,
            "concept_loss_weight": 6.0,
        },
    )
    rsbench_metrics = rsbench_result.record.metrics
    _require_metric(rsbench_metrics, "id_accuracy", "rsbench id_accuracy")
    _require_metric(rsbench_metrics, "ood_accuracy", "rsbench ood_accuracy")
    _require_metric(rsbench_metrics, "id_performance", "rsbench id_performance")
    _require_metric(rsbench_metrics, "ood_performance", "rsbench ood_performance")
    _require_metric(
        rsbench_metrics,
        "benchmark_primary_score",
        "rsbench benchmark_primary_score",
    )
    _require_metric(
        rsbench_metrics,
        "rsbench_external_repo_present",
        "rsbench external repo presence metric",
    )
    _require_metric(
        rsbench_metrics,
        "rsbench_official_xor_model_count",
        "rsbench official xor model count metric",
    )
    print(
        "[OK] rsbench run stored ID/OOD metrics: "
        f"id={rsbench_metrics['id_accuracy']:.4f}, "
        f"ood={rsbench_metrics['ood_accuracy']:.4f}, "
        f"xor_models={rsbench_metrics['rsbench_official_xor_model_count']:.0f}"
    )

    print("[OK] Benchmark adapter smoke check passed.")
    return 0


def _require_metric(metrics: dict[str, float], metric_name: str, label: str) -> None:
    if metric_name not in metrics:
        _fail(f"Missing {label}: {metric_name}")


def _fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())

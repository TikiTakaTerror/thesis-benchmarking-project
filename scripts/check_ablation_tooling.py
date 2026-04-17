#!/usr/bin/env python3
"""Verify Phase 12 ablation and intervention tooling."""

from __future__ import annotations

import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.runtime import get_project_config, get_run_manager
from src.train.synthetic import execute_synthetic_managed_run


def main() -> int:
    project_config = get_project_config()
    run_manager = get_run_manager()

    pipeline_result = execute_synthetic_managed_run(
        run_manager,
        project_config=project_config,
        model_family="pipeline",
        seed=51,
        run_name="phase12_pipeline_ablation_seed_51",
        total_samples=32,
        train_size=24,
        training_overrides={
            "epochs": 4,
            "learning_rate": 0.003,
            "concept_loss_weight": 6.0,
        },
    )
    _require_metric(
        pipeline_result.record.metrics,
        "test_concept_intervention_accuracy",
        "pipeline concept intervention accuracy",
    )
    _require_metric(
        pipeline_result.record.metrics,
        "test_concept_intervention_gain",
        "pipeline concept intervention gain",
    )
    print(
        "[OK] Pipeline stored concept intervention metrics: "
        f"gain={pipeline_result.record.metrics['test_concept_intervention_gain']:.4f}"
    )

    ltn_result = execute_synthetic_managed_run(
        run_manager,
        project_config=project_config,
        model_family="ltn",
        seed=53,
        run_name="phase12_ltn_ablation_seed_53",
        total_samples=32,
        train_size=24,
        training_overrides={
            "epochs": 4,
            "learning_rate": 0.003,
            "concept_loss_weight": 2.0,
            "satisfaction_weight": 1.0,
        },
    )
    _require_metric(
        ltn_result.record.metrics,
        "test_symbolic_layer_ablation_gain",
        "ltn symbolic layer ablation gain",
    )
    _require_metric(
        ltn_result.record.metrics,
        "test_symbolic_layer_ablated_accuracy",
        "ltn symbolic layer ablated accuracy",
    )
    _require_metric(
        ltn_result.record.metrics,
        "test_concept_intervention_gain",
        "ltn concept intervention gain",
    )
    print(
        "[OK] LTN stored ablation and intervention metrics: "
        f"ablation_gain={ltn_result.record.metrics['test_symbolic_layer_ablation_gain']:.4f}, "
        f"intervention_gain={ltn_result.record.metrics['test_concept_intervention_gain']:.4f}"
    )

    deepproblog_result = execute_synthetic_managed_run(
        run_manager,
        project_config=project_config,
        model_family="deepproblog",
        seed=59,
        run_name="phase12_deepproblog_ablation_seed_59",
        total_samples=20,
        train_size=12,
        training_overrides={
            "epochs": 2,
            "learning_rate": 0.01,
            "concept_loss_weight": 4.0,
        },
    )
    _require_metric(
        deepproblog_result.record.metrics,
        "test_concept_intervention_accuracy",
        "deepproblog concept intervention accuracy",
    )
    _require_metric(
        deepproblog_result.record.metrics,
        "test_concept_intervention_gain",
        "deepproblog concept intervention gain",
    )
    print(
        "[OK] DeepProbLog stored concept intervention metrics: "
        f"gain={deepproblog_result.record.metrics['test_concept_intervention_gain']:.4f}"
    )

    if "test_symbolic_layer_ablation_gain" in pipeline_result.record.metrics:
        _fail("Pipeline should not expose symbolic layer ablation metrics in Phase 12")
    if "test_symbolic_layer_ablation_gain" in deepproblog_result.record.metrics:
        _fail("DeepProbLog should not expose symbolic layer ablation metrics in Phase 12")
    print("[OK] Optional symbolic ablation metrics stay absent for unsupported families")

    print("[OK] Ablation and intervention tooling smoke check passed.")
    return 0


def _require_metric(metrics: dict[str, float], metric_name: str, label: str) -> None:
    if metric_name not in metrics:
        _fail(f"Missing {label}: {metric_name}")
    metric_value = metrics[metric_name]
    if not isinstance(metric_value, (int, float)) or not math.isfinite(float(metric_value)):
        _fail(f"Invalid {label}: {metric_value}")


def _fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())

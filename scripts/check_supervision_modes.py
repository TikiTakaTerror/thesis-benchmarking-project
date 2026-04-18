#!/usr/bin/env python3
"""Verify config-driven supervision modes for managed runs."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.runtime import get_project_config, get_run_manager
from src.train import execute_real_mnlogic_managed_run, load_supervision_config


def main() -> int:
    for expected_name, expected_fraction in (
        ("full", 1.0),
        ("label_only", 0.0),
        ("concept_50", 0.5),
    ):
        config = load_supervision_config(expected_name)
        if config.concept_supervision_fraction != expected_fraction:
            _fail(
                f"Supervision config {expected_name} has fraction "
                f"{config.concept_supervision_fraction}, expected {expected_fraction}"
            )
        print(
            f"[OK] Loaded supervision config: {expected_name} "
            f"(fraction={config.concept_supervision_fraction:.2f})"
        )

    manager = get_run_manager()
    project_config = get_project_config()

    full_result = execute_real_mnlogic_managed_run(
        manager,
        project_config=project_config,
        model_family="pipeline",
        seed=201,
        benchmark_suite="core_eval",
        supervision="full",
        run_name="r5_pipeline_full",
        training_overrides={"epochs": 1, "learning_rate": 0.003},
        limit_per_split=32,
    )
    full_config = _load_run_config(full_result.record.config_path)
    _require_close(
        full_config["supervision_policy"]["effective_concept_supervision_fraction"],
        1.0,
        "full supervision fraction",
    )
    if "train_concept_loss" not in full_result.training_metrics:
        _fail("Full supervision run is missing train_concept_loss")
    print(
        f"[OK] Full supervision stored train concept metrics: "
        f"fraction={full_result.training_metrics['train_concept_supervision_fraction']:.4f}"
    )

    label_only_result = execute_real_mnlogic_managed_run(
        manager,
        project_config=project_config,
        model_family="pipeline",
        seed=202,
        benchmark_suite="core_eval",
        supervision="label_only",
        run_name="r5_pipeline_label_only",
        training_overrides={"epochs": 1, "learning_rate": 0.003},
        limit_per_split=32,
    )
    label_only_config = _load_run_config(label_only_result.record.config_path)
    _require_close(
        label_only_config["supervision_policy"]["effective_concept_supervision_fraction"],
        0.0,
        "label_only supervision fraction",
    )
    if "train_concept_loss" in label_only_result.training_metrics:
        _fail("Label-only run should not report train_concept_loss")
    if float(label_only_config["training"]["concept_loss_weight"]) != 0.0:
        _fail("Label-only run should store effective concept_loss_weight=0.0")
    print("[OK] Label-only supervision removed train concept loss")

    concept_50_result = execute_real_mnlogic_managed_run(
        manager,
        project_config=project_config,
        model_family="pipeline",
        seed=203,
        benchmark_suite="core_eval",
        supervision="concept_50",
        run_name="r5_pipeline_concept_50",
        training_overrides={"epochs": 1, "learning_rate": 0.003},
        limit_per_split=32,
    )
    concept_50_config = _load_run_config(concept_50_result.record.config_path)
    effective_fraction = float(
        concept_50_config["supervision_policy"]["effective_concept_supervision_fraction"]
    )
    _require_close(effective_fraction, 0.5, "concept_50 supervision fraction")
    _require_close(
        float(concept_50_result.training_metrics["train_concept_supervision_fraction"]),
        0.5,
        "concept_50 train metric fraction",
    )
    print(
        "[OK] concept_50 supervision stored partial concept masking: "
        f"fraction={effective_fraction:.4f}"
    )

    ltn_label_only_result = execute_real_mnlogic_managed_run(
        manager,
        project_config=project_config,
        model_family="ltn",
        seed=204,
        benchmark_suite="core_eval",
        supervision="label_only",
        run_name="r5_ltn_label_only",
        training_overrides={"epochs": 1, "learning_rate": 0.003},
        limit_per_split=32,
    )
    ltn_label_only_config = _load_run_config(ltn_label_only_result.record.config_path)
    if float(ltn_label_only_config["training"]["satisfaction_weight"]) != 0.0:
        _fail("LTN label-only run should store effective satisfaction_weight=0.0")
    print("[OK] LTN label-only supervision disabled the extra logic-constraint loss")

    print("[OK] Supervision mode check passed.")
    return 0


def _load_run_config(config_path: str) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        _fail(f"Invalid run config snapshot: {config_path}")
    return payload


def _require_close(actual: float, expected: float, label: str, *, tol: float = 1e-6) -> None:
    if abs(actual - expected) > tol:
        _fail(f"{label} mismatch: got {actual}, expected {expected}")


def _fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())

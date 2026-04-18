"""Minimal run execution helpers for Phase 8 orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch

from ..eval import evaluate_named_splits
from ..models import create_model_adapter, create_model_adapter_from_config
from ..services.run_manager import RunManager, RunRecord, RunSelection


@dataclass(frozen=True)
class RunExecutionResult:
    """Outcome of one managed run execution."""

    record: RunRecord
    checkpoint_path: str
    training_metrics: dict[str, float]
    evaluation_metrics: dict[str, float]


def execute_training_run(
    run_manager: RunManager,
    *,
    run_name: str,
    selection: RunSelection,
    config_snapshot: Mapping[str, Any],
    model_config: Mapping[str, Any] | None = None,
    train_batches: Iterable[dict[str, torch.Tensor]],
    evaluation_splits: Mapping[str, Iterable[dict[str, torch.Tensor]]] | None = None,
    evaluation_callback: Callable[[Any, Mapping[str, Iterable[dict[str, torch.Tensor]]]], Mapping[str, Any]]
    | None = None,
    train_kwargs: Mapping[str, Any] | None = None,
) -> RunExecutionResult:
    """Train one model-family run and persist its artifacts and metrics."""

    record = run_manager.create_run(
        run_name=run_name,
        selection=selection,
        config_snapshot=config_snapshot,
    )
    record = run_manager.mark_run_started(record.run_id)

    try:
        started_at = time.perf_counter()
        model = (
            create_model_adapter_from_config(dict(model_config))
            if model_config is not None
            else create_model_adapter(selection.model_family)
        )
        training_metrics = _normalize_metrics(
            model.train(train_batches, **dict(train_kwargs or {}))
        )
        evaluation_metrics = (
            _normalize_metrics(
                (
                    evaluation_callback(model, evaluation_splits)
                    if evaluation_callback is not None
                    else evaluate_named_splits(model, evaluation_splits)
                )
            )
            if evaluation_splits
            else {}
        )
        runtime_seconds = time.perf_counter() - started_at

        checkpoint_path = Path(record.run_dir) / "artifacts" / "model_checkpoint.pt"
        model.save_checkpoint(checkpoint_path)

        all_metrics = {
            **training_metrics,
            **evaluation_metrics,
            "run_runtime_seconds": float(runtime_seconds),
        }
        artifacts = {"checkpoint": str(checkpoint_path)}
        record = run_manager.complete_run(
            record.run_id,
            metrics=all_metrics,
            artifacts=artifacts,
            checkpoint_path=checkpoint_path,
        )
        return RunExecutionResult(
            record=record,
            checkpoint_path=str(checkpoint_path),
            training_metrics=training_metrics,
            evaluation_metrics=evaluation_metrics,
        )
    except Exception as exc:
        run_manager.fail_run(record.run_id, str(exc))
        raise


def _normalize_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric_name, metric_value in payload.items():
        if isinstance(metric_value, bool):
            metrics[str(metric_name)] = float(metric_value)
        elif isinstance(metric_value, (int, float)):
            metrics[str(metric_name)] = float(metric_value)
    return metrics

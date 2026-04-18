"""Internal benchmark suite that uses the project's shared evaluator directly."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .base import BenchmarkSuiteAdapter


class CoreEvalBenchmarkAdapter(BenchmarkSuiteAdapter):
    """Simple internal benchmark suite for direct shared-evaluator comparisons."""

    suite_name = "core_eval"

    def _prepare_dataset(
        self,
        dataset_name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        images = _require_tensor(kwargs, "images")
        label_ids = _require_tensor(kwargs, "label_ids")
        concept_targets = _require_tensor(kwargs, "concept_targets")
        train_size = int(kwargs["train_size"])
        batch_size = int(kwargs["batch_size"])

        train_batches = _make_batches(
            images[:train_size],
            label_ids[:train_size],
            concept_targets[:train_size],
            batch_size=batch_size,
        )
        test_batches = _make_batches(
            images[train_size:],
            label_ids[train_size:],
            concept_targets[train_size:],
            batch_size=batch_size,
        )
        return {
            "dataset_name": dataset_name,
            "train_batches": train_batches,
            "evaluation_splits": {"test": test_batches},
            "suite_context": {"has_ood_split": False},
        }

    def compute_suite_specific_metrics(
        self,
        metrics: Mapping[str, float],
    ) -> dict[str, float]:
        primary_score = metrics.get("test_accuracy")
        suite_metrics = {
            "benchmark_has_ood": 0.0,
            "benchmark_num_splits": 1.0,
        }
        if primary_score is not None:
            suite_metrics["benchmark_primary_score"] = float(primary_score)
            suite_metrics["core_eval_primary_score"] = float(primary_score)
        return suite_metrics


def _require_tensor(payload: Mapping[str, Any], field_name: str) -> torch.Tensor:
    if field_name not in payload:
        raise ValueError(f"core_eval prepare_dataset requires '{field_name}'")
    tensor = payload[field_name]
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"core_eval field '{field_name}' must be a torch.Tensor")
    return tensor


def _make_batches(
    images: torch.Tensor,
    label_ids: torch.Tensor,
    concept_targets: torch.Tensor,
    *,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for start_index in range(0, images.shape[0], batch_size):
        end_index = start_index + batch_size
        batches.append(
            {
                "images": images[start_index:end_index],
                "label_ids": label_ids[start_index:end_index],
                "concept_targets": concept_targets[start_index:end_index],
            }
        )
    return batches

"""rsbench-style benchmark adapter with ID/OOD split handling."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .base import BenchmarkSuiteAdapter


class RSBenchBenchmarkAdapter(BenchmarkSuiteAdapter):
    """Benchmark adapter that mimics an rsbench-style ID/OOD evaluation surface."""

    suite_name = "rsbench"

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
        seed = int(kwargs.get("seed", 0))

        train_batches = _make_batches(
            images[:train_size],
            label_ids[:train_size],
            concept_targets[:train_size],
            batch_size=batch_size,
        )
        id_images = images[train_size:]
        id_labels = label_ids[train_size:]
        id_concepts = concept_targets[train_size:]
        ood_images = _build_ood_images(id_images, seed=seed)

        return {
            "dataset_name": dataset_name,
            "train_batches": train_batches,
            "evaluation_splits": {
                "id": _make_batches(
                    id_images,
                    id_labels,
                    id_concepts,
                    batch_size=batch_size,
                ),
                "ood": _make_batches(
                    ood_images,
                    id_labels,
                    id_concepts,
                    batch_size=batch_size,
                ),
            },
            "suite_context": {"has_ood_split": True},
        }

    def compute_suite_specific_metrics(
        self,
        metrics: Mapping[str, float],
    ) -> dict[str, float]:
        id_score = metrics.get("id_accuracy")
        ood_score = metrics.get("ood_accuracy")

        suite_metrics = {
            "benchmark_has_ood": 1.0 if ood_score is not None else 0.0,
            "benchmark_num_splits": 2.0 if ood_score is not None else 1.0,
        }
        if id_score is not None:
            suite_metrics["id_performance"] = float(id_score)
        if ood_score is not None:
            suite_metrics["ood_performance"] = float(ood_score)

        if id_score is not None and ood_score is not None:
            primary_score = 0.5 * (float(id_score) + float(ood_score))
            suite_metrics["benchmark_primary_score"] = primary_score
            suite_metrics["rsbench_primary_score"] = primary_score
        elif id_score is not None:
            suite_metrics["benchmark_primary_score"] = float(id_score)
            suite_metrics["rsbench_primary_score"] = float(id_score)

        return suite_metrics


def _build_ood_images(images: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=images.device if images.is_cuda else "cpu")
    generator.manual_seed(seed + 1000)
    shifted = torch.roll(images, shifts=5, dims=-1)
    mirrored = torch.flip(shifted, dims=[-2])
    noise = 0.06 * torch.randn(
        images.shape,
        generator=generator,
        device=images.device,
        dtype=images.dtype,
    )
    return (0.75 * shifted + 0.25 * mirrored + noise).clamp(0.0, 1.0)


def _require_tensor(payload: Mapping[str, Any], field_name: str) -> torch.Tensor:
    if field_name not in payload:
        raise ValueError(f"rsbench prepare_dataset requires '{field_name}'")
    tensor = payload[field_name]
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"rsbench field '{field_name}' must be a torch.Tensor")
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

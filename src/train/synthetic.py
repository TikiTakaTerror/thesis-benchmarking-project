"""Synthetic managed-run helpers used for smoke checks and the backend API."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ..benchmarks import create_benchmark_adapter
from ..models.registry import load_model_config
from ..services import ProjectConfig, RunManager, RunSelection
from .runner import RunExecutionResult, execute_training_run
from .supervision import apply_supervision


SYNTHETIC_DATASET_NAME = "synthetic_mnlogic"


def execute_synthetic_managed_run(
    run_manager: RunManager,
    *,
    project_config: ProjectConfig,
    model_family: str,
    seed: int,
    benchmark_suite: str = "rsbench",
    supervision: str = "full",
    run_name: str | None = None,
    training_overrides: Mapping[str, Any] | None = None,
    total_samples: int = 64,
    train_size: int = 48,
) -> RunExecutionResult:
    """Execute one synthetic managed run for a supported model family."""

    benchmark_adapter = create_benchmark_adapter(benchmark_suite)
    external_environment = benchmark_adapter.build_external_environment(
        dataset_name=SYNTHETIC_DATASET_NAME,
        model_family=model_family,
    )
    selection = RunSelection(
        dataset=SYNTHETIC_DATASET_NAME,
        model_family=model_family,
        benchmark_suite=benchmark_suite,
        supervision=supervision,
        seed=seed,
    )
    model_config = load_model_config(model_family)
    training_payload = {
        **default_synthetic_training_kwargs(model_family, model_config),
        **dict(training_overrides or {}),
    }
    batch_size = int(training_payload.pop("batch_size", 16))

    images, label_ids, concept_targets = build_synthetic_dataset(
        model_family=model_family,
        total_samples=total_samples,
        seed=seed,
        input_channels=int(model_config["shared_encoder"]["input_channels"]),
        input_size=tuple(model_config["shared_encoder"]["input_size"]),
    )
    prepared_suite = benchmark_adapter.prepare_dataset(
        SYNTHETIC_DATASET_NAME,
        images=images,
        label_ids=label_ids,
        concept_targets=concept_targets,
        batch_size=batch_size,
        train_size=train_size,
        seed=seed,
    )
    supervision_result = apply_supervision(
        model_family=model_family,
        supervision_name=supervision,
        seed=seed,
        train_batches=prepared_suite["train_batches"],
        train_kwargs=training_payload,
    )
    train_batches = supervision_result.train_batches
    effective_training_payload = supervision_result.train_kwargs
    evaluation_splits = prepared_suite["evaluation_splits"]

    resolved_run_name = run_name or f"api_{model_family}_seed_{seed}"
    config_snapshot = {
        "project": project_config.to_dict(),
        "run": {
            "name": resolved_run_name,
            "phase": 13,
            "seed": seed,
            "source": "phase13_benchmark_adapters",
        },
        "selection": selection.to_dict(),
        "model": model_config,
        "benchmark": {
            "suite": benchmark_suite,
            "config": benchmark_adapter.config.to_dict(),
            "external_environment": external_environment,
            "prepared_dataset": {
                key: value
                for key, value in prepared_suite.items()
                if key not in {"train_batches", "evaluation_splits"}
            },
        },
        "training": effective_training_payload,
        "supervision_policy": supervision_result.summary,
        "synthetic_data": {
            "total_samples": total_samples,
            "train_size": train_size,
            "batch_size": batch_size,
        },
    }

    return execute_training_run(
        run_manager,
        run_name=resolved_run_name,
        selection=selection,
        config_snapshot=config_snapshot,
        train_batches=train_batches,
        evaluation_splits=evaluation_splits,
        evaluation_callback=lambda model, split_batches: benchmark_adapter.run_evaluation(
            model,
            split_batches,
            seed=seed,
            label_loss_weight=float(effective_training_payload.get("label_loss_weight", 1.0)),
            concept_loss_weight=float(effective_training_payload.get("concept_loss_weight", 1.0)),
            external_environment=external_environment,
        ),
        train_kwargs=effective_training_payload,
    )


def default_synthetic_training_kwargs(
    model_family: str,
    model_config: Mapping[str, Any],
) -> dict[str, float]:
    """Return stable synthetic-run defaults per model family."""

    training_defaults = dict(model_config.get("training_defaults", {}))
    payload = {
        "epochs": float(training_defaults.get("max_epochs", 12)),
        "learning_rate": float(training_defaults.get("learning_rate", 1e-3)),
    }

    if "label_loss_weight" in training_defaults:
        payload["label_loss_weight"] = float(training_defaults["label_loss_weight"])
    if "concept_loss_weight" in training_defaults:
        payload["concept_loss_weight"] = float(training_defaults["concept_loss_weight"])

    # Pipeline synthetic runs need stronger concept supervision for stable stored metrics.
    if model_family == "pipeline":
        payload["epochs"] = 12.0
        payload["learning_rate"] = 0.003
        payload["concept_loss_weight"] = 6.0

    return payload


def build_synthetic_dataset(
    *,
    model_family: str,
    total_samples: int,
    seed: int,
    input_channels: int,
    input_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build synthetic image, label, and concept tensors for one model family."""

    torch.manual_seed(seed)

    base_patterns = torch.tensor(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=torch.float32,
    )
    repeats = (total_samples + len(base_patterns) - 1) // len(base_patterns)
    concept_targets = base_patterns.repeat(repeats, 1)[:total_samples]
    concept_targets = concept_targets[torch.randperm(total_samples)]

    if model_family == "deepproblog":
        positive_mask = (
            (concept_targets[:, 0] == 1)
            & (concept_targets[:, 1] == 1)
            & (concept_targets[:, 2] == 1)
        )
    else:
        positive_mask = (concept_targets[:, 0] == 1) & (concept_targets[:, 1] == 1)
    label_ids = positive_mask.long()

    image_height, image_width = input_size
    images = torch.zeros(total_samples, input_channels, image_height, image_width)
    for sample_index, concepts in enumerate(concept_targets):
        if concepts[0] > 0.5:
            images[sample_index, 0, 4:28, 4:28] = 1.0
        if concepts[1] > 0.5 and input_channels >= 2:
            images[sample_index, 1, 4:28, 36:60] = 1.0
        if concepts[2] > 0.5 and input_channels >= 3:
            images[sample_index, 2, 36:60, 20:44] = 1.0

    noise_scale = 0.01 if model_family != "ltn" else 0.03
    images = (images + noise_scale * torch.randn_like(images)).clamp(0.0, 1.0)
    return images, label_ids, concept_targets


def split_tensor_batches(
    images: torch.Tensor,
    label_ids: torch.Tensor,
    concept_targets: torch.Tensor,
    *,
    batch_size: int,
    train_size: int,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    """Split synthetic tensors into train/test batches."""

    train_batches = make_batches(
        images[:train_size],
        label_ids[:train_size],
        concept_targets[:train_size],
        batch_size=batch_size,
    )
    test_batches = make_batches(
        images[train_size:],
        label_ids[train_size:],
        concept_targets[train_size:],
        batch_size=batch_size,
    )
    return train_batches, test_batches


def make_batches(
    images: torch.Tensor,
    label_ids: torch.Tensor,
    concept_targets: torch.Tensor,
    *,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    """Convert tensors into the common batch dictionary format."""

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

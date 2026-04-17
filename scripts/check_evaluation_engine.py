#!/usr/bin/env python3
"""Smoke-check the shared evaluation engine with the custom pipeline model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import evaluate_named_splits
from src.models.pipeline import PipelineModelAdapter
from src.models.registry import load_model_config


def main() -> int:
    torch.manual_seed(19)

    model = PipelineModelAdapter.from_config_dict(load_model_config("pipeline"))

    train_images, train_labels, train_concepts = build_dataset(
        model,
        total_samples=96,
        noise_scale=0.04,
        shift_pixels=0,
    )
    id_images, id_labels, id_concepts = build_dataset(
        model,
        total_samples=32,
        noise_scale=0.06,
        shift_pixels=0,
    )
    ood_images, ood_labels, ood_concepts = build_dataset(
        model,
        total_samples=32,
        noise_scale=0.10,
        shift_pixels=6,
    )

    train_batches = make_batches(
        train_images,
        train_labels,
        train_concepts,
        batch_size=model.config.training_defaults.batch_size,
    )
    id_batches = make_batches(
        id_images,
        id_labels,
        id_concepts,
        batch_size=model.config.training_defaults.batch_size,
    )
    ood_batches = make_batches(
        ood_images,
        ood_labels,
        ood_concepts,
        batch_size=model.config.training_defaults.batch_size,
    )

    model.train(
        train_batches,
        epochs=model.config.training_defaults.max_epochs,
        learning_rate=model.config.training_defaults.learning_rate,
    )
    print("[OK] Pipeline model trained for evaluation smoke check")

    metrics = evaluate_named_splits(
        model,
        {"id": id_batches, "ood": ood_batches},
        seed=19,
        label_loss_weight=model.config.training_defaults.label_loss_weight,
        concept_loss_weight=model.config.training_defaults.concept_loss_weight,
    )

    required_metrics = [
        "id_accuracy",
        "id_macro_f1",
        "id_concept_accuracy",
        "id_concept_macro_f1",
        "id_exact_concept_vector_match",
        "id_rule_satisfaction_rate",
        "id_violation_rate",
        "id_concept_label_consistency",
        "ood_accuracy",
        "ood_macro_f1",
        "ood_concept_accuracy",
        "ood_rule_satisfaction_rate",
    ]
    missing_metrics = [metric_name for metric_name in required_metrics if metric_name not in metrics]
    if missing_metrics:
        print(f"[ERROR] Missing expected evaluation metrics: {missing_metrics}", file=sys.stderr)
        return 1

    print(f"[OK] id accuracy: {metrics['id_accuracy']:.4f}")
    print(f"[OK] id macro_f1: {metrics['id_macro_f1']:.4f}")
    print(f"[OK] id concept_accuracy: {metrics['id_concept_accuracy']:.4f}")
    print(
        f"[OK] id rule_satisfaction_rate: {metrics['id_rule_satisfaction_rate']:.4f}"
    )
    print(f"[OK] ood accuracy: {metrics['ood_accuracy']:.4f}")
    print(f"[OK] ood macro_f1: {metrics['ood_macro_f1']:.4f}")

    if metrics["id_accuracy"] < 0.90:
        print("[ERROR] ID accuracy is too low.", file=sys.stderr)
        return 1
    if metrics["id_concept_accuracy"] < 0.90:
        print("[ERROR] ID concept accuracy is too low.", file=sys.stderr)
        return 1
    if metrics["id_rule_satisfaction_rate"] < 0.90:
        print("[ERROR] ID rule satisfaction is too low.", file=sys.stderr)
        return 1

    print("[OK] Evaluation engine smoke check passed.")
    return 0


def build_dataset(
    model: PipelineModelAdapter,
    *,
    total_samples: int,
    noise_scale: float,
    shift_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    concept_patterns = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.float32,
    )
    repeats = (total_samples + len(concept_patterns) - 1) // len(concept_patterns)
    concept_targets = concept_patterns.repeat(repeats, 1)[:total_samples]
    concept_targets = concept_targets[torch.randperm(total_samples)]

    label_ids = model.symbolic_executor.predict_label_ids(concept_targets).long()

    image_height, image_width = model.config.shared_encoder.input_size
    images = torch.zeros(
        total_samples,
        model.config.shared_encoder.input_channels,
        image_height,
        image_width,
    )

    shift = int(shift_pixels)
    for sample_index, concepts in enumerate(concept_targets):
        if concepts[0] > 0.5:
            images[sample_index, 0, 4 + shift : 20 + shift, 4:20] = 1.0
        if concepts[1] > 0.5:
            images[sample_index, 1, 4:20, 44 - shift : 60 - shift] = 1.0
        if concepts[2] > 0.5:
            images[sample_index, 2, 40 - shift : 56 - shift, 24:40] = 1.0

    images = (images + noise_scale * torch.randn_like(images)).clamp(0.0, 1.0)
    return images, label_ids, concept_targets


def make_batches(
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


if __name__ == "__main__":
    raise SystemExit(main())

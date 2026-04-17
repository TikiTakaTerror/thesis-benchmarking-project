#!/usr/bin/env python3
"""Train and validate the DeepProbLog model family on a synthetic dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.deepproblog_model import DeepProbLogModelAdapter
from src.models.registry import load_model_config


def main() -> int:
    torch.manual_seed(23)

    model = DeepProbLogModelAdapter.from_config_dict(load_model_config("deepproblog"))
    print(
        f"[OK] Loaded DeepProbLog config with {model.config.num_concepts} concepts "
        f"and {model.config.num_labels} labels"
    )

    images, label_ids, concept_targets = build_synthetic_dataset(
        model,
        total_samples=64,
    )
    train_batches, val_batches = split_into_batches(
        images,
        label_ids,
        concept_targets,
        batch_size=model.config.training_defaults.batch_size,
        train_size=48,
    )
    print("[OK] Synthetic DeepProbLog dataset created: train=48, val=16")

    train_metrics = model.train(
        train_batches,
        val_batches=val_batches,
        epochs=model.config.training_defaults.max_epochs,
        learning_rate=model.config.training_defaults.learning_rate,
        concept_loss_weight=model.config.training_defaults.concept_loss_weight,
    )
    print("[OK] Training completed")
    print(
        f"[OK] Validation label accuracy: {train_metrics['val_label_accuracy']:.4f}"
    )
    print(
        f"[OK] Validation concept accuracy: {train_metrics['val_concept_accuracy']:.4f}"
    )
    print(
        "[OK] Validation rule satisfaction rate: "
        f"{train_metrics['val_rule_satisfaction_rate']:.4f}"
    )

    if train_metrics["val_label_accuracy"] < 0.95:
        print("[ERROR] Validation label accuracy is too low.", file=sys.stderr)
        return 1
    if train_metrics["val_concept_accuracy"] < 0.90:
        print("[ERROR] Validation concept accuracy is too low.", file=sys.stderr)
        return 1

    checkpoint_path = (
        PROJECT_ROOT
        / "results"
        / "runs"
        / "phase7_deepproblog_smoke"
        / "deepproblog_checkpoint.pt"
    )
    model.save_checkpoint(checkpoint_path)

    reloaded_model = DeepProbLogModelAdapter.load_checkpoint(checkpoint_path)
    reloaded_model.to("cpu")

    val_images = torch.cat([batch["images"] for batch in val_batches], dim=0)
    original_predictions = model.predict(val_images)
    reloaded_predictions = reloaded_model.predict(val_images)
    if not torch.equal(original_predictions, reloaded_predictions):
        print("[ERROR] Checkpoint reload mismatch.", file=sys.stderr)
        return 1

    print("[OK] Checkpoint reload matched predictions.")
    print("[OK] DeepProbLog smoke check passed.")
    return 0


def build_synthetic_dataset(
    model: DeepProbLogModelAdapter,
    *,
    total_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    positive_mask = (
        (concept_targets[:, 0] == 1)
        & (concept_targets[:, 1] == 1)
        & (concept_targets[:, 2] == 1)
    )
    label_ids = positive_mask.long()

    image_height, image_width = model.config.shared_encoder.input_size
    images = torch.zeros(
        total_samples,
        model.config.shared_encoder.input_channels,
        image_height,
        image_width,
    )

    for sample_index, concepts in enumerate(concept_targets):
        if concepts[0] > 0.5:
            images[sample_index, 0, 4:28, 4:28] = 1.0
        if concepts[1] > 0.5:
            images[sample_index, 1, 4:28, 36:60] = 1.0
        if concepts[2] > 0.5:
            images[sample_index, 2, 36:60, 20:44] = 1.0

    images = (images + 0.01 * torch.randn_like(images)).clamp(0.0, 1.0)
    return images, label_ids, concept_targets


def split_into_batches(
    images: torch.Tensor,
    label_ids: torch.Tensor,
    concept_targets: torch.Tensor,
    *,
    batch_size: int,
    train_size: int,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    train_batches = make_batches(
        images[:train_size],
        label_ids[:train_size],
        concept_targets[:train_size],
        batch_size=batch_size,
    )
    val_batches = make_batches(
        images[train_size:],
        label_ids[train_size:],
        concept_targets[train_size:],
        batch_size=batch_size,
    )
    return train_batches, val_batches


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

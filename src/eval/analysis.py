"""Ablation and intervention helpers for model-family evaluation."""

from __future__ import annotations

from typing import Any

import torch

from .metrics import compute_classification_metrics


def compute_ablation_and_intervention_metrics(
    model: Any,
    batches: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Compute symbolic-layer ablation and concept intervention metrics when supported."""

    if not batches:
        return {}

    if hasattr(model, "_set_training_mode"):
        model._set_training_mode(False)

    device = _resolve_model_device(model)
    label_targets: list[torch.Tensor] = []
    full_predictions: list[torch.Tensor] = []
    ablated_targets: list[torch.Tensor] = []
    ablated_predictions: list[torch.Tensor] = []
    intervention_targets: list[torch.Tensor] = []
    intervention_predictions: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in batches:
            images = _require_batch_field(batch, "images").to(device).float()
            batch_label_targets = batch.get("label_ids")
            batch_concept_targets = batch.get("concept_targets")
            if batch_label_targets is None:
                continue

            outputs = model.forward(images) if hasattr(model, "forward") else None
            if outputs is None:
                raise ValueError(
                    "Ablation tooling expects model.forward(...) to be available."
                )

            batch_full_predictions = outputs.extras.get("hard_label_predictions")
            if batch_full_predictions is None:
                if outputs.label_logits is None:
                    raise ValueError("Model outputs do not provide label predictions.")
                batch_full_predictions = outputs.label_logits.argmax(dim=-1)

            label_targets.append(batch_label_targets.detach().cpu().long())
            full_predictions.append(batch_full_predictions.detach().cpu().long())

            if _supports_symbolic_ablation(model):
                batch_ablated = model.predict_without_symbolic_layer(
                    images,
                    reference_outputs=outputs,
                )
                ablated_targets.append(batch_label_targets.detach().cpu().long())
                ablated_predictions.append(batch_ablated.detach().cpu().long())

            if batch_concept_targets is not None and _supports_concept_intervention(model):
                batch_intervened = model.predict_from_concepts(
                    batch_concept_targets.to(device).float(),
                    reference_outputs=outputs,
                )
                intervention_targets.append(batch_label_targets.detach().cpu().long())
                intervention_predictions.append(batch_intervened.detach().cpu().long())

    if not label_targets or not full_predictions:
        return {}

    target_tensor = torch.cat(label_targets, dim=0)
    full_prediction_tensor = torch.cat(full_predictions, dim=0)
    full_metrics = compute_classification_metrics(target_tensor, full_prediction_tensor)

    metrics: dict[str, float] = {}

    if ablated_targets and ablated_predictions:
        ablated_target_tensor = torch.cat(ablated_targets, dim=0)
        ablated_tensor = torch.cat(ablated_predictions, dim=0)
        ablated_metrics = compute_classification_metrics(ablated_target_tensor, ablated_tensor)
        metrics["symbolic_layer_ablated_accuracy"] = ablated_metrics["accuracy"]
        metrics["symbolic_layer_ablated_macro_f1"] = ablated_metrics["macro_f1"]
        metrics["symbolic_layer_ablation_gain"] = (
            full_metrics["accuracy"] - ablated_metrics["accuracy"]
        )

    if intervention_targets and intervention_predictions:
        intervention_target_tensor = torch.cat(intervention_targets, dim=0)
        intervention_tensor = torch.cat(intervention_predictions, dim=0)
        intervention_metrics = compute_classification_metrics(
            intervention_target_tensor,
            intervention_tensor,
        )
        metrics["concept_intervention_accuracy"] = intervention_metrics["accuracy"]
        metrics["concept_intervention_macro_f1"] = intervention_metrics["macro_f1"]
        metrics["concept_intervention_gain"] = (
            intervention_metrics["accuracy"] - full_metrics["accuracy"]
        )

    return metrics


def _supports_symbolic_ablation(model: Any) -> bool:
    if hasattr(model, "supports_symbolic_ablation"):
        return bool(model.supports_symbolic_ablation())
    return False


def _supports_concept_intervention(model: Any) -> bool:
    if hasattr(model, "supports_concept_intervention"):
        return bool(model.supports_concept_intervention())
    return False


def _resolve_model_device(model: Any) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)
    return torch.device("cpu")


def _require_batch_field(
    batch: dict[str, torch.Tensor],
    field_name: str,
) -> torch.Tensor:
    if field_name not in batch:
        raise ValueError(f"Each evaluation batch must contain '{field_name}'")
    return batch[field_name]

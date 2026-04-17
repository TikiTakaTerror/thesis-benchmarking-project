"""Shared metric computation for task, concept, semantic, and control views."""

from __future__ import annotations

from typing import Any

import torch
from sklearn.metrics import f1_score


def compute_classification_metrics(
    label_targets: torch.Tensor | None,
    label_predictions: torch.Tensor | None,
) -> dict[str, float]:
    """Compute task-level classification metrics."""

    if label_targets is None or label_predictions is None:
        return {}

    y_true = label_targets.detach().cpu().numpy()
    y_pred = label_predictions.detach().cpu().numpy()
    accuracy = float((label_targets == label_predictions).float().mean().item())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {
        "accuracy": accuracy,
        "label_accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def compute_concept_metrics(
    concept_targets: torch.Tensor | None,
    concept_predictions: torch.Tensor | None,
) -> dict[str, float]:
    """Compute concept quality metrics."""

    if concept_targets is None or concept_predictions is None:
        return {}

    concept_targets = concept_targets.float()
    concept_predictions = concept_predictions.float()
    concept_accuracy = float(
        (concept_targets == concept_predictions).float().mean().item()
    )
    exact_match = float(
        (concept_targets == concept_predictions).all(dim=1).float().mean().item()
    )

    per_concept_f1: list[float] = []
    target_array = concept_targets.detach().cpu().numpy()
    prediction_array = concept_predictions.detach().cpu().numpy()
    for concept_index in range(concept_targets.shape[1]):
        per_concept_f1.append(
            float(
                f1_score(
                    target_array[:, concept_index],
                    prediction_array[:, concept_index],
                    average="macro",
                    zero_division=0,
                )
            )
        )

    return {
        "concept_accuracy": concept_accuracy,
        "concept_macro_f1": float(sum(per_concept_f1) / len(per_concept_f1)),
        "exact_concept_vector_match": exact_match,
    }


def compute_semantic_metrics(
    *,
    label_targets: torch.Tensor | None,
    label_predictions: torch.Tensor | None,
    symbolic_label_predictions: torch.Tensor | None,
    hard_rule_scores: torch.Tensor | None,
) -> dict[str, float]:
    """Compute semantic consistency and rule-based metrics."""

    metrics: dict[str, float] = {}

    if label_predictions is not None and symbolic_label_predictions is not None:
        metrics["concept_label_consistency"] = float(
            (label_predictions == symbolic_label_predictions).float().mean().item()
        )

    if label_targets is not None and hard_rule_scores is not None:
        indices = torch.arange(label_targets.shape[0], device=label_targets.device)
        satisfied = hard_rule_scores[indices, label_targets.long()]
        satisfaction_rate = float(satisfied.float().mean().item())
        metrics["rule_satisfaction_rate"] = satisfaction_rate
        metrics["violation_rate"] = 1.0 - satisfaction_rate

    return metrics


def compute_control_metrics(
    *,
    model: Any,
    num_examples: int,
    num_batches: int,
    evaluation_time_seconds: float,
    seed: int | None = None,
) -> dict[str, float]:
    """Compute lightweight control metrics available during evaluation."""

    parameter_count = 0
    if hasattr(model, "parameters"):
        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if getattr(parameter, "requires_grad", False)
        )

    metrics = {
        "parameter_count": float(parameter_count),
        "num_examples": float(num_examples),
        "num_batches": float(num_batches),
        "evaluation_time_seconds": float(evaluation_time_seconds),
    }
    if seed is not None:
        metrics["seed"] = float(seed)

    return metrics
